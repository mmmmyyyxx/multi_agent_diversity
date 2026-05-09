
# 一、重新总结当前实验暴露的问题

## 1. 最大问题仍然是 batch family judge 的组内同化

实验文件中已经观察到，batch family judge 会把一组 agent 的 traces 一起输入模型，再要求模型输出每个 agent 的 family。实际结果中：

```text
test_epoch1: all_same_primary = 91%, all_same_pair = 89%
test_epoch2: all_same_primary = 92%, all_same_pair = 89%
```

这说明 family judge 对同一题内 agent 之间的差异非常不敏感，甚至可能把整组 traces 概括成同一种策略。这个问题会直接影响 reward、同质化压力、更新选择和后续 textual gradient。

因此，**最优先修改仍然是把 family judge 从 batch judge 改成 per-agent single-trace judge**。

---

## 2. Summary 的问题不是“出现组级表述”，而是“信息量不足”

你说得对：\
简单禁止 summary 里出现 “all agents / peers / group” 这类词并不能真正解决问题。LLM 可能只是换一种说法，而且确实存在所有 agent 推理方法都很相似的情况。

从实验观察看，summary 的主要问题不是组级污染比例高，而是：

```text
summary 平均只有 21.6 words
median 21
p90 26
```

这类 summary 往往只是：

```text
The agent evaluates options...
The reasoning applies algebraic manipulation...
The reasoning decomposes the problem...
```

它们太短、太泛化，难以保留 trace 的语义差异，也难以支撑后续 embedding 或 rewriter。

因此，不应该靠“禁词”解决，而应该改成：

> **由 single-trace judge 生成更详细、更结构化、可回溯的 reasoning profile，其中 reasoning\_summary 本身要尽量保留 trace 的语义信息，长度控制在 512 tokens 内。**

---

## 3. Group diagnosis 不是当前最需要修的地方

你提出去掉 taxonomy-constrained group diagnosis 和后处理校验是合理的。\
虽然日志里确实出现过 `missing_modes` 非法、group diagnosis 频繁把所有 agents 判为 redundant 的现象，但这些问题很大程度来自上游 family judge 和 summary 质量差。

如果先改 group diagnosis 的后处理，可能只是修表面症状。\
更合理的优先级是：

```text
先修 family judge 和 summary profile
再观察 group diagnosis 是否仍然异常
```

因此本轮修改说明中删除原任务 5 和任务 6。

---

## 4. Rewriter 的问题是“泛化口号”，不是“缺少硬约束”

实验里已经出现了这类 prompt：

```text
Broaden your approach by integrating probabilistic, heuristic, and sensory reasoning techniques...
Focus on minimizing redundancy by adopting diverse reasoning strategies...
systematically employ a variety of reasoning styles...
```

这类 prompt 的问题不是不够受约束，而是**没有可执行推理行为**。它不会稳定诱导不同推理轨迹，只是在说“要多样”。

但你说得也对：不能给候选 prompt 加太多硬约束，不能强制某个 agent 在所有任务上都使用某个 family。尤其在 MMLU 上，有些题目的合理解法有限，强行绑定 family 反而会制造不自然推理。

所以 rewriter 应该生成的是：

> **任务无关、可迁移、能诱导不同推理轨迹的通用 reasoning prompt。**

不是：

```text
你必须使用 counterexample_search。
```

而是：

```text
当题目存在多个可疑选项时，优先从边界条件、反例或例外情况检查选项；如果该方法不适用，则退回到概念匹配和选项对比。
```

也就是说，prompt 应该提供**推理倾向和 fallback**，而不是硬绑定 family。

---

## 5. Low-confidence family label 需要用已有审核模型复判

你指出任务 10 需要改成：\
**低 confidence 时调用项目中已有的审核模型重新给出 family label。**

这是更合适的做法。\
实验里已有大量低 confidence 或 confidence=0 的情况，如果这些标签直接进入 reward，会让训练信号不可靠。

因此不应简单降权，而应优先复判：

```text
low confidence family label
→ existing review/audit model
→ revised legal family label
→ 再进入 reward
```

---

# 二、重新整理后的 Copilot 修改说明

下面是新的修改任务清单，已按你的意见删掉不重要或不合适的任务。

---

# Task 1：将 batch family judge 改成 per-agent single-trace judge

## 目标

当前 family judge 一次性看多个 agents 的 traces，容易产生组内同化。需要改为每个 agent 独立判别 family。

## 修改要求

在 `system.py` 中新增或重构：

```python
async def _judge_strategy_family_single(
    self,
    agent_id: int,
    trace: str,
    answer: str,
    question: str,
) -> Dict[str, Any]:
    ...
```

该函数只允许输入：

```text
agent_id
question
single agent trace
single agent answer
valid family taxonomy
task_type
```

禁止输入：

```text
其他 agents 的 trace
其他 agents 的 summary
其他 agents 的 answer
其他 agents 的 family label
```

输出严格 JSON：

```json
{
  "agent_id": 0,
  "primary_family": "option_elimination",
  "secondary_family": "definition_application",
  "reasoning_summary": "...",
  "strategy_steps": ["...", "..."],
  "distinctive_features": ["...", "..."],
  "evidence_spans": ["...", "..."],
  "confidence": 0.78
}
```

原来的 batch 调用改成并发 single judge：

```python
judgments = await asyncio.gather(*[
    self._judge_strategy_family_single(i, traces[i], answers[i], question)
    for i in range(len(traces))
])
```

保留 trace hash cache：

```python
strategy_family_cache[trace_hash] = judgment
```

若 cache 命中，不重复调用 LLM。

---

# Task 2：低 confidence 时调用已有审核模型复判 family label

## 目标

低 confidence family label 不应直接进入 reward。\
在 family judge confidence 低于阈值时，调用项目中已有的审核模型重新判断 family。

## 新增配置

在 `config.py` 中新增：

```python
family_confidence_threshold: float = 0.4
family_rejudge_on_low_confidence: int = 1
```

## 修改逻辑

在 single family judge 之后增加：

```python
if judgment["confidence"] < cfg.family_confidence_threshold:
    judgment = await self._review_or_rejudge_family_label(
        trace=trace,
        answer=answer,
        question=question,
        original_judgment=judgment,
        taxonomy=taxonomy,
    )
```

复判结果必须返回：

```json
{
  "primary_family": "...",
  "secondary_family": "...",
  "reasoning_summary": "...",
  "strategy_steps": [...],
  "distinctive_features": [...],
  "evidence_spans": [...],
  "confidence": 0.0-1.0,
  "source": "review_model_rejudge"
}
```

日志中记录：

```json
{
  "low_confidence_before_rejudge": true,
  "original_confidence": 0.21,
  "rejudged_confidence": 0.74,
  "family_source": "review_model_rejudge"
}
```

不要采用简单降权作为主方案。\
低 confidence 的首选处理是复判。

---

# Task 3：将 reasoning\_summary 升级为详细结构化 reasoning profile

## 目标

当前 summary 太短，不能保留 trace 的语义信息，也不适合后续 embedding。\
需要把 summary 改成结构化 profile，同时让 `reasoning_summary` 更详细，最多 512 tokens。

## 新增数据结构

```python
@dataclass
class ReasoningProfile:
    agent_id: int
    trace_hash: str
    primary_family: str
    secondary_family: str
    reasoning_summary: str
    strategy_steps: List[str]
    distinctive_features: List[str]
    evidence_spans: List[str]
    confidence: float
    source: str
```

## reasoning\_summary 要求

`reasoning_summary` 不能只是 1 句话。\
它应该在 512 tokens 内尽量保留 trace 的语义信息，包括：

```text
该 agent 如何理解问题
它优先关注什么信息
它如何组织推理步骤
它是否逐项比较、反向验证、构造约束、代数推导、估计判断等
它如何处理不确定性
它最终如何收敛到答案
```

推荐 prompt 中加入：

```text
Write a detailed but task-agnostic reasoning summary of the target agent.
Preserve the semantic structure of the trace as much as possible.
Focus on reasoning trajectory rather than final answer.
The summary should be suitable for embedding-based comparison.
Maximum length: 512 tokens.
```

## 输出格式

single family judge 输出：

```json
{
  "agent_id": 0,
  "primary_family": "option_elimination",
  "secondary_family": "definition_application",
  "reasoning_summary": "The agent first identifies the conceptual focus of the question, then compares the answer options against that concept. It eliminates options whose wording conflicts with the expected definition or historical pattern. The reasoning relies on option-level plausibility comparison rather than formal derivation. It finally selects the option that best matches the inferred concept.",
  "strategy_steps": [
    "identify the concept being tested",
    "compare options against the concept",
    "eliminate mismatched options",
    "select the most plausible remaining option"
  ],
  "distinctive_features": [
    "option-level comparison",
    "concept matching",
    "plausibility-based elimination"
  ],
  "evidence_spans": [
    "short excerpt from trace",
    "short excerpt from trace"
  ],
  "confidence": 0.82
}
```

## 日志修改

`reasoning_summary_history.jsonl` 每条记录改为：

```json
{
  "epoch": 1,
  "step": 10,
  "question_hash": "...",
  "agent_id": 0,
  "trace_hash": "...",
  "primary_family": "...",
  "secondary_family": "...",
  "reasoning_summary": "...",
  "summary_token_count": 312,
  "strategy_steps": [...],
  "distinctive_features": [...],
  "evidence_spans": [...],
  "confidence": 0.82,
  "source": "single_trace_judge"
}
```

后续如果使用 summary embedding，应优先使用：

```python
profile["reasoning_summary"]
```

而不是完整 trace。

---

# Task 4：不要用禁词方式处理 summary 组级污染，改用证据支撑与单 trace 隔离

## 目标

不再通过禁止 “all agents / peers / group” 这类词来解决 summary 污染。\
更好的做法是：

```text
输入隔离：summary 生成时只看单条 trace
证据支撑：summary 中的关键判断应能由 evidence_spans 支持
```

## 修改要求

因为 family judge 已经改成 single-trace judge，summary 生成天然不再看到其他 agents。\
不要新增简单的 banned words 过滤。

新增一个轻量质量检查：

```python
def check_summary_support(
    reasoning_summary: str,
    evidence_spans: List[str],
    trace: str,
) -> Dict[str, Any]:
    ...
```

检查：

```text
evidence_spans 是否为空
evidence_spans 是否真的出现在 trace 中
reasoning_summary 是否过短
reasoning_summary 是否明显没有描述推理过程
```

不要因为 summary 和其他 agents 相似就强制改写。\
如果所有 agents 真实推理相同，summary 相似是合理现象。

推荐规则：

```python
if len(reasoning_summary.split()) < min_summary_words:
    regenerate_summary = True

if len(evidence_spans) == 0:
    regenerate_summary = True
```

新增配置：

```python
min_summary_words: int = 60
max_summary_tokens: int = 512
min_evidence_spans: int = 1
```

---

# Task 5：Textual gradient 使用结构化 profile，而不是完整 trace

## 目标

下游 textual gradient 不要使用完整 trace，也不要只用一句泛化 summary。\
它应该使用结构化 reasoning profile。

## 修改输入

`generate_textual_gradient()` 输入改为：

```python
target_profile = {
    "agent_id": i,
    "primary_family": ...,
    "secondary_family": ...,
    "reasoning_summary": ...,
    "strategy_steps": [...],
    "distinctive_features": [...],
    "evidence_spans": [...],
    "confidence": ...,
    "rho_i": ...,
    "invalid_trace_penalty": ...
}

peer_profiles = [
    {
        "agent_id": j,
        "primary_family": ...,
        "secondary_family": ...,
        "reasoning_summary": ...,
        "distinctive_features": [...]
    }
]
```

不传完整 peer traces。

## 输出格式

```json
{
  "diagnosis": "...",
  "redundant_pattern": "...",
  "desired_shift": "...",
  "prompt_edit_instruction": "..."
}
```

注意：\
不要要求 textual gradient 给出“必须绑定某个 family”的指令。\
它应该给出**通用推理轨迹转移方向**，例如：

```text
从直接选项比较转向先识别概念边界；
从单路径计算转向先构造约束再验证；
从直觉判断转向显式处理不确定性和干扰项。
```

---

# Task 6：Rewriter 候选应生成通用、可迁移、能诱导不同轨迹的 prompt

## 目标

避免 rewriter 生成没有操作性的多样性口号，但也不要加入过强约束。\
候选 prompt 应该是**通用推理偏好**，能在多种任务中诱导与 peers 不同的推理轨迹。

## 不要做的事情

不要强制：

```text
每个 candidate 必须绑定 target_family
每个 candidate 必须使用某个固定 family
每个题目都必须使用某种特定方法
candidate 必须在当前题目上改变 family
```

## 要做的事情

候选 prompt 应该包含：

```json
{
  "name": "anti_redundancy_shift",
  "reasoning_bias": "...",
  "trajectory_shift": "...",
  "applicability_condition": "...",
  "fallback_strategy": "...",
  "task_agnostic_prompt": "..."
}
```

其中：

- `reasoning_bias`：通用推理倾向；
- `trajectory_shift`：希望 agent 的推理轨迹如何不同；
- `applicability_condition`：什么情况下优先采用这种方式；
- `fallback_strategy`：如果这种方式不适合当前题目，如何退回到合理方法；
- `task_agnostic_prompt`：最终可写入 agent 的通用 prompt。

## 合理候选示例

```json
{
  "name": "boundary_checking_bias",
  "reasoning_bias": "boundary and exception checking",
  "trajectory_shift": "Instead of immediately comparing options, first identify boundary cases, exceptions, or conditions under which an option would fail.",
  "applicability_condition": "Use this when the problem contains conceptual distinctions, rules, definitions, or options with absolute wording.",
  "fallback_strategy": "If no meaningful boundary case exists, fall back to concise concept matching and option contrast.",
  "task_agnostic_prompt": "When solving a problem, first check whether the candidate answers involve boundaries, exceptions, or overly broad claims. Test these before making a direct selection. If boundary testing is not applicable, use concise concept matching and option comparison."
}
```

另一个例子：

```json
{
  "name": "reverse_validation_bias",
  "reasoning_bias": "answer-to-question backward validation",
  "trajectory_shift": "Start from each plausible answer and ask what must be true for it to explain the question, rather than reasoning only from the question forward.",
  "applicability_condition": "Use this when multiple options appear plausible or differ by subtle conceptual conditions.",
  "fallback_strategy": "If backward validation is not useful, return to direct reasoning with explicit uncertainty checks.",
  "task_agnostic_prompt": "For each plausible answer, briefly reason backward: if this answer were correct, what conditions would need to hold? Compare those required conditions with the problem statement before selecting. If this backward check is not informative, use direct reasoning and state the key uncertainty."
}
```

## 候选校验

新增轻量校验：

```python
def validate_candidate_prompt(candidate, question) -> Tuple[bool, str]:
    ...
```

只检查必要项：

```text
必须是 task-agnostic
不能泄漏当前题目的实体、数字、选项、答案
不能只是“be diverse / use diverse strategies”这类口号
必须包含可执行 reasoning behavior
必须包含 fallback_strategy
```

不要检查：

```text
是否绑定特定 family
是否覆盖某个 taxonomy leaf
是否强制改变当前题目的 family
```

---

# Task 7：候选采纳保留 reward 比较，但增加行为诊断日志

## 目标

不要硬性要求 candidate 改变 family，因为某些 MMLU 题目合理解法有限。\
但需要记录它是否实际改变了 agent 的行为。

## 修改 `evaluate_candidate_minibatch()`

记录：

```python
family_before_agent_i
family_after_agent_i
rho_before_i
rho_after_i
reward_before_i
reward_after_i
invalid_before_i
invalid_after_i
summary_before_i
summary_after_i
```

计算：

```python
family_shift_rate = count(after_family != before_family) / batch_size
rho_reduction = mean(rho_before_i - rho_after_i)
invalid_delta = mean(invalid_after_i - invalid_before_i)
summary_embedding_shift = cosine_distance(summary_before_emb, summary_after_emb)
```

采纳规则保持宽松：

```python
accept if:
    candidate_mean_reward >= current_mean_reward
    and invalid_after_mean <= invalid_before_mean + invalid_tolerance
```

不要强制：

```python
family_shift_rate >= threshold
```

但在 reward 相近时，可以 tie-break：

```python
if abs(candidate_reward - current_reward) < reward_tie_eps:
    prefer candidate with:
        higher summary_embedding_shift
        lower rho_i
        lower invalid penalty
```

新增配置：

```python
reward_tie_eps: float = 0.03
invalid_tolerance: float = 0.1
```

---

# Task 8：保留完整 trace 记录，但下游默认使用 detailed summary

## 目标

完整 trace 很长，不适合全部传给 group diagnosis / textual gradient / rewriter。\
但完整 trace 必须保留用于人工分析、复判和审计。

## 修改要求

继续保存：

```text
trace_history.jsonl
```

下游默认使用：

```text
reasoning_summary
strategy_steps
distinctive_features
evidence_spans
```

只有以下情况允许使用完整 trace：

```text
single family judge
low confidence rejudge
family expansion review
人工分析脚本
```

不要把 group diagnosis、textual gradient、rewriter 全部改成完整 trace。

---

# Task 9：新增 summary embedding 预留接口

## 目标

你之后可能使用 `reasoning_summary` 计算 embedding，因此提前规范接口。

## 新增函数

```python
def build_summary_embedding_text(profile: Dict[str, Any]) -> str:
    """
    Build a <=512-token text for embedding-based reasoning-trajectory comparison.
    Prefer detailed reasoning_summary, then strategy_steps and distinctive_features.
    """
```

逻辑：

```python
text = profile["reasoning_summary"]

if strategy_steps exists:
    text += "\nStrategy steps:\n" + "\n".join(strategy_steps)

if distinctive_features exists:
    text += "\nDistinctive features:\n" + "\n".join(distinctive_features)

truncate_to_512_tokens(text)
```

注意：

```text
embedding text 不包含答案正确性
不包含 gold answer
不包含其他 agents 信息
尽量保留该 agent 推理轨迹语义
```

---

# Task 10：新增实验验收指标

## 目标

修改后需要验证问题是否改善。

在 `train_step_logs.jsonl` 和 `test_epoch*_predictions.jsonl` 中新增：

```json
{
  "all_same_primary": true,
  "all_same_pair": false,
  "primary_dominant_share": 0.8,
  "pair_dominant_share": 0.6,
  "mean_family_confidence": 0.72,
  "low_confidence_share": 0.1,
  "rejudge_count": 3,
  "mean_summary_words": 95,
  "mean_summary_tokens": 130,
  "mean_evidence_spans": 2.1,
  "generic_prompt_candidate_rate": 0.05
}
```

新增或更新分析脚本，输出：

```text
all_same_primary rate
all_same_pair rate
mean confidence
low confidence rate
rejudge count
mean summary length
generic prompt candidate rate
family_shift_rate during candidate eval
summary_embedding_shift during candidate eval
```

验收目标：

```text
all_same_primary 不应长期接近 90%
low confidence label 不应直接进入 reward
summary 平均长度应明显高于旧版 21 words
generic prompt candidate rate 应明显下降
rewriter prompt 应包含可执行 reasoning behavior 和 fallback
```
# 三、最终建议

这次修改的核心不应是“更强约束 rewriter”，而是：

```text
更可靠地识别单个 agent 的推理轨迹
更完整地总结 trace 语义
更少地让 batch 判断抹平差异
更少地产生泛化口号式 prompt
更谨慎地处理 MMLU 这类低多样性可供性的题目
```

你现在的方向应该是：

> **从“强制多样性”转向“可迁移的软分化”。**

也就是让 agent 形成不同的通用推理倾向，而不是让它们在每一道题上都强行走不同 family。
