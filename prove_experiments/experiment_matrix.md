# 实验矩阵

每个实验都是证明链的一部分。重点是让每个结果都可证伪：提前定义什么结果支持指标，什么结果反驳指标。

## P1. Judge 可靠性与标签稳定性

问题：
策略树 judge 对同一条完整 trace 的分类是否稳定？

设计：

- 从 `runs_experiments` 中抽取 100 到 200 条完整 trace，四组设置 `shared_div`、`bank_div`、`shared_baseline`、`bank_baseline` 尽量均衡。
- 每条 trace 用 temperature 0 重判 3 次。
- 如果预算允许，用第二个 critic model 对子集重判。
- 输入只包含完整 trace，不传 prompt、答案正确性、投票结果或其他 agent 的 trace。

对照：

- 同一条 trace 重复输入，应产生接近一致的标签。
- 打乱 trace 顺序，标签不应变化。
- 隐藏实验设置名称，标签不应依赖 run identity。

主要结果：

- primary family 一致率。
- primary+secondary pair 一致率。
- major family 一致率。
- 平均 confidence 和 low-confidence share。

通过标准：

- major family 一致率不低于 0.85。
- primary family 一致率不低于 0.70。
- pair 一致率可以更低，但必须明显高于随机。
- low-confidence case 应集中在真实混合策略或信息不足的 trace 上。

失败解释：

- 如果同一条 trace 标签剧烈波动，指标作为 reward 太受 judge 噪声影响。
- 如果 major 稳定但 leaf 不稳定，训练 reward 应使用 major 或 softened tree similarity，leaf label 只做诊断。

## P2. 同策略改写负对照

问题：
当只是 prompt 措辞不同、实际策略相同时，指标是否避免虚假多样性？

设计：

- 构造 5 个 prompt 变体，它们都要求同一种 broad strategy，例如 option elimination。
- 使用同一个 solver model、同一批问题、同样 temperature。
- 用 5 个 agent 在 100 道题上运行。

同策略 prompt 示例：

- "Solve by eliminating impossible options."
- "Compare choices and remove choices contradicted by the stem."
- "Use option-by-option elimination before answering."
- "Reject distractors first, then choose the remaining option."
- "Check each candidate answer against the problem and discard inconsistent ones."

期望：

- prompt 文本多样性可能很高。
- trace embedding diversity 可能中等。
- strategy-tree diversity 应低或中等，same-major 或 same-leaf overlap 应较高。

通过标准：

- mean family homogeneity 更接近 `shared_baseline`，而不是显式混合策略实验。
- 至少 70% trace 落在目标 major family 或直接相邻 family。

失败解释：

- 如果指标仅因 paraphrase 就报告高多样性，说明它过度依赖表面风格或 judge 关键词。

## P3. 显式策略干预

问题：
当 agent 被明确要求使用不同策略时，实际 trace 和标签是否改变？

设计：

- 使用相同的 100 道均衡 MMLU 题。
- 使用 4 个低成本同级 solver model：`gpt-4o-mini`、`gemini-2.5-flash-lite`、`qwen2.5-7b-instruct`、`deepseek-chat`。
- 每个模型都跑 same 条件和 mixed 条件；same 条件 5 个 agent 都使用同一宽策略，mixed 条件 5 个 agent 分别使用来自 5 个不同 major family 的策略。
- 使用 test-only 模式，不训练。
- 所有主结论默认基于质量控制后的有效 trace；无效或退化 trace 只进入污染风险诊断。

mixed 条件的 5 个目标策略必须属于 5 个不同主类：

| agent | target leaf | target major | 中文解释 |
|---:|---|---|---|
| 0 | `distractor_elimination` | `mmlu_option_semantics` | 逐项排除干扰项，保留最符合题干的选项。 |
| 1 | `rule_or_principle_application` | `mmlu_domain_reasoning` | 先识别领域规则、定理、原则或机制，再应用到题干。 |
| 2 | `decomposition` | `representation_formalization` | 把题干拆成事实、约束和子问题，再合并得到答案。 |
| 3 | `case_analysis` | `logical_proof` | 枚举相关条件、情形或分支并逐一检验。 |
| 4 | `edge_case_analysis` | `optimization_boundary_meta` | 检查边界条件、限定词、例外或极端情形。 |

主要结果：

- mixed 相对 same 的 `team_family_diversity`、`team_major_family_diversity`、`team_family_homogeneity_rate` 配对差异。
- 每个 agent 的 target exact hit、target same-major hit、top primary 分布，用于诊断 prompt 遵循和 judge/taxonomy 偏差。
- 有效 trace 数量和退化 trace 比例，确认结果不是由坏 trace 污染造成。
- GPT-5.5 normal taxonomy judge：给出与正式 judge 尽量相同的信息，重新判断 taxonomy 标签。
- GPT-5.5 prompt-following：只看原始策略指令和 trace，辅助判断模型是否真的遵循 prompt。

通过标准：

- mixed 条件相对 same 条件的 `team_family_diversity` 和 `team_major_family_diversity` 配对差异为正，且 bootstrap/Wilcoxon 检验支持显著提升。
- mixed 条件相对 same 条件的 `team_family_homogeneity_rate` 下降。
- 在只保留有效 trace 后，上述方向仍成立。
- target exact hit 不作为硬性通过标准；它是 leaf 粒度、prompt 可执行性和 judge/taxonomy 吸附的诊断项。
- GPT-5.5 normal taxonomy judge 若系统性不同意原 judge 的 primary label，应优先解释为 taxonomy/judge 诊断风险，而不是直接否定 team-level 多样性结果。

失败解释：

- 如果 prompt 没改变 trace，说明底座模型不听策略指令。
- 如果 trace 改变但 label 不改变，说明 taxonomy 或 judge 漏检。
- 如果 label 改变但 trace 没变，说明 judge 可能被 trace 里的提示性措辞误导。
- 如果只有部分策略难以遵循，说明策略 prompt 可执行性不均衡，应单独重写弱策略 prompt，而不是直接否定整个指标。

## P4. 跨 LLM 策略迁移

问题：
指标测到的是策略 prompt 效应，还是模型身份/输出风格效应？

设计：

- 在四个低成本、同级别 solver model 上重复 same-elimination、same-definition、mixed-strategy 三类 prompt family。
- 推荐组合：`gpt-4o-mini`、`gemini-2.5-flash-lite`、`qwen2.5-7b-instruct`、`deepseek-chat`。
- 不使用太大、太新的模型，避免把模型能力提升误当成策略迁移，也控制 API 成本。
- 主分析固定 critic/judge model 为 `gpt-4o-mini`。
- 对 P3 相关疑点用 GPT-5.5 做子集 audit。

推荐模型表：

| alias | 推荐模型 | 目的 |
|---|---|---|
| `gpt4omini` | `gpt-4o-mini` | OpenAI 低成本基准 |
| `gemini_flash_lite` | `gemini-2.5-flash-lite` | Google 低成本轻量模型 |
| `qwen25_7b` | `qwen2.5-7b-instruct` | 开源 Qwen 系列低成本模型 |
| `deepseek_chat` | `deepseek-chat` | DeepSeek 低成本通用模型 |

不同网关的模型 id 可能不同；实际运行时以 `prove_experiments/p4_low_cost_models.json` 为准。

条件：

- 相同模型，相同 prompt：同一个 run 内 5 agent 的 team 多样性。
- 同一模型，不同策略 prompt。
- 不同模型，同一策略 prompt。
- 不同模型，不同策略 prompt。

期望：

- 同一模型内，不同策略 prompt 应提高策略多样性。
- 不同模型即使使用同一 prompt，也可能因为默认推理风格不同而产生策略树差异。
- trace 风格和长度可能受模型影响；embedding 指标预计比策略树指标更强地反映模型身份。

通过标准：

- P4 不强行要求 prompt 效应大于模型身份效应；它的核心目标是定量分解二者。
- 如果同模型不同 prompt 的距离为正，说明策略 prompt 是有效多样性来源。
- 如果不同模型同 prompt 的距离更大，说明模型身份也会成为多样性来源，跨模型实验不能被直接解释为纯策略差异。
- 策略树指标若比 embedding 指标更少受模型身份影响，则说明它比文本相似度更接近结构化策略差异。

失败解释：

- 如果标签更按模型聚类，而不是按策略聚类，说明 judge 捕获了模型风格而非推理方法。
- 如果 embedding 对模型身份极敏感而策略树相对稳定，说明策略树指标仍有增量价值，但跨模型使用时必须报告模型身份效应。

## P5. Reward 权重 sweep

问题：
策略树 reward 是否太严格，导致难以优化？

设计：

- 从 `shared` 初始化训练 5 个 agents。
- 使用均衡 train/val/test 划分。
- 预算允许时用 seeds `{42, 43, 44}`；先用 seed 42 做 pilot。
- sweep diversity、homogeneity 和 same-major 权重。

推荐网格：

| condition | lambda_diversity | lambda_homogeneity | same_major_family_weight | 目的 |
|---|---:|---:|---:|---|
| no_div | 0.0 | 0.0 | 0.5 | 无多样性 reward 的训练对照 |
| weak | 0.25 | 0.15 | 0.5 | 检查弱约束是否容易提升 |
| default | 0.50 | 0.35 | 0.5 | 当前设置 |
| strong | 0.80 | 0.55 | 0.5 | 过强约束压力测试 |
| softened_tree | 0.50 | 0.35 | 0.7 | 同 major family 内惩罚更轻 |
| strict_tree | 0.50 | 0.35 | 0.25 | 更严格要求 leaf 分离 |

主要结果：

- validation family diversity。
- validation family homogeneity。
- update applied rate。
- candidate family shift rate。
- prompt drift。
- invalid trace penalty。
- early stopping epoch。
- validation-selected best prompts 的最终 test family diversity。

通过标准：

- 至少一个非零 reward 设置在验证集 family diversity 上超过 no_div 和 shared_baseline。
- 提升不应依赖极端 prompt drift 或 invalid trace 增加。
- strong 或 strict 设置表现差不等于指标无效，反而可能说明中等或 softened reward 更合理。

失败解释：

- 如果所有设置都不能提升验证集 diversity，且 candidate shift rate 很低，说明指标太难优化或 rewriter 太弱。
- 如果 diversity 只靠 invalid trace 增加，说明 reward 可被钻空子，必须加强 invalid-trace 约束。

## P6. Taxonomy 粒度敏感性

问题：
当前策略树是太细、太粗，还是刚好？

设计：

- 在同一批 trace 记录上离线重算三种粒度：
  - major-only。
  - 当前 primary+secondary weighted tree。
  - strict leaf-only。
- 尽量不重新跑 solver，只做离线 recompute。

期望：

- major-only 稳定，但可能漏掉同一 major 内的真实差异。
- strict leaf-only 可能更敏感，但更噪声、更难优化。
- 当前 weighted tree 应处在两者之间。

通过标准：

- 当前 weighted tree 的相关性应与 trace embedding/token 指标分开报告；GPT-5.5 盲评更贴近哪一类指标本身就是 P7 的诊断对象。
- 当前 weighted tree 比 strict leaf-only 有更好的优化信号。

失败解释：

- 如果 major-only 最好，说明当前 leaf taxonomy 对 reward 过细。
- 如果 strict leaf-only 最好且稳定，说明当前 same-major smoothing 可能太强。

## P7. GPT-5.5 盲评验证

问题：
独立 GPT-5.5 评估器只看匿名完整 trace 时，是否也认为高指标多样性的组使用了不同方法？

设计：

- 抽样 120 个 question-level trace group，默认从 cleaned runs 的全部可用记录中按极值筛选：
  - 20 个高 trace embedding diversity。
  - 20 个低 trace embedding diversity。
  - 20 个高 strategy-tree diversity。
  - 20 个低 strategy-tree diversity。
  - 20 个高 trace embedding diversity 但低 strategy diversity。
  - 20 个低 trace embedding diversity 但高 strategy diversity，如存在。
- 隐藏 run setting、prompt、model、label 和答案。
- 要求 GPT-5.5 给 1 到 5 的 method diversity 分数、confidence、distinct method count 和可选 coarse method tag。
- 评估 prompt 明确要求忽略措辞、长度、流畅度和答案正确性，只判断推理方法差异。

通过标准：

- strategy-tree diversity、trace embedding diversity、trace token diversity 与 GPT-5.5 method-diversity score 的 Spearman 相关分开报告。
- 如果 GPT-5.5 更贴近 trace embedding/token diversity，应把它解释为“可见 trace 展开差异”的证据，而不是策略树构念的直接胜利。

失败解释：

- 如果 GPT-5.5 盲评判断与指标在常见 case 上系统不一致，应先检查 taxonomy 定义和 judge prompt，再继续训练。
- 如果 GPT-5.5 判断本身不稳定，可增加第二评估模型或重复评估做一致性审计。

## P8. 任务依赖检查

问题：
MMLU 是否天然限制策略空间，导致指标看起来难提升？

设计：

- 按 MMLU subject 和题型切分。
- 与至少一个天然支持多解法的数据集对比，例如 GSM8K、AQuA-RAT、StrategyQA、ARC-Challenge 或 BBH。
- 使用与 P2/P3 相同的干预逻辑。

期望：

- 一些 MMLU 学科会呈现较低 reachable diversity，因为题目本身确实更约束策略。
- 更开放的推理数据集应有更大的 intervention effect。

通过标准：

- 指标能识别受限任务上的低可达多样性，也能识别多方法任务上的高可达多样性。
- 这支持指标的物理意义：它测的是可用推理路径差异，而不是任意制造差别。

