# Prove Experiments：策略树多样性指标证明实验

本目录用于设计一组严谨实验，证明“基于策略树分类的多样性指标”是否有效。这里的目标不是证明准确率提升，而是证明两个更核心的问题：

1. 策略树标签及其派生的多样性分数，是否真的测到了推理轨迹中的真实策略差异。
2. 这个指标作为 reward 时，是否约束过强，导致训练或 prompt 搜索很难得到稳定提升。

这组实验采用可证伪的证据链。单个正结果不够，需要同时通过受控干预、负对照、judge 稳定性、GPT-5.5 盲评代理和训练可优化性检查。

## 核心主张

### C1. 构念有效性

当两条 trace 用不同方法解决同一道题时，策略树指标应该给出更低的 pairwise similarity 和更高的 team diversity。

期望证据：

- 显式策略 prompt 会让同一批题上的 primary 或 secondary family 发生预期变化。
- GPT-5.5 盲评代理认为“方法不同”的 trace 组，应能帮助区分策略树多样性与完整 trace 文本展开差异。
- 策略树多样性与 trace embedding 语义差异可以相关，但二者不能混作同一个构念；P7 默认用 `trace_embedding_div` 作为文本多样性口径。

### C2. 区分效度

当 trace 只是措辞、长度或格式不同，但推理方法相同时，策略树指标不应该报告很高的多样性。

期望证据：

- 同策略不同措辞的 prompt，仍然落在相同或相邻 family。
- 同一条 trace 多次重判时标签稳定。
- embedding diversity 可能因为措辞变化而升高，但 strategy-tree diversity 应该保持较低。

### C3. 干预敏感性

当 prompt 明确要求使用不同策略时，生成出的完整 trace 应该真的向对应策略变化。

期望证据：

- 受控策略 prompt 能把各 agent 的标签分布推向目标 family 或同一 major family。
- 变化能在完整 trace 中看到，而不是只出现在 judge 摘要或标签里。
- 这个现象在多个 solver LLM 和多个 MMLU 学科上仍然成立。

### C4. 优化可行性

reward 必须有足够可用的优化信号。如果指标太严格，候选 prompt 很少能改善，早停会很快触发，验证集 diversity 不会提升。

期望证据：

- candidate mini-batch evaluation 能找到 diversity delta 为正且 invalid delta 可接受的 prompt 候选。
- 中等强度 reward 能提升验证集 family diversity 或降低 homogeneity。
- 强 reward 或 strict leaf 设置可以表现较差，但 sweep 应该显示存在可用的中间区域。

## 推荐证据层次

1. 离线 judge 验证。
   不训练，只检查策略标签是否稳定、是否有意义。

2. 受控 prompt 干预。
   显式要求 agent 使用指定策略，检查完整 trace 和 family label 是否真的改变。

3. 跨 LLM trace 对比。
   用同一批策略 prompt 跑不同 solver model，检查指标测到的是策略差异，而不是模型风格。

4. reward 可优化性。
   做小规模训练和 candidate-eval sweep，检查指标是否能被优化，是否会导致 invalid trace 或过度特化。

5. 压力测试与负对照。
   针对常见质疑逐个回应：judge 偏差、关键词泄漏、prompt 表面差异、任务依赖、taxonomy 粒度、reward 过强。

## 现有 baseline 的使用方式

当前 `runs_experiments` 可以作为初步背景，但不能单独作为指标有效性的证明：

- `bank_baseline` 说明人工设计的多角色 prompt 可以产生更高策略多样性。
- `shared_baseline` 提供低 prompt 多样性的参考。
- `shared_div` 和 `bank_div` 说明训练会改变 prompt，但单 seed、少 epoch 不足以证明指标有效或 reward 可优化。
- 现有 `test_epoch*_predictions.jsonl` 和 `test_trace_history.jsonl` 可能存在 `question_hash` 不对齐问题；除非修复对齐，否则不要强行 join。

证明实验应尽量生成同一批题上对齐的 prediction 和 trace 记录。

## 最小运行顺序

为了控制成本，建议按以下顺序运行：

1. `P1_judge_reliability`
2. `P2_same_strategy_negative_control`
3. `P3_explicit_strategy_intervention`
4. `P4_cross_llm_strategy_transfer`
5. `P5_reward_weight_sweep`
6. `P6_taxonomy_granularity_sensitivity`
7. `P7_gpt55_blind_validation`
8. `P8_task_dependence_check`

如果 P1 失败，应先停止。judge 对同一条 trace 都不稳定时，后续训练结果无法解释。

## 结果解释决策树

在修改训练算法前，先按下面逻辑解释结果：

- P1 失败：judge 或 taxonomy 不够可靠。不能用训练结果证明指标有效。
- P1 通过、P2 失败：指标过度敏感于措辞或 judge 关键词，需要先修 judge prompt 和 taxonomy 定义。
- P1/P2 通过、P3 失败且完整 trace 没变：solver model 没有遵循策略指令，这是可控性问题，不是直接证明指标无效。
- P1/P2 通过、P3 中 trace 明显变化但 label 不变：taxonomy 或 judge 漏掉了真实策略差异。
- P1/P2/P3 通过、P5 失败：指标可能作为测量是有效的，但 reward、rewriter 或搜索过程太弱或太严格。
- 只有 `softened_tree` 有效：当前指标可用，但训练 reward 中同 major family 的相似度应放宽。
- 只有 `bank` 初始化有效：人工角色能产生多样性，但从 shared 初始化自动角色分化仍未解决。

## 已实现脚本

- `scripts/rejudge_strategy_traces.py`：P1，同 trace 多次重判，输出 judge 稳定性。
- `scripts/run_strategy_probe.py`：P2/P3/P4，自定义 per-agent prompt 的 test-only probe，支持 solver 和 critic 使用不同 OpenAI-compatible endpoint。
- `scripts/run_p4_cross_llm_matrix.py`：P4，低成本四模型跨 LLM 策略迁移矩阵。
- `scripts/run_prove_reward_sweep.py`：P5，reward 权重 sweep。
- `scripts/analyze_prove_experiments.py`：P2/P3/P4/P5 汇总，包含 P5 candidate 诊断、paired bootstrap CI、Wilcoxon 近似检验和 model identity check。
- `scripts/analyze_taxonomy_granularity.py`：P6，离线重算 major-only、weighted tree、strict leaf-only。
- `scripts/run_gpt_blind_validation.py`：P7，生成盲评包，并调用 GPT-5.5 做独立盲评分数与 Spearman 分析。
- `scripts/prepare_human_blind_validation.py`：P7 备选工具，只生成盲评包或接收人工分数回填。
- `scripts/analyze_task_dependence.py`：P8，按 subject 或数据集分析 reachable strategy diversity。

