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

- 使用相同的 100 到 200 道均衡 MMLU 题。
- 使用 5 个 agent，每个 agent 给不同策略 prompt。
- 使用 test-only 模式，不训练。

推荐策略 prompt：

1. 概念或定义匹配。
2. 选项对比与干扰项排除。
3. 从候选答案反向检查题干。
4. 规则、定理或原则应用。
5. 分解题干事实、约束和推论。

主要结果：

- 每个 agent 的 instructed-family hit rate。
- pairwise strategy-tree diversity。
- pairwise full-trace embedding diversity。
- GPT-5.5 盲评代理：完整 trace 是否真的表现出不同方法。

通过标准：

- instructed family 或 same-major hit rate 每个 agent 不低于 0.60。
- mixed-strategy team diversity 明显高于同策略负对照。
- GPT-5.5 盲评分数的 method diversity 与指标在至少 70% 抽样样例上方向一致。

失败解释：

- 如果 prompt 没改变 trace，说明底座模型不听策略指令。
- 如果 trace 改变但 label 不改变，说明 taxonomy 或 judge 漏检。
- 如果 label 改变但 trace 没变，说明 judge 可能被 trace 里的提示性措辞误导。

## P4. 跨 LLM 策略迁移

问题：
指标测到的是策略差异，还是模型身份差异？

设计：

- 在四个低成本、同级别 solver model 上重复 P2 和 P3。
- 推荐组合：`gpt-4o-mini`、Gemini Flash-Lite、Llama 3.1/3.2 8B Instruct、Qwen2.5 7B Instruct。
- 不使用太大、太新的模型，避免把模型能力提升误当成策略迁移，也控制 API 成本。
- 主分析固定 critic/judge model 为 `gpt-4o-mini`。
- 可选：用第二个 critic model 对子集做 audit。

推荐模型表：

| alias | 推荐模型 | 目的 |
|---|---|---|
| `gpt4omini` | `gpt-4o-mini` | OpenAI 低成本基准 |
| `gemini_flash_lite` | `gemini-2.5-flash-lite` | Google 低成本轻量模型 |
| `llama31_8b` | `Meta-Llama-3.1-8B-Instruct` 或网关同级 8B instruct id | 开源 Llama 系列 |
| `qwen25_7b` | `Qwen2.5-7B-Instruct` 或网关同级 7B instruct id | 开源 Qwen 系列 |

不同网关的模型 id 可能不同；实际运行时以 `prove_experiments/p4_low_cost_models.json` 为准。

条件：

- 同一模型，不同策略 prompt。
- 不同模型，同一策略 prompt。
- 不同模型，不同策略 prompt。

期望：

- 同一模型内，不同策略 prompt 应提高策略多样性。
- 不同模型但同一策略 prompt，不应产生比“同模型不同策略”更高的策略多样性。
- trace 风格和长度可能受模型影响，但策略树指标应比原始文本 embedding 更少受模型身份影响。

通过标准：

- strategy prompt 对 family label 的效应量大于 model identity 效应量。
- same-strategy cross-model trace 的 same-major 比例高于 mixed-strategy same-model trace。

失败解释：

- 如果标签更按模型聚类，而不是按策略聚类，说明 judge 捕获了模型风格而非推理方法。

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

- 当前 weighted tree 与 GPT-5.5 盲评 method-diversity 判断的相关性优于或接近其他粒度。
- 当前 weighted tree 比 strict leaf-only 有更好的优化信号。

失败解释：

- 如果 major-only 最好，说明当前 leaf taxonomy 对 reward 过细。
- 如果 strict leaf-only 最好且稳定，说明当前 same-major smoothing 可能太强。

## P7. GPT-5.5 盲评验证

问题：
独立 GPT-5.5 评估器只看匿名完整 trace 时，是否也认为高指标多样性的组使用了不同方法？

设计：

- 抽样 80 个 question-level trace group：
  - 20 个高 strategy-tree diversity。
  - 20 个低 strategy-tree diversity。
  - 20 个高 embedding diversity 但低 strategy diversity。
  - 20 个低 embedding diversity 但高 strategy diversity，如存在。
- 隐藏 run setting、prompt、model、label 和答案。
- 要求 GPT-5.5 给 1 到 5 的 method diversity 分数、confidence、distinct method count 和可选 coarse method tag。
- 评估 prompt 明确要求忽略措辞、长度、流畅度和答案正确性，只判断推理方法差异。

通过标准：

- strategy-tree diversity 与 GPT-5.5 method-diversity score 有正 Spearman 相关。
- strategy-tree diversity 比 raw trace embedding 更好地区分 GPT-5.5 高/低多样性组。

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

