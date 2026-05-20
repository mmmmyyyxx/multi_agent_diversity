# 证明实验结果总分析（按实验矩阵）

本文按 `prove_experiments/experiment_matrix.md` 的顺序汇总当前证明实验结果。P5 当前未纳入，因为本轮明确先不跑 reward sweep。本文默认采用已完成质量控制后的可用 trace 作为解释口径；当某些模型仍有较多无效 trace 时，只作为风险说明，不再把历史目录作为对照口径展开。

## 0. 指标中文含义

| 指标 | 中文含义 | 解读方向 |
|---|---|---|
| `family_div` / `mean_family_diversity` | 策略树加权多样性。综合 primary/secondary leaf 与同主类相似度后计算 5 个 agent 的策略分散程度。 | 越高表示团队策略越分散。 |
| `homogeneity` / `mean_family_homogeneity_rate` | 策略同质性。衡量 5 个 agent 是否集中在相同或相近策略。 | 越低越好，表示不容易塌缩到同一策略。 |
| `major_div` / `mean_major_family_diversity` | 主类策略多样性。只看 major family 层面的分散程度。 | 越高表示跨大类策略更多。 |
| `target_exact_hit_rate` | 目标 leaf 精确命中率。自动 judge 的 primary/secondary 是否命中 prompt 指定 leaf。 | 严格，但容易受 leaf 粒度和 `option_contrast` 吸附影响。 |
| `target_same_major_hit_rate` | 目标主类命中率。自动 judge 的 primary/secondary 是否落在目标 leaf 所属 major。 | 比 exact 更稳健。 |
| `vote_acc` | 5 个 agent 多数投票答案准确率。 | 用于确认多样性提升是否牺牲答题效果。 |
| `prompt_embedding_cosine_diversity` | prompt 组本身的完整文本 embedding 多样性。对 P2/P3/P4 的 prompt JSON 里的 5 条策略指令做同样的完整文本 embedding 后计算。 | 作为输入提示词本身的客观参考列。 |
| `trace_embedding_cosine_diversity` | 完整 trace embedding 的平均余弦差异，等于 1 - 平均余弦相似度。 | 本报告只使用完整 trace embedding，不使用 summary embedding。 |
| `major_distribution_distance` | 同一道题两个 team 的 major-family 分布距离。 | 用于 P4 分解 prompt 效应和模型身份效应。 |

后文表格中的 `trace_embedding_div` 都指完整 trace embedding 的 `1 - mean cosine similarity`。口径上，P2/P4 表格使用 `prove_experiments/cleaned_runs` 下同一 run 的 cleaned prediction 与 trace history；P3 主结果使用 `p3_valid_trace_quality` 标出的同一批 valid questions 重新计算 embedding/token；`prompt_embedding_div` 则来自对应 prompt JSON 的 5 条策略指令本身。不要把 cleaned family 指标与原始 `p4_embedding_metrics.csv` 或 all-question P3 embedding 均值混用。

## 1. 总结论

当前结果支持一个谨慎但正向的结论：策略树分类指标确实能测到显式策略 prompt 引起的结构化策略差异，尤其在主类层面更清楚；但它不是纯粹的“真实策略真值”，仍会受到模型身份、输出风格、MMLU 多选题形态和 taxonomy primary 判定规则影响。

最关键的证据链如下：

| 问题 | 当前证据 | 结论 |
|---|---|---|
| judge 是否随机不稳定 | P1 同 trace 重判：major 一致率 0.9700，primary 一致率 0.9533，pair 一致率 0.8617。 | 随机 judge 噪声不是主问题。 |
| 显式不同策略是否提高策略树多样性 | P3 原四模型主结果：有效 trace 口径下 mixed - same 的 `family_div` +0.0900、`major_div` +0.1755、`homogeneity` -0.1213，Wilcoxon p 均小于 1e-11；qwen3.5-plus 同源补跑也保持同方向。 | 策略 prompt 能系统改变 team-level 策略分布。 |
| 指标是否只是在测模型身份/风格 | P4 同模型不同 prompt 距离 0.1939；不同模型同 prompt 距离 0.2920。 | prompt 有效，但模型身份效应更强，不能把跨模型差异都解释成策略差异。 |
| leaf exact 是否过严 | P3 mixed exact 低，GPT-5.5 normal judge 在 776 条复核样本中只有 21.78% 继续支持原 `option_contrast` 主判定，78.22% 质疑原判。 | leaf exact 不能作为唯一有效性证据，major/weighted-tree 更可靠。 |
| GPT-5.5 盲评是否支持策略树 | P7 中 `family_div` 与 GPT-5.5 方法多样性 Spearman 约 0.0075，`major_div` 约 -0.0592，`trace_embedding_div` 约 0.6722，`trace_token_div` 约 0.6829。 | 以 GPT-5.5 作为人类感知参考时，完整 trace embedding/token 多样性更对齐；策略树指标测的是 taxonomy-defined family 分布，不等同于感知到的展开差异。 |
| taxonomy 粒度是否合理 | P6 平均 major-only 0.4060、weighted-tree 0.4735、strict-leaf 0.5875。 | weighted-tree 是折中口径；strict leaf 更敏感但更可能噪声化。 |

## P1. Judge 可靠性

P1 检查同一条 trace 重复交给 judge 时，标签是否稳定。

| trace_count | judgment_count | mean_major_agreement | mean_primary_agreement | mean_pair_agreement | mean_confidence |
|---:|---:|---:|---:|---:|---:|
| 200 | 600 | 0.9700 | 0.9533 | 0.8617 | 0.8433 |

预设通过标准是 major 一致率不低于 0.85，primary 一致率不低于 0.70。当前结果明显通过。因此后续异常更应优先解释为系统性 taxonomy 偏差、prompt 可执行性差异、模型身份差异或 trace 质量问题，而不是随机重判噪声。

```mermaid
flowchart LR
    A[P1 重判稳定] --> B[随机 judge 噪声不是主问题]
    B --> C[后续重点检查系统性偏差]
    C --> D[option_contrast 吸附]
    C --> E[leaf exact 过严]
    C --> F[模型身份/风格混入]
```

## P2. 同策略负对照

P2 的目的不是证明多样性提升，而是确认“同一宽策略的不同措辞”不会被指标误判成强策略多样性。这里把 `family_div`、`major_div`、`homogeneity`、`prompt_embedding_div`、`trace_embedding_div`、`trace_token_div` 和 `vote_acc` 放在同一张表里。P2/P4 表格统一使用 `cleaned_runs` 下同一批 P4 run：`family_div` 来自 cleaned prediction 指标，prompt embedding 来自 prompt JSON，embedding/token 来自同一 run 的 `test_trace_history.jsonl` 完整 trace；我已核对 prediction 与 trace history 的 `question_hash` 集合和 agent family 标签完全一致。

| model | family_div | major_div | homogeneity | prompt_embedding_div | trace_embedding_div | trace_token_div | target exact | target same-major | vote_acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-chat | 0.4786 | 0.4889 | 0.6915 | 0.1506 | 0.0371 | 0.1465 | 0.5680 | 0.5920 | 0.9000 |
| gemini-2.5-flash-lite | 0.4034 | 0.3051 | 0.8256 | 0.1506 | 0.0285 | 0.0767 | 0.7660 | 0.7720 | 0.8000 |
| gpt-4o-mini | 0.4210 | 0.3486 | 0.7618 | 0.1506 | 0.0328 | 0.1102 | 0.7760 | 0.8040 | 0.8400 |
| qwen2.5-7b-instruct | 0.4565 | 0.2851 | 0.7277 | 0.1506 | 0.2087 | 0.4530 | 0.7200 | 0.8180 | 0.7700 |

这个表同时说明两件事：一是 same-elimination 的 target 命中总体仍然高，确实更集中；二是 embedding 多样性并没有把 same-elimination 和 mixed 的结构差异放大到和 family_div 同等的程度，它更多是在反映模型风格和展开方式。qwen 的 embedding 和 token 多样性都明显更高，也再次说明它更像风格离群值，而不是最干净的同策略负对照。

## P3. 显式策略干预

P3 当前正式设计使用 4 个 solver model，每个模型跑 100 道 MMLU；same 条件是同一宽策略的选项排除，mixed 条件是 5 个 agent 分别使用来自 5 个不同主类的策略。后续补跑的 qwen3.5-plus 也是同一套设置，这里作为同源补充一起看，但不改变原四模型主结果的分母。必须先确认这 5 个策略确实属于不同主类：

| agent | target leaf | target major | 中文解释 |
|---|---|---|---|
| 0 | `distractor_elimination` | `mmlu_option_semantics` | 逐项排除干扰项，保留最符合题干的选项。 |
| 1 | `rule_or_principle_application` | `mmlu_domain_reasoning` | 先识别领域规则、定理、原则或机制，再应用到题干。 |
| 2 | `decomposition` | `representation_formalization` | 把题干拆成事实、约束和子问题，再合并得到答案。 |
| 3 | `case_analysis` | `logical_proof` | 枚举相关条件、情形或分支并逐一检验。 |
| 4 | `edge_case_analysis` | `optimization_boundary_meta` | 检查边界条件、限定词、例外或极端情形。 |

### P3 主结果

有效 trace 口径下，mixed 条件在总体上比 same 条件更分散。这里把 `family_div`、`major_div`、`homogeneity`、`prompt_embedding_div`、`trace_embedding_div`、`trace_token_div` 和 `vote_acc` 放在同一张表里，所有 embedding 都只用完整 trace，不用 summary embedding。P3 的 prompt embedding 来自 `same_elimination_mmlu.json` 和 `mixed_strategy_mmlu.json` 的 5 条策略指令；trace embedding/token 列按 `p3_valid_trace_quality/p3_valid_trace_question_rows.csv` 中 `valid_all_agents=1` 的同一批题重算，和 valid `family_div` 保持同一题集口径。

| condition | valid questions | family_div | major_div | homogeneity | prompt_embedding_div | trace_embedding_div | trace_token_div | vote_acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| same | 331 / 400 | 0.4405 | 0.3914 | 0.7336 | 0.1506 | 0.0336 | 0.1137 | 0.8144 |
| mixed | 380 / 400 | 0.5285 | 0.5772 | 0.6019 | 0.1899 | 0.0461 | 0.1490 | 0.8359 |
| mixed - same | 329 paired | +0.0900 | +0.1755 | -0.1213 | +0.0393 | +0.0096 | +0.0237 | +0.0215 |

paired 检验结果：

| metric | paired n | mixed - same | 95% CI | Wilcoxon p |
|---|---:|---:|---:|---:|
| `team_family_diversity` | 329 | +0.0900 | [0.0665, 0.1146] | 6.45e-12 |
| `team_family_homogeneity_rate` | 329 | -0.1213 | [-0.1527, -0.0896] | 1.24e-13 |
| `team_major_family_diversity` | 329 | +0.1755 | [0.1325, 0.2200] | 2.91e-13 |

```mermaid
flowchart LR
    A[same: family_div 0.4405] --> B[mixed: family_div 0.5285]
    C[same: major_div 0.3914] --> D[mixed: major_div 0.5772]
    E[same: homogeneity 0.7336] --> F[mixed: homogeneity 0.6019]
```

按模型拆开看，把 embedding 列放在同一张表里：

| model | condition | valid questions | family_div | major_div | homogeneity | prompt_embedding_div | trace_embedding_div | trace_token_div | vote_acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek-chat | same | 100/100 | 0.4563 | 0.5023 | 0.6625 | 0.1506 | 0.0366 | 0.1479 | 0.9300 |
| deepseek-chat | mixed | 100/100 | 0.5564 | 0.7584 | 0.4745 | 0.1899 | 0.0492 | 0.1854 | 0.9100 |
| gemini-2.5-flash-lite | same | 100/100 | 0.4124 | 0.3162 | 0.8058 | 0.1506 | 0.0270 | 0.0736 | 0.7900 |
| gemini-2.5-flash-lite | mixed | 99/100 | 0.4546 | 0.3902 | 0.7575 | 0.1899 | 0.0329 | 0.0838 | 0.8081 |
| gpt-4o-mini | same | 100/100 | 0.3395 | 0.2156 | 0.8566 | 0.1506 | 0.0281 | 0.0967 | 0.8600 |
| gpt-4o-mini | mixed | 100/100 | 0.4807 | 0.4243 | 0.7157 | 0.1899 | 0.0411 | 0.1234 | 0.8600 |
| qwen2.5-7b-instruct | same | 31/100 | 0.5538 | 0.5313 | 0.6095 | 0.1506 | 0.0627 | 0.1880 | 0.6774 |
| qwen2.5-7b-instruct | mixed | 81/100 | 0.6224 | 0.7362 | 0.4601 | 0.1899 | 0.0647 | 0.2154 | 0.7654 |
| qwen3.5-plus | same | 100/100 | 0.5016 | 0.3982 | 0.7426 | 0.1506 | 0.0409 | 0.1278 | 0.9300 |
| qwen3.5-plus | mixed | 100/100 | 0.5626 | 0.5812 | 0.6020 | 0.1899 | 0.0532 | 0.1378 | 0.9400 |

qwen2.5-7b-instruct 在 same 条件下有效题数只有 31/100，因此它的 P3 same 估计方差很大；而且按当前统一的 valid-trace 口径重算后，它的 `trace_embedding_div` 也并不是负值，而是和 `family_div` 同向上升。先前“family_div 为正但 trace_embedding_div 为负”的印象，主要来自把不同层级、不同汇总口径混看，尤其是把 all-question 与 valid-question 口径错开，再叠加 qwen2.5 same 条件样本过少带来的波动。对照同设置补跑的 qwen3.5-plus，same 与 mixed 两边都保持 `family_div` 和 `trace_embedding_div` 同向变化，更像干净的同源对照，也说明那一处负号更像口径/样本稳定性问题，而不是普遍现象。

### P3 目标策略命中

P3 的关键不是每个 agent 都稳定命中指定 leaf，而是 team-level 策略分布是否随 prompt 改变。目标命中拆解如下：

| condition | agent | target leaf | target major | exact | same-major(any) | top primary | top primary share |
|---|---:|---|---|---:|---:|---|---:|
| mixed | 0 | `distractor_elimination` | `mmlu_option_semantics` | 0.2275 | 0.6900 | `option_contrast` | 0.4550 |
| mixed | 1 | `rule_or_principle_application` | `mmlu_domain_reasoning` | 0.1675 | 0.3400 | `option_contrast` | 0.2825 |
| mixed | 2 | `decomposition` | `representation_formalization` | 0.3725 | 0.3875 | `decomposition` | 0.3700 |
| mixed | 3 | `case_analysis` | `logical_proof` | 0.1175 | 0.1375 | `option_contrast` | 0.4450 |
| mixed | 4 | `edge_case_analysis` | `optimization_boundary_meta` | 0.0525 | 0.0525 | `option_contrast` | 0.4625 |

`distractor_elimination` 的 same-major 命中最高，`decomposition` 的 exact 命中最高；`edge_case_analysis` 最弱。`option_contrast` 仍是很多 mixed trace 的 top primary，说明 MMLU 多选题的可见推理形态容易被 judge 判成选项比较。这是 P3 暴露出的主要风险：策略树分数提升是真实的，但 leaf exact compliance 不能直接解释为“每个 agent 都按指定 leaf 严格执行”。

### P3 GPT-5.5 复核

GPT-5.5 normal taxonomy judge 是更强证据，因为它拿到与正常 judge 接近的信息并重新打 taxonomy 标签。这里把原自动 judge 与 GPT-5.5 rejudge 放到同一张表里，便于直接看 `option_contrast` 吸附有多强。

| judge | primary option | pair option | target exact | target same-major | questioned | confidence |
|---|---:|---:|---:|---:|---:|---:|
| original auto judge | 1.0000 | 1.0000 | 0.0979 | 0.2307 | - | - |
| GPT-5.5 rejudge | 0.2178 | 0.4845 | 0.2088 | 0.2590 | 0.7822 | 0.9169 |

逐策略看时，原自动 judge 与 GPT-5.5 rejudge 的目标策略命中、以及 GPT-5.5 prompt-following 诊断放在同一张表里。`exact` 是目标 leaf 精确命中，`same-major` 是目标主类命中。

| agent | target | normal n | judge exact | GPT exact | judge same-major | GPT same-major | judge primary option | GPT primary option | prompt n | GPT followed | mean score | judge taxonomy likely | model prompt likely | ambiguous |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `distractor_elimination` | 179 | 0.4246 | 0.7877 | 1.0000 | 0.9609 | 1.0000 | 0.2011 | 179 | 0.7598 | 4.0559 | 0.7542 | 0.1508 | 0.0950 |
| 1 | `rule_or_principle_application` | 117 | 0.0000 | 0.1709 | 0.0000 | 0.2222 | 1.0000 | 0.2479 | 117 | 0.6239 | 3.6410 | 0.6154 | 0.2906 | 0.0940 |
| 2 | `decomposition` | 131 | 0.0000 | 0.0076 | 0.0000 | 0.0076 | 1.0000 | 0.2748 | 131 | 0.6870 | 3.8244 | 0.6870 | 0.2519 | 0.0611 |
| 3 | `case_analysis` | 171 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2456 | 171 | 0.7602 | 3.9240 | 0.7602 | 0.1696 | 0.0702 |
| 4 | `edge_case_analysis` | 178 | 0.0000 | 0.0000 | 0.0000 | 0.0112 | 1.0000 | 0.1461 | 178 | 0.5843 | 3.5393 | 0.5843 | 0.3034 | 0.1124 |

更完整的策略 prompt 原文与样本对应关系已另存于 `prove_experiments/prompt_sets.md` 和各子实验的 key/packet 文件中，因此这里不再重复展开。

P3 综合结论：策略树指标确实测到了显式策略 prompt 导致的结构化变化；但 exact leaf hit 过严，并且部分策略 prompt 可执行性不足。对齐到同一批有效题后，`trace_embedding_div` 和 `trace_token_div` 也随 mixed 策略上升，但幅度小于 `family_div` / `major_div`，说明 embedding/token 能看到部分完整 trace 展开差异，而 taxonomy family 指标对显式策略干预更敏感。qwen3.5-plus 的同源补跑把这一点钉得更稳，因为它在 same 和 mixed 两边都保持了与 `family_div` 同向的 `trace_embedding` 变化。

## P4. 跨 LLM 策略迁移

P4 的目标是回答：策略树指标测到的是策略 prompt 效应，还是模型身份/风格效应？因此分成四组对比：

| 对比组 | 含义 | 本实验中的实现 |
|---|---|---|
| 相同模型相同 prompt | 同一个 run 内 5 agent 的 team 多样性。 | 作为基线 team 内多样性。 |
| 相同模型不同 prompt | 同一 solver model 下比较 same-elimination / same-definition / mixed-strategy。 | 估计 prompt 对策略分布的影响。 |
| 不同模型相同 prompt | 四个 solver model 在同一 prompt family 上的差异。 | 估计模型身份/风格效应。 |
| 不同模型不同 prompt | 模型和 prompt 都变。 | 估计混合效应。 |

P4 run 级汇总同样使用 cleaned P4 run 的同源指标；prompt embedding 来自对应 prompt JSON 的 5 条策略指令本身：

| prompt family | family_div | major_div | homogeneity | prompt_embedding_div | trace_embedding_div | trace_token_div | vote_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_elimination | 0.4399 | 0.3569 | 0.7516 | 0.1506 | 0.0863 | 0.2303 | 0.8125 |
| same_definition | 0.4765 | 0.4329 | 0.6975 | 0.1682 | 0.0843 | 0.2229 | 0.8300 |
| mixed_strategy | 0.5043 | 0.4764 | 0.6576 | 0.1899 | 0.0940 | 0.2524 | 0.8125 |

按模型聚合：

| model | family_div | major_div | homogeneity | prompt_embedding_div | trace_embedding_div | trace_token_div |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-chat | 0.5137 | 0.6100 | 0.5793 | 0.1682 | 0.0431 | 0.1674 |
| gemini-2.5-flash-lite | 0.4300 | 0.3455 | 0.8016 | 0.1682 | 0.0295 | 0.0772 |
| gpt-4o-mini | 0.4579 | 0.3595 | 0.7503 | 0.1682 | 0.0413 | 0.1291 |
| qwen2.5-7b-instruct | 0.4926 | 0.3733 | 0.6778 | 0.1682 | 0.1960 | 0.4265 |

四组因子对比使用同题两个 team 的 `major_family_distribution` 距离：

| contrast | unit | n | mean major distribution distance |
|---|---|---:|---:|
| same_model_same_prompt | within-team | 1200 | 0.4221 |
| same_model_different_prompt | between-team same-question | 1200 | 0.1939 |
| different_model_same_prompt | between-team same-question | 1800 | 0.2920 |
| different_model_different_prompt | between-team same-question | 3600 | 0.2998 |

这说明：prompt family 的确改变策略分布，但不同模型在相同 prompt 下的差异更大。换句话说，模型身份可以成为多样性来源，但这也意味着跨模型多样性不能直接当作纯策略多样性。

把这些指标并排看，结论更清楚：family_div 和 major_div 对 prompt family 的变化最敏感；trace_embedding_div 和 trace_token_div 更像模型风格指标，尤其 qwen 显著更高。因此 P4 不把 embedding 另列为旁证，而是作为与 family_div、major_div 同表的客观文本对照。

## P5. Reward 权重 Sweep

P5 尚未运行，因此不能证明“策略树 reward 一定可优化”或“不会过强”。当前能说的只有：

| 待验证问题 | 需要 P5 给出的证据 |
|---|---|
| 指标是否约束太强 | `weak/default/strong/strict/softened_tree` sweep 中是否存在可提升区间。 |
| 提升是否靠坏 trace | diversity 提升是否伴随 invalid trace 增加。 |
| reward 是否可优化 | update applied rate、candidate family shift、validation/test family diversity 是否同步改善。 |
| strict leaf 是否过硬 | strict_tree 是否比 softened_tree 更难优化或更不稳定。 |

在 P5 未完成前，关于“该指标是否约束太强导致很难提升”的结论只能保留为开放问题。P3 说明 prompt 干预能提升指标，但不等同于训练 reward 一定容易优化。

## P6. Taxonomy 粒度敏感性

P6 在同一批 trace 上离线重算三种粒度：

| 粒度 | 平均 diversity | 含义 |
|---|---:|---|
| major-only | 0.4060 | 只看主类，最稳但可能漏掉同主类内部差异。 |
| weighted-tree | 0.4735 | 当前主指标，兼顾 leaf 与 same-major 相似度。 |
| strict leaf-only | 0.5875 | 最敏感，但也最容易把近义 leaf 或 judge 边界误差放大。 |

P6 说明 weighted-tree 是合理折中：它比 major-only 更能捕捉细分策略，比 strict leaf-only 更不容易过度依赖 leaf exact。结合 P3 GPT-5.5 复核，当前最稳的报告口径应是 weighted-tree + major-level，而不是 strict leaf exact。

```mermaid
flowchart LR
    A[major-only: 稳但粗] --> B[weighted-tree: 折中主指标]
    B --> C[strict leaf-only: 敏感但噪声更高]
```

## P7. GPT-5.5 盲评

P7 让 GPT-5.5 不看 model、prompt、label 和自动指标，只看匿名 traces，判断真实方法多样性。

| bucket | n | family_div | major_div | trace_embedding_div | trace_token_div | GPT-5.5 score |
|---|---:|---:|---:|---:|---:|---:|
| high_strategy | 20 | 0.7445 | 0.8697 | 0.1370 | 0.3986 | 1.8000 |
| low_strategy | 20 | 0.2623 | 0.0722 | 0.0870 | 0.2116 | 1.2500 |
| high_text_low_strategy | 20 | 0.2962 | 0.0325 | 0.2058 | 0.5217 | 1.8000 |
| low_text_high_strategy | 20 | 0.6878 | 0.7973 | 0.0278 | 0.0672 | 1.2000 |

| metric | Spearman rho vs GPT-5.5 score | n |
|---|---:|---:|
| family_div | 0.0075 | 80 |
| major_div | -0.0592 | 80 |
| trace_embedding_div | 0.6722 | 80 |
| trace_token_div | 0.6829 | 80 |

这个结果把三类“多样性”分开了：GPT-5.5 盲评作为人类感知参考，最接近完整 trace 的 embedding/token 多样性；family/major 策略树多样性与 GPT-5.5 分数几乎不相关。高策略树组和低策略树组的 GPT-5.5 差值为 -0.0250，95% CI [-0.3000, 0.2750]，也不支持“策略树越高，GPT-5.5 感知到的方法越多样”这一强说法。

因此 P7 的结论不是策略树指标无效，而是指标定位不同：`family_div` / `major_div` 测 taxonomy-defined family 分布，`trace_embedding_div` / `trace_token_div` 更接近读者对完整 trace 展开差异的感知。后续若引入真实人工标注，应同时保留这三列，直接检验三者与人工感知的对齐情况。

## P8. 任务依赖性

P8 检查策略多样性干预是否对不同 MMLU subject 都同样有效。

| dataset | subject_count | paired_subject_count | mean_subject_intervention_effect | positive_subject_rate |
|---|---:|---:|---:|---:|
| mmlu | 212 | 212 | 0.0694 | 0.6368 |

平均看，mixed 相对 same 有正向干预效果；但只有约 63.68% 的 subject 为正，说明可提升空间有明显任务依赖。某些 subject 的题目天然更容易被同一种解题模板覆盖，或者样本太少导致估计不稳定。

P8 的结论是：策略树指标不是在所有任务上都同等可提升。后续如果要做 reward 优化，最好按 subject 或题型分层报告，而不是只给总体均值。

## 结论与风险

当前最稳妥的结论是：

1. 策略树分类指标测到了真实的 prompt-induced 策略分布变化，P3 原四模型主结果给出强正证据，qwen3.5-plus 同源补跑没有改变这个方向。
2. 该指标不应被解释为纯粹的人类感知方法多样性；P7 已经显示 GPT-5.5 盲评更随文本/trace 表面差异变化。
3. leaf exact hit 过严，且 `option_contrast` 存在吸附；应优先报告 weighted-tree 和 major-level 指标。
4. 模型身份是重要多样性来源；P4 显示不同模型同 prompt 的差异大于同模型不同 prompt。
5. P5 尚未运行，因此“reward 是否过强、是否难以提升”仍是开放问题。

后续最有价值的补充是 P5 sweep，以及对 `edge_case_analysis`、`rule_or_principle_application` 这两个弱遵循策略重写 prompt 后复测。
