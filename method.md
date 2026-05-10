# Method

本文档描述项目的抽象方法。代码入口、参数、日志和运行方式见 `README.md`。

## 1. 问题目标

给定任务数据集 $\mathcal{D}=\{(x,y)\}$，训练一组同构 LLM agents，使它们在解决同一问题时形成更分散、更互补的推理策略。这里的“多样性”不是表面措辞不同，而是不同 agents 倾向于使用不同 reasoning family，例如分解、代数推导、分类讨论、反例搜索、反向验证等。

本方法的直接优化目标是降低团队内部策略同质化，而不是直接最大化准确率。准确率仍会作为评估指标记录，用于观察多样性训练是否损害任务表现。

## 2. 多智能体生成

设团队包含 $N$ 个 agents。第 $i$ 个 agent 在训练步 $t$ 的 prompt 为 $p_i^{(t)}$。给定问题 $x$，agent 生成推理轨迹 $z_i$ 与答案 $\hat y_i$：

$$
(z_i,\hat y_i)=f_\theta(x,p_i^{(t)})
$$

所有 agents 调用同一底座模型 $f_\theta$，差异只来自各自 prompt。初始化方式有两种：

- `shared`：所有 agents 从同一个初始 prompt 出发。
- `bank`：从人工设计的 prompt bank 中分配不同初始角色。

## 3. Reasoning Family 判别

每条推理轨迹 $z_i$ 会被映射到 reasoning family。当前基础 family 是两层结构：主类描述推理功能区域，叶子策略参与判别和统计。

| 主类 | 叶子策略 |
| --- | --- |
| `representation_formalization` | `decomposition`, `symbolic_formulation`, `spatial_visualization`, `dimensional_unit_analysis` |
| `algebra_computation` | `algebraic_derivation`, `equation_solving`, `direct_computation`, `combinatorial_counting` |
| `logical_proof` | `case_analysis`, `exhaustive_enumeration`, `constraint_propagation`, `option_elimination`, `backward_reasoning`, `consistency_verification`, `counterexample_search`, `proof_by_contradiction`, `invariant_reasoning`, `symmetry_reasoning`, `definition_application`, `rule_based_classification`, `theorem_property_application` |
| `probability_statistics` | `probabilistic_reasoning`, `expected_value_reasoning` |
| `induction_pattern` | `pattern_generalization`, `inductive_reasoning`, `analogy_mapping`, `comparative_reasoning` |
| `process_structure_simulation` | `simulation_tracing`, `recursive_reasoning`, `temporal_sequential_reasoning`, `causal_reasoning` |
| `optimization_boundary_meta` | `optimization_extremal_reasoning`, `approximation_bounding`, `edge_case_analysis`, `abductive_inference`, `counterfactual_reasoning` |

判别流程：

1. **Single-trace LLM judge**：critic model 每次只查看一个 agent 的完整 trace，并输出 `primary_family`、`secondary_family`、详细 `reasoning_summary`、`strategy_steps`、`distinctive_features`、`evidence_spans`、`confidence` 和 `reason`。该阶段不接收其他 agents 的 trace、summary、answer 或 family label。
2. **低置信复判**：当 `confidence < family_confidence_threshold`，或 summary 过短、缺少证据片段、证据片段无法在 trace 中找到时，系统调用审核模型重新判别该单条 trace。
3. **审核与扩展**：当 judge 提出未知新标签时，审核模型判断它是否是可复用的新 family。接纳则写入 taxonomy，拒绝则映射到现有 family。
4. **规则兜底**：当 LLM 输出缺失、格式异常或无法解析时，使用关键词启发式估计 family。

系统不使用 `other` 分类。空输出、异常输出或未知标签会被映射到最接近的有效策略，避免无意义类别参与 diversity reward。

## 4. Reasoning Summary

`reasoning_summary` 是一个不超过 `max_summary_tokens` 的详细自然语言 reasoning profile。它应覆盖 agent 如何理解问题、优先关注哪些信息、如何组织中间推理、是否进行选项比较、反向验证、约束构造、代数推导、估计判断、如何处理不确定性，以及如何收敛到答案。

summary prompt 明确要求不包含“推理很细致/很稳健/很有效”这类质量评价句，因为这些不是推理路径本身。后续 summary embedding 只使用这段自然语言摘要；`strategy_steps`、`distinctive_features`、`evidence_spans` 作为结构化回溯和诊断字段保留。

`max_summary_tokens` 优先通过 `tiktoken` 计算和截断。如果 tokenizer 不可用，则退回单词数近似。日志中的 `summary_token_count` 和 `mean_summary_tokens` 同样优先使用真实 token 数。

## 5. Family 统计量

默认启用主/子策略模式。每条轨迹得到两个叶子策略：

$$
v_i(k)=
\begin{cases}
0.7, & k=\text{primary}_i\\
0.3, & k=\text{secondary}_i\\
0, & \text{otherwise}
\end{cases}
$$

如果主策略和子策略一致，则该策略权重为 1.0。上述 0.7/0.3 可由 `primary_family_weight` 和 `secondary_family_weight` 调整。关闭双策略模式时，每条轨迹退化为单一策略分布。

团队叶子策略分布为：

$$
p_k=\frac{1}{N}\sum_i v_i(k)
$$

团队多样性同时考虑跨主类和主类内分化：

$$
D_{\text{family}}
=
\alpha D_{\text{macro}}+(1-\alpha)D_{\text{intra}}
$$

其中 `macro_diversity_weight` 对应 $\alpha$。$D_{\text{macro}}$ 是主类分布的归一化熵，$D_{\text{intra}}$ 是各主类内部叶子策略熵的加权平均。

同质率使用层级相似度。两个叶子策略完全相同相似度为 1；属于同一主类但不同叶子策略相似度为 `same_major_family_weight`；属于不同主类相似度为 0。两个 agent 的期望相似度为：

$$
sim_{ij}
=
\sum_k\sum_l v_i(k)v_j(l)sim(k,l)
$$

团队同质率和个体重叠比例为：

$$
R_{\text{family}}
=
\frac{2}{N(N-1)}
\sum_{i<j}sim_{ij}
$$

$$
\rho_i
=
\frac{1}{N-1}
\sum_{j\ne i}sim_{ij}
$$

因此 $\rho_i$ 不再只是“同 family 数量比例”，而是平均策略重叠度。

## 6. 无效轨迹惩罚

为了避免 agents 通过无意义输出制造伪多样性，系统定义无效轨迹惩罚 $V_i\in[0,1]$。当前检测包括：

- trace 过短；
- 缺少 `FINAL_ANSWER:`；
- token 数过少；
- bigram 重复比例过高；
- 抽取到的 answer 为空。

该项不是准确率奖励，也不判断答案是否正确；它只约束输出必须像一条有效推理轨迹。

## 7. Reward

第 $i$ 个 agent 的 reward 为：

$$
r_i
=
\lambda_f
\left(
0.75D_{\text{family}}
+0.25(1-\rho_i)
\right)
-\lambda_h\rho_i
-\lambda_vV_i
$$

其中：

- $\lambda_f$ 对应 `lambda_diversity`，表示策略分化强度。
- $\lambda_h$ 对应 `lambda_homogeneity`，表示同质化惩罚强度。
- $\lambda_v$ 对应 `lambda_invalid_trace`，表示无效轨迹惩罚强度。

reward 不直接使用答案正确性，也不使用 skeleton 或原始文本相似度。

## 8. 窗口级更新触发

系统不在每个样本后都更新 prompt，而是维护一个与 `update_every` 对齐的窗口。每个 agent 在当前样本上的同质化压力为：

$$
P_i=0.85\rho_i+0.15V_i
$$

当 $P_i>0$ 时，该 agent 在窗口内记一次同质化/压力信号。窗口未填满时只统计不更新；窗口填满且当前 step 满足 `step % update_every == 0` 时，系统选择压力最高的 1 到 2 个 agents 进入更新阶段。

## 9. Group Diagnosis

更新前，系统构造窗口级 group context。它包含：

- 当前窗口中同质化较高的样本摘要；
- 其他样本的简短摘要；
- 当前样本的 family 分布、同质化统计和 agent 角色信息；
- 每个 agent 的 compact trace profile 和 family 重叠信息。

Group Diagnosis 是纯多样性诊断链路，不接收 gold answer、各 agent 的预测答案、vote answer 或 vote correctness。投票和答案只保留在 reward 计算、评估指标与回溯日志中，不作为 group critic、textual gradient 或 rewriter 的输入。

## 10. Prompt Rewriting

对被选中的 agent，rewriter 接收当前 prompt、当前 trace profile、peer agents 的压缩摘要、group diagnosis 的稳定子集和目标 role hint，生成若干候选 prompt。候选 prompt 必须包含可执行 reasoning behavior 和 fallback strategy，不能只是“be diverse / avoid redundancy”这类口号。

所有候选 prompt 都会经过 sanitize，防止泄漏具体题目、答案或样本内容。

## 11. Bandit 选择与采纳

每个 agent 维护一个轻量 bandit policy。动作集合为：

$$
\mathcal{A}=\{\text{keep\_current}\}\cup\{\text{candidate prompts}\}
$$

bandit 根据 softmax preference 采样动作。若采样到候选 prompt，系统在小批样本上比较当前 prompt 与候选 prompt 的 mean reward，并记录 `family_shift_rate`、`rho_reduction`、`invalid_delta` 和 `summary_embedding_shift`。

若候选不劣于当前 prompt 且无效轨迹增量可接受，则接受更新；否则拒绝更新。bandit 根据采样动作获得的 reward 做 policy-gradient 风格更新：

$$
\theta_a
\leftarrow
\theta_a
+\eta(r-b)(\mathbb{I}[a]-\pi(a))
$$

其中 $b$ 是移动平均 baseline，$\eta$ 是 `bandit_lr`。

## 12. 实验设置

实验脚本包含四个设置：

| 设置 | 初始化 | 训练 | diversity reward |
| --- | --- | --- | --- |
| `shared_div` | shared | 是 | 开启 |
| `bank_div` | bank | 是 | 开启 |
| `shared_baseline` | shared | 否，只测试 | 关闭 |
| `bank_baseline` | bank | 否，只测试 | 关闭 |

`scripts/run_experiments.py` 负责运行四个设置并生成 run 目录；`scripts/analyze_experiments.py` 读取已有结果，统一调用指标计算和画图脚本。

结果分析会额外使用 `BAAI/bge-small-en-v1.5` 计算 embedding cosine similarity/diversity。prompt embedding 使用最终 agent prompt；summary embedding 直接使用 `summary_embedding_text`；trace embedding 使用完整 trace，若 trace 较长则按固定词数切分为多个 chunk，分别编码后进行平均池化，得到单条 trace 的向量表示。所有 embedding 指标统一放入 embedding 可视化图，文本级 trace/summary cosine、family、prompt 和行为指标按语义分别展示。
