# Method

本文档描述项目的抽象方法。代码入口、参数、日志和运行方式见 `README.md`。

## 1. 问题目标

给定一个任务数据集 $\mathcal{D}=\{(x,y)\}$，训练一组同构 LLM agents，使它们在解决同一问题时形成更分散、更互补的推理策略。这里的“多样性”不是表面措辞不同，而是不同 agents 倾向于使用不同 reasoning family，例如分解、代数推导、分类讨论、反例搜索、反向验证等。

本方法的直接优化目标是降低团队内部策略同质化，而不是直接最大化准确率。准确率仍被记录为评估指标，用于观察多样性训练是否损害任务表现。

当前版本不使用 skeleton / 骨架相似度，也不使用原始文本相似度作为 reward。完整推理轨迹会保存用于人工分析，但 reward 只依赖 reasoning family 分布和基础无效输出惩罚。

## 2. 多智能体生成

设团队包含 $N$ 个 agents。第 $i$ 个 agent 在训练步 $t$ 的 prompt 为 $p_i^{(t)}$。给定问题 $x$，agent 生成推理轨迹 $z_i$ 与答案 $\hat y_i$：

$$
(z_i,\hat y_i)=f_\theta(x,p_i^{(t)})
$$

所有 agents 调用同一底座模型 $f_\theta$，区别仅来自各自 prompt。初始化方式有两种：

- `shared`：所有 agents 从同一个初始 prompt 出发，用来观察训练是否能从完全同质状态中分化。
- `bank`：从人工设计的 prompt bank 中分配不同初始角色，用来观察预设角色是否增强多样性。

## 3. Reasoning Family 判别

每个推理轨迹 $z_i$ 会被映射到一个 reasoning family：

$$
f_i \in \mathcal{F}
$$

当前基础 family 采用两层树结构：7 个主类负责描述推理功能区域，具体方法作为叶子策略参与判别和统计。

| 主类 | 具体方法 |
| --- | --- |
| `representation_formalization` 问题表征与形式化 | `decomposition`, `symbolic_formulation`, `spatial_visualization`, `dimensional_unit_analysis` |
| `algebra_computation` 代数与计算执行 | `algebraic_derivation`, `equation_solving`, `direct_computation`, `combinatorial_counting` |
| `logical_proof` 逻辑推演与证明 | `case_analysis`, `exhaustive_enumeration`, `constraint_propagation`, `option_elimination`, `backward_reasoning`, `consistency_verification`, `counterexample_search`, `proof_by_contradiction`, `invariant_reasoning`, `symmetry_reasoning`, `definition_application`, `rule_based_classification`, `theorem_property_application` |
| `probability_statistics` 概率、期望与统计推理 | `probabilistic_reasoning`, `expected_value_reasoning` |
| `induction_pattern` 归纳与模式推广 | `pattern_generalization`, `inductive_reasoning`, `analogy_mapping`, `comparative_reasoning` |
| `process_structure_simulation` 过程与结构模拟 | `simulation_tracing`, `recursive_reasoning`, `temporal_sequential_reasoning`, `causal_reasoning` |
| `optimization_boundary_meta` 优化、边界与元推理 | `optimization_extremal_reasoning`, `approximation_bounding`, `edge_case_analysis`, `abductive_inference`, `counterfactual_reasoning` |

当模型提出未见过的新策略时，系统会进入“动态扩展”流程：由审核模型判断是否接纳新 family；若接纳则加入集合并写入持久化文件；若拒绝则映射到最接近的旧 family。这样 $\mathcal{F}$ 会随时间扩展为 $\mathcal{F}_{\text{base}}$ 的超集。

审核模型的判断信息仅来自推理轨迹与家族判别信号，主要包括：

- 新标签候选（raw label）；
- 现有 family 列表；
- 基础 family 的定义文本；
- family judge 生成的 `reasoning_summary`；
- family judge 的 reason 与 confidence；
- 是否允许扩展的开关状态。

判别分四层：

1. **Single-trace LLM judge**：critic model 每次只查看一个 agent 的完整 trace，并发地为各 agent 输出 `primary_family`、`secondary_family`、详细 `reasoning_summary`、`strategy_steps`、`distinctive_features`、`evidence_spans`、`confidence` 和 `reason`。该阶段不接收其他 agents 的 trace、summary、answer 或 family label。
2. **低置信复判**：当 `confidence < family_confidence_threshold`，或 summary 过短、缺少证据片段、证据片段无法在 trace 中找到时，系统调用审核模型对该单条 trace 重新判别 family 和 reasoning profile。复判结果以 `family_source="review_model_rejudge"` 进入 reward。
3. **审核与扩展**：审核模型判断未见新标签是否应被接纳。接纳则写入 taxonomy 文件并扩充 $\mathcal{F}$；拒绝则映射到现有 family。关闭 `family_expansion_enabled` 时，未知标签直接本地映射回已有 family，不再调用扩展审核。
4. **规则兜底**：当 LLM 输出缺失、格式异常或无法解析时，用关键词启发式规则估计 family。

`reasoning_summary` 不再是一两句话的短摘，而是一个不超过 `max_summary_tokens` 的详细自然语言 reasoning profile。它应尽量覆盖：agent 如何理解问题、优先关注哪些信息、如何组织中间推理、是否进行选项比较/反向验证/约束构造/代数推导/估计判断、如何处理不确定性，以及最终如何收敛到答案。后续 summary embedding 接口只使用这段自然语言摘要；`strategy_steps`、`distinctive_features`、`evidence_spans` 作为结构化回溯和诊断字段保留，供 group critic、textual gradient 与 rewriter 使用。完整 trace 仍保留在 trace history 中，仅用于 single judge、复判、扩展审核和人工分析。

系统不再设置 `other` 分类。空输出、异常输出或未知标签会被映射到最接近的有效策略，避免无意义类别参与 diversity reward。

这个设计把复杂的推理路径比较转化为较稳定、较低成本的策略族分类问题。

## 4. Family 统计量

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

其中 `macro_diversity_weight` 对应 $\alpha$。$D_{\text{macro}}$ 是 7 个主类分布的归一化熵，$D_{\text{intra}}$ 是各主类内部叶子策略熵的加权平均。

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

因此 $\rho_i$ 不再只是“同 family 数量比例”，而是“平均策略重叠度”。它能区分同主类不同子方法和完全相同子方法。

## 5. 无效轨迹惩罚

为了避免 agents 通过无意义输出制造“伪多样性”，系统定义无效轨迹惩罚 $V_i\in[0,1]$。当前检测包括：

- trace 过短；
- 缺少 `FINAL_ANSWER:`；
- token 数过少；
- bigram 重复比例过高；
- 抽取到的 answer 为空。

该项不是准确率奖励，也不判断答案是否正确；它只约束输出必须像一条有效推理轨迹。

## 6. Reward

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

- $\lambda_f$ 对应 `lambda_diversity`；
- $\lambda_h$ 对应 `lambda_homogeneity`；
- $\lambda_v$ 对应 `lambda_invalid_trace`。

三个参数的物理意义如下：

- $\lambda_f$：策略分化强度。值越大，越鼓励团队覆盖更多 family，同时鼓励个体远离拥挤 family。
- $\lambda_h$：同族碰撞惩罚。值越大，同 family 重叠越难获得高 reward。
- $\lambda_v$：无效多样性惩罚。值越大，格式错误、空输出、重复输出等投机行为越难获得高 reward。

其中 $0.75D_{\text{family}}$ 是团队级多样性项，所有 agents 共享；$0.25(1-\rho_i)$ 是个体新颖性项，鼓励 agent 从拥挤角色中分化出来。

## 7. 窗口级更新触发

系统不在每个样本后都更新 prompt，而是维护一个与 `update_every` 对齐的窗口。每个 agent 在当前样本上的同质化压力为：

$$
P_i=0.85\rho_i+0.15V_i
$$

当 $P_i>0$ 时，该 agent 在窗口内记一次同质化/压力信号。窗口未填满时只统计不更新；窗口填满且当前 step 满足 `step % update_every == 0` 时，系统选择压力最高的 1 到 2 个 agents 进入更新阶段。

这种机制的作用是避免被单个样本的偶然 family 判别牵着走，而是根据最近一批样本中的稳定压力选择更新对象。

## 8. Group Diagnosis

更新前，系统构造窗口级 group context。它包含：

- 当前窗口中同质化较高的样本摘要；
- 其他样本的简短摘要；
- 当前样本的 family 分布、同质化统计和 agent 角色信息；
- 每个 agent 的 compact trace profile 和 family 重叠信息。

Group Diagnosis 是纯多样性诊断链路，不接收 gold answer、各 agent 的预测答案、vote answer 或 vote correctness。投票和答案只保留在 reward 计算、评估指标与回溯日志中，不作为 group critic、textual gradient 或 rewriter 的输入。

Group critic 根据这些信息输出稳定格式的诊断：

```text
group_summary: PATTERN=...;GAP=...;ACTION=...
target_role_hints[agent_id]: ROLE=...;FOCUS=...;AVOID=...
```

其中：

- `PATTERN` 描述团队近期重复出现的策略模式；
- `GAP` 描述缺失或不足的策略角色；
- `ACTION` 给出下一步分化方向；
- `ROLE/FOCUS/AVOID` 为具体 agent 提供重写提示。

## 9. Prompt Rewriting

对被选中的 agent，rewriter 接收：

- 当前 agent 的 prompt；
- 当前 agent 的 trace profile；
- peer agents 的压缩摘要；
- group diagnosis 的稳定子集；
- 目标 agent 的 role hint。

rewriter 生成若干候选 prompt，通常覆盖三类方向：

- `conservative_specialization`：在原角色附近加强专化；
- `coverage_gap_shift`：转向团队缺失的策略；
- `anti_redundancy_shift`：显式避开当前冗余策略。

每个候选不强制绑定某个固定 family，而是给出可迁移的通用推理偏好：

```json
{
  "name": "...",
  "reasoning_bias": "...",
  "trajectory_shift": "...",
  "applicability_condition": "...",
  "fallback_strategy": "...",
  "task_agnostic_prompt": "..."
}
```

候选 prompt 必须包含可执行 reasoning behavior 和 fallback strategy，不能只是“be diverse / avoid redundancy”这类口号。系统会做轻量校验并记录 `generic_prompt_candidate_rate`。所有候选 prompt 都会经过 sanitize，防止泄漏具体题目、答案或样本内容。

## 10. Bandit 选择与采纳

每个 agent 维护一个轻量 bandit policy。动作集合为：

$$
\mathcal{A}=\{\text{keep\_current}\}\cup\{\text{candidate prompts}\}
$$

bandit 根据 softmax preference 采样动作。若采样到候选 prompt，系统会在小批样本上比较：

- 当前 prompt 的 mean reward；
- 候选 prompt 的 mean reward。

系统同时记录行为诊断：`family_shift_rate`、`rho_reduction`、`invalid_delta` 和 `summary_embedding_shift`。采纳规则仍以 reward 比较为主，不强制候选必须改变 family；但候选的 invalid trace penalty 增量不能超过 `invalid_tolerance`。当 reward 差距小于 `reward_tie_eps` 时，系统优先考虑 summary 轨迹变化更大、$\rho_i$ 更低、invalid penalty 更低的候选。

若候选不劣于当前 prompt 且无效轨迹增量可接受，则接受更新；否则拒绝更新。bandit 根据采样动作获得的 reward 做 policy-gradient 风格更新：

$$
\theta_a
\leftarrow
\theta_a
+\eta(r-b)(\mathbb{I}[a]-\pi(a))
$$

其中 $b$ 是移动平均 baseline，$\eta$ 是 `bandit_lr`。

## 11. 训练循环摘要

完整训练过程可以概括为：

1. agents 基于当前 prompts 并行解题；
2. LLM judge 判别 reasoning family；
3. 计算 family 多样性、同质率、无效轨迹惩罚与 reward；
4. 更新窗口压力统计；
5. 到达更新步时选择高压力 agents；
6. group critic 生成窗口级 textual gradient；
7. rewriter 生成候选 prompts；
8. bandit 采样候选并通过小批 reward 对比决定是否采纳；
9. 保存训练日志、更新日志、trace 与状态快照。

## 12. 方法边界

当前方法的优势是成本较低、日志较清晰、reward 信号稳定；主要优化目标是 family-level 策略多样性。

它不直接保证完整推理路径在细粒度结构上不同，因为 skeleton similarity 和语义 pairwise judge 已被移除。如果未来需要更细粒度的轨迹多样性，可以在不改变主循环的情况下替换或增强 family judge，例如加入更可靠的策略子类判别、人工标注校准或低频抽样的语义评估。
