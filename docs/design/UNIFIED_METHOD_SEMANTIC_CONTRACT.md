# Unified Team Prompt Search：方法语义契约

本契约用于固定当前研究方法的**科学含义、模块职责和实验解释边界**。其目的不是规定具体代码实现，而是防止后续实验迭代、重构或 Codex 实现过程中，模块虽然名字不变，但实际回答的问题发生漂移。

---

## 1. 总体研究目标

本工作的目标不是最大化多智能体之间的通用 diversity，也不是让每个成员独立性能都尽可能高。

研究目标是：

> **在固定团队聚合机制下，根据当前团队失败结构，把有限的 prompt optimization 机会优先分配给更有责任、也更可能通过更新产生团队收益的成员，并进一步通过细粒度错误模式指导局部 prompt search，使各成员形成对团队有用的能力补充。**

因此核心对象始终是：

\[
\boxed{\text{Team Performance}}
\]

成员级责任、Pattern、局部性能和 Memory 都是为了改善团队搜索过程，而不是独立优化目标。

方法可概括为：

\[
\text{Team Failure}
\rightarrow
\text{Member Responsibility}
\rightarrow
\text{Opportunity Allocation}
\rightarrow
\text{Focused Local Optimization}
\rightarrow
\text{Team-Level Selection}
\rightarrow
\text{Experience Accumulation}
\]

---

# 2. Responsibility 的含义：它首先是“优化机会分配信号”

Responsibility 模块回答的问题是：

> **当前团队状态下，把一次有限的优化机会分配给哪个成员，更可能修复团队仍未解决的问题？**

它不是：

- 单纯衡量哪个成员 accuracy 最低；
- 给每个成员公平分配同样多的优化次数；
- 强制每个 epoch 每个成员 exactly once；
- 给成员贴一个长期固定的“角色标签”。

当前责任信号可以继续采用已有形式：

\[
V_i=\max(4D_i,2N_i,C_i)
\]

其中 \(D_i,N_i,C_i\) 表示成员 \(i\) 在当前团队状态下具有的不同等级的合法 team-repair opportunities。

Responsibility 的本质是：

\[
\boxed{
\text{当前状态下，该成员拥有多少有价值的团队修复机会}
}
\]

而不是该成员“有多差”。

---

# 3. Opportunity Allocation：允许不均匀，而且不均匀本身可能正是机制效果

本方法**不要求**所有成员获得相同数量的 optimization opportunities。

也不要求：

\[
n_1=n_2=\cdots=n_K
\]

或者：

> 每个 team epoch 每个 eligible member exactly once。

允许某些成员因为当前责任长期较高而获得明显更多机会。

例如：

```text
Agent 1: 2 opportunities
Agent 2: 9 opportunities
Agent 3: 3 opportunities
Agent 4: 1 opportunity
Agent 5: 6 opportunities
```

这种分配本身不是问题。

真正需要验证的是这种非均匀分配是否**有价值**。

---

# 4. Responsibility-guided allocation 的核心有效性定义

责任分配机制“有效”至少包含三个相互关联但不同的含义。

## 4.1 Allocation alignment

责任更高的成员确实应该倾向获得更多优化机会：

\[
V_i > V_j
\Rightarrow
P(i\text{ receives next opportunity})
>
P(j\text{ receives next opportunity})
\]

这是最基础的机制一致性：

> scheduler 确实按照 responsibility 配置搜索资源。

---

## 4.2 Opportunity-to-team-gain conversion

如果某成员获得很多机会，这些机会应该能够转化为实际团队收益。

不能出现：

> 一个成员因为责任长期最高，占用了大多数优化机会，但这些机会最终几乎没有改善团队。

所以需要关注：

\[
\text{Opportunity Count}_i
\]

与：

\[
\text{Realized Team Gain}_i
\]

之间的关系。

高 opportunity share 本身不是成功。

它必须对应合理的：

\[
\text{Team Gain Share}
\]

或至少较好的 opportunity-to-gain conversion。

---

## 4.3 Counterfactual allocation value

更强的 claim 是：

> 在相同 parent team、相同搜索预算和相同其他条件下，把优化机会给责任更高的成员，平均比把机会给责任较低的成员产生更大的团队收益。

即希望观察：

\[
E[\Delta Team\mid target=i]
>
E[\Delta Team\mid target=j]
\]

当：

\[
V_i>V_j
\]

时。

这才对应 responsibility 作为 **resource-allocation signal** 的真正科学价值。

---

# 5. Failure discount 的语义保持现状

当前已有的 failure-discount target score：

\[
\frac{V_i}{1+f_i}
\]

本轮方法契约**不重新定义、不修改，也暂不进一步讨论其替代方案**。

它属于当前 responsibility-based target-selection implementation 的既有组成。

本契约需要固定的是：

> Responsibility 机制最终要通过“机会是否被合理分配，以及这些机会是否产生团队收益”来验证。

而不是在此阶段重新设计 \(f_i\) 的具体定义。

---

# 6. “所有成员有机会”的含义

“所有成员都应该有机会”不等于强制 round-robin。

它的研究含义是：

> 搜索过程不能因为责任分配机制造成未经验证的永久成员排除，从而无法判断其他成员是否也具有可利用的更新价值。

但在成员已经被充分暴露之后，后续 opportunity distribution 完全可以高度不均匀。

因此：

\[
\boxed{
\text{Opportunity equality is not an objective}
}
\]

核心仍是：

\[
\boxed{
\text{Opportunity allocation efficiency}
}
\]

---

# 7. Pattern 的真实角色：把粗粒度 responsibility 转成更细粒度 optimization gradient

Responsibility 回答：

> **优化谁？**

Pattern 回答：

> **这一次具体应该围绕这个成员的哪一种失败机制优化？**

二者不能混淆。

流程是：

\[
\text{Responsibility}
\rightarrow
\text{Target Member}
\rightarrow
\text{Pattern Analysis}
\rightarrow
\text{Focused Optimization}
\]

Pattern 不参与 target selection。

它是在 target 已经确定之后，对该成员当前 residual failures 做更精细的机制级分解。

---

# 8. Pattern 不是一个“是否允许优化”的前置门槛

这里必须防止一个重要语义漂移。

本方法不是：

```text
先分析 Pattern
    ↓
Pattern 足够集中？
    ↓
YES → 用 Pattern
NO  → 不用 Pattern / 不优化
```

Pattern concentration 不是 intervention gate。

真实原意是：

> 无论当前错误是否高度集中，都尽量识别若干不同 failure mechanisms；但是一次 local optimization 只针对其中一个 pattern。

因此：

\[
\boxed{
\text{one optimization opportunity}
\rightarrow
\text{one focus pattern}
}
\]

---

# 9. 为什么一次只优化一个 Pattern

如果当前 residuals 包含：

```text
Pattern A
Pattern B
Pattern C
```

不希望把：

```text
A + B + C
```

同时交给 optimizer，并要求 LLM 一次 rewrite 同时修复所有机制。

原因是这种 mixed gradient 容易使 LLM：

- 总结出过于宽泛的规则；
- 无法针对任何一个机制形成清晰修改；
- 引入 collateral regression；
- 出现“看似综合、实际什么都没解决”的 prompt rewrite。

因此本方法刻意把更新 gradient 收窄：

```text
Multiple observed patterns
          ↓
Select ONE focus pattern
          ↓
Focused evidence
          ↓
GEPA reflection / mutation
```

核心原则是：

\[
\boxed{
\text{one coherent failure mechanism per local update}
}
\]

---

# 10. Pattern 不集中时仍然只选一个 Pattern

即使 residual failure structure 很混合，例如：

```text
Pattern A: 30%
Pattern B: 25%
Pattern C: 20%
Others/unassigned: 25%
```

仍然不应该把多个 pattern 一次性交给 optimizer。

仍然选择一个 focus pattern。

可以是当前定义下与 responsibility primary lane 最相关的主要 pattern，也可以在机制实验中与其他选择策略比较。

但主方法的语义始终是：

> **低 concentration 改变的是“当前错误结构有多集中”的诊断结论，而不是“一次 update 是否只针对一个 pattern”的原则。**

---

# 11. Dominant Pattern 与 Focus Pattern 应概念区分

“dominant pattern”容易让人误解成：

> 所有 residual 中数量绝对最多的 pattern。

但本方法真正需要的是：

> 当前 responsibility context 下，这一次最值得作为 optimization gradient 的 pattern。

因此概念上更准确的是：

\[
\boxed{\text{Focus Pattern}}
\]

Focus Pattern 可以综合：

- 与 primary responsibility lane 的重叠；
- support 数量；
- pattern confidence；
- 稳定 tie-breaking。

它不必严格等于全局最大簇。

---

# 12. Pattern concentration 指标的语义必须明确

如果报告：

\[
DPR
\]

必须明确分母是谁。

研究真正关心的是：

\[
\boxed{
DPR_{all}
=
\frac{
|\text{focus-pattern support}|
}{
|\text{all target legal residuals}|
}
}
\]

这个指标回答：

> 当前 target 的全部 legal residuals 中，有多大比例真正由 focus pattern 解释？

与此同时可以另外存在：

\[
ConditionalDPR
=
\frac{
|\text{focus support}|
}{
|\text{all pattern-assigned residuals}|
}
\]

它回答的是另一个问题：

> 在 Pattern Analyzer 成功解释的那些 residual 中，focus pattern 有多占优势？

二者不能混用。

同样，Pattern Coverage 应理解为：

\[
Coverage
=
\frac{
|\text{pattern-assigned residuals}|
}{
|\text{all target legal residuals}|
}
\]

因此：

```text
Coverage
DPR_all
ConditionalDPR
Entropy
```

描述的是不同性质。

尤其不能因为：

\[
ConditionalDPR
\]

很高，就直接声称：

> 全部 residual 高度集中。

---

# 13. Pattern 的 evidence 原则

一次 Pattern-guided update 中，真正用于 repair gradient 的 evidence 应围绕**一个 focus pattern**。

可以同时存在：

- focus-pattern support；
- focus-pattern counterexamples；
- focus-pattern risks；
- preservation evidence；
- transition anchors/focus；
- 必要的 safety/team-hard context。

但这些额外样本的作用是：

\[
\text{preservation / boundary / safety}
\]

而不是引入第二个独立 repair mechanism。

因此必须避免把：

> “为了满足 evidence 数量”

误解成：

> “可以重新混入其他 pattern 的 repair examples。”

一次 update 的 repair target 仍然只有一个 coherent mechanism。

---

# 14. GEPA 的角色保持清楚

GEPA 不是本工作的责任分配模块，也不是 Pattern Analyzer。

它回答的是：

> 给定 target member、focused evidence 和 optimizer context，具体怎样产生更好的 prompt revision？

因此职责边界是：

```text
Responsibility → WHO
Pattern        → WHAT
GEPA           → HOW
```

GEPA 内部仍可保留：

- reflective mutation；
- Pareto/frontier；
- parent selection；
- local acceptance；
- lineage。

本工作的重点不要求重新发明单 prompt optimizer。

---

# 15. Local search 与 Team deployment 必须分离

本方法坚持：

\[
\boxed{
\text{Local usefulness}
\neq
\text{Team usefulness}
}
\]

GEPA local acceptance/rejection 主要服务于：

> local search trajectory 和 lineage。

真正决定 prompt 能否进入团队的是：

> team-level evaluation。

因此一个局部 rejected proposal 仍可能：

```text
local reject
    ↓
team-level evaluation
    ↓
team improvement
    ↓
commit
```

这种情况并不矛盾。

它恰恰体现：

> 单成员 local objective 与 multi-agent team objective 并非完全一致。

---

# 16. 当前阶段的实验优先级：先证明整体有效，再拆机制

当前研究阶段不要求所有 V2 内部机制立刻获得单独 causal attribution。

即使 V2 同时包含：

- candidate exposure；
- variable evidence；
- variable feasibility；
- decoupled local/team admission；

当前首要问题仍可以是：

\[
\boxed{
\text{整体方法是否能稳定产生 team improvement？}
}
\]

如果整体无效，没有必要过早把实验矩阵拆得非常细。

如果整体有效，再分别验证：

- responsibility allocation；
- focused Pattern；
- candidate exposure；
- Memory；
- shared risk；

各自贡献。

因此：

> “当前几个机制被 bundle 在 V2 中”

不等于当前实验设计错误。

它只是意味着：

> **整体结果不能被错误解释成某一个子机制已经被独立证明。**

---

# 17. Team-level transition：允许有价值的 specialization

这是另一个必须固定的重要语义。

研究目标不是要求：

\[
A_i^{t+1}\ge A_i^t
\]

即每个成员性能随每一步都严格单调不降。

这种限制会阻止真正的 role specialization。

允许：

```text
target member:
72 → 68

team:
82 → 87
```

只要这个成员没有跌破最初的基本能力底线。

---

# 18. 成员能力约束的真实含义

期望保持：

\[
\boxed{
A_i^t\ge A_i^0
}
\]

其中：

\[
A_i^0
\]

是初始团队中成员 \(i\) 的能力。

而不是要求：

\[
A_i^t\ge A_i^{t-1}
\]

因此，一个成员之前可能已经从：

\[
65\rightarrow72
\]

积累了一部分额外 competence。

后来为了获得更有价值的 complementary role，可以：

\[
72\rightarrow68
\]

只要：

\[
68\ge65
\]

并且团队因此变得更好。

---

# 19. Specialization 不是允许任意牺牲成员

允许 specialization 不等于：

> 只要 team accuracy 上升，就可以把某个成员性能降得很低。

成员仍然有初始能力底线：

\[
A_i^{candidate}\ge A_i^{initial}
\]

这保证：

> specialization 是在保留基础通用能力之后形成的任务分工，而不是把某个 agent 牺牲成一个极端 narrow specialist。

因此研究目标更准确是：

\[
\boxed{
\text{competence-preserving specialization}
}
\]

其中 preservation 相对于**initial competence floor**，而不是每一步 incumbent。

---

# 20. Team gain 与 specialization 的关系

如果 candidate 相对当前 member performance 有下降：

\[
A_i^{candidate}<A_i^{current}
\]

那么这种下降只有在换来实际团队收益时才有研究意义：

\[
Team^{candidate}>Team^{current}
\]

否则只是退化，不是 specialization。

因此：

> member regression relative to incumbent 本身并不是目标；它只是可能被允许的 trade-off。

真正目标始终是：

\[
\boxed{\text{useful team-level complementarity}}
\]

---

# 21. Memory 的研究目标不是“保存历史记录”

Memory 不是为了让系统知道：

```text
第 3 轮优化过 Agent 2
第 4 轮修了 3 个样本
第 5 轮被 Shadow 拒绝
```

这种信息属于 History / telemetry。

Memory 应回答：

> **过去的优化过程产生了什么可以迁移到未来搜索中的经验？**

因此：

\[
\boxed{
History \neq Memory
}
\]

History 记录事实。

Memory 抽取经验。

---

# 22. Memory 的基本单元应该是 Experience

一条 Memory 的语义应接近：

\[
\boxed{
Situation
\rightarrow
Action
\rightarrow
Outcome
\rightarrow
Lesson
}
\]

### Situation

当时是什么 failure mechanism / responsibility context。

### Action

采取了什么抽象 prompt-update strategy。

### Outcome

对 target 和 team 产生了什么结果。

### Lesson

以后遇到类似情境时，有什么可复用的原则。

---

# 23. Private Success Memory 的原意

Private Memory 主要保存：

> 这个成员自己过去成功优化时学到的可迁移经验。

例如：

```text
Situation:
multiple plausible referents survive local lexical cues

Action:
enumerate candidate referents first, then check global consistency

Outcome:
repaired several team residuals without losing initial competence

Lesson:
for ambiguous-reference cases, delay commitment until all
candidate referents have been tested against the full sentence
```

这才是真正的 optimization experience。

不是单纯：

```text
pattern_hash = ...
fixed = 4
broken = 0
success = true
```

后者只能作为 provenance / telemetry。

---

# 24. Shared Risk Memory 的原意

Shared Memory 不主要传播“其他 agent 成功学到了什么”。

这样容易导致所有成员吸收同样的 positive strategy，反而趋于 homogenization。

Shared Memory 更适合传播：

\[
\boxed{\text{structural risks / failed strategies}}
\]

例如：

> 某种过于宽泛的 rewrite 在另一个成员上导致了 TeamProbe regression。

于是其他成员以后可以避免重复踩相同的坑。

因此基本信息流是：

```text
Positive experience
→ target-private

Structural failure / risk
→ team-shared
```

这与“让成员形成不同能力，但共享安全知识”的目标一致。

---

# 25. Memory 需要保存“抽象经验”，而不是 prompt 或题目本身

Memory 不应该成为：

- raw prompt archive；
- raw question archive；
- gold-answer store；
- hidden reasoning store；
- example-copy mechanism。

应保留的是抽象后的：

```text
failure mechanism
edit strategy
applicability condition
success principle
avoidance rule
outcome summary
```

即：

\[
\boxed{\text{generalizable optimization knowledge}}
\]

而不是：

\[
\text{past optimization log}
\]

---

# 26. Memory 与 Pattern 的关系

Pattern 描述：

> **当前**遇到的 failure mechanism。

Memory 描述：

> **过去**遇到类似 mechanism 时，什么样的 update strategy 有效或危险。

因此可以形成：

```text
Current Focus Pattern
        ↓
Retrieve relevant past experience
        ↓
GEPA reflection
```

但 Memory 不能反过来改变：

- responsibility attribution；
- target member selection。

它服务的是 local search，而不是 WHO。

---

# 27. Pattern identity 与 Memory identity 的语义

Memory 如果要说：

> “当前情况和过去是同一种 Pattern”

这里的 “same pattern” 应指：

\[
\text{same or sufficiently equivalent failure mechanism}
\]

而不是简单：

> provider 恰好生成了相同的 `pattern_id` 字符串。

因此 `pattern_id` 在科学语义上只是标识符。

真正的 Pattern identity 应来自其 mechanism / corrective principle 等结构语义。

后续任何实现不能把：

> 名字相同

直接等同于：

> 机制相同。

---

# 28. 整个方法最终想形成的搜索动态

理想动态不是：

```text
所有成员一起变成更强的通用 solver
```

而是：

```text
Current Team Errors
        ↓
Responsibility identifies where repair leverage lies
        ↓
More useful opportunities flow toward those members
        ↓
Pattern isolates one coherent deficiency at a time
        ↓
GEPA produces a focused revision
        ↓
Team selection keeps revisions useful to the ensemble
        ↓
Members gradually acquire complementary strengths
        ↓
Memory reduces repeated search and repeated mistakes
```

因此最终希望出现的是：

\[
\boxed{
\text{task-aligned differentiation}
}
\]

而不是人为最大化文本差异或行为 diversity。

---

# 29. 实验中必须避免的几个错误解释

### 错误解释 1

> “某 member 获得最多 opportunities，所以 responsibility allocation 有效。”

不成立。

还需要看它是否转化为 team gain。

---

### 错误解释 2

> “所有成员 opportunities 很均匀，所以 scheduler 很公平。”

公平不是目标。

如果 responsibility 不均匀，均匀反而可能说明 scheduler 没有发挥作用。

---

### 错误解释 3

> “DPR 很高，所以全部 residual 高度集中。”

必须先确认 DPR 的分母是不是 all legal residuals。

---

### 错误解释 4

> “Pattern 很混合，所以这一轮不应该使用 Pattern。”

不对。

即使混合，也应该只选择一个 focus pattern，避免 mixed gradient。

---

### 错误解释 5

> “Pattern Analyzer 输出了三个 patterns，所以 GEPA 应该一起修三个。”

不对。

一次 local update 原则上只围绕一个 focus mechanism。

---

### 错误解释 6

> “target member 相比上一步下降，所以 candidate 一定不好。”

不对。

只要仍高于 initial competence floor，并且产生真实 team gain，这可能是有价值 specialization。

---

### 错误解释 7

> “Memory 写入了很多 entries，所以 Memory 有效。”

不成立。

真正的问题是这些 entries 是否包含可复用经验，以及后续读取是否提高搜索效率/成功率或减少重复失败。

---

### 错误解释 8

> “V2 比 V1 好，所以 candidate exposure 已被独立证明。”

不成立。

当前阶段可以先证明整体有效，但 bundled method 的结果不能自动归因到单一子机制。

---

# 30. 当前核心科学问题

整个方法可以最终归纳成三个主问题。

## Q1. Responsibility-guided Resource Allocation

> 能否根据当前团队失败结构，判断把下一单位 optimization budget 花在哪个成员上最有价值？

验证重点：

\[
Responsibility
\rightarrow
Opportunity Allocation
\rightarrow
Realized Team Gain
\]

---

## Q2. Pattern-focused Local Optimization

> 在选定 target 后，把 heterogeneous residuals 分解为 failure mechanisms，并一次只针对一个 coherent pattern 优化，是否比混合梯度更容易产生有效且低 collateral 的 update？

验证重点：

\[
Focused\ Gradient
\rightarrow
Repair
-
Collateral
\]

---

## Q3. Experience-guided Search

> 能否从过去实际 optimization trajectory 中抽取可迁移经验，让未来 search 少走重复路径、减少重复失败并提高有效 update 的产生效率？

验证重点：

\[
Past\ Experience
\rightarrow
Future\ Search\ Efficiency
\]

---

# 31. 方法边界

以下不属于本工作的当前核心目标：

- weighted voting；
- learned router；
- aggregation optimization；
- generic diversity maximization；
- 强制 role assignment；
- 每个成员相同 optimization budget；
- 每步所有成员性能单调上升；
- 自己重新发明一个完整 single-prompt optimizer；
- 一开始就独立证明每一个 V2 子机制的 causal contribution。

这些都不应在后续实现中悄悄进入主方法。

---

# 32. 最终一句话定义

> **Unified Team Prompt Search uses member-level responsibility to allocate limited optimization opportunities toward members with greater potential to repair current team failures; for each selected member, it isolates a single coherent failure pattern to provide a focused optimization gradient to a replaceable local prompt optimizer; team-level evaluation then preserves only revisions that improve the ensemble while allowing specialization above an initial competence floor; successful and failed optimization trajectories are finally distilled into reusable private experience and shared structural-risk knowledge for future search.**

中文可以概括为：

> **本方法首先根据当前团队失败结构判断有限优化预算应该优先花在哪个成员上，再将该成员的错误进一步分解为细粒度机制，并在一次更新中只围绕一个 focus pattern 进行局部 Prompt Search；最终由团队收益而非局部优化器决定是否部署，同时允许成员在不跌破初始能力的前提下形成有价值的 specialization，并将真实优化过程抽象为可供后续搜索复用的个人经验和共享风险知识。**