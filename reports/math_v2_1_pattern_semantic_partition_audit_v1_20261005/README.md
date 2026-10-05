# A4 Canary Pattern semantic partition audit

本次零 API 审计没有找到可以确认为同一可修复失败机制的重复 pair，也没有证明七个 singleton 对应七种不同的真实根因。它们来自 Pattern 原始返回；V3 guard 没有拆簇。现有证据支持继续准备原方法的 A4 开发 Pilot，暂不修改算法或提示词。准备、冻结和真实执行均尚未开展，API 与各数据访问范围需要另行授权。

审计对象是 attempt4、A4、Seed81 的一次完整 opportunity。原 Canary 的 `VALID_OPERATIONAL_CANARY` 保留；本报告分类为 `VALID_FORENSIC_ZERO_API`。本报告是回顾性、单一 owner、非盲审阅，首次查看私有材料发生在审计 rubric 冻结之前。它不是预注册的效果检验，也没有独立标注一致性估计。

## 五个问题的答案

1. **七条内容是否完全不同？** 内容和数学表示确实多样，但并非毫无联系。可形成离散计数、几何面积等粗家族。A03/A04 是最接近的一对，共享坐标面积计算操作；A03/A04/A07 也能归入更宽的几何面积家族。题目家族相同不能直接证明错误根因相同。
2. **七个机制描述是否只是不同措辞的重复？** 审阅全部 21 个 unordered pairs，发现 1 对共享计算子过程但需要不同修复、6 对只有粗家族相似、3 对只有泛化的检查建议，另外 11 对未找到有依据的共同修复。没有确立动作与缺陷均等价的重复 pair。这个结果不证明潜在根因彼此不同。
3. **identity 或 V3 guard 是否制造了 singleton？** 原始返回已含七个单例支持。精确别名解码、14 次 guard 检查及完整 `score_partition` 复算都保留原分区。`mechanism_identity` 对 mechanism 和 direction 做 NFKC、大小写与空白归一化，再计算哈希；七个 mechanism、七个 direction 单独归一化后也都不同。它按合同仅合并相同描述对，不解析语义同义改写。该限制确实可能保留未来出现的同义重复，本次却没有证据认定它造成了过度碎片化。
4. **提示词是否鼓励逐题独立诊断？** 冻结提示明确要求在整个 wrong set、跨 responsibility lanes 寻找 recurring mechanisms，没有逐题生成一个 mechanism 的要求。它允许有理由的 singleton，也允许 unassigned。不过它没有明确的先比较再分组过程、共享修复标准或 singleton 理由字段；示意输出也是单个 support ID。其分组指导可能偏弱，但一次 generation 无法证明这种写法导致 singleton。提示词保持原样。
5. **人工能否自然合并两三条 deficiency？** owner 会自然识别上述粗家族，但尚不能为同一个已观察到的缺陷给出可信合并。最近的 A03/A04 虽共享面积计算，修复分别涉及离散合法性与计数、单次计算核验。把它们统一为泛化的“检查面积”会丢失区分作用。本次没有修改 support、重算人工分区的 F 或把标注反馈给优化器。

## 比簇数更关键的证据限制

**Pattern 七条输入都只有最终预测值，没有 Solver 推理过程；六条原始 Solver 响应也仅有最终答案。** 只有 A07 的私有原始响应提供了可审阅的计算过程，且该过程没有进入 Pattern 请求。A04 是实际选中的 singleton，其具体算术错误位置没有被观察到。多数诊断是根据问题类型和错误终值提出的合理假设，不能当作已定位的稳定 reasoning deficiency。

A02 的修复描述另有独立性范围不够明确的问题：分离独立的抽样组件可以成立，而固定总体内的联合事件需要条件处理。这是 owner 对诊断措辞的质量标记，未观察到 Solver 的内部错误过程，也不足以直接认定该建议错误。原文本留在本地私有材料中。

| 证据层面 | 本次结论 |
|---|---|
| 七个单例支持、零 recurring support | 确认 |
| WHO、same-F、wide → narrow、Memory 与 Layer2 链路 | 原独立审计 PASS 保留 |
| guard 拆分了原有共同簇 | 否，原返回已经是单例 |
| 七种不同的潜在错误根因 | 未证明 |
| 已确认的可合并重复缺陷 | 未发现；证据有限 |
| Pattern 效果或群体碎片化率 | 未估计 |

## 对 A4 Pilot 的判断

**建议有条件准备原方法 A4 Pilot。** 这一次单例观察不足以支持强制簇数、最小 support、similarity merge、cluster-size preference 或修改 Layer1/team gate。更大的冻结 Optimize 集合和多个 opportunities 可以观察 recurring support 是否出现，但它们仍不能单独识别 Pattern 的因果增益。

如果后续决定开展 Pilot，可在全新预注册中冻结 support-size distribution、singleton/unassigned fraction 和诊断证据类别等观察指标；它们不参与选择、门槛、调参、效果驱动停止或重跑。任何新的 Pattern 方法修改均需要独立 amendment 与新身份。本次只给出审计判断，没有创建 Pilot 运行范围或授予执行权限。

## 验证与公开范围

零真实 provider 调用、零 API tokens、审计进程网络尝试为零。347 个原运行 artifact 与 54 个原公开报告文件的字节哈希保持不变；累计 token ledger 保持不变。只将七条初始 selected-member Optimize 失败及一次 Pattern exchange 选入语义审阅；混合 trace 仅用于 role/stage 定位，没有将 Shadow、Validation 或 Test outcome 用于判断。

公开材料仅包含哈希、计数、类别和审计结论。原请求、问题、参考值、预测、原始响应及具体诊断文本留在 ignored 本地材料。规范、runtime、Pattern prompt、原 Canary manifest 与原报告均未修改。当前真实执行、Pilot、Validation 仍未授权，Test 封存。

`partition_replay_audit.json` 保存精确复算和逐项类别；`owner_annotations.json` 保存全部 21 对 owner 审阅；`provenance.json` 绑定输入和本地审计程序哈希；`verification_summary.json` 保存本次实际检查范围。

本次 compileall、manifest/schema、仓库治理、精确 replay 与脱敏通过。治理及执行合同范围首次检查中的两个失败在定稿后修正并单独复验通过，具体检查名与范围见 `verification_summary.json`。未重复运行整个 current suite；原生产源码的既有完整 current suite 通过记录仅作为上下文。历史 private-artifact tests 与完整 historical replay 本次未运行，不作通过声明。
