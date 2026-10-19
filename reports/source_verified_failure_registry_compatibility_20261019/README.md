# `source_verified` 治理兼容性修复与重新准备

本次仅修复失败证据分类的兼容性；审计起点为
`917ca8ba3e8734e1091efcd9f6c29d871437e4ae`。没有新模型调用，没有 Canary、
正式训练、Validation/Test 评估或推送。旧实验、旧冻结清单和旧报告保持原样。

规范见 [失败证据分类 V1](../../docs/design/FAILURE_EVIDENCE_LEVELS_V1.md)。
`source_verified` 表示有可追溯源码和离线检查依据的实现/流程核对，**不表示**
真实 API 故障、因果归因、性能收益或泛化能力。四类证据不是强弱排序。
`MITIGATED` 仍然不能绕过校验或授予运行权限。

## 修复与边界

`validate_failure_registry` 显式支持已使用的 `source_verified`；这一类别还必须
提供非空的证据和回归测试引用列表。所有类别继续检查未知枚举、必填项、状态、
重复 ID 以及引用存在性。校验器检查元数据，不自动判断说明文本是否真实。

两条原始记录 `FAIL-ABLATION-PARTIAL-FORMAL-ADAPTER-CONFORMANCE`、
`FAIL-ABLATION-PATTERN-DIAGNOSIS-UNIVERSE` 的 V5/V6 证据说明对应源码流程，
有现存回归测试支撑，因此保留 `source_verified` 与 `MITIGATED`。失败 YAML
本身完全未变，没有将它们提升为 `observed` 或 `causally_supported`。
重新生成 `KNOWN_FAILURES.md`，展示当前登记记录并链接证据定义。

C0/C1/C2 调度、Gradient/Cluster、代表采样、候选结构 Guard、模型、SDK、
Optimize/Validation/Test、缓存、重试、接纳、并发及预算不变。
没有新增 Responsibility、ComBEE、Population 或其他优化模块。
独立 Manifest 与配置保持原样；旧 SHA 下的准备不因修复自动获得执行资格。

## 验证

新运行的检查与复用证据分开记录在 [verification.json](verification.json)。
治理测试55项通过；准备及授权边界7项通过，均处于离线网络 Guard 下。
最初一次测试入口因未创建临时目录的父目录而出现 setup 错误，创建父目录后
使用新的目录完成检查；没有修改测试断言来隐藏失败。

此前完整 current 的1205通过、2跳过及调度功能37通过证据只用于未改变的
优化/测评代码，不冒充这次新跑的完整回归，也不将重叠用例相加。
参见 [原始准备记录](../single_prompt_pattern_scheduling_preparation_20261019/README.md)。
本次重新运行受影响的治理与准备测试、生产编译、完整治理/Manifest预检、
生成视图检查、脱敏与差异检查；不宣称历史/私有资产回放全部通过。

## 新冻结与下一阶段

干净提交后，用现有 `prepare.py --pattern-directions` 在全新目录重新冻结。
新 source SHA、输入哈希、scope 与 handoff 必须重新核对；旧准备保持不可变。
新清单仍为 `approved=false`、`READY_TO_RUN=false`，训练、自适应Validation
及最终Test均未授权。完整预检通过不等于真实模型链路已经验证。

独立低预算 Canary 需要另行具体冻结并获得授权，不读取 Test，也不作为正式
九条轨迹数据。之后正式优化需要独立授权；最终 Test300 在完整训练审计后
另行申请访问。历史 Test 已有暴露，不称新的独立盲测。
