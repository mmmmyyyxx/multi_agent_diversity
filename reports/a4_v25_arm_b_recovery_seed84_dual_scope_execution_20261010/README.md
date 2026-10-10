# Seed84 recovery Canary / Pilot execution

精确双 Scope 获得用户授权后，Canary 首批八个并发 Solver 请求均被供应商以 HTTP 404 / NotFoundError 拒绝；全部八份错误回执的供应商错误码均为 model_not_found，请求模型为冻结的 qwen3-8b；供应商端具体原因尚未确定。没有取得任何模型响应，也没有完成初始评估。源码、启动身份、请求配置、授权消费、八份失败回执、账本链和中止证据目录核对通过。

Canary 为 EXECUTION_ABORTED；完整实验效力 NOT_EVALUABLE。Gradient、Pattern、Layer1、Probe、Full、Shadow 与 Commit 均 NOT_OBSERVED。这次供应商阻断不能用于评价生成恢复、解析器或推理能力。没有更改模型、供应商配置、Prompt、预算或停止规则，也没有追加真实 API 诊断请求。

依据冻结的工程与完整性推进门槛，Pilot 未启动。两项 Scope 均已关闭；Canary 授权已消费，Pilot 授权未消费。未重跑，不继续使用本次余额，也不擅自启动其他实验。供应商模型/通道可用性需要另行解决；新尝试需要独立冻结和授权。

账本有 17 个有效哈希链事件：一次授权、八次预留、八次失败结算。由于没有可靠供应商 usage，67,355 tokens 全部是按请求预留上限记入的保守 fallback；这不是供应商实际计费证明。终止时未决预留为 0，峰值并发为 8，charged + reserved 峰值为 67,355，低于每项 2,000,000 上限。Pilot、Validation 和 Test 调用均为 0。

此前离线修复证据保持不变：完整当前套件 583 通过、2 跳过；追加 HTTP 边界 2 项、冻结治理 126 项通过。220 个历史模块未执行，未声称历史重放全部通过。原准备报告、旧成绩与历史证据均未追溯修改。本报告只包含脱敏哈希、计数、类别与指标；本地提交，不推送。

中止后的治理、仓库登记与样本隔离检查共核验 126 项：首轮 115 项通过、11 项因收尾 frontier 仍指向关闭 Scope 而失败；纠正登记后，原 11 项失败身份精确重检全部通过。没有改测试代码；首次失败证据保留。compileall、Manifest schema、冻结证据不变、脱敏、报告哈希和 diff 检查通过。所有后续审计 API 调用和网络尝试均为 0。
