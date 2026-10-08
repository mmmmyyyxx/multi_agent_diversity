# A4 可见数学解题文本：V6 接口与反馈修复

本轮实现显式版本化的 Solver 输出及反馈接口修复，范围为零 API、合成数据和当前生产图验证。搜索算法及数学评分规则保持原样；Solver 指令和优化器可见证据改变，因此未来执行的科学、运行及请求身份必须重新冻结。没有启动 canary、Pilot 或 held-out 评估。

原来抑制解题文本的指令有两处：V3–V5 system 要求只输出一行并禁止 reasoning，V4/V5 user suffix 再次要求仅输出最终答案。新的 `MATH_SOLVER_INTERFACE_V6` 请求普通 assistant `content` 中的清晰、有序数学解答，最后保留唯一 `FINAL_ANSWER:` 行，payload 仅为数学答案，之后不得输出文本。新的接口没有 answer-only suffix，也没有增加数学求解策略。初始五个可变 prompt 保持原样。

现有 `final_payload()` 已支持最终标记之前的多行文本。本次复用它，没有改写提取器、数学等价判定、invalid 分类、四次恢复或 plurality。分数仅来自最终 payload；不能从解题文本猜答案。缺失、重复、非末行标记和响应截断仍遵循原规则。分数不额外惩罚缺失的可见解答。

新的私有 profile 保留完整 resolved prediction 及所有恢复尝试的原响应。反馈投影绑定成员、可变 prompt hash、题面 hash、example、split、实际请求/缓存身份、响应 hash 和 prediction 身份。进入反馈之前再次检查所选成员、当前 prompt 与题面；不会以其他成员、旧候选或不同实现的输出替代它。

逐题 Gradient 使用 `PER_EXAMPLE_TEXTUAL_GRADIENT_SCHEMA_V2` 和 prompt V4。每道错误 Optimize 题仍独立产生一个 Gradient，然后仅将 corrective gradients 交给原 clustering。Layer1 使用 `PATTERN_GRADIENT_VISIBLE_SOLUTIONS_MEMORY_INPUT_V5`：当前候选的 panel observation 带实际解题文本，下一代 mutation 可以查看；root failure representatives 则保留各自的 root 来源。责任分数、Pattern 排名、六代/36 metrics/panel cap 6/export 4/Full 2、Probe、transition 和 Memory 写入规则不变。

反馈最多暴露每条解答的 4096 个 Unicode 字符前缀，明确记录原长度、暴露长度和反馈截断。状态区分解答存在、解答缺失、最终边界异常和响应截断；异常边界的文本仅作为未分段、不完整证据。完整原响应仍保存在私有运行证据。普通书面步骤并不证明模型的隐藏推理或失败原因。

只有 Optimize 轨迹可进入 Gradient/Layer1。Shadow 留在原私有 gate 内，Validation/Test 被反馈入口拒绝。Solver 请求不含 reference，轨迹不会自动写入 Memory、共享经验或公开报告。模型仍为 qwen3-8b、thinking=false，采样配置、3600 输出上限和其他预算不变；没有增加 Solver 请求或读取 provider-private reasoning 字段。

新执行需要 `MATH_V2_2_VISIBLE_TRAJECTORY_EXECUTION_BINDING_V1`。历史 V2–V5、旧 binding/manifest、历史报告和原运行证据未被重解释。仓库提供 HOLD 状态的 opt-in profile；完整冻结、初始 profiling、独立 cache、source/startup 与一次性 API 授权仍须另行完成。已有 Validation 请求长度元数据可按精确空请求格式字节差派生，新文件需重新哈希；此过程不读取 held-out 内容。

离线验证覆盖数学整数、分数、表达式、集合，错误与格式异常，响应/反馈截断，恢复与缓存，来源错配及 held-out 拒绝。完整当前图验证了所选成员的实际响应进入每个独立 Gradient，以及当前候选响应进入下一代 Layer1 mutation；并与历史接口配对比较 WHO、调用粒度、clustering、排名、预算、Probe/Full、transition、Memory 和 plurality。新冻结 binding 的依赖检查和生产 manifest 构造也已零 API 通过。最终测试计数与完整性结果见 [verification.json](verification.json)。历史私有用例不在当前套件通过声明中。

最终当前套件为 **1,640 passed、2 skipped、1,468 deselected**，专项测试 **33 passed**，此前 Gradient/cluster 回归 **142 passed**。当前治理、compileall、manifest smoke、脱敏和字节完整性检查通过；旧治理 CLI 的三个既有 failure registry 兼容错误单独记录。本轮新增 API 调用及计费 tokens 均为 **0**。

真实 qwen3-8b 是否持续在普通 content 中输出合规解答、现有上限是否足够、实际恢复率、反馈文本质量及优化效果尚未评估。后续 canary 必须使用 [canary preparation](canary_preparation.md) 的新冻结与授权边界；不能沿用关闭的旧授权。离线 conformance 不支持优化效果提升结论。

参考实现的共同原则是将书面解答和最终答案分开，评分只使用最终答案。依据为 [GEPA optimize-anything](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)、[DSPy GEPA mathematics tutorial](https://dspy.ai/3.1.1/tutorials/gepa_aime/) 和 [GEPA MATH adapter](https://github.com/gepa-ai/gepa/blob/main/src/gepa/adapters/dspy_full_program_adapter/README.md)。本实现没有复制 AIME 的整数限制或 DSPy 模块结构。
