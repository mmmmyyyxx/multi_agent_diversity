# 验证与用户停止记录

**最终发布状态：PARTIALLY_VERIFIED_USER_STOP。** 最终完整 current suite 按用户指令中断，没有完整结果，不计作 PASS；停止后没有新测试运行。

最终代码的新修订专项为 22 passed。开发阶段完整套件为 1,657 passed、2 skipped、1,468 历史/私有用例排除；该结果早于最终补齐，不替代最终完整回归。source、tests、design 与 schema 共 865 个文件自最后一次回归启动后未变。

下列参数均以前置 Python 执行。Python 指仓库 ignored runtime，公开报告不包含主机路径。已完成应用测试均先安装 network guard、清除真实 credentials，network attempts 0。中断运行没有最终 network counter 回执，记录为未知而非补造 0；账本与授权核验证明本轮真实 API 调用和新增计费为零。

| 精确参数 | 结果与范围 |
|---|---|
| `tests/formal_zero_api_runner.py tests/current/test_optimization_evidence_v23.py --suite current --junitxml=runs/math_optimization_evidence_v23_20261008/final_memory_bounds_complete_v2.xml` | 22 passed、0 skipped；FINAL_SOURCE_FOCUSED_ONLY |
| `tests/formal_zero_api_runner.py tests --suite current --junitxml=runs/math_optimization_evidence_v23_20261008/current_suite.xml` | 1657 passed、2 skipped；EARLIER_DEVELOPMENT_SOURCE_NOT_FINAL；1,468 deselected |
| `tests/formal_zero_api_runner.py tests/current/test_optimization_evidence_v23.py tests/current/test_visible_solver_trajectories.py tests/test_unified_execution_governance.py --suite current --junitxml=runs/math_optimization_evidence_v23_20261008/focused_final.xml` | 267 passed、0 skipped；EARLIER_DEVELOPMENT_SOURCE_NOT_FINAL |
| `tests/formal_zero_api_runner.py tests --suite current --junitxml=runs/math_optimization_evidence_v23_20261008/current_suite_final.xml` | INTERRUPTED：OWNER_RESTART_AFTER_NORMATIVE_SOURCE_HASH_REPAIR；无完整结果 |
| `tests/formal_zero_api_runner.py tests --suite current --junitxml=runs/math_optimization_evidence_v23_20261008/current_suite_release.xml` | INTERRUPTED：STOPPED_BY_USER；无完整结果 |
| `-m compileall -q multi_dataset_diverse_rl scripts tests/current/test_optimization_evidence_v23.py` | PASS，停止前完成 |
| `tests/formal_zero_api_runner.py --offline-command runs/math_optimization_evidence_v23_20261008/synthetic_report.py` | PASS，停止前完成；预设合成 provider，不是效能证据 |
| `tests/formal_zero_api_runner.py --offline-command scripts/audit_repository_governance.py --generate --out runs/math_optimization_evidence_v23_20261008/governance_stop_publication.json` | PASS；停止后的确定性发布审计，没有启动测试或模型 |
| `tests/formal_zero_api_runner.py --offline-command runs/math_optimization_evidence_v23_20261008/integrity.py` | PASS；历史 raw/reports、parent binding/manifest、账本、关闭授权、source closure 与脱敏 |
| `tests/formal_zero_api_runner.py --offline-command runs/math_optimization_evidence_v23_20261008/legacy_check.py` | BLOCKED_BY_EXISTING_BASELINE；3 个相同 registry 错误，新增 0 |
| `git diff --check`、`git diff --cached --check` | PASS |

22 项专项覆盖独立 Optimize roles、reference provenance、validation 负迁移、拒绝 Full 的同成员记忆、实际 diff/意图、bootstrap/提交覆盖、rotation、预调用容量检查、六代 42 评估、原存储容量和新规范 source hash。

两个 skipped 是 HoVer/PuPa 未冻结真实数据。旧 CLI 在 registry 阶段遇到 resource-bound 条目缺少 symptom 和两项旧 evidence_level 不兼容，因此未执行后续校验；不标为 PASS。

中间开发修复和主动重启记录保留在 ignored runs，没有伪造成最终通过。机器计数、XML hashes、source hashes 与账本 witness 见 [verification.json](verification.json)。完整状态梳理见 [stop_summary.md](stop_summary.md)。

后续需要另行获准完成最终完整回归；真实实验还需新的冻结和单次授权。当前 HOLD，不建立有效性、泛化或隐藏推理因果结论。
