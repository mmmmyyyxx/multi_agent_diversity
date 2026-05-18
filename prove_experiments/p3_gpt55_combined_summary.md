# P3 GPT-5.5 综合结论

本文合并两类 GPT-5.5 复核。证据优先级是：Normal Taxonomy Judge 为主，Prompt Following 为补充。

## 证据优先级

1. **Normal Taxonomy Judge**：GPT-5.5 得到与正式 judge 尽量相同的信息，直接重新给 trace 打 taxonomy 标签。这是更强证据。
2. **Prompt Following**：GPT-5.5 只看原始策略指令和 trace，判断是否遵循 prompt。这是辅助诊断，用来区分 judge/taxonomy 吸附和模型/prompt 遵循不足。

## 核心统计

| normal n | GPT primary option | GPT target exact | GPT target same-major | judge/taxonomy questioned |
|---|---|---|---|---|
| 40 | 0.2750 | 0.2250 | 0.3000 | 0.7250 |


| prompt n | followed rate | mean score | judge taxonomy likely | model prompt likely | ambiguous |
|---|---|---|---|---|---|
| 40 | 0.6000 | 3.7250 | 0.6000 | 0.2250 | 0.1750 |

## 综合判断

- 优先看 normal taxonomy judge：GPT-5.5 并没有大规模继续支持原自动 judge 的 `option_contrast` 主判定，说明自动 judge/taxonomy 存在明显的 `option_contrast` 吸附风险。
- 再看 prompt-following：GPT-5.5 认为 60.00% 的抽样 trace 基本遵循了原始策略指令，80.00% 至少部分遵循。这进一步说明 leaf exact hit 偏低不能直接等同于模型完全不听策略 prompt。
- 同时，prompt-following 也显示不同策略可执行性不均衡：`distractor_elimination`、`decomposition`、`case_analysis` 更容易被执行；`rule_or_principle_application` 和 `edge_case_analysis` 更弱。
- 因此，P3 的主证据应是 team-level diversity、major diversity 和 homogeneity 的系统变化；exact target hit 更适合作为诊断指标，而不是最终有效性的唯一标准。
