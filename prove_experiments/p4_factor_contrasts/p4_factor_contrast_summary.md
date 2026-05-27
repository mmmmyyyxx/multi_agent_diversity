# P4 模型身份与 Prompt 因子对比

本分析区分两种口径：`*_family` 是当前 family-level prompt 设定，`same_model_same_prompt` 和 `different_model_same_prompt` 是后补的 exact same-prompt baseline。

| contrast | prompt_mode | unit | left_model | right_model | left_prompt_family | right_prompt_family | n | mean major distribution distance | mean family diversity | mean major diversity | mean homogeneity | mean prompt embedding diversity | mean trace embedding diversity | mean trace token diversity | mean vote acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | gemini-2.5-flash-lite | mixed_strategy | same_definition | 100 | 0.3608 | 0.5679 | 0.6363 | 0.5756 | 0.1762 | 0.0536 | 0.1947 | 0.8650 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | gemini-2.5-flash-lite | mixed_strategy | same_elimination | 100 | 0.3780 | 0.5572 | 0.6155 | 0.5895 | 0.1798 | 0.0530 | 0.1950 | 0.8450 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | gemini-2.5-flash-lite | same_definition | same_elimination | 100 | 0.4436 | 0.6058 | 0.6334 | 0.5186 | 0.1923 | 0.0547 | 0.1958 | 0.8550 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | gpt-4o-mini | mixed_strategy | same_definition | 100 | 0.3944 | 0.5934 | 0.6304 | 0.5642 | 0.1762 | 0.0635 | 0.2325 | 0.8800 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | gpt-4o-mini | mixed_strategy | same_elimination | 100 | 0.3918 | 0.5801 | 0.6440 | 0.5513 | 0.1798 | 0.0603 | 0.2215 | 0.8650 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | gpt-4o-mini | same_definition | same_elimination | 100 | 0.4628 | 0.6117 | 0.6572 | 0.4859 | 0.1923 | 0.0627 | 0.2246 | 0.8750 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | qwen2.5-7b-instruct | mixed_strategy | same_definition | 100 | 0.3958 | 0.6102 | 0.6467 | 0.5235 | 0.1762 | 0.1637 | 0.4143 | 0.8100 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | qwen2.5-7b-instruct | mixed_strategy | same_elimination | 100 | 0.4108 | 0.6042 | 0.6375 | 0.5277 | 0.1798 | 0.1718 | 0.4266 | 0.8300 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | qwen2.5-7b-instruct | same_definition | same_elimination | 100 | 0.4918 | 0.6244 | 0.6546 | 0.4720 | 0.1923 | 0.1729 | 0.4245 | 0.8400 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | qwen3.5-plus | mixed_strategy | same_definition | 100 | 0.3764 | 0.5917 | 0.6493 | 0.5629 | 0.1762 | 0.0621 | 0.2235 | 0.9200 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | qwen3.5-plus | mixed_strategy | same_elimination | 100 | 0.3564 | 0.5969 | 0.6349 | 0.5673 | 0.1798 | 0.0578 | 0.2175 | 0.9250 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | qwen3.5-plus | same_definition | same_elimination | 100 | 0.4208 | 0.6180 | 0.6272 | 0.5159 | 0.1923 | 0.0590 | 0.2185 | 0.9350 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | deepseek-chat | mixed_strategy | same_definition | 100 | 0.4382 | 0.6012 | 0.6325 | 0.5183 | 0.1762 | 0.0550 | 0.1966 | 0.8550 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | deepseek-chat | mixed_strategy | same_elimination | 100 | 0.3478 | 0.5180 | 0.5422 | 0.6520 | 0.1798 | 0.0517 | 0.1867 | 0.8500 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | deepseek-chat | same_definition | same_elimination | 100 | 0.3472 | 0.5303 | 0.5643 | 0.6321 | 0.1923 | 0.0516 | 0.1860 | 0.8700 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gpt-4o-mini | mixed_strategy | same_definition | 100 | 0.2008 | 0.4808 | 0.3796 | 0.7485 | 0.1762 | 0.0532 | 0.1525 | 0.8350 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gpt-4o-mini | mixed_strategy | same_elimination | 100 | 0.2112 | 0.4532 | 0.3925 | 0.7467 | 0.1798 | 0.0481 | 0.1412 | 0.8200 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gpt-4o-mini | same_definition | same_elimination | 100 | 0.2248 | 0.4868 | 0.4442 | 0.7187 | 0.1923 | 0.0474 | 0.1405 | 0.8400 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen2.5-7b-instruct | mixed_strategy | same_definition | 100 | 0.1906 | 0.4954 | 0.3886 | 0.7038 | 0.1762 | 0.1594 | 0.3585 | 0.7650 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen2.5-7b-instruct | mixed_strategy | same_elimination | 100 | 0.2066 | 0.4756 | 0.3683 | 0.7181 | 0.1798 | 0.1673 | 0.3736 | 0.7850 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen2.5-7b-instruct | same_definition | same_elimination | 100 | 0.2222 | 0.4979 | 0.4131 | 0.6941 | 0.1923 | 0.1666 | 0.3711 | 0.8050 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen3.5-plus | mixed_strategy | same_definition | 100 | 0.2046 | 0.4915 | 0.4083 | 0.7279 | 0.1762 | 0.0564 | 0.1579 | 0.8750 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen3.5-plus | mixed_strategy | same_elimination | 100 | 0.1592 | 0.4826 | 0.3844 | 0.7495 | 0.1798 | 0.0517 | 0.1515 | 0.8800 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen3.5-plus | same_definition | same_elimination | 100 | 0.1876 | 0.5109 | 0.4326 | 0.7226 | 0.1923 | 0.0513 | 0.1505 | 0.9000 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | deepseek-chat | mixed_strategy | same_definition | 100 | 0.4414 | 0.6119 | 0.6641 | 0.4839 | 0.1762 | 0.0634 | 0.2301 | 0.8850 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | deepseek-chat | mixed_strategy | same_elimination | 100 | 0.3478 | 0.5507 | 0.5718 | 0.6140 | 0.1798 | 0.0595 | 0.2169 | 0.8800 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | deepseek-chat | same_definition | same_elimination | 100 | 0.3720 | 0.5521 | 0.5424 | 0.6273 | 0.1923 | 0.0623 | 0.2242 | 0.8850 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | gemini-2.5-flash-lite | mixed_strategy | same_definition | 100 | 0.2332 | 0.4945 | 0.4562 | 0.6946 | 0.1762 | 0.0501 | 0.1469 | 0.8500 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | gemini-2.5-flash-lite | mixed_strategy | same_elimination | 100 | 0.2134 | 0.4662 | 0.4087 | 0.7266 | 0.1798 | 0.0496 | 0.1470 | 0.8300 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | gemini-2.5-flash-lite | same_definition | same_elimination | 100 | 0.1926 | 0.4766 | 0.3701 | 0.7517 | 0.1923 | 0.0521 | 0.1516 | 0.8350 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen2.5-7b-instruct | mixed_strategy | same_definition | 100 | 0.1852 | 0.5244 | 0.4373 | 0.6666 | 0.1762 | 0.1604 | 0.3648 | 0.7950 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen2.5-7b-instruct | mixed_strategy | same_elimination | 100 | 0.1842 | 0.4916 | 0.3897 | 0.6882 | 0.1798 | 0.1685 | 0.3776 | 0.8150 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen2.5-7b-instruct | same_definition | same_elimination | 100 | 0.1818 | 0.4911 | 0.3528 | 0.7147 | 0.1923 | 0.1702 | 0.3757 | 0.8200 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen3.5-plus | mixed_strategy | same_definition | 100 | 0.2518 | 0.5215 | 0.4572 | 0.6745 | 0.1762 | 0.0670 | 0.1993 | 0.9050 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen3.5-plus | mixed_strategy | same_elimination | 100 | 0.2036 | 0.5138 | 0.4297 | 0.6971 | 0.1798 | 0.0612 | 0.1903 | 0.9100 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen3.5-plus | same_definition | same_elimination | 100 | 0.1822 | 0.5090 | 0.3882 | 0.7336 | 0.1923 | 0.0625 | 0.1922 | 0.9150 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | deepseek-chat | mixed_strategy | same_definition | 100 | 0.4416 | 0.6403 | 0.6814 | 0.4551 | 0.1762 | 0.1483 | 0.3879 | 0.8300 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | deepseek-chat | mixed_strategy | same_elimination | 100 | 0.3594 | 0.5770 | 0.6100 | 0.5663 | 0.1798 | 0.1456 | 0.3818 | 0.8250 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | deepseek-chat | same_definition | same_elimination | 100 | 0.3742 | 0.5732 | 0.5751 | 0.5837 | 0.1923 | 0.1631 | 0.4090 | 0.8150 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | gemini-2.5-flash-lite | mixed_strategy | same_definition | 100 | 0.2284 | 0.5196 | 0.4836 | 0.6535 | 0.1762 | 0.1411 | 0.3250 | 0.7950 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | gemini-2.5-flash-lite | mixed_strategy | same_elimination | 100 | 0.2014 | 0.4848 | 0.4282 | 0.6875 | 0.1798 | 0.1410 | 0.3264 | 0.7750 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | gemini-2.5-flash-lite | same_definition | same_elimination | 100 | 0.1866 | 0.4891 | 0.3859 | 0.7111 | 0.1923 | 0.1591 | 0.3574 | 0.7650 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | gpt-4o-mini | mixed_strategy | same_definition | 100 | 0.1982 | 0.5243 | 0.4425 | 0.6741 | 0.1762 | 0.1436 | 0.3326 | 0.8100 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | gpt-4o-mini | mixed_strategy | same_elimination | 100 | 0.1618 | 0.4971 | 0.4300 | 0.6703 | 0.1798 | 0.1399 | 0.3285 | 0.7950 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | gpt-4o-mini | same_definition | same_elimination | 100 | 0.1696 | 0.4985 | 0.3883 | 0.6900 | 0.1923 | 0.1593 | 0.3612 | 0.7850 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen3.5-plus | mixed_strategy | same_definition | 100 | 0.2570 | 0.5447 | 0.4972 | 0.6334 | 0.1762 | 0.1531 | 0.3622 | 0.8500 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen3.5-plus | mixed_strategy | same_elimination | 100 | 0.2028 | 0.5350 | 0.4492 | 0.6489 | 0.1798 | 0.1488 | 0.3576 | 0.8550 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen3.5-plus | same_definition | same_elimination | 100 | 0.1950 | 0.5295 | 0.4057 | 0.6789 | 0.1923 | 0.1648 | 0.3852 | 0.8450 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | deepseek-chat | mixed_strategy | same_definition | 100 | 0.4144 | 0.6361 | 0.6758 | 0.4639 | 0.1922 | 0.0646 | 0.2214 | 0.9300 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | deepseek-chat | mixed_strategy | same_elimination | 100 | 0.4130 | 0.5871 | 0.6476 | 0.5522 | 0.1937 | 0.0625 | 0.2148 | 0.9250 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | deepseek-chat | same_definition | same_elimination | 100 | 0.3800 | 0.5662 | 0.5887 | 0.6058 | 0.1923 | 0.0621 | 0.2202 | 0.9250 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | gemini-2.5-flash-lite | mixed_strategy | same_definition | 100 | 0.2632 | 0.5594 | 0.5480 | 0.6402 | 0.1922 | 0.0560 | 0.1499 | 0.8950 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | gemini-2.5-flash-lite | mixed_strategy | same_elimination | 100 | 0.2662 | 0.5377 | 0.5153 | 0.6590 | 0.1937 | 0.0561 | 0.1506 | 0.8750 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | gemini-2.5-flash-lite | same_definition | same_elimination | 100 | 0.2376 | 0.5055 | 0.4295 | 0.7140 | 0.1923 | 0.0563 | 0.1577 | 0.8750 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | gpt-4o-mini | mixed_strategy | same_definition | 100 | 0.2644 | 0.5584 | 0.5171 | 0.6520 | 0.1922 | 0.0681 | 0.1934 | 0.9100 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | gpt-4o-mini | mixed_strategy | same_elimination | 100 | 0.3140 | 0.5593 | 0.5407 | 0.6152 | 0.1937 | 0.0652 | 0.1870 | 0.8950 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | gpt-4o-mini | same_definition | same_elimination | 100 | 0.2816 | 0.5296 | 0.4566 | 0.6687 | 0.1923 | 0.0665 | 0.1952 | 0.8950 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | qwen2.5-7b-instruct | mixed_strategy | same_definition | 100 | 0.2834 | 0.5828 | 0.5329 | 0.5989 | 0.1922 | 0.1687 | 0.3870 | 0.8400 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | qwen2.5-7b-instruct | mixed_strategy | same_elimination | 100 | 0.3316 | 0.5760 | 0.5352 | 0.5930 | 0.1937 | 0.1769 | 0.4007 | 0.8600 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | qwen2.5-7b-instruct | same_definition | same_elimination | 100 | 0.2788 | 0.5275 | 0.4355 | 0.6583 | 0.1923 | 0.1776 | 0.4021 | 0.8600 |
| different_model_same_prompt | exact | between_team_same_question | gpt-4o-mini | qwen3.5-plus | same_prompt | same_prompt | 100 | 0.1816 | 0.4440 | 0.3288 | 0.7735 | 0.0000 | 0.0537 | 0.1658 | 0.9050 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | gemini-2.5-flash-lite | mixed_strategy | mixed_strategy | 100 | 0.3548 | 0.5609 | 0.6170 | 0.5930 | 0.1554 | 0.0534 | 0.1958 | 0.8450 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | gemini-2.5-flash-lite | same_definition | same_definition | 100 | 0.4266 | 0.6148 | 0.6496 | 0.5137 | 0.1496 | 0.0549 | 0.1955 | 0.8750 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | gemini-2.5-flash-lite | same_elimination | same_elimination | 100 | 0.3528 | 0.5147 | 0.5403 | 0.6562 | 0.1339 | 0.0509 | 0.1855 | 0.8500 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | gpt-4o-mini | mixed_strategy | mixed_strategy | 100 | 0.3622 | 0.5872 | 0.6479 | 0.5441 | 0.1554 | 0.0618 | 0.2287 | 0.8750 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | gpt-4o-mini | same_definition | same_definition | 100 | 0.4650 | 0.6051 | 0.6466 | 0.5145 | 0.1496 | 0.0633 | 0.2305 | 0.8900 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | gpt-4o-mini | same_elimination | same_elimination | 100 | 0.3548 | 0.5286 | 0.5447 | 0.6306 | 0.1339 | 0.0567 | 0.2069 | 0.8700 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | qwen2.5-7b-instruct | mixed_strategy | mixed_strategy | 100 | 0.3590 | 0.6019 | 0.6674 | 0.5069 | 0.1554 | 0.1474 | 0.3887 | 0.8200 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | qwen2.5-7b-instruct | same_definition | same_definition | 100 | 0.4524 | 0.6213 | 0.6479 | 0.4868 | 0.1496 | 0.1632 | 0.4103 | 0.8200 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | qwen2.5-7b-instruct | same_elimination | same_elimination | 100 | 0.3706 | 0.5514 | 0.5370 | 0.6038 | 0.1339 | 0.1693 | 0.4180 | 0.8350 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | qwen3.5-plus | mixed_strategy | mixed_strategy | 100 | 0.3668 | 0.6193 | 0.6972 | 0.5018 | 0.1833 | 0.0633 | 0.2203 | 0.9200 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | qwen3.5-plus | same_definition | same_definition | 100 | 0.4288 | 0.5898 | 0.6492 | 0.5382 | 0.1496 | 0.0622 | 0.2210 | 0.9300 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | qwen3.5-plus | same_elimination | same_elimination | 100 | 0.3368 | 0.5540 | 0.5468 | 0.6318 | 0.1339 | 0.0565 | 0.2102 | 0.9300 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gpt-4o-mini | mixed_strategy | mixed_strategy | 100 | 0.2112 | 0.4723 | 0.4144 | 0.7218 | 0.1554 | 0.0509 | 0.1483 | 0.8300 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gpt-4o-mini | same_definition | same_definition | 100 | 0.2228 | 0.5061 | 0.4266 | 0.7218 | 0.1496 | 0.0520 | 0.1502 | 0.8550 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gpt-4o-mini | same_elimination | same_elimination | 100 | 0.2154 | 0.4469 | 0.3967 | 0.7561 | 0.1339 | 0.0463 | 0.1385 | 0.8200 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen2.5-7b-instruct | mixed_strategy | mixed_strategy | 100 | 0.2110 | 0.4972 | 0.4329 | 0.6776 | 0.1554 | 0.1415 | 0.3274 | 0.7750 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen2.5-7b-instruct | same_definition | same_definition | 100 | 0.2096 | 0.5259 | 0.4415 | 0.6828 | 0.1496 | 0.1586 | 0.3562 | 0.7850 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen2.5-7b-instruct | same_elimination | same_elimination | 100 | 0.1894 | 0.4667 | 0.3519 | 0.7298 | 0.1339 | 0.1663 | 0.3721 | 0.7850 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen3.5-plus | mixed_strategy | mixed_strategy | 100 | 0.2522 | 0.5299 | 0.5077 | 0.6640 | 0.1833 | 0.0565 | 0.1516 | 0.8750 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen3.5-plus | same_definition | same_definition | 100 | 0.2442 | 0.5263 | 0.4649 | 0.7004 | 0.1496 | 0.0557 | 0.1557 | 0.8950 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen3.5-plus | same_elimination | same_elimination | 100 | 0.1822 | 0.4833 | 0.3878 | 0.7472 | 0.1339 | 0.0510 | 0.1503 | 0.8800 |
| different_model_same_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen2.5-7b-instruct | mixed_strategy | mixed_strategy | 100 | 0.1860 | 0.5277 | 0.4743 | 0.6430 | 0.1554 | 0.1424 | 0.3342 | 0.8050 |
| different_model_same_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen2.5-7b-instruct | same_definition | same_definition | 100 | 0.1658 | 0.5057 | 0.3870 | 0.7109 | 0.1496 | 0.1593 | 0.3593 | 0.8000 |
| different_model_same_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen2.5-7b-instruct | same_elimination | same_elimination | 100 | 0.1460 | 0.4670 | 0.3501 | 0.7223 | 0.1339 | 0.1650 | 0.3699 | 0.8050 |
| different_model_same_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen3.5-plus | mixed_strategy | mixed_strategy | 100 | 0.2840 | 0.5666 | 0.5537 | 0.6093 | 0.1833 | 0.0671 | 0.1925 | 0.9050 |
| different_model_same_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen3.5-plus | same_definition | same_definition | 100 | 0.2066 | 0.5105 | 0.4234 | 0.7327 | 0.1496 | 0.0660 | 0.1976 | 0.9100 |
| different_model_same_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen3.5-plus | same_elimination | same_elimination | 100 | 0.1906 | 0.4923 | 0.3928 | 0.7163 | 0.1339 | 0.0592 | 0.1824 | 0.9000 |
| different_model_same_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen3.5-plus | mixed_strategy | mixed_strategy | 100 | 0.2928 | 0.5742 | 0.5616 | 0.5677 | 0.1833 | 0.1535 | 0.3603 | 0.8500 |
| different_model_same_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen3.5-plus | same_definition | same_definition | 100 | 0.2324 | 0.5355 | 0.4485 | 0.6778 | 0.1496 | 0.1674 | 0.3868 | 0.8400 |
| different_model_same_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen3.5-plus | same_elimination | same_elimination | 100 | 0.2062 | 0.5130 | 0.3827 | 0.6880 | 0.1339 | 0.1726 | 0.3965 | 0.8650 |
| same_model_different_prompt_family | family | between_team_same_question | deepseek-chat | deepseek-chat | mixed_strategy | same_definition | 100 | 0.3692 | 0.6403 | 0.7279 | 0.4461 | 0.1762 | 0.0466 | 0.1806 | 0.9000 |
| same_model_different_prompt_family | family | between_team_same_question | deepseek-chat | deepseek-chat | mixed_strategy | same_elimination | 100 | 0.2486 | 0.5604 | 0.6467 | 0.5891 | 0.1798 | 0.0424 | 0.1671 | 0.8950 |
| same_model_different_prompt_family | family | between_team_same_question | deepseek-chat | deepseek-chat | same_definition | same_elimination | 100 | 0.4444 | 0.6250 | 0.6985 | 0.4761 | 0.1923 | 0.0471 | 0.1810 | 0.9050 |
| same_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gemini-2.5-flash-lite | mixed_strategy | same_definition | 100 | 0.1188 | 0.4539 | 0.3846 | 0.7882 | 0.1762 | 0.0303 | 0.0782 | 0.8200 |
| same_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gemini-2.5-flash-lite | mixed_strategy | same_elimination | 100 | 0.1090 | 0.4265 | 0.3385 | 0.8170 | 0.1798 | 0.0297 | 0.0788 | 0.8000 |
| same_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gemini-2.5-flash-lite | same_definition | same_elimination | 100 | 0.1304 | 0.4489 | 0.3794 | 0.7875 | 0.1923 | 0.0298 | 0.0796 | 0.8200 |
| same_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | gpt-4o-mini | mixed_strategy | same_definition | 100 | 0.1490 | 0.4955 | 0.3944 | 0.7357 | 0.1762 | 0.0467 | 0.1411 | 0.8650 |
| same_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | gpt-4o-mini | mixed_strategy | same_elimination | 100 | 0.1804 | 0.4794 | 0.4342 | 0.7234 | 0.1798 | 0.0403 | 0.1288 | 0.8500 |
| same_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | gpt-4o-mini | same_definition | same_elimination | 100 | 0.1542 | 0.4781 | 0.3668 | 0.7447 | 0.1923 | 0.0467 | 0.1420 | 0.8550 |
| same_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen2.5-7b-instruct | mixed_strategy | same_definition | 100 | 0.1498 | 0.5214 | 0.4337 | 0.6533 | 0.1762 | 0.1877 | 0.4098 | 0.7400 |
| same_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen2.5-7b-instruct | mixed_strategy | same_elimination | 100 | 0.1350 | 0.4954 | 0.3958 | 0.6731 | 0.1798 | 0.1969 | 0.4301 | 0.7600 |
| same_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen2.5-7b-instruct | same_definition | same_elimination | 100 | 0.1382 | 0.4809 | 0.3475 | 0.6953 | 0.1923 | 0.1962 | 0.4263 | 0.7500 |
| same_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | qwen3.5-plus | mixed_strategy | same_definition | 100 | 0.1860 | 0.5498 | 0.5150 | 0.6782 | 0.1922 | 0.0528 | 0.1426 | 0.9500 |
| same_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | qwen3.5-plus | mixed_strategy | same_elimination | 100 | 0.2008 | 0.5470 | 0.5033 | 0.6672 | 0.1937 | 0.0483 | 0.1365 | 0.9550 |
| same_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | qwen3.5-plus | same_definition | same_elimination | 100 | 0.1468 | 0.5093 | 0.4132 | 0.7407 | 0.1923 | 0.0475 | 0.1405 | 0.9550 |
| same_model_same_prompt | exact | within_team | gpt-4o-mini | gpt-4o-mini | same_prompt | same_prompt | 100 | 0.1905 | 0.3181 | 0.1905 | 0.8820 | 0.0000 | 0.0213 | 0.0746 | 0.8500 |
| same_model_same_prompt | exact | within_team | qwen3.5-plus | qwen3.5-plus | same_prompt | same_prompt | 100 | 0.3258 | 0.4362 | 0.3258 | 0.8025 | 0.0000 | 0.0256 | 0.0858 | 0.9600 |
| same_model_same_prompt_family | family | within_team | deepseek-chat | deepseek-chat | mixed_strategy | mixed_strategy | 100 | 0.7033 | 0.5819 | 0.7033 | 0.5092 | 0.1748 | 0.0444 | 0.1774 | 0.8900 |
| same_model_same_prompt_family | family | within_team | deepseek-chat | deepseek-chat | same_definition | same_definition | 100 | 0.6377 | 0.4807 | 0.6377 | 0.5372 | 0.1682 | 0.0477 | 0.1795 | 0.9100 |
| same_model_same_prompt_family | family | within_team | deepseek-chat | deepseek-chat | same_elimination | same_elimination | 100 | 0.4889 | 0.4786 | 0.4889 | 0.6915 | 0.1506 | 0.0371 | 0.1464 | 0.9000 |
| same_model_same_prompt_family | family | within_team | gemini-2.5-flash-lite | gemini-2.5-flash-lite | mixed_strategy | mixed_strategy | 100 | 0.3276 | 0.4268 | 0.3276 | 0.8141 | 0.1748 | 0.0305 | 0.0786 | 0.8000 |
| same_model_same_prompt_family | family | within_team | gemini-2.5-flash-lite | gemini-2.5-flash-lite | same_definition | same_definition | 100 | 0.4038 | 0.4598 | 0.4038 | 0.7653 | 0.1682 | 0.0293 | 0.0762 | 0.8400 |
| same_model_same_prompt_family | family | within_team | gemini-2.5-flash-lite | gemini-2.5-flash-lite | same_elimination | same_elimination | 100 | 0.3051 | 0.4057 | 0.3064 | 0.8260 | 0.1506 | 0.0285 | 0.0767 | 0.8000 |
| same_model_same_prompt_family | family | within_team | gpt-4o-mini | gpt-4o-mini | mixed_strategy | mixed_strategy | 100 | 0.4096 | 0.4805 | 0.4096 | 0.6995 | 0.1748 | 0.0453 | 0.1403 | 0.8600 |
| same_model_same_prompt_family | family | within_team | gpt-4o-mini | gpt-4o-mini | same_definition | same_definition | 100 | 0.3205 | 0.4722 | 0.3205 | 0.7896 | 0.1682 | 0.0453 | 0.1339 | 0.8700 |
| same_model_same_prompt_family | family | within_team | gpt-4o-mini | gpt-4o-mini | same_elimination | same_elimination | 100 | 0.3486 | 0.4210 | 0.3486 | 0.7618 | 0.1506 | 0.0328 | 0.1102 | 0.8400 |
| same_model_same_prompt_family | family | within_team | qwen2.5-7b-instruct | qwen2.5-7b-instruct | mixed_strategy | mixed_strategy | 100 | 0.4650 | 0.5278 | 0.4650 | 0.6076 | 0.1748 | 0.1948 | 0.4274 | 0.7500 |
| same_model_same_prompt_family | family | within_team | qwen2.5-7b-instruct | qwen2.5-7b-instruct | same_definition | same_definition | 100 | 0.3699 | 0.4953 | 0.3693 | 0.6990 | 0.1682 | 0.1844 | 0.3991 | 0.7300 |
| same_model_same_prompt_family | family | within_team | qwen2.5-7b-instruct | qwen2.5-7b-instruct | same_elimination | same_elimination | 100 | 0.2851 | 0.4565 | 0.2851 | 0.7277 | 0.1506 | 0.2087 | 0.4530 | 0.7700 |
| same_model_same_prompt_family | family | within_team | qwen3.5-plus | qwen3.5-plus | mixed_strategy | mixed_strategy | 100 | 0.5892 | 0.5810 | 0.5892 | 0.5969 | 0.1899 | 0.0537 | 0.1395 | 0.9500 |
| same_model_same_prompt_family | family | within_team | qwen3.5-plus | qwen3.5-plus | same_definition | same_definition | 100 | 0.3976 | 0.4716 | 0.3976 | 0.7785 | 0.1682 | 0.0483 | 0.1368 | 0.9500 |
| same_model_same_prompt_family | family | within_team | qwen3.5-plus | qwen3.5-plus | same_elimination | same_elimination | 100 | 0.3709 | 0.5017 | 0.3709 | 0.7490 | 0.1506 | 0.0416 | 0.1309 | 0.9600 |

先把 baseline 口径拆开。`same_model_same_prompt` 是逐字同 prompt 的 exact baseline；`same_model_same_prompt_family` 是同一模型、同一 prompt family 下的 family-level baseline。前者才是后面 signed delta 的零点，后者只是“同家族不同表述”的对照。

## Full Metrics

| contrast | prompt_mode | unit | left_model | right_model | left_prompt_family | right_prompt_family | n | family_div | major_div | homogeneity | prompt_embedding_div | trace_embedding_div | trace_token_div | vote_acc | major_dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | gemini-2.5-flash-lite | mixed_strategy | same_definition | 100 | 0.5679 | 0.6363 | 0.5756 | 0.1762 | 0.0536 | 0.1947 | 0.8650 | 0.3608 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | gemini-2.5-flash-lite | mixed_strategy | same_elimination | 100 | 0.5572 | 0.6155 | 0.5895 | 0.1798 | 0.0530 | 0.1950 | 0.8450 | 0.3780 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | gemini-2.5-flash-lite | same_definition | same_elimination | 100 | 0.6058 | 0.6334 | 0.5186 | 0.1923 | 0.0547 | 0.1958 | 0.8550 | 0.4436 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | gpt-4o-mini | mixed_strategy | same_definition | 100 | 0.5934 | 0.6304 | 0.5642 | 0.1762 | 0.0635 | 0.2325 | 0.8800 | 0.3944 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | gpt-4o-mini | mixed_strategy | same_elimination | 100 | 0.5801 | 0.6440 | 0.5513 | 0.1798 | 0.0603 | 0.2215 | 0.8650 | 0.3918 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | gpt-4o-mini | same_definition | same_elimination | 100 | 0.6117 | 0.6572 | 0.4859 | 0.1923 | 0.0627 | 0.2246 | 0.8750 | 0.4628 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | qwen2.5-7b-instruct | mixed_strategy | same_definition | 100 | 0.6102 | 0.6467 | 0.5235 | 0.1762 | 0.1637 | 0.4143 | 0.8100 | 0.3958 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | qwen2.5-7b-instruct | mixed_strategy | same_elimination | 100 | 0.6042 | 0.6375 | 0.5277 | 0.1798 | 0.1718 | 0.4266 | 0.8300 | 0.4108 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | qwen2.5-7b-instruct | same_definition | same_elimination | 100 | 0.6244 | 0.6546 | 0.4720 | 0.1923 | 0.1729 | 0.4245 | 0.8400 | 0.4918 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | qwen3.5-plus | mixed_strategy | same_definition | 100 | 0.5917 | 0.6493 | 0.5629 | 0.1762 | 0.0621 | 0.2235 | 0.9200 | 0.3764 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | qwen3.5-plus | mixed_strategy | same_elimination | 100 | 0.5969 | 0.6349 | 0.5673 | 0.1798 | 0.0578 | 0.2175 | 0.9250 | 0.3564 |
| different_model_different_prompt_family | family | between_team_same_question | deepseek-chat | qwen3.5-plus | same_definition | same_elimination | 100 | 0.6180 | 0.6272 | 0.5159 | 0.1923 | 0.0590 | 0.2185 | 0.9350 | 0.4208 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | deepseek-chat | mixed_strategy | same_definition | 100 | 0.6012 | 0.6325 | 0.5183 | 0.1762 | 0.0550 | 0.1966 | 0.8550 | 0.4382 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | deepseek-chat | mixed_strategy | same_elimination | 100 | 0.5180 | 0.5422 | 0.6520 | 0.1798 | 0.0517 | 0.1867 | 0.8500 | 0.3478 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | deepseek-chat | same_definition | same_elimination | 100 | 0.5303 | 0.5643 | 0.6321 | 0.1923 | 0.0516 | 0.1860 | 0.8700 | 0.3472 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gpt-4o-mini | mixed_strategy | same_definition | 100 | 0.4808 | 0.3796 | 0.7485 | 0.1762 | 0.0532 | 0.1525 | 0.8350 | 0.2008 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gpt-4o-mini | mixed_strategy | same_elimination | 100 | 0.4532 | 0.3925 | 0.7467 | 0.1798 | 0.0481 | 0.1412 | 0.8200 | 0.2112 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gpt-4o-mini | same_definition | same_elimination | 100 | 0.4868 | 0.4442 | 0.7187 | 0.1923 | 0.0474 | 0.1405 | 0.8400 | 0.2248 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen2.5-7b-instruct | mixed_strategy | same_definition | 100 | 0.4954 | 0.3886 | 0.7038 | 0.1762 | 0.1594 | 0.3585 | 0.7650 | 0.1906 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen2.5-7b-instruct | mixed_strategy | same_elimination | 100 | 0.4756 | 0.3683 | 0.7181 | 0.1798 | 0.1673 | 0.3736 | 0.7850 | 0.2066 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen2.5-7b-instruct | same_definition | same_elimination | 100 | 0.4979 | 0.4131 | 0.6941 | 0.1923 | 0.1666 | 0.3711 | 0.8050 | 0.2222 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen3.5-plus | mixed_strategy | same_definition | 100 | 0.4915 | 0.4083 | 0.7279 | 0.1762 | 0.0564 | 0.1579 | 0.8750 | 0.2046 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen3.5-plus | mixed_strategy | same_elimination | 100 | 0.4826 | 0.3844 | 0.7495 | 0.1798 | 0.0517 | 0.1515 | 0.8800 | 0.1592 |
| different_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen3.5-plus | same_definition | same_elimination | 100 | 0.5109 | 0.4326 | 0.7226 | 0.1923 | 0.0513 | 0.1505 | 0.9000 | 0.1876 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | deepseek-chat | mixed_strategy | same_definition | 100 | 0.6119 | 0.6641 | 0.4839 | 0.1762 | 0.0634 | 0.2301 | 0.8850 | 0.4414 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | deepseek-chat | mixed_strategy | same_elimination | 100 | 0.5507 | 0.5718 | 0.6140 | 0.1798 | 0.0595 | 0.2169 | 0.8800 | 0.3478 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | deepseek-chat | same_definition | same_elimination | 100 | 0.5521 | 0.5424 | 0.6273 | 0.1923 | 0.0623 | 0.2242 | 0.8850 | 0.3720 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | gemini-2.5-flash-lite | mixed_strategy | same_definition | 100 | 0.4945 | 0.4562 | 0.6946 | 0.1762 | 0.0501 | 0.1469 | 0.8500 | 0.2332 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | gemini-2.5-flash-lite | mixed_strategy | same_elimination | 100 | 0.4662 | 0.4087 | 0.7266 | 0.1798 | 0.0496 | 0.1470 | 0.8300 | 0.2134 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | gemini-2.5-flash-lite | same_definition | same_elimination | 100 | 0.4766 | 0.3701 | 0.7517 | 0.1923 | 0.0521 | 0.1516 | 0.8350 | 0.1926 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen2.5-7b-instruct | mixed_strategy | same_definition | 100 | 0.5244 | 0.4373 | 0.6666 | 0.1762 | 0.1604 | 0.3648 | 0.7950 | 0.1852 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen2.5-7b-instruct | mixed_strategy | same_elimination | 100 | 0.4916 | 0.3897 | 0.6882 | 0.1798 | 0.1685 | 0.3776 | 0.8150 | 0.1842 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen2.5-7b-instruct | same_definition | same_elimination | 100 | 0.4911 | 0.3528 | 0.7147 | 0.1923 | 0.1702 | 0.3757 | 0.8200 | 0.1818 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen3.5-plus | mixed_strategy | same_definition | 100 | 0.5215 | 0.4572 | 0.6745 | 0.1762 | 0.0670 | 0.1993 | 0.9050 | 0.2518 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen3.5-plus | mixed_strategy | same_elimination | 100 | 0.5138 | 0.4297 | 0.6971 | 0.1798 | 0.0612 | 0.1903 | 0.9100 | 0.2036 |
| different_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen3.5-plus | same_definition | same_elimination | 100 | 0.5090 | 0.3882 | 0.7336 | 0.1923 | 0.0625 | 0.1922 | 0.9150 | 0.1822 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | deepseek-chat | mixed_strategy | same_definition | 100 | 0.6403 | 0.6814 | 0.4551 | 0.1762 | 0.1483 | 0.3879 | 0.8300 | 0.4416 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | deepseek-chat | mixed_strategy | same_elimination | 100 | 0.5770 | 0.6100 | 0.5663 | 0.1798 | 0.1456 | 0.3818 | 0.8250 | 0.3594 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | deepseek-chat | same_definition | same_elimination | 100 | 0.5732 | 0.5751 | 0.5837 | 0.1923 | 0.1631 | 0.4090 | 0.8150 | 0.3742 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | gemini-2.5-flash-lite | mixed_strategy | same_definition | 100 | 0.5196 | 0.4836 | 0.6535 | 0.1762 | 0.1411 | 0.3250 | 0.7950 | 0.2284 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | gemini-2.5-flash-lite | mixed_strategy | same_elimination | 100 | 0.4848 | 0.4282 | 0.6875 | 0.1798 | 0.1410 | 0.3264 | 0.7750 | 0.2014 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | gemini-2.5-flash-lite | same_definition | same_elimination | 100 | 0.4891 | 0.3859 | 0.7111 | 0.1923 | 0.1591 | 0.3574 | 0.7650 | 0.1866 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | gpt-4o-mini | mixed_strategy | same_definition | 100 | 0.5243 | 0.4425 | 0.6741 | 0.1762 | 0.1436 | 0.3326 | 0.8100 | 0.1982 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | gpt-4o-mini | mixed_strategy | same_elimination | 100 | 0.4971 | 0.4300 | 0.6703 | 0.1798 | 0.1399 | 0.3285 | 0.7950 | 0.1618 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | gpt-4o-mini | same_definition | same_elimination | 100 | 0.4985 | 0.3883 | 0.6900 | 0.1923 | 0.1593 | 0.3612 | 0.7850 | 0.1696 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen3.5-plus | mixed_strategy | same_definition | 100 | 0.5447 | 0.4972 | 0.6334 | 0.1762 | 0.1531 | 0.3622 | 0.8500 | 0.2570 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen3.5-plus | mixed_strategy | same_elimination | 100 | 0.5350 | 0.4492 | 0.6489 | 0.1798 | 0.1488 | 0.3576 | 0.8550 | 0.2028 |
| different_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen3.5-plus | same_definition | same_elimination | 100 | 0.5295 | 0.4057 | 0.6789 | 0.1923 | 0.1648 | 0.3852 | 0.8450 | 0.1950 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | deepseek-chat | mixed_strategy | same_definition | 100 | 0.6361 | 0.6758 | 0.4639 | 0.1922 | 0.0646 | 0.2214 | 0.9300 | 0.4144 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | deepseek-chat | mixed_strategy | same_elimination | 100 | 0.5871 | 0.6476 | 0.5522 | 0.1937 | 0.0625 | 0.2148 | 0.9250 | 0.4130 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | deepseek-chat | same_definition | same_elimination | 100 | 0.5662 | 0.5887 | 0.6058 | 0.1923 | 0.0621 | 0.2202 | 0.9250 | 0.3800 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | gemini-2.5-flash-lite | mixed_strategy | same_definition | 100 | 0.5594 | 0.5480 | 0.6402 | 0.1922 | 0.0560 | 0.1499 | 0.8950 | 0.2632 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | gemini-2.5-flash-lite | mixed_strategy | same_elimination | 100 | 0.5377 | 0.5153 | 0.6590 | 0.1937 | 0.0561 | 0.1506 | 0.8750 | 0.2662 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | gemini-2.5-flash-lite | same_definition | same_elimination | 100 | 0.5055 | 0.4295 | 0.7140 | 0.1923 | 0.0563 | 0.1577 | 0.8750 | 0.2376 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | gpt-4o-mini | mixed_strategy | same_definition | 100 | 0.5584 | 0.5171 | 0.6520 | 0.1922 | 0.0681 | 0.1934 | 0.9100 | 0.2644 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | gpt-4o-mini | mixed_strategy | same_elimination | 100 | 0.5593 | 0.5407 | 0.6152 | 0.1937 | 0.0652 | 0.1870 | 0.8950 | 0.3140 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | gpt-4o-mini | same_definition | same_elimination | 100 | 0.5296 | 0.4566 | 0.6687 | 0.1923 | 0.0665 | 0.1952 | 0.8950 | 0.2816 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | qwen2.5-7b-instruct | mixed_strategy | same_definition | 100 | 0.5828 | 0.5329 | 0.5989 | 0.1922 | 0.1687 | 0.3870 | 0.8400 | 0.2834 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | qwen2.5-7b-instruct | mixed_strategy | same_elimination | 100 | 0.5760 | 0.5352 | 0.5930 | 0.1937 | 0.1769 | 0.4007 | 0.8600 | 0.3316 |
| different_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | qwen2.5-7b-instruct | same_definition | same_elimination | 100 | 0.5275 | 0.4355 | 0.6583 | 0.1923 | 0.1776 | 0.4021 | 0.8600 | 0.2788 |
| different_model_same_prompt | exact | between_team_same_question | gpt-4o-mini | qwen3.5-plus | same_prompt | same_prompt | 100 | 0.4440 | 0.3288 | 0.7735 | 0.0000 | 0.0537 | 0.1658 | 0.9050 | 0.1816 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | gemini-2.5-flash-lite | mixed_strategy | mixed_strategy | 100 | 0.5609 | 0.6170 | 0.5930 | 0.1554 | 0.0534 | 0.1958 | 0.8450 | 0.3548 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | gemini-2.5-flash-lite | same_definition | same_definition | 100 | 0.6148 | 0.6496 | 0.5137 | 0.1496 | 0.0549 | 0.1955 | 0.8750 | 0.4266 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | gemini-2.5-flash-lite | same_elimination | same_elimination | 100 | 0.5147 | 0.5403 | 0.6562 | 0.1339 | 0.0509 | 0.1855 | 0.8500 | 0.3528 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | gpt-4o-mini | mixed_strategy | mixed_strategy | 100 | 0.5872 | 0.6479 | 0.5441 | 0.1554 | 0.0618 | 0.2287 | 0.8750 | 0.3622 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | gpt-4o-mini | same_definition | same_definition | 100 | 0.6051 | 0.6466 | 0.5145 | 0.1496 | 0.0633 | 0.2305 | 0.8900 | 0.4650 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | gpt-4o-mini | same_elimination | same_elimination | 100 | 0.5286 | 0.5447 | 0.6306 | 0.1339 | 0.0567 | 0.2069 | 0.8700 | 0.3548 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | qwen2.5-7b-instruct | mixed_strategy | mixed_strategy | 100 | 0.6019 | 0.6674 | 0.5069 | 0.1554 | 0.1474 | 0.3887 | 0.8200 | 0.3590 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | qwen2.5-7b-instruct | same_definition | same_definition | 100 | 0.6213 | 0.6479 | 0.4868 | 0.1496 | 0.1632 | 0.4103 | 0.8200 | 0.4524 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | qwen2.5-7b-instruct | same_elimination | same_elimination | 100 | 0.5514 | 0.5370 | 0.6038 | 0.1339 | 0.1693 | 0.4180 | 0.8350 | 0.3706 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | qwen3.5-plus | mixed_strategy | mixed_strategy | 100 | 0.6193 | 0.6972 | 0.5018 | 0.1833 | 0.0633 | 0.2203 | 0.9200 | 0.3668 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | qwen3.5-plus | same_definition | same_definition | 100 | 0.5898 | 0.6492 | 0.5382 | 0.1496 | 0.0622 | 0.2210 | 0.9300 | 0.4288 |
| different_model_same_prompt_family | family | between_team_same_question | deepseek-chat | qwen3.5-plus | same_elimination | same_elimination | 100 | 0.5540 | 0.5468 | 0.6318 | 0.1339 | 0.0565 | 0.2102 | 0.9300 | 0.3368 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gpt-4o-mini | mixed_strategy | mixed_strategy | 100 | 0.4723 | 0.4144 | 0.7218 | 0.1554 | 0.0509 | 0.1483 | 0.8300 | 0.2112 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gpt-4o-mini | same_definition | same_definition | 100 | 0.5061 | 0.4266 | 0.7218 | 0.1496 | 0.0520 | 0.1502 | 0.8550 | 0.2228 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gpt-4o-mini | same_elimination | same_elimination | 100 | 0.4469 | 0.3967 | 0.7561 | 0.1339 | 0.0463 | 0.1385 | 0.8200 | 0.2154 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen2.5-7b-instruct | mixed_strategy | mixed_strategy | 100 | 0.4972 | 0.4329 | 0.6776 | 0.1554 | 0.1415 | 0.3274 | 0.7750 | 0.2110 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen2.5-7b-instruct | same_definition | same_definition | 100 | 0.5259 | 0.4415 | 0.6828 | 0.1496 | 0.1586 | 0.3562 | 0.7850 | 0.2096 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen2.5-7b-instruct | same_elimination | same_elimination | 100 | 0.4667 | 0.3519 | 0.7298 | 0.1339 | 0.1663 | 0.3721 | 0.7850 | 0.1894 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen3.5-plus | mixed_strategy | mixed_strategy | 100 | 0.5299 | 0.5077 | 0.6640 | 0.1833 | 0.0565 | 0.1516 | 0.8750 | 0.2522 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen3.5-plus | same_definition | same_definition | 100 | 0.5263 | 0.4649 | 0.7004 | 0.1496 | 0.0557 | 0.1557 | 0.8950 | 0.2442 |
| different_model_same_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | qwen3.5-plus | same_elimination | same_elimination | 100 | 0.4833 | 0.3878 | 0.7472 | 0.1339 | 0.0510 | 0.1503 | 0.8800 | 0.1822 |
| different_model_same_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen2.5-7b-instruct | mixed_strategy | mixed_strategy | 100 | 0.5277 | 0.4743 | 0.6430 | 0.1554 | 0.1424 | 0.3342 | 0.8050 | 0.1860 |
| different_model_same_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen2.5-7b-instruct | same_definition | same_definition | 100 | 0.5057 | 0.3870 | 0.7109 | 0.1496 | 0.1593 | 0.3593 | 0.8000 | 0.1658 |
| different_model_same_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen2.5-7b-instruct | same_elimination | same_elimination | 100 | 0.4670 | 0.3501 | 0.7223 | 0.1339 | 0.1650 | 0.3699 | 0.8050 | 0.1460 |
| different_model_same_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen3.5-plus | mixed_strategy | mixed_strategy | 100 | 0.5666 | 0.5537 | 0.6093 | 0.1833 | 0.0671 | 0.1925 | 0.9050 | 0.2840 |
| different_model_same_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen3.5-plus | same_definition | same_definition | 100 | 0.5105 | 0.4234 | 0.7327 | 0.1496 | 0.0660 | 0.1976 | 0.9100 | 0.2066 |
| different_model_same_prompt_family | family | between_team_same_question | gpt-4o-mini | qwen3.5-plus | same_elimination | same_elimination | 100 | 0.4923 | 0.3928 | 0.7163 | 0.1339 | 0.0592 | 0.1824 | 0.9000 | 0.1906 |
| different_model_same_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen3.5-plus | mixed_strategy | mixed_strategy | 100 | 0.5742 | 0.5616 | 0.5677 | 0.1833 | 0.1535 | 0.3603 | 0.8500 | 0.2928 |
| different_model_same_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen3.5-plus | same_definition | same_definition | 100 | 0.5355 | 0.4485 | 0.6778 | 0.1496 | 0.1674 | 0.3868 | 0.8400 | 0.2324 |
| different_model_same_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen3.5-plus | same_elimination | same_elimination | 100 | 0.5130 | 0.3827 | 0.6880 | 0.1339 | 0.1726 | 0.3965 | 0.8650 | 0.2062 |
| same_model_different_prompt_family | family | between_team_same_question | deepseek-chat | deepseek-chat | mixed_strategy | same_definition | 100 | 0.6403 | 0.7279 | 0.4461 | 0.1762 | 0.0466 | 0.1806 | 0.9000 | 0.3692 |
| same_model_different_prompt_family | family | between_team_same_question | deepseek-chat | deepseek-chat | mixed_strategy | same_elimination | 100 | 0.5604 | 0.6467 | 0.5891 | 0.1798 | 0.0424 | 0.1671 | 0.8950 | 0.2486 |
| same_model_different_prompt_family | family | between_team_same_question | deepseek-chat | deepseek-chat | same_definition | same_elimination | 100 | 0.6250 | 0.6985 | 0.4761 | 0.1923 | 0.0471 | 0.1810 | 0.9050 | 0.4444 |
| same_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gemini-2.5-flash-lite | mixed_strategy | same_definition | 100 | 0.4539 | 0.3846 | 0.7882 | 0.1762 | 0.0303 | 0.0782 | 0.8200 | 0.1188 |
| same_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gemini-2.5-flash-lite | mixed_strategy | same_elimination | 100 | 0.4265 | 0.3385 | 0.8170 | 0.1798 | 0.0297 | 0.0788 | 0.8000 | 0.1090 |
| same_model_different_prompt_family | family | between_team_same_question | gemini-2.5-flash-lite | gemini-2.5-flash-lite | same_definition | same_elimination | 100 | 0.4489 | 0.3794 | 0.7875 | 0.1923 | 0.0298 | 0.0796 | 0.8200 | 0.1304 |
| same_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | gpt-4o-mini | mixed_strategy | same_definition | 100 | 0.4955 | 0.3944 | 0.7357 | 0.1762 | 0.0467 | 0.1411 | 0.8650 | 0.1490 |
| same_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | gpt-4o-mini | mixed_strategy | same_elimination | 100 | 0.4794 | 0.4342 | 0.7234 | 0.1798 | 0.0403 | 0.1288 | 0.8500 | 0.1804 |
| same_model_different_prompt_family | family | between_team_same_question | gpt-4o-mini | gpt-4o-mini | same_definition | same_elimination | 100 | 0.4781 | 0.3668 | 0.7447 | 0.1923 | 0.0467 | 0.1420 | 0.8550 | 0.1542 |
| same_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen2.5-7b-instruct | mixed_strategy | same_definition | 100 | 0.5214 | 0.4337 | 0.6533 | 0.1762 | 0.1877 | 0.4098 | 0.7400 | 0.1498 |
| same_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen2.5-7b-instruct | mixed_strategy | same_elimination | 100 | 0.4954 | 0.3958 | 0.6731 | 0.1798 | 0.1969 | 0.4301 | 0.7600 | 0.1350 |
| same_model_different_prompt_family | family | between_team_same_question | qwen2.5-7b-instruct | qwen2.5-7b-instruct | same_definition | same_elimination | 100 | 0.4809 | 0.3475 | 0.6953 | 0.1923 | 0.1962 | 0.4263 | 0.7500 | 0.1382 |
| same_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | qwen3.5-plus | mixed_strategy | same_definition | 100 | 0.5498 | 0.5150 | 0.6782 | 0.1922 | 0.0528 | 0.1426 | 0.9500 | 0.1860 |
| same_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | qwen3.5-plus | mixed_strategy | same_elimination | 100 | 0.5470 | 0.5033 | 0.6672 | 0.1937 | 0.0483 | 0.1365 | 0.9550 | 0.2008 |
| same_model_different_prompt_family | family | between_team_same_question | qwen3.5-plus | qwen3.5-plus | same_definition | same_elimination | 100 | 0.5093 | 0.4132 | 0.7407 | 0.1923 | 0.0475 | 0.1405 | 0.9550 | 0.1468 |
| same_model_same_prompt | exact | within_team | gpt-4o-mini | gpt-4o-mini | same_prompt | same_prompt | 100 | 0.3181 | 0.1905 | 0.8820 | 0.0000 | 0.0213 | 0.0746 | 0.8500 | 0.1905 |
| same_model_same_prompt | exact | within_team | qwen3.5-plus | qwen3.5-plus | same_prompt | same_prompt | 100 | 0.4362 | 0.3258 | 0.8025 | 0.0000 | 0.0256 | 0.0858 | 0.9600 | 0.3258 |
| same_model_same_prompt_family | family | within_team | deepseek-chat | deepseek-chat | mixed_strategy | mixed_strategy | 100 | 0.5819 | 0.7033 | 0.5092 | 0.1748 | 0.0444 | 0.1774 | 0.8900 | 0.7033 |
| same_model_same_prompt_family | family | within_team | deepseek-chat | deepseek-chat | same_definition | same_definition | 100 | 0.4807 | 0.6377 | 0.5372 | 0.1682 | 0.0477 | 0.1795 | 0.9100 | 0.6377 |
| same_model_same_prompt_family | family | within_team | deepseek-chat | deepseek-chat | same_elimination | same_elimination | 100 | 0.4786 | 0.4889 | 0.6915 | 0.1506 | 0.0371 | 0.1464 | 0.9000 | 0.4889 |
| same_model_same_prompt_family | family | within_team | gemini-2.5-flash-lite | gemini-2.5-flash-lite | mixed_strategy | mixed_strategy | 100 | 0.4268 | 0.3276 | 0.8141 | 0.1748 | 0.0305 | 0.0786 | 0.8000 | 0.3276 |
| same_model_same_prompt_family | family | within_team | gemini-2.5-flash-lite | gemini-2.5-flash-lite | same_definition | same_definition | 100 | 0.4598 | 0.4038 | 0.7653 | 0.1682 | 0.0293 | 0.0762 | 0.8400 | 0.4038 |
| same_model_same_prompt_family | family | within_team | gemini-2.5-flash-lite | gemini-2.5-flash-lite | same_elimination | same_elimination | 100 | 0.4057 | 0.3064 | 0.8260 | 0.1506 | 0.0285 | 0.0767 | 0.8000 | 0.3051 |
| same_model_same_prompt_family | family | within_team | gpt-4o-mini | gpt-4o-mini | mixed_strategy | mixed_strategy | 100 | 0.4805 | 0.4096 | 0.6995 | 0.1748 | 0.0453 | 0.1403 | 0.8600 | 0.4096 |
| same_model_same_prompt_family | family | within_team | gpt-4o-mini | gpt-4o-mini | same_definition | same_definition | 100 | 0.4722 | 0.3205 | 0.7896 | 0.1682 | 0.0453 | 0.1339 | 0.8700 | 0.3205 |
| same_model_same_prompt_family | family | within_team | gpt-4o-mini | gpt-4o-mini | same_elimination | same_elimination | 100 | 0.4210 | 0.3486 | 0.7618 | 0.1506 | 0.0328 | 0.1102 | 0.8400 | 0.3486 |
| same_model_same_prompt_family | family | within_team | qwen2.5-7b-instruct | qwen2.5-7b-instruct | mixed_strategy | mixed_strategy | 100 | 0.5278 | 0.4650 | 0.6076 | 0.1748 | 0.1948 | 0.4274 | 0.7500 | 0.4650 |
| same_model_same_prompt_family | family | within_team | qwen2.5-7b-instruct | qwen2.5-7b-instruct | same_definition | same_definition | 100 | 0.4953 | 0.3693 | 0.6990 | 0.1682 | 0.1844 | 0.3991 | 0.7300 | 0.3699 |
| same_model_same_prompt_family | family | within_team | qwen2.5-7b-instruct | qwen2.5-7b-instruct | same_elimination | same_elimination | 100 | 0.4565 | 0.2851 | 0.7277 | 0.1506 | 0.2087 | 0.4530 | 0.7700 | 0.2851 |
| same_model_same_prompt_family | family | within_team | qwen3.5-plus | qwen3.5-plus | mixed_strategy | mixed_strategy | 100 | 0.5810 | 0.5892 | 0.5969 | 0.1899 | 0.0537 | 0.1395 | 0.9500 | 0.5892 |
| same_model_same_prompt_family | family | within_team | qwen3.5-plus | qwen3.5-plus | same_definition | same_definition | 100 | 0.4716 | 0.3976 | 0.7785 | 0.1682 | 0.0483 | 0.1368 | 0.9500 | 0.3976 |
| same_model_same_prompt_family | family | within_team | qwen3.5-plus | qwen3.5-plus | same_elimination | same_elimination | 100 | 0.5017 | 0.3709 | 0.7490 | 0.1506 | 0.0416 | 0.1309 | 0.9600 | 0.3709 |

## Signed Delta vs exact same_model_same_prompt

| contrast | left_model | right_model | left_prompt_family | right_prompt_family | Δ family_div | Δ major_div | Δ homogeneity | Δ prompt_embedding_div | Δ trace_embedding_div | Δ trace_token_div | Δ vote_acc | Δ major_dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| different_model_different_prompt_family | deepseek-chat | gemini-2.5-flash-lite | mixed_strategy | same_definition |  |  |  |  |  |  |  | +0.1026 |
| different_model_different_prompt_family | deepseek-chat | gemini-2.5-flash-lite | mixed_strategy | same_elimination |  |  |  |  |  |  |  | +0.1198 |
| different_model_different_prompt_family | deepseek-chat | gemini-2.5-flash-lite | same_definition | same_elimination |  |  |  |  |  |  |  | +0.1854 |
| different_model_different_prompt_family | deepseek-chat | gpt-4o-mini | mixed_strategy | same_definition |  |  |  |  |  |  |  | +0.1362 |
| different_model_different_prompt_family | deepseek-chat | gpt-4o-mini | mixed_strategy | same_elimination |  |  |  |  |  |  |  | +0.1336 |
| different_model_different_prompt_family | deepseek-chat | gpt-4o-mini | same_definition | same_elimination |  |  |  |  |  |  |  | +0.2046 |
| different_model_different_prompt_family | deepseek-chat | qwen2.5-7b-instruct | mixed_strategy | same_definition |  |  |  |  |  |  |  | +0.1376 |
| different_model_different_prompt_family | deepseek-chat | qwen2.5-7b-instruct | mixed_strategy | same_elimination |  |  |  |  |  |  |  | +0.1526 |
| different_model_different_prompt_family | deepseek-chat | qwen2.5-7b-instruct | same_definition | same_elimination |  |  |  |  |  |  |  | +0.2336 |
| different_model_different_prompt_family | deepseek-chat | qwen3.5-plus | mixed_strategy | same_definition |  |  |  |  |  |  |  | +0.1182 |
| different_model_different_prompt_family | deepseek-chat | qwen3.5-plus | mixed_strategy | same_elimination |  |  |  |  |  |  |  | +0.0982 |
| different_model_different_prompt_family | deepseek-chat | qwen3.5-plus | same_definition | same_elimination |  |  |  |  |  |  |  | +0.1626 |
| different_model_different_prompt_family | gemini-2.5-flash-lite | deepseek-chat | mixed_strategy | same_definition |  |  |  |  |  |  |  | +0.1800 |
| different_model_different_prompt_family | gemini-2.5-flash-lite | deepseek-chat | mixed_strategy | same_elimination |  |  |  |  |  |  |  | +0.0896 |
| different_model_different_prompt_family | gemini-2.5-flash-lite | deepseek-chat | same_definition | same_elimination |  |  |  |  |  |  |  | +0.0890 |
| different_model_different_prompt_family | gemini-2.5-flash-lite | gpt-4o-mini | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.0574 |
| different_model_different_prompt_family | gemini-2.5-flash-lite | gpt-4o-mini | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.0470 |
| different_model_different_prompt_family | gemini-2.5-flash-lite | gpt-4o-mini | same_definition | same_elimination |  |  |  |  |  |  |  | -0.0334 |
| different_model_different_prompt_family | gemini-2.5-flash-lite | qwen2.5-7b-instruct | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.0676 |
| different_model_different_prompt_family | gemini-2.5-flash-lite | qwen2.5-7b-instruct | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.0516 |
| different_model_different_prompt_family | gemini-2.5-flash-lite | qwen2.5-7b-instruct | same_definition | same_elimination |  |  |  |  |  |  |  | -0.0360 |
| different_model_different_prompt_family | gemini-2.5-flash-lite | qwen3.5-plus | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.0536 |
| different_model_different_prompt_family | gemini-2.5-flash-lite | qwen3.5-plus | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.0990 |
| different_model_different_prompt_family | gemini-2.5-flash-lite | qwen3.5-plus | same_definition | same_elimination |  |  |  |  |  |  |  | -0.0706 |
| different_model_different_prompt_family | gpt-4o-mini | deepseek-chat | mixed_strategy | same_definition |  |  |  |  |  |  |  | +0.1832 |
| different_model_different_prompt_family | gpt-4o-mini | deepseek-chat | mixed_strategy | same_elimination |  |  |  |  |  |  |  | +0.0896 |
| different_model_different_prompt_family | gpt-4o-mini | deepseek-chat | same_definition | same_elimination |  |  |  |  |  |  |  | +0.1138 |
| different_model_different_prompt_family | gpt-4o-mini | gemini-2.5-flash-lite | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.0250 |
| different_model_different_prompt_family | gpt-4o-mini | gemini-2.5-flash-lite | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.0448 |
| different_model_different_prompt_family | gpt-4o-mini | gemini-2.5-flash-lite | same_definition | same_elimination |  |  |  |  |  |  |  | -0.0656 |
| different_model_different_prompt_family | gpt-4o-mini | qwen2.5-7b-instruct | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.0730 |
| different_model_different_prompt_family | gpt-4o-mini | qwen2.5-7b-instruct | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.0740 |
| different_model_different_prompt_family | gpt-4o-mini | qwen2.5-7b-instruct | same_definition | same_elimination |  |  |  |  |  |  |  | -0.0764 |
| different_model_different_prompt_family | gpt-4o-mini | qwen3.5-plus | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.0064 |
| different_model_different_prompt_family | gpt-4o-mini | qwen3.5-plus | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.0546 |
| different_model_different_prompt_family | gpt-4o-mini | qwen3.5-plus | same_definition | same_elimination |  |  |  |  |  |  |  | -0.0760 |
| different_model_different_prompt_family | qwen2.5-7b-instruct | deepseek-chat | mixed_strategy | same_definition |  |  |  |  |  |  |  | +0.1834 |
| different_model_different_prompt_family | qwen2.5-7b-instruct | deepseek-chat | mixed_strategy | same_elimination |  |  |  |  |  |  |  | +0.1012 |
| different_model_different_prompt_family | qwen2.5-7b-instruct | deepseek-chat | same_definition | same_elimination |  |  |  |  |  |  |  | +0.1160 |
| different_model_different_prompt_family | qwen2.5-7b-instruct | gemini-2.5-flash-lite | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.0298 |
| different_model_different_prompt_family | qwen2.5-7b-instruct | gemini-2.5-flash-lite | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.0568 |
| different_model_different_prompt_family | qwen2.5-7b-instruct | gemini-2.5-flash-lite | same_definition | same_elimination |  |  |  |  |  |  |  | -0.0716 |
| different_model_different_prompt_family | qwen2.5-7b-instruct | gpt-4o-mini | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.0600 |
| different_model_different_prompt_family | qwen2.5-7b-instruct | gpt-4o-mini | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.0964 |
| different_model_different_prompt_family | qwen2.5-7b-instruct | gpt-4o-mini | same_definition | same_elimination |  |  |  |  |  |  |  | -0.0886 |
| different_model_different_prompt_family | qwen2.5-7b-instruct | qwen3.5-plus | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.0012 |
| different_model_different_prompt_family | qwen2.5-7b-instruct | qwen3.5-plus | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.0554 |
| different_model_different_prompt_family | qwen2.5-7b-instruct | qwen3.5-plus | same_definition | same_elimination |  |  |  |  |  |  |  | -0.0632 |
| different_model_different_prompt_family | qwen3.5-plus | deepseek-chat | mixed_strategy | same_definition |  |  |  |  |  |  |  | +0.1562 |
| different_model_different_prompt_family | qwen3.5-plus | deepseek-chat | mixed_strategy | same_elimination |  |  |  |  |  |  |  | +0.1548 |
| different_model_different_prompt_family | qwen3.5-plus | deepseek-chat | same_definition | same_elimination |  |  |  |  |  |  |  | +0.1218 |
| different_model_different_prompt_family | qwen3.5-plus | gemini-2.5-flash-lite | mixed_strategy | same_definition |  |  |  |  |  |  |  | +0.0050 |
| different_model_different_prompt_family | qwen3.5-plus | gemini-2.5-flash-lite | mixed_strategy | same_elimination |  |  |  |  |  |  |  | +0.0080 |
| different_model_different_prompt_family | qwen3.5-plus | gemini-2.5-flash-lite | same_definition | same_elimination |  |  |  |  |  |  |  | -0.0206 |
| different_model_different_prompt_family | qwen3.5-plus | gpt-4o-mini | mixed_strategy | same_definition |  |  |  |  |  |  |  | +0.0062 |
| different_model_different_prompt_family | qwen3.5-plus | gpt-4o-mini | mixed_strategy | same_elimination |  |  |  |  |  |  |  | +0.0558 |
| different_model_different_prompt_family | qwen3.5-plus | gpt-4o-mini | same_definition | same_elimination |  |  |  |  |  |  |  | +0.0234 |
| different_model_different_prompt_family | qwen3.5-plus | qwen2.5-7b-instruct | mixed_strategy | same_definition |  |  |  |  |  |  |  | +0.0252 |
| different_model_different_prompt_family | qwen3.5-plus | qwen2.5-7b-instruct | mixed_strategy | same_elimination |  |  |  |  |  |  |  | +0.0734 |
| different_model_different_prompt_family | qwen3.5-plus | qwen2.5-7b-instruct | same_definition | same_elimination |  |  |  |  |  |  |  | +0.0206 |
| different_model_same_prompt | gpt-4o-mini | qwen3.5-plus | same_prompt | same_prompt |  |  |  |  |  |  |  | -0.0766 |
| different_model_same_prompt_family | deepseek-chat | gemini-2.5-flash-lite | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | +0.0966 |
| different_model_same_prompt_family | deepseek-chat | gemini-2.5-flash-lite | same_definition | same_definition |  |  |  |  |  |  |  | +0.1684 |
| different_model_same_prompt_family | deepseek-chat | gemini-2.5-flash-lite | same_elimination | same_elimination |  |  |  |  |  |  |  | +0.0946 |
| different_model_same_prompt_family | deepseek-chat | gpt-4o-mini | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | +0.1040 |
| different_model_same_prompt_family | deepseek-chat | gpt-4o-mini | same_definition | same_definition |  |  |  |  |  |  |  | +0.2068 |
| different_model_same_prompt_family | deepseek-chat | gpt-4o-mini | same_elimination | same_elimination |  |  |  |  |  |  |  | +0.0966 |
| different_model_same_prompt_family | deepseek-chat | qwen2.5-7b-instruct | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | +0.1008 |
| different_model_same_prompt_family | deepseek-chat | qwen2.5-7b-instruct | same_definition | same_definition |  |  |  |  |  |  |  | +0.1942 |
| different_model_same_prompt_family | deepseek-chat | qwen2.5-7b-instruct | same_elimination | same_elimination |  |  |  |  |  |  |  | +0.1124 |
| different_model_same_prompt_family | deepseek-chat | qwen3.5-plus | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | +0.1086 |
| different_model_same_prompt_family | deepseek-chat | qwen3.5-plus | same_definition | same_definition |  |  |  |  |  |  |  | +0.1706 |
| different_model_same_prompt_family | deepseek-chat | qwen3.5-plus | same_elimination | same_elimination |  |  |  |  |  |  |  | +0.0786 |
| different_model_same_prompt_family | gemini-2.5-flash-lite | gpt-4o-mini | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | -0.0470 |
| different_model_same_prompt_family | gemini-2.5-flash-lite | gpt-4o-mini | same_definition | same_definition |  |  |  |  |  |  |  | -0.0354 |
| different_model_same_prompt_family | gemini-2.5-flash-lite | gpt-4o-mini | same_elimination | same_elimination |  |  |  |  |  |  |  | -0.0428 |
| different_model_same_prompt_family | gemini-2.5-flash-lite | qwen2.5-7b-instruct | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | -0.0472 |
| different_model_same_prompt_family | gemini-2.5-flash-lite | qwen2.5-7b-instruct | same_definition | same_definition |  |  |  |  |  |  |  | -0.0486 |
| different_model_same_prompt_family | gemini-2.5-flash-lite | qwen2.5-7b-instruct | same_elimination | same_elimination |  |  |  |  |  |  |  | -0.0688 |
| different_model_same_prompt_family | gemini-2.5-flash-lite | qwen3.5-plus | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | -0.0060 |
| different_model_same_prompt_family | gemini-2.5-flash-lite | qwen3.5-plus | same_definition | same_definition |  |  |  |  |  |  |  | -0.0140 |
| different_model_same_prompt_family | gemini-2.5-flash-lite | qwen3.5-plus | same_elimination | same_elimination |  |  |  |  |  |  |  | -0.0760 |
| different_model_same_prompt_family | gpt-4o-mini | qwen2.5-7b-instruct | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | -0.0722 |
| different_model_same_prompt_family | gpt-4o-mini | qwen2.5-7b-instruct | same_definition | same_definition |  |  |  |  |  |  |  | -0.0924 |
| different_model_same_prompt_family | gpt-4o-mini | qwen2.5-7b-instruct | same_elimination | same_elimination |  |  |  |  |  |  |  | -0.1122 |
| different_model_same_prompt_family | gpt-4o-mini | qwen3.5-plus | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | +0.0258 |
| different_model_same_prompt_family | gpt-4o-mini | qwen3.5-plus | same_definition | same_definition |  |  |  |  |  |  |  | -0.0516 |
| different_model_same_prompt_family | gpt-4o-mini | qwen3.5-plus | same_elimination | same_elimination |  |  |  |  |  |  |  | -0.0676 |
| different_model_same_prompt_family | qwen2.5-7b-instruct | qwen3.5-plus | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | +0.0346 |
| different_model_same_prompt_family | qwen2.5-7b-instruct | qwen3.5-plus | same_definition | same_definition |  |  |  |  |  |  |  | -0.0258 |
| different_model_same_prompt_family | qwen2.5-7b-instruct | qwen3.5-plus | same_elimination | same_elimination |  |  |  |  |  |  |  | -0.0520 |
| same_model_different_prompt_family | deepseek-chat | deepseek-chat | mixed_strategy | same_definition |  |  |  |  |  |  |  | +0.1110 |
| same_model_different_prompt_family | deepseek-chat | deepseek-chat | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.0096 |
| same_model_different_prompt_family | deepseek-chat | deepseek-chat | same_definition | same_elimination |  |  |  |  |  |  |  | +0.1862 |
| same_model_different_prompt_family | gemini-2.5-flash-lite | gemini-2.5-flash-lite | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.1394 |
| same_model_different_prompt_family | gemini-2.5-flash-lite | gemini-2.5-flash-lite | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.1492 |
| same_model_different_prompt_family | gemini-2.5-flash-lite | gemini-2.5-flash-lite | same_definition | same_elimination |  |  |  |  |  |  |  | -0.1278 |
| same_model_different_prompt_family | gpt-4o-mini | gpt-4o-mini | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.1092 |
| same_model_different_prompt_family | gpt-4o-mini | gpt-4o-mini | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.0778 |
| same_model_different_prompt_family | gpt-4o-mini | gpt-4o-mini | same_definition | same_elimination |  |  |  |  |  |  |  | -0.1040 |
| same_model_different_prompt_family | qwen2.5-7b-instruct | qwen2.5-7b-instruct | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.1084 |
| same_model_different_prompt_family | qwen2.5-7b-instruct | qwen2.5-7b-instruct | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.1232 |
| same_model_different_prompt_family | qwen2.5-7b-instruct | qwen2.5-7b-instruct | same_definition | same_elimination |  |  |  |  |  |  |  | -0.1200 |
| same_model_different_prompt_family | qwen3.5-plus | qwen3.5-plus | mixed_strategy | same_definition |  |  |  |  |  |  |  | -0.0722 |
| same_model_different_prompt_family | qwen3.5-plus | qwen3.5-plus | mixed_strategy | same_elimination |  |  |  |  |  |  |  | -0.0574 |
| same_model_different_prompt_family | qwen3.5-plus | qwen3.5-plus | same_definition | same_elimination |  |  |  |  |  |  |  | -0.1114 |
| same_model_same_prompt | gpt-4o-mini | gpt-4o-mini | same_prompt | same_prompt |  |  |  |  |  |  |  | -0.0676 |
| same_model_same_prompt | qwen3.5-plus | qwen3.5-plus | same_prompt | same_prompt |  |  |  |  |  |  |  | +0.0676 |
| same_model_same_prompt_family | deepseek-chat | deepseek-chat | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | +0.4451 |
| same_model_same_prompt_family | deepseek-chat | deepseek-chat | same_definition | same_definition |  |  |  |  |  |  |  | +0.3795 |
| same_model_same_prompt_family | deepseek-chat | deepseek-chat | same_elimination | same_elimination |  |  |  |  |  |  |  | +0.2308 |
| same_model_same_prompt_family | gemini-2.5-flash-lite | gemini-2.5-flash-lite | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | +0.0695 |
| same_model_same_prompt_family | gemini-2.5-flash-lite | gemini-2.5-flash-lite | same_definition | same_definition |  |  |  |  |  |  |  | +0.1456 |
| same_model_same_prompt_family | gemini-2.5-flash-lite | gemini-2.5-flash-lite | same_elimination | same_elimination |  |  |  |  |  |  |  | +0.0470 |
| same_model_same_prompt_family | gpt-4o-mini | gpt-4o-mini | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | +0.1514 |
| same_model_same_prompt_family | gpt-4o-mini | gpt-4o-mini | same_definition | same_definition |  |  |  |  |  |  |  | +0.0623 |
| same_model_same_prompt_family | gpt-4o-mini | gpt-4o-mini | same_elimination | same_elimination |  |  |  |  |  |  |  | +0.0904 |
| same_model_same_prompt_family | qwen2.5-7b-instruct | qwen2.5-7b-instruct | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | +0.2069 |
| same_model_same_prompt_family | qwen2.5-7b-instruct | qwen2.5-7b-instruct | same_definition | same_definition |  |  |  |  |  |  |  | +0.1117 |
| same_model_same_prompt_family | qwen2.5-7b-instruct | qwen2.5-7b-instruct | same_elimination | same_elimination |  |  |  |  |  |  |  | +0.0269 |
| same_model_same_prompt_family | qwen3.5-plus | qwen3.5-plus | mixed_strategy | mixed_strategy |  |  |  |  |  |  |  | +0.3310 |
| same_model_same_prompt_family | qwen3.5-plus | qwen3.5-plus | same_definition | same_definition |  |  |  |  |  |  |  | +0.1395 |
| same_model_same_prompt_family | qwen3.5-plus | qwen3.5-plus | same_elimination | same_elimination |  |  |  |  |  |  |  | +0.1128 |

signed delta 使用 `contrast_mean - same_model_same_prompt_mean`，不取绝对值。
