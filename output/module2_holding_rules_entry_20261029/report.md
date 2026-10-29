# Module 2: earlier-only holding and exit tuning

{'name': 'rules_entry_ema3_confirm3_hold5_exit-0.15', 'original': False, 'ema_span': 3, 'exit_confirmation': 3, 'min_hold': 5, 'exit_threshold': -0.15, 'trend_protection': True, 'cooldown': 2, 'entry_threshold': 0.35, 'rule_led_entry': True, 'require_price_deterioration': True}

| Strategy | Net return per issuer window | Mean drawdown | Buy fills |
|---|---:|---:|---:|
| tuned | 2.453% | -10.450% | 181 |
| original_ml | 1.768% | -8.480% | 649 |
| current_rules | 2.599% | -11.145% | 198 |
| always_hold | 1.942% | -14.337% | 159 |

Qualified on both earlier periods: False. Paired comparisons: {'current_rules': {'mean_gain_pct': -0.14529224250343725, 'bootstrap95_pct': [-0.8523225975149359, 0.5040004224761061], 'improved_issuers': 27, 'worsened_issuers': 29, 'ties': 103, 'limitation': 'Issuer-block interval retains each issuer path; shared market-date correlation is not fully modeled.'}, 'original_ml': {'mean_gain_pct': 0.6849371360401608, 'bootstrap95_pct': [-1.1738908776017003, 2.5254237026938773], 'improved_issuers': 67, 'worsened_issuers': 88, 'ties': 4, 'limitation': 'Issuer-block interval retains each issuer path; shared market-date correlation is not fully modeled.'}, 'always_hold': {'mean_gain_pct': 0.5117160941037082, 'bootstrap95_pct': [-0.909207228971547, 1.8906518109432924], 'improved_issuers': 42, 'worsened_issuers': 44, 'ties': 73, 'limitation': 'Issuer-block interval retains each issuer path; shared market-date correlation is not fully modeled.'}}.

18:00 signal, next-open fills, carry shares overnight, 17.5bps each buy/sell, final liquidation at last common close. Risk exits override minimum holding periods.

Earlier-only selection; later results are an inspected retrospective replay, not an untouched future test. Direction-model predictions are unchanged by holding-policy tuning.
Promotion gate passed: False; live promoted: False.
