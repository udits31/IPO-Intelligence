# Module 2: earlier-only holding and exit tuning

{'name': 'ema3_confirm2_hold5_exit-0.15_trendFalse', 'original': False, 'ema_span': 3, 'exit_confirmation': 2, 'min_hold': 5, 'exit_threshold': -0.15, 'trend_protection': False, 'cooldown': 2, 'entry_threshold': 0.35}

| Strategy | Net return per issuer window | Mean drawdown | Buy fills |
|---|---:|---:|---:|
| tuned | 1.405% | -9.333% | 191 |
| original_ml | 1.768% | -8.480% | 649 |
| current_rules | 2.599% | -11.145% | 198 |
| always_hold | 1.942% | -14.337% | 159 |

Qualified on both earlier periods: False. Paired comparisons: {'current_rules': {'mean_gain_pct': -1.1936203382753388, 'bootstrap95_pct': [-2.5843454039448086, -0.03630379326385379], 'improved_issuers': 50, 'worsened_issuers': 65, 'ties': 44, 'limitation': 'Issuer-block interval retains each issuer path; shared market-date correlation is not fully modeled.'}, 'original_ml': {'mean_gain_pct': -0.36339095973174035, 'bootstrap95_pct': [-2.1351957246407864, 1.4409738683267828], 'improved_issuers': 66, 'worsened_issuers': 83, 'ties': 10, 'limitation': 'Issuer-block interval retains each issuer path; shared market-date correlation is not fully modeled.'}, 'always_hold': {'mean_gain_pct': -0.5366120016681927, 'bootstrap95_pct': [-2.4676256881807745, 1.2961540202143218], 'improved_issuers': 63, 'worsened_issuers': 74, 'ties': 22, 'limitation': 'Issuer-block interval retains each issuer path; shared market-date correlation is not fully modeled.'}}.

18:00 signal, next-open fills, carry shares overnight, 17.5bps each buy/sell, final liquidation at last common close. Risk exits override minimum holding periods.

Earlier-only selection; later results are an inspected retrospective replay, not an untouched future test. Direction-model predictions are unchanged by holding-policy tuning.
Promotion gate passed: False; live promoted: False.
