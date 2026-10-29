# Module 2 multi-horizon exit research

{'group': 'market', 'family': 'boosting', 'alpha': 20.0, 'policy': {'weekly_exit': -0.5, 'confirmation': 2, 'require_daily': False, 'ema_span': 3, 'min_hold': 5, 'cooldown': 2}, 'folds': {'2023': {'weekly_train_issuers': 83, 'validation_issuers': 38, 'last_training_label': '2023-02-01', 'first_validation_decision': '2023-02-02', 'strategy': {'issuers': 38, 'mean_net_return_pct': 10.617423480873317, 'median_net_return_pct': 6.116926005483803, 'mean_max_drawdown_pct': -10.129442007605872, 'worst_decile_mean_return_pct': -16.65246401007637, 'buys': 52, 'sells': 52, 'exposure_sessions': 1173}, 'rules': {'issuers': 38, 'mean_net_return_pct': 10.077961143274823, 'median_net_return_pct': 6.116926005483803, 'mean_max_drawdown_pct': -11.197818778894465, 'worst_decile_mean_return_pct': -16.379919650341733, 'buys': 41, 'sells': 41, 'exposure_sessions': 1294}, 'mean_gain_vs_rules_pct': 0.5394623375984935, 'drawdown_gain_vs_rules_pct': 1.0683767712885928}, '2024': {'weekly_train_issuers': 136, 'validation_issuers': 72, 'last_training_label': '2024-02-14', 'first_validation_decision': '2024-02-15', 'strategy': {'issuers': 72, 'mean_net_return_pct': 1.777579971426714, 'median_net_return_pct': 0.0, 'mean_max_drawdown_pct': -13.381180173589705, 'worst_decile_mean_return_pct': -20.4193239707732, 'buys': 84, 'sells': 84, 'exposure_sessions': 2035}, 'rules': {'issuers': 72, 'mean_net_return_pct': 1.4750151256085742, 'median_net_return_pct': -1.24291515971523, 'mean_max_drawdown_pct': -13.749353871907333, 'worst_decile_mean_return_pct': -23.029457275980192, 'buys': 84, 'sells': 84, 'exposure_sessions': 2108}, 'mean_gain_vs_rules_pct': 0.3025648458181398, 'drawdown_gain_vs_rules_pct': 0.36817369831762825}}}

| Strategy | Mean net return per IPO window | Mean drawdown | Buy fills |
|---|---:|---:|---:|
| multihorizon | 2.688% | -10.719% | 185 |
| current_rules | 2.599% | -11.145% | 198 |
| always_hold | 1.942% | -14.337% | 159 |

{'current_rules': {'mean_gain_pct': 0.0896844346361056, 'bootstrap95_pct': [-0.4167643366274747, 0.5946277927128358], 'improved_issuers': 28, 'worsened_issuers': 21, 'ties': 110, 'limitation': 'Issuer-block interval retains each issuer path; shared market-date correlation is not fully modeled.'}, 'always_hold': {'mean_gain_pct': 0.7466927712432513, 'bootstrap95_pct': [-0.5685777209099825, 2.124379276668819], 'improved_issuers': 41, 'worsened_issuers': 38, 'ties': 80, 'limitation': 'Issuer-block interval retains each issuer path; shared market-date correlation is not fully modeled.'}}

Five-session target: open[t+6]/open[t+1]-1; targets must mature before the next cohort. Signals at 18:00 IST execute next open. Same later IPOs and dates; 17.5bps on each buy/sell including terminal liquidation. Daily forecast is unchanged.

Research after repeated inspection of later outcomes. All model/policy settings selected on 2023-2024, but this is not an independent prospective test. No shorting; terminal windows and circuit/slippage assumptions remain limitations.
Promotion gate: False; live promoted: False.
