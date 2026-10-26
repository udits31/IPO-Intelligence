# Module 2 — Dated news and the unchanged monitor

This experiment uses actual archived headlines, not generated news or fabricated sentiment labels. The original price/news policy is executed before testing a bounded rule overlay; no live rule is promoted.

Later coverage: 132 IPOs, 39 with eligible preceding-date headlines. Generous observed-news-only accuracy ceiling: 64.39% (assumes every news-covered case becomes correct; not measured predictive accuracy).

| Policy, same later IPOs | 30-session direction accuracy |
|---|---:|
| Price-only current rules | 50.00% |
| Current rules plus dated news | 50.00% |
| Earlier-data-selected news overlay | 50.00% |
| Always sell | 58.33% |

Current news governance changed 0 binary calls. Selected overlay changed 0. Selected parameters: {'mode': 'unchanged', 'threshold': None, 'min_articles': None, 'lookback_days': None}.

Selection saw 214 matured 2021–2024 outcomes; 30 later outcomes of 2024 IPOs were not yet known. 2025–2026 outcomes did not choose the rule.

The goal is not complete unless the full timing-correct requirements are actually met. No report of final-book Module 1 accuracy is substituted for this Module 2 result.

## Limits

- Retrospective secondary RSS index; incomplete news coverage and no immutable original-publisher text.
- Same-day items excluded because their original publisher clocks were not independently verified.
- The existing keyword scorer is heuristic sentiment, not a validated causal event model.
- Current NSE issuer master is used for entity disambiguation only; not a historical as-published security master.
- Later outcomes were previously inspected; no blind or prospective accuracy claim.
- Cash dividends and executable profit are not measured; prices use dated bonus/split adjustments.
