# Full enrichment audit — 25 October 2026

The new extraction, exchange backfill, model challengers, capital calculator and dated monitor preview are implemented. The challengers have not earned promotion.

Original models and forecast rows preserved: **True**

## Module 1: same-case comparisons

| Cohort | Frozen model | Enriched calibrated | GMP-only fitted | Always positive |
|---|---:|---:|---:|---:|
| 69 historical 2026 IPOs | 56/69 (81.2%) | 51/69 (73.9%) | 55/69 (79.7%) | 46/69 (66.7%) |
| 23 recent verified openings | 15/23 (65.2%) | 17/23 (73.9%) | 13/23 (56.5%) | 17/23 (73.9%) |

The fixed historical cohorts have already been inspected. These are retrospective comparisons, not new prospective accuracy.

## Module 2: same 132 matured IPOs

- Existing rule engine: **46.97%**
- Always sell: **58.33%**
- Enriched rules: **47.73%**
- Enriched classifier: **42.42%**

## Remaining evidence gaps

- Recover versioned pre-close RHP/anchor/BRLM evidence and bind category split tables to their own update clocks.
- Verify BSE-only NSE issuer opening: archive access has returned HTTP 403; no NSE self-listing substitute is used.
- BSE delivery, filings and deals need a verified exchange adapter.
- Historical delivery/bulk original publication clocks and institutional client classifications remain incomplete.
- Full filing attachments are not extracted by this exchange-summary collector.
- Trendlyne metrics without historical publication clocks remain supplementary context.
- Do not promote an enriched monitor until it beats baselines under a frozen prospective protocol.

No exact allotment-adjusted portfolio return is claimed without the application counts. The calculator supports separately labeled assumptions and financing costs.

Detailed field coverage, failed requests, cohort membership and integrity hashes are in the JSON report.
