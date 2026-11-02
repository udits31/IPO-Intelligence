# IPO return forecast and priority upgrade — 21 October 2026

## Delivered

Module 1 has a separately versioned return regressor and early-book probability
classifier. It reports a central listing-opening return and lower/upper residual
scenarios, rather than treating probability of a green opening as the amount of
profit. A dated GMP quote is used by the matched challenger; the bid-only model
remains available when no usable quote exists. The original final-book model
bundle and the conference paper were not replaced in this work.

The research-priority board compares open offers using central return, downside
and a disclosed cost scenario. It can select none. Closed offers, missing inputs,
future timestamps, stale publisher clocks and out-of-training-range features
cannot receive a fresh research rank. Old readings remain visible with their
actual clocks. The publisher clock is recovered from hashed archived evidence
when the last complete book is needed; downloading an old table does not reset
its age. NSE receipt timestamps now record completion, not request start.

Each new return forecast is frozen inside the append-only decision input with
its inputs, bid timestamp, model version and artifact hash. Joint priority and
GMP-only comparisons are frozen separately in `module1_priority_v1` at scheduled
collection slots. New return predictions were not inserted into earlier records.
Official listing openings are joined later; absent outcomes remain pending.

## Test result: no established GMP-selection advantage

- Training universe: 186 matched pre-close daily books from 2021–2024.
- GMP-paired training coverage: 102 IPOs. The recovered archive has 112 matched
  issuer histories and 1,350 dated quote rows before the cohort/clock join.
- Candidate selection: chronological earlier-year predictions in 2022/2023.
- Retrospective 2024 confirmation: 75 bid-only cases, 39 GMP-paired cases.
- On the same 39 cases: GMP-plus-bids return MAE **24.26 percentage points**;
  raw GMP-only return MAE **24.96 percentage points**.
- Error improvement: **0.70 pp**, with a descriptive bootstrap 95% interval
  **−4.04 to +5.33 pp**. This does not establish a reliable improvement.
- Four eligible same-date choice panels: model and GMP picked the same IPO in
  all four. There is no demonstrated selection-profit advantage.
- Nominal 80% residual-band coverage on the paired cases: **53.8%**. The app
  therefore does not call the band a valid 80% live prediction interval or a
  maximum-loss boundary.

The benchmark is retrospective and source-limited, not a blind or prospective
trading backtest. Daily pages may have been edited, daily GMP clocks are unknown,
and canonical offer terms lack independently timestamped document extracts.
Daily GMP must precede the bid date. Training on penultimate-session books does
not validate day-one, intraday or final-hour predictions. The 0.5 pp cost is an
illustrative scenario. Reported selection returns are conditional on allotment,
not wallet profits, and do not estimate allotment odds or capital reuse.

Full calculations, every paired choice and the largest misses are in
`output/module1_return_benchmark.md`; machine-readable results and every evaluated
prediction are in the matching JSON and `module1_return_prediction_ledger.csv`.
No post-2024 IPOs were added to training. No claim that the model will make the
user wealthier is justified by these results.

## Module 2 improvements

- Newer dated prices are requested for recent listings on analysis and through
  the existing collector; unavailable NSE archives have retry backoff.
- A future or unfinished daily candle is not used as a closing price.
- A stale retained price is shown as a reference, without issuing a current
  hold/sell action from it.
- Durable news caching retains the last good evidence on source failure, but
  stale sentiment cannot be treated as a fresh signal. Check and success clocks
  reach the screen.
- Quote directories, stock-chart pages and generic financial-profile pages are
  filtered out of company-event sentiment, including already cached results.
- The monitor distinguishes the price-rule call from whether news changed it.
- An absent verdict no longer silently becomes HOLD. The bright supplementary
  quote panel is removed; untimed quotes are labelled indicative, not decision
  prices.
- Stored anchor dates without official source evidence are labelled unverified;
  the app no longer claims a legal selling ban from a listing-date heuristic.
  An unverified unlock calendar cannot authorise accumulation. Actual allotment
  or unlock documents still need to be connected for verified calendars. SEBI's
  issuer-document wording ties lock-in periods to allotment, not simply listing:
  https://www.sebi.gov.in/sebi_data/attachdocs/dec-2024/1735015686891.pdf

Module 2 remains a rule-based risk monitor. These engineering fixes do not
establish improved forward-return accuracy or convert headline sentiment into a
validated return-prediction model. Its frozen price/news decisions must be scored
against future outcomes separately from Module 1.

## Validation and operation

The full backend suite passed **136 tests**; the final timestamp-contract
adjustment also passed its **43 targeted tests**. Frontend forecast-contract
tests passed **4 tests**, and the production frontend build succeeded. Browser
checks covered the return forecast, saved history, research priorities, GMP
comparison and a listed-stock price/news monitor.

The application runs on port 3000 and the backend on port 8000. Existing polling
continues; keep the backend and machine awake. No commit or push was performed.
The next research milestone is a larger, genuinely point-in-time pre-close
cohort and matured prospective outcomes, not a higher number produced by
retuning against already inspected results.
