# IPO Intelligence — What is built, what was verified, and what still fails

Updated 21 October 2026. This is a development status report, not investment
advice or a claim that the requested 75–80% accuracy has been achieved.

## Module 1: pre-close return and priority forecasts

The application has a separately versioned experimental gain/loss estimator,
empirical downside/upside scenarios, and a risk-adjusted ranking of eligible
open offers. Both the estimates and the GMP-only comparison are saved before
the application cutoff. A later calculation is not backdated into that record.
The original final-book three-bid logistic model and the paper remain unchanged.

### Does the return model beat GMP-only selection?

Not established. After checking the opening outcomes against exchange archives,
the **same original saved forecasts, without retraining**, have:

| Same 39 paired 2024 IPOs | Mean absolute return error |
|---|---:|
| Original return forecasts | 25.17 percentage points |
| Dated GMP-only estimate | 25.92 percentage points |

The average error advantage is 0.75 percentage points. Its descriptive 95%
bootstrap interval is −3.96 to +5.34 points: it includes no advantage. The four
eligible same-date selection panels remain ties between the model and GMP.
The nominal 80% scenario band covers only 53.8% of these outcomes. It must not
be described as a calibrated 80% live interval or a maximum loss.

A separate source-corrected research refit has 25.08 pp error versus 25.92 pp
for GMP. That refit is **not deployed** and is not substituted for the unchanged
forecasts when the application displays its source recheck.

### What does the final-book direction score mean?

All 252 original 2021–2024 training outcomes and 179 of 189 later records were
verified from final exchange archives. NSE openings are used where the stock
was listed there at its IPO date, with exact-code BSE outcomes otherwise.
Historical renamed symbols are resolved from official exchange records.

On those 179 verified later records, the original logistic model has 76.0%
direction accuracy, 71.7% balanced accuracy and AUC 0.8141, versus a 67.0%
always-positive baseline. A same-family refit using the verified opening labels
has 74.9% accuracy. The unverified ten records are listed in the report and are
not counted as successes. Neither score establishes application-time accuracy:
that classifier uses the final subscription book, which can arrive too late
for a retail application. It also does not measure return-size prediction.

The exact original cohort includes the Ruchi Soya follow-on offer. It was
retained for cohort parity in this audit, not relabelled as a newly listed IPO.
Issue prices and historical bid-publication clocks still need independent
point-in-time offer-document verification.

Evidence:

- `module1_exchange_outcome_replay_20261021.md` and matching JSON: full results,
  separate refit versus unchanged forecasts, and every paired choice.
- `module1_exchange_outcome_replay_20261021_frozen_forecasts_ledger.csv`: the
  unchanged predictions joined to exchange-verified actual openings.
- `module1_exchange_listing_open_audit_20261021.json`: source coverage and gaps.
- `data/audit/module1_exchange_listing_open_audit.csv`: raw archive paths,
  hashes, exchange, symbol-change evidence, opening prices and discrepancies.

## Module 2: corrected price-only monitor replay

The historical data pipeline had two substantial gaps: missing early BE-series
sessions and late-2024 IPO histories ending at the calendar-year boundary.
Recovery added 361 real BE closes and 1,860 later outcome sessions. Complete
development coverage rose from 181 to 244 IPO histories; source conflicts were
resolved from final daily exchange archives without overwriting canonical data.

Two later apparent crashes were mechanical share changes: Aastha's 1:1 bonus
and Cordelia's 10-to-1 face-value split. The replay now values the shares held
per original listing-day share, applying each official ratio only from its
ex-date onward. Future actions cannot alter earlier decision inputs. Dividends
are not included; these are not total-return or execution-profit estimates.

| 2025–2026, same 132 matured IPOs | 30-session direction accuracy |
|---|---:|
| Existing monitor rules | 50.00% |
| Always sell / stay in cash | 58.33% |
| Always hold | 41.67% |
| Earlier-data-calibrated version of the same rules | 46.97% |

The calibrated rule change was rejected. It did not establish an improvement.
The existing rule's next-session direction accuracy is 53.79%; that is a
different horizon and cannot replace the primary 30-session result.

The calibration used only 2021–2024 IPOs. At the 31 December 2024 parameter
freeze, 214 outcomes had matured; 30 late-2024 IPO outcomes had not and could
not be used to select settings. No synthetic price or IPO rows were added.
All 132 matured later cases are retained; another 43 histories lack the fixed
future horizon and remain explicitly excluded for immaturity, not poor results.

This is a one-checkpoint price-only replay (31st close, then 30 later sessions),
not a blind test of the continuously updated live news/disclosure monitor.
The current monitor is a risk-rule system, not a validated 75–80% forward-return
classifier. Threshold changes alone have not supplied the missing predictive
information. The goal remains incomplete.

Evidence:

- `module2_same_rule_reconciled_20261021.md` and matching JSON: calibration,
  all candidate settings and the rejected result.
- `module2_holdout_accuracy_2025_2026_goal_corporate_actions_20261021.json`:
  corrected replay, class counts, baseline comparisons and coverage.
- `module2_holdout_decision_ledger_goal_corporate_actions_20261021.csv`:
  each decision, execution date, target, drawdown and outcome.
- `module2_corporate_action_reconciliation_20261021.json`: official event
  source windows, archived hashes, applied ratios and exclusion checks.

Sources: [NSE final reports](https://www.nseindia.com/all-reports),
[NSE corporate actions](https://www.nseindia.com/companies-listing/corporate-filings-actions),
[NSE symbol-change records](https://www.nseindia.com/static/market-data/securities-available-for-trading),
[BSE final bhavcopy downloads](https://www.bseindia.com/markets/MarketInfo/BhavCopy.aspx).

## Next scientific gates

1. Expand genuinely dated pre-close bid/GMP choice sets and independently verify
   offer terms. Do not present final-book accuracy as the investor-time result.
2. Connect dated, issuer-specific filings/news to the same Module 2 policy, with
   historical publication cutoffs and separate coverage reporting; measure the
   overlay on identical cases rather than borrowing the price-only score.
3. Continue saving prospective predictions before outcomes, including misses,
   skipped offers and source failures. Score only matured outcomes.
4. Promote a model/rule change only on a documented timing-correct protocol,
   not because retuning already inspected years happens to produce 75–80%.

No model-family replacement, paper edit, commit or push was performed in this
audit. The app's source-recheck panel explicitly distinguishes rescoring the
same saved forecasts from research refits.

## Follow-up: actual dated news, issuer matching and market-rule tests

The next goal turn completed 376 historical news-source attempts, covering
every one of the 244 development and 132 matured later cases. All original RSS
bytes and source hashes were retained. Reprocessing with distinct-issuer
matching leaves 202 eligible preceding-date headlines across 81 cases. On the
132 later cases, 39 have eligible headlines and 93 do not. An empty indexed
search is not proof that no company news existed.

Two concrete defects were found and fixed in the application's news inputs:

- General Indian Railways stories had been admitted as IRFC company news, and
  Kalpataru Projects stories had been admitted for Kalpataru Limited. The
  matcher now uses the official NSE list of 2,608 issuer names/symbols, expands
  shared prefixes until the issuer is distinguished, and rejects ambiguous
  bare-brand matches. Old cached headlines are rechecked, not merely new fetches.
- The original keyword scorer could label a headline saying “Profit Decline”
  bullish because it contained “profit” and “revenue.” Metric direction now
  controls these phrases; a mention alone does not imply improvement. The
  family remains deterministic keyword/context scoring, with a recorded version.

The final checks also reject restaurant-hygiene stories for Five-Star Business
Finance and automated “is rated Strong Sell” directory pages as company-event
news. The scorer correctly treats an explicit Strong Sell phrase as negative
instead of treating “strong” alone as a positive word. Its version and loaded
source-code hashes are included in new decision evidence.

The existing live news governance only vetoes accumulation; it does not alter
ordinary Hold/Sell calls. Executing that exact policy on the actual dated news
changed zero binary decisions and retained 50.0% 30-session accuracy. A fixed
25-option bounded news-overlay experiment, selected only on matured earlier
outcomes, selected the unchanged policy. It was not promoted as an improvement.
With this news coverage, even an unrealistically perfect correction of every
news-covered case would have a 64.4% ceiling while leaving other price calls
unchanged. That is a diagnostic bound, not a measured accuracy result.

A separate, fixed 17-option calibration of the same loss/relative-lag/trailing
rules by the observed Nifty regime reached 53.03% on the same 132 later IPOs,
versus 50.0% current rules and 58.33% always sell. It corrected ten errors and
introduced six; the descriptive interval for the accuracy change was −3.03 to
+9.09 percentage points. It did not establish a reliable improvement or the
requested target, and no market-rule change was promoted.

Current detailed evidence:

- `module2_dated_news_rule_replay_issuer_verified_20261021.md` and JSON:
  actual-policy replay, news-overlay selection, coverage and limitations.
- `module2_news_issuer_recheck_20261021.json`: removed headline examples,
  preserved raw hashes and the identity-master audit.
- `module2_market_rule_calibration_20261021.md` and JSON: all market-conditioned
  candidates, earlier-only selection, same-case later results and per-IPO ledger.

The current issuer master is retrospective identity evidence, not a historical
as-published master. Historical publisher texts and completeness remain
unverified. No synthetic news, IPO or price training rows were added. The
75–80% goal remains active and incomplete; these results must not be presented
as an achieved accuracy gain. The expanded targeted verification passed 71
tests. A live API check confirmed the revised scorer and current price date;
browser visual inspection remains unavailable under the browser tool's policy.

### 22 October: quarterly financial sources and comparative-earnings preparation

Exact filing-sequence checks recovered 68 additional current-quarter rows from
malformed NSE WEB-XBRL exports, bringing verified financial coverage to 254
cases: 144 development and 110 later. These are dated post-listing financial
observations; they must not be inserted into Module 1 pre-application features.
The eleven-policy profit-margin-conditioned experiment retained all 132 later
cases and scored 50.0%, unchanged from the baseline monitor and below 58.33%
always-sell. It was not promoted.

The original-PDF collector archived 176 matching sources from those 254 cases.
Two local OCR workers are collecting complete pages and preserving numeric,
date, unit, model and source hashes. A separate offline verification pass
retains all 376 cases and accepts a comparative only after independent
current-quarter source checks and explicit full-page assistant visual review.
This is not independent human audit. The running workers are not evidence
that extraction is complete. Current progress is recorded in
`module2_financial_comparative_recheck_20261022.json`.

The parser now prevents a merged three-quarter caption's last column from
being misclassified as annual. Source-based checks cover that geometry,
negative prior-year profits, total group rather than owners-only profit,
wrong issuers, late publications, altered reviews and missing raw OCR.
The comparative-earnings replay predeclares seven policies within the existing
rule family and permits only matured 2021-2024 outcomes to select settings.
Every later case and the 30-session target remain unchanged. No forecast is
backdated, no synthetic training observation is added, and no live policy or
paper score is promoted. The 75-80% objective remains incomplete.

The first comparative-earnings pipeline verification used seven source- and
visual-checked tables (six development and one later). All 132 later cases
remained in the replay; no later call changed, so accuracy remains 50.0%.
This is partial-source pipeline verification, not a completed test of earnings
predictive value. The result and per-case ledger are recorded in
`module2_comparative_earnings_rule_replay_20261022.json` and its ledger CSV.
Twenty-nine targeted source, gating, replay and performance-integrity checks
passed. Both running collectors still have outstanding source pages; their
saved raw output is not automatically accepted into the reviewed data.

### Structural check: why the current threshold ranges cannot reach the goal

The comparative source audit now accepts sixteen fully checked tables: eleven
development and five later. All 132 later outcomes remain in the earnings
replay, which still changes no later call and remains at 50.0%. Roman/lettered
row labels and unlabelled standalone table titles are handled with explicit
authority notes: the latter require verified non-consolidated NSE metadata,
current-quarter numeric parity and visual review, not an invented PDF label.

An additional pointwise reachability diagnostic changes the next action.
Even if a hindsight oracle selected the best of the three earnings presets
separately for every IPO, it could be correct on only 81/132 (61.36%). The
broader 75 tested price-threshold combinations reach at most 94/132 (71.21%).
These are NOT achieved scores, are never training features or trading calls,
and apply only to those specified settings. The requested target cannot be
reached merely by assigning those settings using more financial data.

A bounded nine-policy observed-market/stock-momentum risk-off experiment was
selected on matured 2021-2024 outcomes, then tested on all 132 later cases. It
scored 49.24% versus 50.0% current monitor and 58.33% always-sell; it corrected
one mistake and introduced two. It was rejected. No learned classifier or
live core was replaced. The report, diagnostic ledger and readable explanation
are `module2_risk_governance_research_20261022` artifacts. The next useful
research must test genuine evidence-linked decisions beyond the insufficient
threshold ranges, while preserving the user's model-family constraint.

### Explicit earnings-event study and direct-PDF source checks

The next study tests thirteen predeclared event-governance policies rather
than assigning the same three price presets. Current loss-making quarters
can be observed from verified XBRL; comparative deterioration/growth requires
checked prior-year figures. Negative events also require adverse observed
price confirmation; positive events cannot waive an underwater structural
stop. Selection still uses only 214 outcomes matured before 2025. With 24
fully checked comparisons (14 development, 10 later) and 254 verified current
quarters (144 development, 110 later), the selected policy remained unchanged.
All 132 later cases scored 50.0%; no improvement or promotion is claimed.
Evidence is in `module2_earnings_event_research_20261022` JSON, ledger and MD.

A complete native-text sweep of the 254 financial cases checked all 176
available original PDFs and recovered eight source-matching tables, seven
additional to the then-existing OCR recoveries. Native text carries no OCR
confidence score. The review gate re-extracts its exact text/coordinates from
the original hash-checked PDF; current-quarter parity, period/basis/unit checks
and full-page visual review remain mandatory. The separate slow OCR jobs
continue; a completed native sweep is not a completed OCR-source collection.

Full-page review found two explicit before/after-tax direction conflicts
(WeWork and Indiqube). The observed tax rows explain why a PAT comparison is
not straightforward operating deterioration/improvement. Original PAT and
current-quarter loss flags are retained. Reviewed PBT facts qualify only the
comparative event; no synthetic normalized profit or removed case is used.

The accepted comparative-source snapshot is now pinned to its exact raw
extraction and hash. An equivalent OCR copy arriving after native-text review
cannot silently replace the source used by an ongoing experiment. Every
accepted figure, render, review and accounting qualifier is replayed; rejected
page-attempt diagnostics are not mistaken for changing model inputs. Actual
native/OCR copies of the Indiqube table reproduce the same reported losses.
Tampered figures or source hashes fail the snapshot check. The expanded
targeted verification passed 65 tests. No paper, live model, commit or push
was changed by this research stage; the 75-80% objective remains incomplete.

### Runtime policy and share-basis fixes verified in the next pass

The separate listed-stock fallback trading tree has been removed. Canonical
IDs, names and eligible additional listings now use the shared core monitor.
An unlisted issue is directed to Module 1 rather than receiving a second
application verdict from a fallback with missing bid inputs. Exact canonical
identity takes precedence over autocomplete; ambiguous matches require a
specific issuer instead of choosing the first match.

Live Yahoo historical closes were already split-adjusted, whereas the original
issue price was not. Dated NSE corporate-action archives now cross-check the
explicit provider ratios before returns are calculated. The latest quote is
unchanged, earlier history is not adjusted twice, and historical replays undo
only provider adjustments whose effective date lies after the replay date.
Raw NSE history and listing-share research history reach the same as-of basis.
Unsupported rights/capital events or missing/conflicting share evidence pause
advice, with the observed quote retained for reference. Cash dividends are not
included. Prospective outcome evaluation likewise reconciles share entitlements
without changing the frozen original decision.

Real-source tests cover Aastha's 1:1 bonus and Cordelia's 10-for-1 split. Live
API checks on 21 October retained the actual quotes and compared Cordelia with
an adjusted issue price of ₹80.80 (original ₹808), and Aastha with ₹68 (original
₹136). The waterfall labels that basis. Portfolio P&L is withheld until the
user confirms entered share quantities and averages already reflect changes;
the application does not invent an acquisition date or adjusted holding count.

BSE-only NSDL now resolves scrip 544467 to NSDL from the hash-checked 6 August
2025 BSE archive, rather than requesting the invalid provider ticker 544467.BO.
Its ID route returned the shared monitor, a 21 October dated close, and the
explicit BSE identity. BSE split/bonus evidence remains secondary-provider
authority, not primary exchange corporate-action coverage. Legacy/unsupported
BSE identities remain a coverage limitation; no tickers are guessed.

The benchmark screen now leads with the exchange-verified opening scores when
the source audit matches the saved prediction ledger. These implementation
corrections do not establish a higher predictive accuracy, change either model
family, or justify the 75–80% goal. No paper edit, commit or push was made.
The production frontend build, seven frontend checks and 79 targeted backend
tests passed, including eight runtime share-basis/identity tests using actual
archived market responses. The live frontend and benchmark/monitor API routes
returned HTTP 200 after restart.
The browser security policy still prevents visual UI inspection in this run;
HTTP/API checks are not described as browser verification.

## 22 October: source-clock, maturity and recovery-veto research

All three studies retain the same 132 later IPOs, 31st-close decision and
30-later-session direction target. No live rule or paper number was replaced.
Financial-source-clock response rules selected from 214 matured earlier IPOs
scored 50.00%, correcting six mistakes and introducing six. Admitting only
late-2024 IPO outcomes after they matured at each later decision increased that
research branch to 51.52% (68/132); no 2025-2026 IPO outcome selected a rule.
This is a distinct expanding-availability protocol, not a replacement for the
frozen-end-2024 evaluation. Both remain below always-sell at 58.33%.

A 21-candidate bounded recovery-veto study tested whether permitting positive
source-clock price responses below issue price helped. Earlier-only selection
retained the issue-price guard and scored 50.00%. Its matching rule with the
guard removed scored 49.24%; this change is rejected for live promotion.
Reports are `module2_filing_response_research_20261022`,
`module2_maturity_asof_research_20261022` and
`module2_recovery_guard_research_20261022` in this directory. These are
exploratory studies after repeated later-cohort inspection, not blind tests.
Ten real-source tests across these studies passed. Passing implementation
checks does not demonstrate the requested predictive accuracy.

The next original-page review increased accepted comparative tables from 24
to 29 (17 development, 12 later). One additional reviewed table, BlueStone's
June-2025 consolidated results, is deliberately blocked: the archived OCR read
prior-year revenue as 3432.44 million while the printed page says 3482.44
million. The review records the printed value without rewriting raw OCR or
manufacturing an accepted feature. A real-source regression test pins this
conflict. All 376 audit cases and all 132 later evaluation IPOs remain present.
The refreshed seven-policy earnings replay still scores 50.00%; additional
coverage has not established an accuracy improvement. Forty-six targeted
source-review, frozen-snapshot and earnings-rule/event checks passed.
The refreshed 13-policy earnings-event replay selected positive-only events
on the earlier cohort and scored 50.76% (67/132), correcting one later call
without introducing another error. This small exploratory change remains
below the 58.33% always-sell baseline and is not promoted to live or paper.

## Stop expanding PDF review: diagnose the existing errors first

Manual PDF expansion was stopped following the user's cost objection, and the
four verified extraction-worker processes were stopped with archived pages and
checkpoints preserved. No documents or collected data were deleted.
`module2_error_diagnosis_20261022.{json,md}` and its full-case ledger now explain
the unchanged baseline without fitting additional parameters. Of 66 later
errors, 41 are HOLD calls followed by flat/down outcomes and 25 are SELL calls
followed by recovery. Later HOLD accuracy is 30/71 (42.25%), versus 53/97
(54.64%) on matured earlier IPOs. The earlier baseline was only 54.21%, so the
problem is not solely a sudden later-year failure. Only 18 of 66 later errors
have subsequent absolute moves at most 5%; 26 exceed 10%. Removing near-flat
outcomes would not honestly solve the primary goal.

Later positive-outcome prevalence differs markedly: 31.68% for the 101 2025
IPOs versus 74.19% for the 31 2026 IPOs. These are outcome diagnostics, not
features available at prediction time or permission to choose a different
policy after seeing the year. No regime classifier or live change is inferred.

The bounded HOLD-evidence hypothesis was then tested without opening later
prices. Four fixed policies (unchanged, positive stock momentum over 10 or 30
sessions, positive 10-session stock plus market-relative strength) were
selected on 141 matured 2021-2023 cases before the first 2024 decision.
Positive 10-session stock momentum was selected. On the 61 2024 IPO outcomes
completed by end-2024, it scored 28/61 (45.90%) versus default 26/61 (42.62%),
but remained below always-hold 50.82% and always-sell 49.18%. Its paired change
interval spans -3.28 to +9.84 points. The gate failed; no 2025-2026 evaluation
was run, no later label selected a policy and no live change was promoted.
See `module2_hold_evidence_research_20261022.{json,md}`. This does not disprove
all possible HOLD governance; it rejects the specified momentum evidence.

An existing-data delivery-confirmation check was also stopped at the earlier
gate, without fetching anything or evaluating later cases. Actual NSE
delivery matched 213/244 earlier IPOs over the 20 sessions strictly preceding
the checkpoint; the remaining 31 retain default actions. Unadjusted exchange
closes must match, and share-basis changes inside a volume window are rejected.
Four fixed deterministic policies tested increased delivery participation and
quantity alongside positive/negative stock momentum. Earlier selection chose
unchanged rules (the accumulation override worsened selection by one call).
The same 61-case 2024 confirmation therefore remains 42.62%; the gate failed.
No claim is made that delivery identifies institutional orders, or that these
four policies exhaust its possible uses. See
`module2_delivery_rules_research_20261022.{json,md}`. No live/paper promotion.

## Module 1: consistent verified-target candidate, not live promotion

Fresh original-archive revalidation found six training classes differing
between the canonical and defined exchange-opening targets (269, 275, 315,
346, 351, 371). An explicit candidate option now uses the audited outcomes
consistently for all 252 training IPOs and 179 verified later evaluations.
The same three-feature logistic model and 2021-2024 training years remain.
Its fixed-0.50 direction accuracy is 74.9%; its APPLY cutoff selected only on
2022-2023 chronological predictions is 0.53 and yields 76.5% accuracy, 73.0%
balanced accuracy and 82.0% APPLY precision later. This is not a uniform
improvement over the old policy's 90% APPLY precision. Ten unverified later
outcomes remain pending. See `module1_exchange_candidate_notes_20261022.md`.
The separate candidate bundle cannot silently overwrite the live bundle;
canonical data, live coefficients and manuscript remain unchanged. Six
targeted candidate/source tests passed. Final-book retrospective scope remains.

## Module 1: explicit stage-transfer check

The existing penultimate-session public tables were scored with the same
three-input logistic family and verified outcomes. On the identical 75
source-complete 2024 cases, a final-book-trained fit scored 56.0% using early
bids, versus 82.7% using final bids. A matched-training-ID control scored
52.0% on early inputs. Training on early books instead scored 81.3% at the
fixed 0.50 cutoff, but always-positive also scored 81.3%; balanced accuracy
was only 58.3%. Its balanced-accuracy-selected earlier cutoff 0.84 scored
69.3% overall with 67.4% balanced accuracy. Thus stage-matched training helps
the probability scale but does not establish the required predictive edge.

The matched-ID control prevents attributing a change in training coverage
entirely to input stage. All training outcomes must precede the first actual
early decision in each test-year fold, not merely have an earlier listing
year. Three tests passed for source-case pairing, this time boundary and
earlier-only threshold selection. Retrospectively revised day-wise tables
still do not prove original investor-time availability. See
`module1_stage_transfer_research_20261022.{json,md}` and its paired ledger.
No live model or manuscript was replaced, and the goal remains incomplete.

The same-family paired GMP check used only already-collected preceding-date
quotes: 23/12/28/39 matched cases in 2021/2022/2023/2024. Bids-only, GMP-only
and combined logistic variants have identical training/test identities and
as-of boundaries. All scored 87.2% at cutoff 0.50 by predicting every one of
the 39 confirmation IPOs positive, exactly matching the 34/39 positive
baseline and failing all five negative cases. This is not goal achievement.
Earlier-selected policies scored 69.2% bids-only, 53.8% GMP-only and 69.2%
combined. Combined balanced accuracy was 73.8% versus 56.8% bids-only and
73.5% GMP-only, with only five negatives: no reliable advantage is established.
The combined variant changes the input count (not logistic model family),
is explicitly research-only, and does not replace paper/live artifacts.
See `module1_dated_gmp_logistic_research_20261022.{json,md}`. No new downloads
or PDF review were performed.

An ownership-channel audit then confirmed the baseline passes no disclosures
and disables news; it is price-only, not a complete ownership-informed replay.
A bounded five-IPO NSE PIT pilot preserved six actual rows: five modes that
are not explicit open-market trades, and one Medplus market purchase whose
person category is unknown. Neither is invented as an anchor/promoter exit.
No rows for three securities do not establish complete absence of sales.
The actual Medplus trade precedes its exchange publication by over a day,
so a three-test as-of/source check prevents trade-date look-ahead. No model,
paper or accuracy score was changed; no full-scale source expansion followed.
See `module2_ownership_input_audit_20261022.md`.
