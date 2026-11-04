# Module 1: recovered inputs and model diagnosis

Fixed comparison: the same 23 verified exchange opens; NSE opening remains unverified. Flat openings count as nonpositive. No model threshold was adjusted on these outcomes.

| Method | Correct | Accuracy |
|---|---:|---:|
| Frozen combined model, recovered inputs | 15/23 | 65.2% |
| Frozen fitted GMP-only classifier | 13/23 | 56.5% |
| Simple GMP > 0 rule | 15/23 | 65.2% |
| Always positive | 17/23 | 73.9% |
| Research candidate selected on 2025 Brier score | 16/23 | 69.6% |

## What was fixed

All 24 issuers now have dated GMP evidence. The GMP parser preserves intraday observations rather than rejecting differing values on the same day. Only quotes available by the bid reading enter the feature vector. Day-only quotes must predate the decision day. Financial extraction now accepts December annual year-ends and excludes June interim figures. Dated annual financial panels were recovered for 23 issuers. Bid trajectories and GMP slopes were connected. Remaining missing valuation/sector fields are listed per issuer; they were not fabricated.

## Why positive calls persisted

The incomplete-input replay supplied missing GMP, financial and trajectory values to a stack studied at penultimate-day close. Frozen imputation concealed the lack of information behind high raw probabilities. Recovering inputs produces 21 positive and 2 nonpositive calls rather than 23 positives. Both nonpositive calls (Nityas and Sonaselection) are false negatives. All six original flat/negative misses remain positive calls. A high closing-day subscription or positive GMP does not separate flat openings reliably in this cohort. Applying a penultimate-day model to closing-day inputs is also a stage mismatch. This is not repaired by moving a threshold until these 23 look good.

## The six original misses

| IPO | Recovered GMP % | Combined probability | Actual opening return |
|---|---:|---:|---:|
| Vishal Nirmiti | 9.09 | 50.8% | -2.27% |
| Acevector | 3.12 | 55.5% | -11.5% |
| Runwal Enterprises | 4.59 | 62.9% | 0.0% |
| Elevate Campuses | 0.69 | 59.8% | -1.91% |
| ArMee Infotech | 5.33 | 59.3% | 0.0% |
| Bonfiglioli Transmissions | 16.21 | 97.7% | 0.0% |

## Deployment decision

The original frozen artifact and append-only pre-close records remain unchanged. The candidate trained only through 2025 was not promoted: it did not beat the always-positive baseline. The 23 already-inspected issuers are a diagnostic cohort, not an untouched test set. Historical tables recovered later do not prove that this application issued these calls before closing. Fresh prospective outcomes and stage-matched historical data are needed before claiming an edge.
