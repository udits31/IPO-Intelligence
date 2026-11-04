# Frozen Module 1 prospective logging

The backend starts the frozen-model collector automatically. It runs on 15-minute slots, increasing to five-minute slots from 14:00 until 17:00 IST. Subscription collection follows the existing market-session schedule; each completed subscription poll also triggers the frozen scorer. The collector checks listing outcomes independently after bidding closes. The backend, computer and internet connection must remain available; missed downtime is never backdated.

Protocol and exact model/baseline fingerprints: `output/prospective_module1_protocol.json`.

SQLite evidence: `data/telemetry/snapshots.db`. The `frozen_module1_forecast_v1` table preserves scores, model/protocol/input hashes, source evidence, missingness, the actual recording time and the offer cutoff. Predictions and verified outcomes have update/delete prevention triggers. Changed observations can append another prediction in the same polling slot. The last genuine call before the recorded cutoff is selected for the IPO scorecard. Computations finishing at/after closing cannot be saved as pre-close calls.

`frozen_module1_eligible_v1` records the observed eligible universe, including IPOs without forecasts. `frozen_module1_attempt_v1` records collection errors and missing calendars. `frozen_module1_outcome_v1` stores independent exchange-verified listing opens. Flat means nonpositive. Missing quotes and bids are not observed zeroes; optional inputs use the frozen training imputation and the screen labels those estimates provisional. Other intraday/pre-bidding stages do not inherit the retrospective 81.2% validation claim.

The exact GMP-only classifier and its earlier-selected 0.625 cutoff are frozen alongside the five-view classifier's 0.5 cutoff. Both calls are saved together, before their outcomes. Source-verified offer-date overlays feed both the forecast logger and subscription harvester without replacing the original source cache.

API: `GET /api/v1/module1/prospective`. Includes latest predictions, actual opens when verified, correctness for both methods, sample counts, coverage, Brier score and collector health. Model Performance shows the live table. Module 1 shows the same saved estimate and actual open directly in each IPO analysis. Earlier genuinely saved legacy calls remain labelled by their original model. If no call was issued, a calculation made now is labelled reconstruction and never enters prospective accuracy.

Initial run on 25 October 2026 saved five pre-bidding provisional estimates. No new opening outcomes were available yet. Verification included append-only behaviour, duplicate/revised polls, missing values, exact cutoff and completion-time enforcement, outcome direction, protocol mutation rejection and reconstruction exclusion. The production frontend build passed and the live scorer reproduced all 69 historical frozen-model probabilities.
