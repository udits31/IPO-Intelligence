# Module 2: automatic forward research test

The production price-and-filing monitor remains active. The learned candidate runs alongside it and does not place orders or replace its verdict.

## What changed

The candidate combines the existing next-session return forecast with a separately learned five-session return forecast. Its exit policy waits for persistent predicted weakness and weak price trend, while preserving existing risk exits. Model and policy selection used 2023–2024 decisions with matured training labels. Later 2025–2026 results are development comparisons, not untouched validation.

Across the same 159 later IPOs, mean net return per observed IPO window was 2.688% versus 2.599% for current rules and 1.942% for holding. Buy fills fell from 198 to 185. The paired gain over rules was 0.090 percentage points, with a 95% issuer-bootstrap interval of -0.417 to +0.595 points. The candidate still trailed the rules in 2026. It has not earned production promotion.

## Automatic operation

- While the backend runs, the existing Module 2 collector checks dated prices and news and invokes research logging after 18:00 IST on weekdays.
- Complete, verified OHLCV and market history and 21–56 completed post-listing sessions are required. Unsupported cases retain an explicit reason; no invented volume, history or prediction is inserted.
- One immutable forecast per issuer, closing date and model artifact stores next-close and five-session estimates, hypothetical position, source hashes and observation time.
- Positions begin in cash on the first genuinely recorded forecast. A hypothetical trade can execute only at a later observed opening; costs are 17.5 basis points per transition.
- Outcome-source collection continues for forecast issuers after they leave the forecasting age window, until enough observed sessions exist to evaluate the five-session target.
- Original source bytes are retained in the local SQLite ledger and mirrored with forecasts to Supabase through the existing sync process.
- Changed model, feature-builder or policy hashes block new logging until an explicit protocol revision. Prior protocol versions and saved forecasts remain intact.

## Viewing results

The performance page includes the five-session comparison and forward research scorecard. The API routes are `/api/v1/module2/multihorizon-evaluation` and `/api/v1/module2/shadow-scorecard`.

At verification on 29 October 2026, there were 22 forecasts across 22 IPOs, no matured next-session outcomes, and no failed forecast-payload checksums. Local and Supabase forecast/source row counts matched. Future counts and outcomes should be read from the API rather than this dated checkpoint.

Next-day directional accuracy, five-session forecast error, hypothetical strategy return and the production monitor's historical directional score are different measures. None should substitute for another. Circuit limits, queues and actual execution costs may differ from the paper-fill assumptions.
