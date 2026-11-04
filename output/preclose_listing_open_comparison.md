# IPO pre-close data recovery and listing-open comparison

Generated: 2026-10-21T15:00:53.473725+05:30

## What this measures

This is a retrospective replay with the current fixed model, not predictions issued live on the historical dates. The target is NSE listing opening price strictly above issue price; an exactly flat opening is NON_POSITIVE. Probability is not a predicted return or price. No model parameters or thresholds were tuned on these results.

Binary direction uses a fixed 50% threshold. The interface bands remain HIGH ≥60%, LOW ≤40%, MIXED otherwise; MIXED is an abstention for the band score, not an automatic correct prediction.

## Evidence coverage and scores

- Cohort: 15 IPOs, including all nine previously unresolved records.
- Matured official-opening comparisons: 9.
- Clean, closing-day, pre-17:00 source-reported timestamp pairs: 6.
- Primary descriptive direction accuracy: 5/6 = 83.3%; Wilson 95% interval 43.6%–97.0%.
- Always-positive baseline on the identical clean cases: 66.7%.
- Confusion counts: {'TP': 3, 'TN': 2, 'FP': 0, 'FN': 1}; Brier score 0.1807.
- Band decisions: 3/4 decisive calls right; 2 MIXED/abstained.
- All recovered matured candidates, INCLUDING warning-flagged inputs: 8/9 = 88.9%. Diagnostic only; do not report as clean or prospective accuracy.

This is a small, recovered, mixed-decision-time cohort with incomplete independent timestamp proof. It is not an unbiased holdout, a general accuracy claim, or proof of an implementable historical trading strategy.

## Every IPO: inputs, model result and actual opening

| IPO | Closing-day bids as of IST | QIB / NII / Retail | P(positive open) · band | Issue → NSE open | Opening return | 50% direction / band call | Evidence |
|---|---|---|---|---|---|---|---|
| Tonbo Imaging India | 2026-10-19T16:43:00+05:30 | 293.56x / 170.83x / 30.68x | 97.2% · HIGH | Pending | — | listing date has not arrived | clean_reported_timestamp |
| Zetwerk Manufacturing Business | 2026-10-19T16:44:00+05:30 | 4.71x / 1.44x / 0.74x | 45.5% · MIXED | Pending | — | listing date has not arrived | clean_reported_timestamp |
| Bonfiglioli Transmissions | 2026-10-19T16:44:00+05:30 | 120.78x / 40.36x / 5.78x | 91.1% · HIGH | Pending | — | listing date has not arrived | clean_reported_timestamp |
| Sahajanand Medical Technologies | 2026-10-19T16:43:00+05:30 | 34.70x / 14.60x / 2.91x | 79.3% · HIGH | Pending | — | listing date has not arrived | clean_reported_timestamp |
| Fusion CX | 2026-10-16T16:44:00+05:30 | 4.21x / 4.72x / 5.94x | 59.7% · MIXED | Pending | — | listing date has not arrived | clean_reported_timestamp |
| HD Fire Protect | 2026-10-15T16:44:00+05:30 | 256.65x / 203.08x / 30.10x | 97.1% · HIGH | Pending | — | official listing-day archive not available locally yet | clean_reported_timestamp |
| Nityas Gems & Jewellery | 2026-10-05T11:57:00+05:30 | 0.47x / 0.94x / 2.54x | 33.7% · LOW | ₹75 → ₹80 | +6.67% | WRONG / WRONG | clean_reported_timestamp |
| Vishal Nirmiti | 2026-10-05T11:18:00+05:30 | 0.96x / 1.19x / 1.02x | 32.7% · LOW | ₹220 → ₹215 | -2.27% | RIGHT / RIGHT | clean_reported_timestamp |
| SRIT India | 2026-09-30T11:15:00+05:30 | 0.08x / 32.36x / 14.77x | 50.9% · MIXED | ₹130 → ₹148 | +13.85% | RIGHT / ABSTAINED | needs_verification |
| Shah Investor's Home | 2026-09-30T14:23:00+05:30 | 5.51x / 47.25x / 10.13x | 71.7% · HIGH | ₹167 → ₹171 | +2.40% | RIGHT / RIGHT | needs_verification |
| Acevector | 2026-09-29T13:57:00+05:30 | 1.50x / 3.11x / 2.89x | 43.3% · MIXED | ₹32 → ₹28.32 | -11.50% | RIGHT / ABSTAINED | needs_verification |
| Orient Cables (India) | 2026-09-29T12:00:00+05:30 | 0.22x / 37.11x / 15.61x | 53.4% · MIXED | ₹272 → ₹450 | +65.44% | RIGHT / ABSTAINED | clean_reported_timestamp |
| German Green Steel & Power | 2026-09-29T12:39:00+05:30 | 1.35x / 22.35x / 13.37x | 59.6% · MIXED | ₹139 → ₹142 | +2.16% | RIGHT / ABSTAINED | clean_reported_timestamp |
| Runwal Enterprises | 2026-09-29T14:24:00+05:30 | 2.20x / 2.85x / 0.82x | 39.4% · LOW | ₹305 → ₹305 | +0.00% | RIGHT / RIGHT | clean_reported_timestamp |
| Moneyview | 2026-09-28T15:18:00+05:30 | 178.29x / 110.17x / 17.05x | 95.4% · HIGH | ₹34 → ₹55 | +61.76% | RIGHT / RIGHT | clean_reported_timestamp |

## Misses and evidence limitations

- Nityas Gems & Jewellery: model 33.7% positive probability; actual NSE opening +6.67%. Descriptive miss; this comparison alone cannot establish the causal reason.
- SRIT India: single_small_news_publisher_not_independently_corroborated. Excluded from the primary clean-input score.
- Shah Investor's Home: article_also_labels_detailed_table_17_05_conflicting_with_intro. Excluded from the primary clean-input score.
- Acevector: publisher_publication_and_modification_precede_claimed_bid_time. Excluded from the primary clean-input score.

## Sources and reproducibility

Bid times below are explicitly reported by publishers. Some are much earlier than 17:00; they are the latest eligible recoveries found, not asserted to be the latest bids that actually existed. Earlier and warning-flagged observations remain in the append-only source ledger.

- Model file SHA-256: `223a2861acc31952c5404cbe9eb0621f2241775404a4912a26fba0d579eacb2f`.
- Official offer identities/dates: NSE public-past-issues response; its SHA-256 is retained per matched IPO.
- Actual prices: listing-day NSE UDiFF `OpnPric`, checked against issuer name, symbol, date and ISIN. Neither day close nor a later session open is substituted.

- Tonbo Imaging India: [bid evidence](https://theipolist.in/ipo-details/tonbo-imaging-india) (SHA-256 `edf1dc341601548221f30febf749fe8f823f59184c79c84bff0833c1d8f6c2a9`)
- Zetwerk Manufacturing Business: [bid evidence](https://theipolist.in/ipo-details/zetwerk) (SHA-256 `05e12ed4053d5b81bf5311eba60a147076b0053c9c32ab1958816c7aad5e969e`)
- Bonfiglioli Transmissions: [bid evidence](https://theipolist.in/ipo-details/bonfiglioli-transmissions) (SHA-256 `3a9230c83e2140aeaa72630c1cc650c00aff5d910a12519e3a7e2c20ac7bf78c`)
- Sahajanand Medical Technologies: [bid evidence](https://theipolist.in/ipo-details/sahajanand-medical-technologies) (SHA-256 `cfbe85a5950a37e8ec0dc1a5a78c171ec3b09f4eafd76b748a55e64b848f990b`)
- Fusion CX: [bid evidence](https://theipolist.in/ipo-details/fusion-cx) (SHA-256 `2133407388f3f880dff80607fcd699f9951259934c5025b7b66741f4d10c9be5`)
- HD Fire Protect: [bid evidence](https://theipolist.in/ipo-details/hd-fire-protect) (SHA-256 `04e8089b13c3054f1d01c7ebf19df7df9f9cd3585030064d23fdb045ccadd757`)
- Nityas Gems & Jewellery: [bid evidence](https://choiceindia.com/news/nityas-gems-and-jewellery-ipo-subscription-day-3) (SHA-256 `8a9e8cbc68a4d9db4469965bc8639a76453491a9c155413d65c5da89e518e74f`); [NSE listing archive](https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20261008_F_0000.csv.zip) (SHA-256 `07fdfc3409b83fac6ee4a736ceb4fde20069c5d1a490d960b332c178cb703e39`)
- Vishal Nirmiti: [bid evidence](https://choiceindia.com/news/vishal-nirmiti-ipo-subscription-day-3) (SHA-256 `f6a5a0efcc0ce336db03c985e9ec11de34199559c29121dd860e702e83463034`); [NSE listing archive](https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20261008_F_0000.csv.zip) (SHA-256 `07fdfc3409b83fac6ee4a736ceb4fde20069c5d1a490d960b332c178cb703e39`)
- SRIT India: [bid evidence](https://rajkotcityguide.com/srit-india-ipo-gmp-today-subscription-status-listing-price/) (SHA-256 `579965d344c57d402d83c9e9e9e351e4e181410b2886fc4594e50a2b06c078ee`); [NSE listing archive](https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20261006_F_0000.csv.zip) (SHA-256 `f444e5ba9821fe09ac3bff69f53144369ce536be6481acc78ab00d175ae55b56`)
- Shah Investor's Home: [bid evidence](https://www.5paisa.com/blog/shah-investors-home-ipo-subscription-status) (SHA-256 `c33e588976a02d9f62245601e4d78acb6b4a7baffe279fe5b10ef50c2b3cf063`); [NSE listing archive](https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20261006_F_0000.csv.zip) (SHA-256 `f444e5ba9821fe09ac3bff69f53144369ce536be6481acc78ab00d175ae55b56`)
- Acevector: [bid evidence](https://choiceindia.com/news/acevector-ipo-subscription-day-3) (SHA-256 `3fc91d37d008be647fe41025d01db33bb9c490cf85ef92692f9d6c3a89d9d83d`); [NSE listing archive](https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20261005_F_0000.csv.zip) (SHA-256 `43913a4ef72663e2d3fbb0c3b9c686ed8c3d23e5df6ee6b756f8a5c954559ceb`)
- Orient Cables (India): [bid evidence](https://choiceindia.com/news/orient-cables-ipo-subscription-status-day-3) (SHA-256 `11761eadaeb3f353b9dc4be3fdba4f4ba7116154f32a971ce4c78babf99e39bb`); [NSE listing archive](https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20261005_F_0000.csv.zip) (SHA-256 `43913a4ef72663e2d3fbb0c3b9c686ed8c3d23e5df6ee6b756f8a5c954559ceb`)
- German Green Steel & Power: [bid evidence](https://choiceindia.com/news/german-green-steel-power-ipo-subscription-status-day-3) (SHA-256 `724eaed5777d9f5bbb69e3c9c859679bb39f21d1daa8a44632194d2134afc0ee`); [NSE listing archive](https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20261005_F_0000.csv.zip) (SHA-256 `43913a4ef72663e2d3fbb0c3b9c686ed8c3d23e5df6ee6b756f8a5c954559ceb`)
- Runwal Enterprises: [bid evidence](https://choiceindia.com/news/lumino-industries-ipo-subscription-day-3) (SHA-256 `3896776d2a977c70d6e36b858fdc1f94a44133f8aaca039c982363795f18b87b`); [NSE listing archive](https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20261005_F_0000.csv.zip) (SHA-256 `43913a4ef72663e2d3fbb0c3b9c686ed8c3d23e5df6ee6b756f8a5c954559ceb`)
- Moneyview: [bid evidence](https://choiceindia.com/news/moneyview-ipo-subscription-day-3) (SHA-256 `47af2f13caeddb93bed3ac87a1310346bec02e55102f4cf0b105de2ca68bb319`); [NSE listing archive](https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20261001_F_0000.csv.zip) (SHA-256 `ccc5fb27872716bbcc99d2d87e522ab304f6620e11c3a5045c30e1ff25cbfb73`)
