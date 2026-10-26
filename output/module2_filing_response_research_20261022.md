# Module 2: price response aligned to a financial-source clock

Same 31st-close checkpoint, next-close execution and 30-later-session direction; all 132 later cases retained.

| Policy | Later direction accuracy |
|---|---:|
| Current monitor | 50.00% |
| Earlier-data-selected source-response rules | 50.00% |
| Always sell | 58.33% |

Verified selected NSE financial-source publication, not necessarily the first public earnings announcement.

Daily close before the source clock to decision close; at least two observed post-source closes; same-date Nifty response subtracted.

## Limitations

- The selected financial source can be later than an earlier PDF or press release; this is not an isolated first-announcement return.
- Daily session-close timing does not establish an intraday quote immediately before a midday publication.
- Nifty prices are retrospective secondary-provider observations, not independently immutable exchange archives.
- Missing anchors, source clocks or market prices retain the default rule; IPOs are not removed and issue price is not a substitute anchor.
- Repeated later-year inspection makes this exploratory, not blind or prospective validation.
- No live policy or paper result is promoted. Literature on announcement responses is not a score for this application.
