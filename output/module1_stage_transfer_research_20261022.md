# Module 1: does final-book accuracy transfer before closing?

Same three-input logistic family, only earlier listing years train each model.

| Training stage | 2024 cases | Early-input direction accuracy | Balanced accuracy | Always positive |
|---|---:|---:|---:|---:|
| final_book_training | 75 | 56.0% | 64.7% | 81.3% |
| matched_id_final_book_training | 75 | 52.0% | 67.7% | 81.3% |
| matched_early_book_training | 75 | 81.3% | 58.3% | 81.3% |

- Daily tables were retrieved retrospectively and may have been revised; not immutable investor-time captures.
- The matched-ID final-book track controls the training-IPO coverage difference when comparing stage-specific fits.
- An article publication date is not proof that a penultimate-session row was published at that time.
- Source-complete early-book coverage is non-random; no claim for uncovered IPOs.
- Final-input comparison is deliberately marked NOT_PRE_CLOSE and cannot be counted as application-time accuracy.
- 2024 outcomes were inspected in previous research; this is retrospective confirmation, not blind validation.
