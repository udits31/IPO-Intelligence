from types import SimpleNamespace

import pandas as pd
import pytest

from ml.test_historical_joins_and_cutoff_20261028 import (
    choose_threshold, eligible_document, nav_candidates,
)


def test_document_gate_rejects_future_and_unclocked_same_day():
    row = SimpleNamespace(company_name='Example Limited', decision_market_date='2023-08-10')
    record = dict(status='dated_document_facts', company_name='Example Ltd',
                  document_date='2023-08-09', facts={'financial_period_end': '2023-03-31'})
    assert eligible_document(record, row)[0]
    for date in ['2023-08-10', '2023-08-11', None]:
        assert not eligible_document(dict(record, document_date=date), row)[0]
    assert not eligible_document(dict(record, company_name='Different Limited'), row)[0]
    assert not eligible_document(dict(record, facts={'financial_period_end': '2023-09-30'}), row)[0]


def test_nav_requires_unique_dated_issuer_value_and_excludes_peers():
    text = 'V. Net Asset Value per Equity Share\nAs on March 31, 2023 52.99\nNotes:\nOther values'
    found = nav_candidates(text, '2023-08-10')
    assert len(found) == 1 and found[0][1] == 52.99
    assert nav_candidates(text, '2023-01-01') == []
    assert nav_candidates('Net Asset Value\nMarch 31, 2023 52.99 64.00', '2023-08-10') == []
    assert nav_candidates('Net Asset Value\nConsolidated / Standalone\nMarch 31, 2023 52.99', '2023-08-10') == []
    assert nav_candidates('Net Asset Value\nComparison with listed industry peers\nMarch 31, 2023 52.99', '2023-08-10') == []


def test_cutoff_cannot_select_on_confirmation_or_test_labels():
    rows = pd.DataFrame({'year': [2023]*4, 'target_binary': [0,0,1,1], 'p': [.4,.6,.7,.8]})
    choice = choose_threshold(rows)
    assert .6 < choice['threshold'] <= .7 and choice['balanced_accuracy'] == 1
    for year in [2024, 2025, 2026]:
        with pytest.raises(ValueError, match='2023'):
            choose_threshold(rows.assign(year=year))


def test_through2025_cutoff_still_excludes_2025_and_2026_selection():
    rows = pd.DataFrame({'year': [2024]*4, 'target_binary': [0,0,1,1], 'p': [.4,.6,.7,.8]})
    assert choose_threshold(rows, selection_year=2024)['balanced_accuracy'] == 1
    for year in [2025, 2026]:
        with pytest.raises(ValueError):
            choose_threshold(rows.assign(year=year), selection_year=year)


def test_through2025_split_rejects_2026_training_and_overlap():
    from ml.test_through2025_cutoff_20261028 import check_split
    from tests.test_model_generation_comparison import frame
    train = frame(2025, [1], ['Example Ltd'])
    test = frame(2026, [2], ['Other Ltd'])
    check_split(train, test)
    with pytest.raises(ValueError, match='years'):
        check_split(frame(2026, [1], ['Example']), test)
    with pytest.raises(ValueError, match='ID leakage'):
        check_split(train, frame(2026, [1], ['Other']))
