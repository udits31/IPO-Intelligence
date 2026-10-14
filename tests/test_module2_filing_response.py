"""Observed financial clocks and real dated quote windows, not synthetic IPOs."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.calibrate_module2_rules import DEFAULT, decisions
from ml.replay_module2_financial_rules import EVIDENCE
from ml.replay_module2_news_overlay import SOURCES
from ml.research_module2_filing_response import (
    GRID, assemble_response, response_features, response_calls, select)


def evidence():
    return {r['ipo_id']: r for r in (json.loads(s) for s in EVIDENCE.read_text(encoding='utf-8').splitlines())}


def test_real_after_close_publication_uses_that_days_close_not_a_later_quote():
    sources = evidence()
    source = sources[504]
    prices = pd.read_csv(SOURCES['holdout'][0], parse_dates=['date'])
    one = prices[prices.ipo_id.eq(504)]
    checkpoint = pd.Timestamp(source['decision_date'])
    publication = source['financials']['published_at_ist']
    assert pd.Timestamp(publication).hour >= 16
    result = response_features(one, checkpoint, publication)
    assert result['filing_response_status'] == 'verified_daily_source_clock_response'
    assert result['source_reference_close_date'] == '2026-01-28'
    # Actual future prices are present in one but cannot affect the features.
    assert result == response_features(one[one.date.le(checkpoint)], checkpoint, publication)


def test_no_price_or_prepublication_anchor_is_manufactured():
    source = evidence()[504]
    prices = pd.read_csv(SOURCES['holdout'][0], parse_dates=['date'])
    one = prices[prices.ipo_id.eq(504)]
    checkpoint = pd.Timestamp(source['decision_date'])
    publication = source['financials']['published_at_ist']
    damaged = one[one.date.gt(pd.Timestamp(publication).normalize())]
    result = response_features(damaged, checkpoint, publication)
    assert result['filing_response_status'] == 'no_observed_prepublication_close'
    assert pd.isna(result['stock_response_since_source_pct'])
    with pytest.raises(ValueError, match='after the decision'):
        response_features(one, checkpoint, checkpoint + pd.Timedelta(hours=16))


def test_all_132_later_ipos_remain_and_missing_windows_keep_the_default():
    frame = assemble_response('holdout', evidence())
    assert len(frame) == 132
    missing = ~frame.filing_response_status.eq('verified_daily_source_clock_response').to_numpy()
    baseline = decisions(frame, DEFAULT)
    underwater = frame.decision_close.lt(frame.issue_price).to_numpy()
    for parameters in GRID:
        calls = response_calls(frame, parameters)
        assert len(calls) == 132
        np.testing.assert_array_equal(calls[missing], baseline[missing])
        assert not np.any(baseline[underwater] & ~calls[underwater])


def test_later_or_unmatured_outcomes_cannot_choose_filing_response_settings():
    sources = evidence()
    with pytest.raises(ValueError, match='matured 2021-2024'):
        select(assemble_response('holdout', sources), '2024-12-31')
    development = assemble_response('development', sources)
    with pytest.raises(ValueError, match='matured 2021-2024'):
        select(development, '2024-12-31')
    known = development[development.outcome_date.le('2024-12-31')]
    choice, candidates = select(known, '2024-12-31')
    assert len(known) == 214 and len(candidates) == 13 and choice in candidates
