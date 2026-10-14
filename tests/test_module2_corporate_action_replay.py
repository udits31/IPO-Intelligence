"""Replay safeguards exercised on archived, real exchange observations."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.reconcile_module2_corporate_actions import (
    ROOT, RAW, SOURCES, AUDIT, action_rows, load_listing_resolutions,
    normalize_prices, parse_bhavcopy, share_change,
)
from ml.calibrate_module2_rules import assemble, decisions, DEFAULT


@pytest.fixture
def exchange_actions():
    paths = list(RAW.glob('actions_*.json.gz'))
    if not paths:
        pytest.skip('Original NSE action responses are not installed')
    rows = []
    for path in paths:
        _, first, last = path.name.split('.')[0].split('_')
        with gzip.open(path,'rt') as stream:
            rows.extend(action_rows(json.load(stream),pd.Timestamp(first),pd.Timestamp(last)))
    return pd.DataFrame(rows)


@pytest.fixture
def actual_prices():
    if not SOURCES['holdout'].exists():
        pytest.skip('Actual official NSE prices are not installed')
    return pd.read_csv(SOURCES['holdout'],parse_dates=['date'])


@pytest.fixture
def canonical():
    return pd.read_csv(ROOT/'data/canonical_listed_ipos.csv',parse_dates=['listing_date']).set_index('id')


def test_real_exchange_ratios_not_guessed_from_price_moves(exchange_actions):
    bonus = exchange_actions[exchange_actions.symbol.eq('AASTHA') & exchange_actions.ex_date.eq('2026-09-28')].iloc[0]
    split = exchange_actions[exchange_actions.symbol.eq('CORDELIA') & exchange_actions.ex_date.eq('2026-08-25')].iloc[0]
    assert share_change(bonus.subject) == ('bonus',2.)
    assert share_change(split.subject) == ('split',10.)
    rights = exchange_actions[exchange_actions.symbol.eq('SHANTIGOLD') & exchange_actions.subject.str.startswith('Rights')].iloc[0]
    assert share_change(rights.subject) == ('unsupported',None)


def test_share_count_changes_only_on_actual_ex_date(exchange_actions,actual_prices,canonical):
    original = actual_prices[actual_prices.ipo_id.isin([659,661])]
    result, changes, excluded = normalize_prices(original,canonical,exchange_actions)
    assert not excluded and len(changes)==2
    for ipo_id,date,factor in [(659,'2026-09-28',2.),(661,'2026-08-25',10.)]:
        group = result[result.ipo_id.eq(ipo_id)]
        assert group.loc[group.date.lt(date),'shares_per_listing_share'].eq(1).all()
        assert group.loc[group.date.ge(date),'shares_per_listing_share'].eq(factor).all()
        np.testing.assert_allclose(group.close_price_adjusted,
                                   group.unadjusted_nse_close*group.shares_per_listing_share)
        # A replay stopped before the action has exactly the same past inputs.
        prefix = original[original.ipo_id.eq(ipo_id)].sort_values('date').head(31)
        earlier,_,_ = normalize_prices(prefix,canonical,exchange_actions)
        np.testing.assert_allclose(group.head(31).close_price_adjusted,earlier.close_price_adjusted)
        assert earlier.shares_per_listing_share.eq(1).all()


def test_later_rights_issue_does_not_censor_prior_frozen_outcomes(exchange_actions,actual_prices,canonical):
    result,_,excluded = normalize_prices(actual_prices[actual_prices.ipo_id.isin([575,603])],canonical,exchange_actions)
    assert not excluded
    assert result.groupby('ipo_id').size().eq(62).all()
    assert result.shares_per_listing_share.eq(1).all()


def test_listing_conflicts_resolved_from_raw_exchange_archives(canonical):
    if not AUDIT.exists():
        pytest.skip('Listing-day exchange archives are not installed')
    verified = load_listing_resolutions(AUDIT,canonical)
    assert set(verified)=={304,404,424,440,664}
    assert verified[304]['nse_close']==812.6
    assert verified[404]['nse_close']==1422.3
    assert verified[440]['series']=='BE'
    assert verified[440]['nse_open']==145.
    audit = pd.read_csv(AUDIT)
    actual = audit.iloc[0]
    with pytest.raises(ValueError,match='identity not unique'):
        parse_bhavcopy((ROOT/actual.raw_archive_path).read_bytes(),pd.Timestamp(actual.listing_date),'CORDELIA')


def test_actual_reconciled_replay_matches_unchanged_rule_and_drawdown_definition():
    for name,source in [('development',SOURCES['development']),('holdout',SOURCES['holdout'])]:
        prices = source.with_name(source.stem+'_ca_reconciled.csv')
        ledger = ROOT/f'output/module2_{name}_decision_ledger_goal_corporate_actions_20261021.csv'
        if not prices.exists() or not ledger.exists():
            pytest.skip('Reconciled actual-price replay is not installed')
        frame = assemble(prices,ledger)
        assert np.array_equal(decisions(frame,DEFAULT),frame.sell.to_numpy(dtype=bool))
        assert frame.future_max_drawdown_pct.le(0).all()
        data = pd.read_csv(prices)
        for row in frame.itertuples():
            observed = data[data.ipo_id.eq(row.ipo_id)].sort_values('date').iloc[31:62]
            expected = ((observed.close_price_adjusted/observed.close_price_adjusted.cummax()-1)*100).min()
            assert np.isclose(row.future_max_drawdown_pct,expected)
