"""Actual archived headlines and prices, with historical cutoff safeguards."""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.monitor_engine import HoldSellMonitor
from services.news_service import NewsService
from ml.collect_module2_dated_news import ROOT,parse_news
from ml.reparse_module2_news import OUTPUT
from ml.replay_module2_news_overlay import (SOURCES,GRID,NO_OVERLAY,load_evidence,
                                            assemble_news,overlay,select,coverage)


@pytest.fixture(scope='module')
def actual_news():
    if not OUTPUT.exists():
        pytest.skip('Real dated RSS snapshots are not installed')
    return [json.loads(line) for line in OUTPUT.read_text(encoding='utf-8').splitlines()]


@pytest.fixture(scope='module')
def an_article_case(actual_news):
    return next(r for r in actual_news if r['status']=='verified_rss_response' and r['articles'])


def test_archive_matches_actual_rss_and_all_news_precedes_decision(actual_news):
    for row in actual_news:
        if row['status']!='verified_rss_response':
            continue
        with gzip.open(ROOT/row['archive_path'],'rb') as stream:
            parsed,_,_=parse_news(stream.read(),row['company_name'],pd.Timestamp(row['cutoff_ist']).to_pydatetime())
        assert parsed==row['articles']
        assert all(pd.Timestamp(a['published_at'])<pd.Timestamp(row['cutoff_ist']) for a in parsed)
        assert all(pd.Timestamp(a['published_at']).tz_convert('Asia/Kolkata').date()<pd.Timestamp(row['cutoff_ist']).date()
                   for a in parsed)


def test_empty_real_feed_is_not_fabricated_neutral_sentiment(actual_news):
    row=next(r for r in actual_news if r['status']=='verified_rss_response' and not r['articles'])
    result=NewsService().summarize_historical_snapshot(row['company_name'],row['decision_date'],row)
    assert result['sentiment_score'] is None
    assert result['status']=='historical_no_eligible_indexed_headlines'


def test_issuer_and_cutoff_mismatch_cannot_become_no_news(an_article_case):
    row=an_article_case
    with pytest.raises(ValueError,match='issuer or decision cutoff'):
        NewsService().summarize_historical_snapshot('Not this issuer',row['decision_date'],row)
    with pytest.raises(ValueError,match='issuer or decision cutoff'):
        NewsService().summarize_historical_snapshot(row['company_name'],'2020-01-01',row)


def test_real_headline_is_not_used_before_its_publication(an_article_case):
    row=copy.deepcopy(an_article_case)
    earlier=min(pd.Timestamp(a['published_at']).tz_convert('Asia/Kolkata') for a in row['articles'])-pd.Timedelta(days=1)
    row['decision_date']=earlier.date().isoformat()
    row['cutoff_ist']=str(earlier.date())+'T15:30:00+05:30'
    result=NewsService().summarize_historical_snapshot(row['company_name'],row['decision_date'],row)
    assert result['articles']==[] and result['sentiment_score'] is None
    assert result['rejected_by_asof_guard']==len(row['articles'])


def test_historical_monitor_uses_only_supplied_dated_snapshot(an_article_case,monkeypatch):
    row=an_article_case
    name='development' if row['listing_year']<=2024 else 'holdout'
    monitor=HoldSellMonitor(prices_path=str(SOURCES[name][0]))
    def forbidden(*a,**k):
        raise AssertionError('Present-day news was fetched for a historical decision')
    monkeypatch.setattr(monitor.news_service,'get_company_news',forbidden)
    result=monitor.evaluate_holding(row['ipo_id'],as_of_date=row['decision_date'],
                                    historical_news=row,disclosures=[])
    assert result['news_sentiment']['status']=='historical_indexed_headlines'
    assert result['news_sentiment']['raw_sha256']==row['raw_sha256']
    assert result['recent_news']==row['articles']
    with pytest.raises(ValueError,match='explicit historical decision'):
        monitor.evaluate_holding(row['ipo_id'],historical_news=row)


@pytest.fixture(scope='module')
def later_with_news(actual_news):
    expected=pd.read_csv(SOURCES['holdout'][1])
    if not set(expected.ipo_id).issubset({r['ipo_id'] for r in actual_news}):
        pytest.skip('Historical news collection is still in progress')
    return assemble_news('holdout',load_evidence())


def test_overlay_does_not_drop_news_missing_cases_or_waive_structural_loss(later_with_news):
    frame=later_with_news
    assert len(frame)==132
    baseline=overlay(frame,NO_OVERLAY)
    assert np.array_equal(baseline,frame.sell.to_numpy(dtype=bool))
    for parameters in GRID:
        call=overlay(frame,parameters)
        assert len(call)==len(frame)
        underwater=baseline & frame.decision_close.lt(frame.issue_price).to_numpy()
        assert call[underwater].all()
        missing=frame.news_score.isna().to_numpy()
        np.testing.assert_array_equal(call[missing],baseline[missing])


def test_later_outcomes_cannot_select_news_policy(later_with_news):
    with pytest.raises(ValueError,match='only matured 2021–2024'):
        select(later_with_news,later_with_news.outcome_date.max())


def test_coverage_upper_bound_is_not_reported_as_a_prediction_score(later_with_news):
    stats=coverage(later_with_news)
    assert stats['n']==132
    assert 'not achievable measured model accuracy' in stats['ceiling_definition']
