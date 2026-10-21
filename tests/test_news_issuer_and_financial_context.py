"""Real RSS examples exercise issuer and financial-phrase corrections."""
import json

import pandas as pd
import pytest

from services.news_identity import ROOT,MASTER,relevant_headline
from services.news_service import NewsService,_quote_directory_title
from ml.collect_module2_dated_news import OUTPUT as ORIGINAL
from ml.calibrate_module2_market_rules import GRID,BASE,market_calls,select
from ml.calibrate_module2_rules import assemble,decisions,DEFAULT
from ml.replay_module2_news_overlay import SOURCES


@pytest.fixture(scope='module')
def actual_archive():
    if not ORIGINAL.exists() or not MASTER.exists():
        pytest.skip('Original real RSS and official issuer names are not installed')
    return {r['ipo_id']:r for r in (json.loads(line) for line in ORIGINAL.read_text(encoding='utf-8').splitlines())}


def test_railway_industry_news_is_not_irfc_company_news(actual_archive):
    record=actual_archive[311]
    wrong=next(a['title'] for a in record['articles'] if 'Hitachi ABB' in a['title'])
    assert not relevant_headline(wrong,record['company_name'])


def test_shared_brand_does_not_merge_two_listed_issuers(actual_archive):
    record=actual_archive[588]
    wrong=next(a['title'] for a in record['articles'] if 'Kalpataru Projects Q1' in a['title'])
    assert not relevant_headline(wrong,record['company_name'])
    assert relevant_headline(wrong,'Kalpataru Projects International Limited')


def test_restaurant_hygiene_rating_is_not_a_finance_issuer_event(actual_archive):
    row=actual_archive[324]
    title=next(a['title'] for a in row['articles'] if 'hygiene rated' in a['title'])
    assert not relevant_headline(title,row['company_name'])


def test_observed_live_automated_rating_is_not_a_corporate_news_catalyst():
    # Observed from the real Kalpataru live RSS/API on 21 October 2026
    # (MarketsMojo, published 16 October), not a generated market observation.
    title='Kalpataru Ltd is Rated Strong Sell'
    assert _quote_directory_title(title)
    assert NewsService().analyze_sentiment(title)<0


def test_real_profit_decline_headline_is_not_bullish(actual_archive):
    headline=actual_archive[544]['articles'][0]['title']
    assert 'Profit Decline' in headline
    assert NewsService().analyze_sentiment(headline)<0


def test_real_growth_and_order_wins_retain_positive_sign(actual_archive):
    scorer=NewsService()
    growth=next(a['title'] for a in actual_archive[332]['articles'] if 'growth in profit' in a['title'])
    order=next(a['title'] for a in actual_archive[467]['articles'] if 'bags 2 orders' in a['title'])
    assert scorer.analyze_sentiment(growth)>0
    assert scorer.analyze_sentiment(order)>0


def test_market_rule_does_not_use_future_outcomes_to_make_a_call():
    frame=assemble(*SOURCES['holdout'])
    for parameters in GRID:
        expected=market_calls(frame,parameters)
        restricted=frame.drop(columns=['day30_return_pct','day1_return_pct','executed_hold_return_pct','outcome_date'])
        assert (expected==market_calls(restricted,parameters)).all()
    assert (market_calls(frame,BASE)==decisions(frame,DEFAULT)).all()


def test_later_year_labels_cannot_choose_market_parameters():
    later=assemble(*SOURCES['holdout'])
    with pytest.raises(ValueError,match='matured 2021–2024'):
        select(later,later.outcome_date.max())
