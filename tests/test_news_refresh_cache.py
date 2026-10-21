from datetime import datetime, timezone
from email.utils import format_datetime

import pytest

from services import news_service as news


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path,monkeypatch):
    monkeypatch.setattr(news,'NEWS_CACHE_PATH',tmp_path/'news.json')
    monkeypatch.setattr(news,'_NEWS_CACHE',{})
    monkeypatch.setattr(news,'_COMPANY_LOCKS',{})


def article(title='Example Corp wins order'):
    return {'title':title,'published_at':format_datetime(datetime.now(timezone.utc)),
            'url':'https://example.test/article','publisher':'Example'}


def test_only_stale_news_refetches_and_survives_a_restart(monkeypatch):
    calls=[]
    monkeypatch.setattr(news,'_download_company_news',lambda *a,**k: calls.append(a) or [article()])
    first = news.get_news_snapshot('Example Corp')
    assert first['freshness'] == 'fresh'
    news.get_news_snapshot('Example Corp')
    monkeypatch.setattr(news,'_NEWS_CACHE',None)
    assert news.get_news_snapshot('Example Corp')['articles'] == first['articles']
    assert len(calls) == 1
    news._NEWS_CACHE['example corp']['last_success_epoch'] -= 601
    news.get_news_snapshot('Example Corp')
    assert len(calls) == 2


def test_failure_retains_last_good_headlines_but_does_not_claim_fresh_sentiment(monkeypatch):
    monkeypatch.setattr(news,'_download_company_news',lambda *a,**k: [article()])
    first = news.get_news_snapshot('Example Corp')
    news._NEWS_CACHE['example corp']['last_success_epoch'] -= 601
    monkeypatch.setattr(news,'_download_company_news',lambda *a,**k: (_ for _ in ()).throw(TimeoutError()))
    result = news.NewsService().get_company_news('Example Corp')
    assert result['articles'] == first['articles']
    assert result['freshness'] == 'stale'
    assert result['last_success_at'] == first['last_success_at']
    assert result['sentiment_score'] is None
    assert result['status'] == 'stale_headlines_source_unavailable'


def test_no_articles_is_distinct_from_fetch_failure(monkeypatch):
    monkeypatch.setattr(news,'_download_company_news',lambda *a,**k: [])
    assert news.NewsService().get_company_news('Example Corp')['status'] == 'no_recent_timestamped_articles'


def test_copilot_uses_saved_forecast_basis():
    from services.ai_copilot_service import ask_ai_copilot
    answer = ask_ai_copilot('Example Corp','explain',{
        'forecast_basis':'historical_reconstruction','forecast_asof_ist':'2026-10-19T16:44:00+05:30',
        'verdict':'HIGH','hybrid_ml_scores':{'hybrid_positive_listing_prob':91.1}})
    assert '91.1%' in answer and 'calculated later' in answer
    assert 'final-book model estimates' not in answer


@pytest.mark.parametrize('title', [
    'SRIT INDIA LTD. Stock/Share price , NSE/BSE Forecast News and Live Quotes',
    'SRIT India Limited (SRIT.NS) stock price, news, quote and history',
    'SRIT India Ltd. Share Price Today',
    'SRIT India Limited (SRIT.NS) interactive stock chart',
    'SRIT India Ltd Share Price Today NSE/BSE Charts',
    'SRIT India Limited (SRIT.NS) Stock Price, News, Quote & History',
    'SRIT India Share Price Today: Price Action and Valuation',
    'SRIT India Ltd Key Financial Ratios – Valuation, Profitability & More',
])
def test_quote_directories_are_not_company_event_news(title):
    assert news._quote_directory_title(title)


@pytest.mark.parametrize('title', [
    'SRIT India share price drops 10% after weak quarterly results',
    'SRIT India wins order for hospital software',
    'SRIT India profit falls after contract cancellation',
])
def test_genuine_event_or_price_move_reports_remain_eligible(title):
    assert not news._quote_directory_title(title)


def test_legacy_cached_quote_pages_are_filtered_without_claiming_a_fetch_failure(monkeypatch):
    monkeypatch.setattr(news, '_download_company_news', lambda *a, **k: [article()])
    news.get_news_snapshot('Example Corp')
    news._NEWS_CACHE['example corp']['articles'] = [article('Example Corp Share Price Today')]
    result = news.NewsService().get_company_news('Example Corp')
    assert result['articles'] == []
    assert result['sentiment_score'] is None
    assert result['status'] == 'no_recent_timestamped_articles'
    assert result['freshness'] == 'fresh'
