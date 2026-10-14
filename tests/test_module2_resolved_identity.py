from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

import services.module2_dynamic_monitor as monitor


def test_name_search_keeps_resolved_source_id_and_handles_first_trading_day(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):
            now=datetime(2026,10,28,14,0,tzinfo=ZoneInfo('Asia/Kolkata'))
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)
    monkeypatch.setattr(monitor,'datetime',Clock)
    stock=pd.DataFrame(dict(date=[pd.Timestamp('2026-10-28')],close=[1390.]))
    stock.attrs['provider_meta']=dict(symbol='JPL.NS',company_name='Jio Platforms Limited')
    monkeypatch.setattr(monitor.AttributionEngine,'_fetch_yahoo_series',lambda *a,**k:stock.copy())
    data=dict(company_name='Jio Platforms',issue_price=1119,listing_date='2026-10-28',nse_symbol='JPL',
              url='https://www.chittorgarh.com/ipo/jio-platforms-ipo/3152/')
    result=monitor.evaluate_dynamic_listing(data,target='Jio Platforms')
    assert result['ipo_id']==3152 and result['status']=='FIRST_TRADING_SESSION'
    assert 'verdict' not in result


def test_cached_scraper_keeps_source_url_for_name_and_url_search(monkeypatch):
    import scrapers.live_ipo_scraper as scraper
    url='https://www.chittorgarh.com/ipo/jio-platforms-ipo/3152/'
    monkeypatch.setattr(scraper,'load_scraped_cache',lambda:{url:dict(company_name='Jio Platforms')})
    monkeypatch.setattr(scraper,'fetch_live_macro_regime',lambda:{})
    for target in ('Jio Platforms',url):
        assert scraper.scrape_chittorgarh_url(target)['data']['url']==url
