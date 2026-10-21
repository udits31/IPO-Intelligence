from datetime import date
from services import market_calendar_feed as feed


def test_cached_read_is_network_free_and_reclassifies_offer_dates(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx,'get',lambda *a,**k: (_ for _ in ()).throw(AssertionError('Cache read contacted network')))
    cache={'source':dict(company_name='Example',source_observed=True,category='OPEN',
        date_range='2026-10-26 to 2026-10-29',gmp_pct=0,subscription_observed_at='2026-10-29T16:45:00+05:30'),
        'unverified':dict(company_name='Unverified',source_observed=False)}
    rows=feed.cached_items(cache,as_of=date(2026,10,30))
    assert len(rows)==1 and rows[0]['category']=='CLOSED'
    assert rows[0]['gmp_pct']==0
    assert rows[0]['subscription_observed_at']==cache['source']['subscription_observed_at']
    assert cache['source']['category']=='OPEN'


def test_refresh_is_single_flight_and_keeps_the_fast_read_separate(monkeypatch):
    from scrapers import live_ipo_scraper as scraper
    monkeypatch.setattr(feed,'_state',dict(status='idle'))
    calls=[]
    monkeypatch.setattr(scraper,'sync_chittorgarh_calendar_delta',lambda **kw:calls.append(kw))
    class Background:
        tasks=[]
        def add_task(self,task):self.tasks.append(task)
    background=Background()
    assert feed.queue_refresh(background) is True
    assert feed.queue_refresh(background) is False
    assert calls==[] and len(background.tasks)==1
    background.tasks.pop()()
    assert calls==[{'force_refresh':False}]
    assert feed.refresh_status()['completed_at_ist']
    assert feed.queue_refresh(background) is True
    background.tasks.pop()()


def test_failed_source_refresh_releases_lock_and_reports_failure(monkeypatch):
    from scrapers import live_ipo_scraper as scraper
    monkeypatch.setattr(feed,'_state',dict(status='idle'))
    def broken(**kwargs):raise ValueError('source unavailable')
    monkeypatch.setattr(scraper,'sync_chittorgarh_calendar_delta',broken)
    class Background:
        def add_task(self,task):self.task=task
    background=Background();feed.queue_refresh(background);background.task()
    assert feed.refresh_status()['status']=='failed'
    assert 'source unavailable' in feed.refresh_status()['error']
    assert feed.queue_refresh(background) is True
    background.task()
