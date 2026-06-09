"""Fast calendar reads; source collection is a separate, single-flight task."""
from datetime import datetime
from threading import Lock
from zoneinfo import ZoneInfo

_refresh_lock = Lock()
_state = dict(status='idle',requested_at_ist=None,completed_at_ist=None,error=None)


def cached_items(cache,*,as_of=None):
    # Preserve the cache's provenance fields and observation clocks. Calendar
    # reconciliation at the API boundary computes today's offer stage.
    from scrapers.live_ipo_scraper import classify_ipo_stage
    today=as_of or datetime.now(ZoneInfo('Asia/Kolkata')).date()
    rows=[]
    for url,item in cache.items():
        if item.get('source_observed') is not True or not item.get('company_name'):continue
        category,status=classify_ipo_stage(item,as_of_date=today)
        rows.append({**item,'url':url,'category':category,'status':status})
    rows.sort(key=lambda r: ({"OPEN":0,"UPCOMING":1,"CLOSED":2,"LISTED":3}.get(r["category"],4),r.get("company_name","")))
    return rows


def queue_refresh(background):
    if not _refresh_lock.acquire(blocking=False):return False
    _state.update(status='refreshing',requested_at_ist=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),error=None)
    background.add_task(_refresh)
    return True


def _refresh():
    try:
        from scrapers.live_ipo_scraper import sync_chittorgarh_calendar_delta
        # Refresh published subscriptions/GMP without unnecessarily rescraping
        # every prospectus and immutable detail page on a UI refresh.
        sync_chittorgarh_calendar_delta(force_refresh=False)
        from scrapers.live_ipo_scraper import CALENDAR_FETCH_STATUS
        _state.update(status=CALENDAR_FETCH_STATUS.get('status','completed'),error=CALENDAR_FETCH_STATUS.get('error'))
    except Exception as exc:
        _state.update(status='failed',error=str(exc)[:250])
    finally:
        _state['completed_at_ist']=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()
        _refresh_lock.release()


def refresh_status():
    return dict(_state)
