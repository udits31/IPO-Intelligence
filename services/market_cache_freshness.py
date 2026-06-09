"""Live cache boundaries; historical scoring code and price ledgers stay fixed."""
from datetime import datetime
from services.point_in_time import IST


def invalidate_preclose_market_cache(now=None):
    from core.attribution_engine import _YAHOO_CACHE
    now=(now or datetime.now(IST)).astimezone(IST)
    if (now.hour,now.minute)<(15,30):return 0
    cutoff=now.replace(hour=15,minute=30,second=0,microsecond=0).timestamp()
    removed=0
    for key,entry in list(_YAHOO_CACHE.items()):
        if entry.get('fetched_at',0)<cutoff and _YAHOO_CACHE.get(key) is entry:
            _YAHOO_CACHE.pop(key,None);removed+=1
    return removed
