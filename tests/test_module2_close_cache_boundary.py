import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import core.attribution_engine as engine
from services.market_cache_freshness import invalidate_preclose_market_cache


@pytest.mark.parametrize('minute,expected,calls_expected',[(29,100.,0),(31,200.,1)])
def test_preclose_cached_daily_candle_is_refetched_after_session_close(tmp_path,monkeypatch,minute,expected,calls_expected):
    zone=ZoneInfo('Asia/Kolkata')
    fetched=datetime(2026,10,28,15,25,tzinfo=zone).timestamp()
    now=datetime(2026,10,28,15,minute,tzinfo=zone).timestamp()
    monkeypatch.setattr(engine.time,'time',lambda:now)
    monkeypatch.setattr(engine,'_YAHOO_CACHE',{('TEST.NS',None):dict(fetched_at=fetched,data=pd.DataFrame(dict(date=[pd.Timestamp('2026-10-28')],close=[100.])))})
    monkeypatch.setattr(engine,'YAHOO_ARCHIVES',tmp_path)
    calls=[]
    class Response:
        url='https://query1.finance.yahoo.com/v8/finance/chart/TEST.NS'
        content=json.dumps({'chart':{'result':[dict(meta=dict(symbol='TEST.NS',currency='INR'),timestamp=[int(fetched)],indicators=dict(quote=[dict(close=[200.])]))]}}).encode()
        def json(self):return json.loads(self.content)
        def raise_for_status(self):pass
    def fetch(*args,**kwargs):calls.append(1);return Response()
    monkeypatch.setattr(engine.requests,'get',fetch)
    invalidate_preclose_market_cache(datetime.fromtimestamp(now,zone))
    result=engine.AttributionEngine._fetch_yahoo_series('TEST.NS')
    assert result.iloc[0]['close']==expected and len(calls)==calls_expected
