import gzip,hashlib,json,sqlite3
import pandas as pd
import numpy as np
from services import module2_shadow_forecast as shadow


def test_forward_scoring_waits_for_actual_next_bar_and_executes_after_signal(tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(weekly_artifact_sha256='model')))
    monkeypatch.setattr(shadow,'MANIFEST',manifest);monkeypatch.setattr(shadow,'YAHOO_ARCHIVES',tmp_path)
    dates=pd.date_range('2026-10-28',periods=2,tz='Asia/Kolkata')+pd.Timedelta(hours=9,minutes=15)
    raw=json.dumps(dict(chart=dict(result=[dict(meta=dict(symbol='ABC.NS',currency='INR'),
        timestamp=[int(d.timestamp()) for d in dates],indicators=dict(quote=[dict(open=[99.,102.],close=[100.,110.])]))]))).encode()
    sha=hashlib.sha256(raw).hexdigest();(tmp_path/(sha+'.json.gz')).write_bytes(gzip.compress(raw))
    db=tmp_path/'shadow.db'
    with sqlite3.connect(db) as c:
        shadow.init(c)
        for i,day in enumerate(['2026-10-28','2026-10-29']):
            payload=dict(source=dict(stock_sha256=sha),symbol='ABC.NS',listing_date='2026-10-01',
                decision_close=[100.,110.][i],features=dict(rule_position=1))
            encoded=json.dumps(payload);digest=hashlib.sha256(encoded.encode()).hexdigest()
            c.execute('INSERT INTO module2_ml_shadow_forecasts_v1 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (str(i),1,'ABC',day,day+'T13:00:00+00:00','model',.7,1.,2.,[1,0][i],['BUY','SELL'][i],encoded,digest))
    r=shadow.scorecard(db_path=db)
    assert r['forecasts']==2 and r['evaluated_next_session']==1 and r['direction_accuracy_pct']==100.
    assert r['weekly_evaluated']==0
    assert sum(x['outcome_status']=='pending_next_session' for x in r['records'])==1
    paper=r['paper_strategies']['shadow_ml']
    assert paper['buys']==1
    assert np.isclose(paper['mean_net_return_pct'],100*((110/102)*(1-.00175)**2-1))
    # Durable later outcome bars grade old forecasts without a new forecast.
    chart=json.loads(raw);observed=chart['chart']['result'][0]
    observed['timestamp'].append(int((dates[-1]+pd.Timedelta(days=1)).timestamp()))
    observed['indicators']['quote'][0]['open'].append(111.)
    observed['indicators']['quote'][0]['close'].append(112.)
    later=json.dumps(chart).encode()
    with sqlite3.connect(db) as c:
        shadow.archive_source(c,1,'ABC.NS','2026-10-30',later,'2026-10-30T13:00:00+00:00')
    r=shadow.scorecard(db_path=db)
    assert r['forecasts']==2 and r['evaluated_next_session']==2
    assert r['paper_strategies']['shadow_ml']['buys']==1
    # The SELL recorded after the second close cannot sell at that day's open.
    with sqlite3.connect(db) as c:c.execute("UPDATE module2_ml_shadow_forecasts_v1 SET payload_sha256='bad' WHERE forecast_id='0'")
    r=shadow.scorecard(db_path=db)
    assert r['unverified_payloads']==1 and r['evaluated_next_session']==1
