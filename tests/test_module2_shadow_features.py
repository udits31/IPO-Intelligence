import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from services import module2_shadow_forecast as shadow


def test_live_features_match_existing_training_examples():
    root=Path(__file__).resolve().parents[1]
    if not (root/'output/module2_next_session_20261029/daily_examples.csv').exists():
        import pytest
        pytest.skip('Historical training-feature snapshot is not installed')
    data=pd.read_csv(root/'output/module2_next_session_20261029/daily_examples.csv')
    raw=pd.read_csv(root/'data/audit/module2_development_ohlcv_2021_2024.csv',parse_dates=['date'])
    ipos=pd.read_csv(root/'data/canonical_listed_ipos.csv').set_index('id')
    checked=0
    for sample in data[data.listing_year.le(2024)].iloc[::100].itertuples():
        g=raw[raw.ipo_id.eq(sample.ipo_id)&raw.date.le(sample.decision_date)].rename(columns={'ohlcv_close':'close'})
        try:f=shadow.features_at(g,float(ipos.loc[sample.ipo_id].issue_price))
        except ValueError:continue
        for k,v in f.items():assert np.isclose(v,getattr(sample,k),rtol=1e-7,atol=1e-8),(k,v,getattr(sample,k))
        checked+=1
        if checked==3:break
    assert checked==3


def test_missing_volume_or_short_history_blocks_forecast():
    import pytest
    g=pd.DataFrame(dict(date=pd.date_range('2026-01-01',periods=21),open=100.,high=102.,low=99.,close=101.,volume=1000.,nifty50_close=22000.,sector_index_close=1000.))
    with pytest.raises(ValueError):shadow.features_at(g.iloc[:20],100.)
    g.loc[20,'volume']=np.nan
    with pytest.raises(ValueError):shadow.features_at(g,100.)


def test_raw_future_bar_is_not_part_of_forecast_history():
    dates=pd.date_range('2026-10-01',periods=3,tz='Asia/Kolkata')+pd.Timedelta(hours=9,minutes=15)
    raw=dict(chart=dict(result=[dict(meta=dict(symbol='ABC.NS',currency='INR'),timestamp=[int(d.timestamp()) for d in dates],
        indicators=dict(quote=[dict(open=[100,101,999],high=[103,104,999],low=[99,100,999],close=[102,103,999],volume=[1000,2000,999])]))]))
    g=shadow.frame_from_archive(raw,'ABC.NS','2026-10-01','2026-10-02')
    assert len(g)==2 and g.close.max()==103


def test_artifact_change_blocks_shadow_logging(tmp_path,monkeypatch):
    artifact=tmp_path/'weekly';artifact.write_bytes(b'changed artifact')
    daily=tmp_path/'daily';daily.write_bytes(b'daily model')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(research_only=True,weekly_artifact_sha256='wrong')))
    monkeypatch.setattr(shadow,'ARTIFACT',artifact);monkeypatch.setattr(shadow,'DAILY',daily);monkeypatch.setattr(shadow,'MANIFEST',manifest)
    result=shadow.capture_shadow({},db_path=tmp_path/'none.db',now=datetime(2026,10,29,19,tzinfo=ZoneInfo('Asia/Kolkata')))
    assert result=='blocked_artifact_or_policy_change'
    assert not (tmp_path/'none.db').exists()


def test_very_new_listing_is_explicitly_blocked_before_network_fetch(tmp_path,monkeypatch):
    weekly=tmp_path/'weekly';weekly.write_bytes(b'weekly')
    daily=tmp_path/'daily';daily.write_bytes(b'daily')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(research_only=True,
        weekly_artifact_sha256=shadow.digest(weekly),daily_artifact_sha256=shadow.digest(daily),
        feature_builder_sha256=shadow.digest(Path(shadow.__file__)),
        policy_code_sha256=shadow.digest(shadow.ROOT/'ml/research_module2_multihorizon.py'))))
    monkeypatch.setattr(shadow,'ARTIFACT',weekly);monkeypatch.setattr(shadow,'DAILY',daily);monkeypatch.setattr(shadow,'MANIFEST',manifest)
    def network(*args,**kw):raise AssertionError('Young listing should not trigger a research data fetch')
    monkeypatch.setattr(shadow.AttributionEngine,'_fetch_yahoo_series',network)
    status=shadow.capture_shadow(dict(ipo_id=1,evaluation_date='2026-10-29',listing_date='2026-10-28'),
        db_path=tmp_path/'shadow.db',now=datetime(2026,10,29,19,tzinfo=ZoneInfo('Asia/Kolkata')))
    assert status.startswith('ineligible: Not enough post-listing history')
