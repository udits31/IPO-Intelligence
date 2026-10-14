import gzip
import hashlib
import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from services.module1_input_contract import financial_cache_eligible,growth_reference,valid_categories

IST=ZoneInfo('Asia/Kolkata')


def test_invalid_categories_never_enter_model():
    assert valid_categories(dict(qib_x=0,nii_x=1,retail_x=2))
    for value in [-1,float('nan'),float('inf'),None]:
        assert not valid_categories(dict(qib_x=value,nii_x=1,retail_x=2))


def test_growth_uses_book_date_not_analysis_date_and_never_itself():
    old=dict(id=1,source_family='nse');new=dict(id=2,source_family='nse')
    a=pd.Timestamp('2026-06-12T16:00:00+05:30');b=pd.Timestamp('2026-06-15T16:00:00+05:30')
    assert growth_reference([(a,old),(b,new)],(b,new))==(a,old)
    assert growth_reference([(b,new)],(b,new)) is None
    assert growth_reference([(a,dict(old,source_family='other')),(b,new)],(b,new)) is None


def test_financial_facts_need_source_hash_capture_time_and_prior_period(tmp_path):
    raw=b'Verified issuer annual table';path=tmp_path/'source.gz';path.write_bytes(gzip.compress(raw))
    fact=dict(captured_at_ist='2026-06-10T12:00:00+05:30',annual_period='2026-03-31',raw_sha256=hashlib.sha256(raw).hexdigest())
    asof=datetime(2026,6,15,12,tzinfo=IST)
    assert financial_cache_eligible(fact,path,asof)
    for changed in [dict(captured_at_ist='2026-06-16T12:00:00+05:30'),dict(annual_period='2026-09-30'),dict(raw_sha256='wrong')]:
        assert not financial_cache_eligible(dict(fact,**changed),path,asof)


def test_live_builder_orders_by_publisher_time_and_keeps_real_growth(tmp_path,monkeypatch):
    import services.prospective_module1 as s
    import services.locked_model_configuration as locked
    db=tmp_path/'test.db';s.init_db(db)
    monkeypatch.setattr(s,'market_frame',lambda:pd.DataFrame())
    monkeypatch.setattr(s,'bundles',lambda:({},dict(expert_configs={'test':dict(feature_set='base')}),{}))
    monkeypatch.setattr(locked,'apply_live_recovered',lambda *args:None)
    monkeypatch.setattr(locked,'cache_live_offer',lambda *args:None)
    url='https://www.chittorgarh.com/ipo/example/99999/'
    item=dict(company_name='Example Ltd',date_range='2026-06-12 to 2026-06-16',issue_price=100,issue_size_cr=100)
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE subscription_capture_v2 (id INTEGER PRIMARY KEY, ipo_id TEXT, company_name TEXT, capture_phase TEXT,status TEXT,fetched_at_ist TEXT,attempted_at_ist TEXT,raw_html_gzip BLOB,raw_sha256 TEXT,source_family TEXT,qib_x REAL,nii_x REAL,retail_x REAL)')
        # A later fetch contains an older published book and must not replace the fresher publisher reading.
        for id,pub,fetched,qib in [(1,'13-Jun-2026 16:00:00','2026-06-15T10:00:00+05:30',2.),(2,'14-Jun-2026 16:00:00','2026-06-14T16:05:00+05:30',4.)]:
            raw=json.dumps({'updateTime':'Updated as on '+pub}).encode()
            c.execute('INSERT INTO subscription_capture_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',(id,'chitto_99999','Example Ltd','pre_deadline','verified',fetched,fetched,gzip.compress(raw),hashlib.sha256(raw).hexdigest(),'nse_exchange',qib,3.,2.))
    features,evidence=s.build_features(item,url,datetime(2026,6,15,12,tzinfo=IST),db_path=db)
    assert np.isclose(features['log_qib'],np.log1p(4.))
    assert np.isclose(features['log_qib_growth'],np.log1p(4.)-np.log1p(2.))
    assert evidence['sources']['bids']['id']==2
    assert evidence['sources']['growth_reference']['id']==1
    assert not evidence['sources']['stage_alignment']['comparable']
