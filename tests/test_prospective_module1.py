from datetime import datetime
import json
import sqlite3
from zoneinfo import ZoneInfo

import pytest

import services.prospective_module1 as service

IST=ZoneInfo('Asia/Kolkata')


@pytest.fixture
def setup(tmp_path,monkeypatch):
    db=tmp_path/'snapshots.db'
    protocol={'_sha256':service.digest(service.PROTOCOL),'model':{'sha256':'fixed'},'gmp_baseline':{'cutoff':.625}}
    monkeypatch.setattr(service,'bundles',lambda:(protocol,{},{}))
    monkeypatch.setattr(service,'predict',lambda features:(.7,.6))
    return db,dict(company_name='Test Future IPO',category='OPEN',date_range='25 - 27 Jan',
        listing_date='2027-02-01',issue_price=100,source_observed=True),'https://www.chittorgarh.com/ipo/test-future/99999/'


def test_append_only_dedup_and_changed_input_same_slot(setup):
    db,item,url=setup;clock=datetime(2027,1,26,16,55,tzinfo=IST)
    for _ in range(2):
        service.record(item,url,db_path=db,clock=clock,features={'gmp_pct':0},evidence={'quality':'provisional_missing_inputs'})
    service.record(item,url,db_path=db,clock=clock,features={'gmp_pct':1},evidence={'quality':'provisional_missing_inputs'})
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT COUNT(*) FROM frozen_module1_forecast_v1').fetchone()[0]==2
        with pytest.raises(sqlite3.IntegrityError,match='immutable'):
            c.execute('UPDATE frozen_module1_forecast_v1 SET probability=.99')
        with pytest.raises(sqlite3.IntegrityError,match='immutable'):
            c.execute('DELETE FROM frozen_module1_forecast_v1')


def test_cutoff_rejects_after_close_and_never_zero_fills(setup):
    db,item,url=setup
    assert service.record(item,url,db_path=db,clock=datetime(2027,1,27,17,0,tzinfo=IST),features={}) is None
    service.record(item,url,db_path=db,clock=datetime(2027,1,26,16,55,tzinfo=IST),features={'gmp_pct':0})
    with sqlite3.connect(db) as c:
        inputs=json.loads(c.execute('SELECT input_json FROM frozen_module1_forecast_v1').fetchone()[0])
    assert inputs['features']=={'gmp_pct':0}
    assert inputs['gmp_missing'] is False
    assert 'log_qib' not in inputs['features']
    assert service.record({**item,'source_observed':False},url,db_path=db,
        clock=datetime(2027,1,26,16,55,tzinfo=IST),features={}) is None


def test_prediction_completion_crossing_cutoff_is_rejected(setup,monkeypatch):
    db,item,url=setup
    clocks=iter([datetime(2027,1,27,16,59,59,tzinfo=IST),datetime(2027,1,27,17,0,1,tzinfo=IST)])
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):
            return next(clocks)
    monkeypatch.setattr(service,'datetime',Clock)
    monkeypatch.setattr(service,'build_features',lambda *args:({},{}))
    assert service.record(item,url,db_path=db) is None
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT COUNT(*) FROM frozen_module1_forecast_v1').fetchone()[0]==0


def test_pre_bidding_estimates_are_distinct_and_flat_counts_negative(setup):
    db,item,url=setup
    service.record({**item,'category':'UPCOMING'},url,db_path=db,clock=datetime(2027,1,24,12,tzinfo=IST),features={})
    first=service.scorecard(db_path=db)['records'][0]
    assert first['kind']=='pre_bidding_estimate'
    outcome={'actual_open':100,'issue_price':100,'positive':False,'return_pct':0}
    with sqlite3.connect(db) as c:
        c.execute('INSERT INTO frozen_module1_outcome_v1 VALUES (?,?,?)',(first['ipo_id'],'2027-02-01T18:00:00+05:30',json.dumps(outcome)))
    result=service.scorecard(db_path=db)
    assert result['records'][0]['direction_correct'] is False
    assert result['records'][0]['gmp_correct'] is True
    assert result['summary']['direction_accuracy_pct']==0


def test_protocol_mutation_stops_logging(setup,monkeypatch):
    db,item,url=setup
    monkeypatch.setattr(service,'digest',lambda path:'changed')
    with pytest.raises(ValueError,match='protocol changed'):
        service.record(item,url,db_path=db,clock=datetime(2027,1,26,12,tzinfo=IST),features={})


def test_listed_without_saved_call_never_calculates_a_new_probability(setup,monkeypatch):
    db,item,url=setup
    def forbidden(*args):
        raise AssertionError('A listed IPO must not be rescored')
    monkeypatch.setattr(service,'build_features',forbidden)
    monkeypatch.setattr(service,'predict',forbidden)
    from services import listing_outcome_service
    monkeypatch.setattr(listing_outcome_service,'verified_listing_outcomes',lambda *args:{'chitto_99999':{
        'issue_price':100,'nse_open_price':100}})
    closed={**item,'category':'LISTED','date_range':'1 - 3 Oct','listing_date':'2026-10-08'}
    result=service.analysis_card(closed,url,db_path=db)
    assert result['mode']=='no_saved_pre_close'
    assert result['probability'] is None
    assert result['outcome']['actual_open']==100
    assert service.scorecard(db_path=db)['summary']['ipos_forecast']==0


def test_latest_genuine_preclose_call_wins_over_later_saved_scores(setup):
    db,item,url=setup
    service.init_db(db)
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE automatic_decision_v1 (id INTEGER PRIMARY KEY,ipo_id TEXT,company_name TEXT,decision_kind TEXT,positive_listing_probability REAL,recorded_at_ist TEXT,source_asof_ist TEXT,input_json TEXT,model_version TEXT)')
        for index,issued,p in [(1,'2026-10-03T15:00:00+05:30',.55),(2,'2026-10-03T16:50:00+05:30',.61),(3,'2026-10-04T10:00:00+05:30',.98)]:
            c.execute('INSERT INTO automatic_decision_v1 VALUES (?,?,?,?,?,?,?,?,?)',(index,'chitto_99999',item['company_name'],'live_bid_forecast',p,issued,'2026-10-03T14:59:00+05:30',json.dumps({'application_cutoff_at_ist':'2026-10-03T17:00:00+05:30'}),'original-model'))
    result=service.analysis_card({**item,'category':'LISTED','date_range':'2026-10-01 to 2026-10-03'},url,db_path=db)
    assert result['mode']=='legacy_saved_pre_close'
    assert result['probability']==.61
    assert result['recorded_at_ist']=='2026-10-03T16:50:00+05:30'
