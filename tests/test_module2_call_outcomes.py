import hashlib
import json
import sqlite3
import math
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from services import module2_call_outcomes as outcomes

IST=ZoneInfo('Asia/Kolkata')


def row(id,day,issued,price,call):
    return dict(id=id,price_date=day,observed_at_utc=issued,payload_sha256='verified',
        payload=dict(decision_close=price,final_verdict=call))


def prices():
    return pd.DataFrame(dict(date=pd.to_datetime(['2026-10-27','2026-10-28','2026-10-29']),close=[100.,90.,95.]))


def test_sell_then_fall_matches_but_sell_before_rise_misses():
    rows=[row(1,'2026-10-27','2026-10-27T13:00:00+00:00',100,'SELL / REDUCE'),
          row(2,'2026-10-28','2026-10-28T13:00:00+00:00',90,'SELL / REDUCE')]
    r=outcomes.evaluate_calls(rows,prices(),{},horizon=1,now=datetime(2026,10,29,18,tzinfo=IST))
    assert r['summary']['matched']==1 and r['summary']['missed']==1
    byid={c['id']:c for c in r['calls']}
    assert math.isclose(byid[1]['actual_return_pct'],-10) and byid[1]['matched'] is True
    assert byid[2]['actual_return_pct']>0 and byid[2]['matched'] is False
    assert byid[1]['called_at_ist'].startswith('2026-10-27T18:30')


def test_horizon_counts_observed_sessions_and_pending_is_not_a_miss():
    rows=[row(1,'2026-10-27','2026-10-27T13:00:00+00:00',100,'HOLD (WATCH)')]
    r=outcomes.evaluate_calls(rows,prices(),{},horizon=5,now=datetime(2026,10,29,18,tzinfo=IST))
    assert r['summary']['evaluated']==0 and r['summary']['pending']==1
    assert r['calls'][0]['matched'] is None and r['calls'][0]['observed_future_sessions']==2


def test_stale_input_cannot_turn_a_rising_stock_into_a_successful_sell():
    rows=[row(1,'2026-10-27','2026-10-29T04:30:00+00:00',100,'SELL / REDUCE')]
    r=outcomes.evaluate_calls(rows,prices(),{},horizon=1,now=datetime(2026,10,29,18,tzinfo=IST))
    call=r['calls'][0]
    assert call['decision_reference_price']==100 and call['decision_reference_date']=='2026-10-27'
    assert call['reference_price']==90 and call['reference_date']=='2026-10-28'
    assert call['stale_decision_input'] is True
    assert call['actual_return_pct']>0 and call['matched'] is False


def test_unfinished_session_is_never_an_actual_outcome():
    rows=[row(1,'2026-10-28','2026-10-29T04:30:00+00:00',90,'HOLD / CONSOLIDATION')]
    r=outcomes.evaluate_calls(rows,prices(),{},horizon=1,now=datetime(2026,10,29,14,tzinfo=IST))
    assert r['calls'][0]['status']=='pending'
    r=outcomes.evaluate_calls(rows,prices(),{},horizon=1,now=datetime(2026,10,29,18,tzinfo=IST))
    assert r['calls'][0]['matched'] is True and r['calls'][0]['outcome_date']=='2026-10-29'


def test_a_call_cannot_use_a_reference_close_that_had_not_happened():
    rows=[row(1,'2026-10-28','2026-10-28T04:30:00+00:00',90,'HOLD (WATCH)')]
    r=outcomes.evaluate_calls(rows,prices(),{},horizon=1,now=datetime(2026,10,29,18,tzinfo=IST))
    assert r['calls'][0]['status']=='reference_not_available_at_call'
    assert r['summary']['evaluated']==0


def test_share_basis_conflict_and_split_are_not_counted_as_misses():
    rows=[row(1,'2026-10-27','2026-10-27T13:00:00+00:00',100,'SELL / REDUCE')]
    source=dict(splits=[{'ex_date':'2026-10-28'}])
    r=outcomes.evaluate_calls(rows,prices(),source,horizon=1,now=datetime(2026,10,29,18,tzinfo=IST))
    assert r['calls'][0]['status']=='share_basis_unverified' and r['calls'][0]['matched'] is None
    rows[0]['payload']['decision_close']=200
    r=outcomes.evaluate_calls(rows,prices(),{},horizon=1,now=datetime(2026,10,29,18,tzinfo=IST))
    assert r['calls'][0]['status']=='share_basis_unverified'


def test_reader_excludes_changed_saved_payload_and_never_updates_ledger(tmp_path,monkeypatch):
    db=tmp_path/'calls.db'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE module2_decisions_v1(id INTEGER,ipo_id INTEGER,price_date TEXT,observed_at_utc TEXT,payload_json TEXT,payload_sha256 TEXT)')
        encoded=json.dumps(dict(decision_close=100,final_verdict='SELL / REDUCE'))
        c.execute('INSERT INTO module2_decisions_v1 VALUES (1,42,?,?,?,?)',('2026-10-27','2026-10-27T13:00:00+00:00',encoded,'wrong'))
    original=db.read_bytes()
    monkeypatch.setattr(outcomes,'archived_prices',lambda rows:(prices(),{}))
    r=outcomes.call_outcomes(42,db_path=db)
    assert r['excluded_unverified_payloads']==1 and r['summary']['saved']==0
    assert db.read_bytes()==original
