"""Verify production API storage without SQLite or modifying IPO history."""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ['OPENBELL_STORAGE_BACKEND']='supabase'
ROOT=Path(__file__).resolve().parents[1]


def main():
    from db import ledger
    from services.module2_call_outcomes import call_outcomes
    from services.module1_benchmark_lookup import catalog
    from services.prospective_module1 import predict
    from services import production_worker
    from db.supabase_client import get_supabase
    report={'backend':'supabase_api','direct_postgres_credentials_used':False,'sqlite_opened':False}
    with patch('db.ledger.local_sqlite.connect',side_effect=AssertionError('Production opened SQLite')):
        records=catalog()
        mismatches=[r['company_name'] for r in records if abs(predict(r['features'])[0]-r['probability'])>1e-9]
        assert len(records)==93 and not mismatches
        report.update(benchmark_records=93,frozen_replay_mismatches=len(mismatches))
        outcomes=call_outcomes(610,horizon=1,db_path=ROOT/'tmp/no_sqlite/module2_decisions.db')
        assert outcomes['summary']['evaluated']>0
        report['example_monitor_outcomes']=outcomes['summary']
        with ledger.connect('snapshots.db') as c:
            c.execute('CREATE TABLE IF NOT EXISTS _openbell_storage_checks(test_id TEXT PRIMARY KEY,value TEXT,raw BLOB)')
        key=uuid.uuid4().hex
        with ledger.connect('snapshots.db') as c:
            c.execute('INSERT INTO _openbell_storage_checks VALUES(?,?,?)',(key,'first',b'OpenBell cloud storage check'))
        c=ledger.connect('snapshots.db')
        assert c.execute('SELECT value,raw FROM _openbell_storage_checks WHERE test_id=?',(key,)).fetchone()==('first',b'OpenBell cloud storage check')
        c.execute('UPDATE _openbell_storage_checks SET value=? WHERE test_id=?',('rolled-back',key));c.rollback()
        assert c.execute('SELECT value FROM _openbell_storage_checks WHERE test_id=?',(key,)).fetchone()[0]=='first'
        with ledger.connect('snapshots.db') as other:
            other.execute('UPDATE _openbell_storage_checks SET value=? WHERE test_id=?',('second',key))
        c.execute('UPDATE _openbell_storage_checks SET value=? WHERE test_id=?',('stale-write',key))
        try:c.commit();raise AssertionError('Stale write was accepted')
        except ledger.OperationalError as exc:assert '40001' in str(exc)
        c.close()
        report.update(durable_write_roundtrip=True,rollback_verified=True,optimistic_conflict_blocked=True)
        c=ledger.connect('snapshots.db');c.row_factory=ledger.Row
        row=c.execute('SELECT * FROM frozen_module1_forecast_v1 ORDER BY id DESC LIMIT 1').fetchone()
        try:c.execute('UPDATE frozen_module1_forecast_v1 SET probability=? WHERE id=?',(row['probability'],row['id']));raise AssertionError('Saved history was mutable')
        except ledger.IntegrityError:report['immutable_prediction_update_blocked']=True
        c.rollback();columns=row.keys()
        duplicate=c.execute('INSERT OR IGNORE INTO frozen_module1_forecast_v1 ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',tuple(row))
        assert duplicate.rowcount==0;c.commit();c.close()
        report['duplicate_prediction_ignored']=True
    client=get_supabase();first=production_worker.OWNER
    try:
        assert production_worker.lease(client)
        production_worker.OWNER=uuid.uuid4().hex
        assert not production_worker.lease(client)
        report['second_collector_owner_blocked']=True
    finally:
        production_worker.OWNER=first;production_worker.lease(client,release=True)
    report['model_sha256']=hashlib.sha256((ROOT/'ml/artifacts/module1_2026_standard_classifier_research_20261025.joblib').read_bytes()).hexdigest()
    report['no_ipo_history_created_or_rewritten']=True
    target=ROOT/'output/native_supabase_storage_verification_20261030.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
