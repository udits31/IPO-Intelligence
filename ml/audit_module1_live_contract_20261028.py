"""Read-only training/live coverage and automatic logger audit."""
import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import requests
from ml.test_historical_joins_and_cutoff_20261028 import ROOT,SOURCE,sha
from services.prospective_module1 import PROTOCOL
from services.module1_input_contract import VERSION


def run():
    out=ROOT/'output/module1_period_stability_20261028'
    frame=pd.read_csv(SOURCE);train=frame[frame.cohort.eq('historical')&frame.year.le(2025)]
    fields=['log_qib','log_hni','log_retail','gmp_pct','log_qib_growth','log_hni_growth','log_retail_growth','pub_ipo_earnings_yield_pct','pub_log_ipo_price_book','macro_vix']
    protocol_sha=sha(PROTOCOL)
    path=ROOT/'data/telemetry/snapshots.db'
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=15) as c:
        c.row_factory=sqlite3.Row
        forecasts=[dict(r) for r in c.execute('''SELECT * FROM frozen_module1_forecast_v1 WHERE id IN
            (SELECT MAX(id) FROM frozen_module1_forecast_v1 WHERE protocol_sha=? GROUP BY ipo_id)
            ORDER BY recorded_at_ist DESC''',(protocol_sha,))]
    live=[]
    for row in forecasts:
        value=json.loads(row['input_json']);features=value.get('features',{});evidence=value.get('evidence',{})
        sources=evidence.get('sources',{})
        live.append(dict(ipo_id=row['ipo_id'],company=row['company_name'],recorded_at_ist=row['recorded_at_ist'],cutoff_at_ist=row['cutoff_at_ist'],
            saved_before_cutoff=pd.Timestamp(row['recorded_at_ist'])<pd.Timestamp(row['cutoff_at_ist']),
            input_contract=sources.get('input_contract_version','earlier_contract'),stage=sources.get('stage_alignment',{}),
            available={f:features.get(f) is not None and pd.notna(features.get(f)) for f in fields}))
    try:
        response=requests.get('http://127.0.0.1:8000/api/v1/health',timeout=30);response.raise_for_status();health=response.json()
        runtime={k:health.get(k) for k in ['status','module1_model','collectors']}
    except Exception as exc:runtime={'error':str(exc)[:200]}
    try:
        response=requests.get('http://127.0.0.1:8000/api/v1/telemetry/data-quality',timeout=30);response.raise_for_status();quality=response.json()
        # Persist only collector/sync health; no credentials or raw response dumps.
        runtime['data_quality_status']=quality.get('status')
        runtime['sync']=quality.get('database_sync') or quality.get('supabase_sync') or quality.get('sync')
    except Exception as exc:runtime['quality_check_error']=str(exc)[:200]
    report=dict(audited_at_ist=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),input_contract_version=VERSION,
        training_n=len(train),live_latest_issuer_n=len(live),coverage={f:dict(training_observed=int(train[f].notna().sum()),training_n=len(train),live_observed=sum(r['available'][f] for r in live),live_n=len(live)) for f in fields},
        forecasts=live,runtime=runtime,
        fixed_bugs=['Negative/nonfinite subscription categories rejected','Books ordered by publisher time where available','Growth compares book publication days; cannot use itself as prior-day reference','Financial cache requires raw SHA, as-of capture time, and a prior annual period'],
        remaining_limitation='The historical model was studied at penultimate-session input stage. Generic live intraday/closing-day snapshots are not stage-equivalent; new outputs carry explicit stage evidence and require prospective scoring. No empirical stage conversion is fabricated.')
    (out/'live_input_audit.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['forecasts']},indent=2,default=str))


if __name__=='__main__':run()
