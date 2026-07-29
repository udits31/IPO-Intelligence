"""Live regime bookkeeping; frozen predictions and model weights stay immutable."""
import json,threading
from db import ledger as sqlite3
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from services.saved_preclose_service import ROOT
from services.point_in_time import IST
from services.ipo_regime_service import rolling_regime
from services.historical_bid_service import name_key

STATE=ROOT/'data/telemetry/progressive_regime_state.json'


def refresh_regime_state():
    path=ROOT/'output/module1_progressive_ablation_20261026_features.csv'
    if not path.exists():return dict(status='research_inputs_not_ready')
    data=pd.read_csv(path,dtype={'issuer_key':str});history=[];seen=set()
    for row in data.sort_values('listing_date').to_dict('records'):
        key=name_key(row['company_name'])
        if key in seen or pd.isna(row.get('target_return')):continue
        seen.add(key);history.append(dict(issuer_key=key,listing_date=row['listing_date'],actual_return_pct=row['target_return'],gmp_pct=row.get('gmp_pct'),gmp_preclose_verified=pd.notna(row.get('gmp_pct'))))
    from services.listing_outcome_service import verified_listing_outcomes
    outcomes=verified_listing_outcomes();forecasts={}
    with sqlite3.connect(ROOT/'data/telemetry/snapshots.db',timeout=10) as conn:
        conn.row_factory=sqlite3.Row
        for r in conn.execute('SELECT * FROM frozen_module1_forecast_v1 ORDER BY recorded_at_ist'):
            if pd.Timestamp(r['recorded_at_ist'])>=pd.Timestamp(r['cutoff_at_ist']):continue
            value=json.loads(r['input_json']);quote=value.get('features',{}).get('gmp_pct')
            if quote is not None:forecasts[r['ipo_id']]=quote
    for id,o in outcomes.items():
        key=name_key(o['company_name']);price=o.get('exchange_open_price') or o.get('nse_open_price')
        if key in seen or price is None:continue
        seen.add(key);history.append(dict(issuer_key=key,listing_date=o['listing_date'],actual_return_pct=100*(price/o['issue_price']-1),gmp_pct=forecasts.get(id,np.nan),gmp_preclose_verified=id in forecasts))
    now=datetime.now(IST);regime=rolling_regime(pd.DataFrame(history),now.date().isoformat())
    # Roll to the next calendar quarter; new folds need a new future cohort.
    report=ROOT/'output/module1_progressive_ablation_20261026.json'
    result=json.loads(report.read_text()) if report.exists() else {}
    fitted=pd.Timestamp(result['created_at_utc']).tz_convert(IST).to_pydatetime() if result.get('created_at_utc') else now
    month=((fitted.month-1)//3+1)*3+1;year=fitted.year+(month>12);month=1 if month>12 else month
    state=dict(updated_at_ist=now.isoformat(),regime=regime,history_issuers=len(history),accepted_components=result.get('accepted_components',[]),
        model_weights_changed=False,prediction_ledgers_modified=False,
        next_quarterly_review_ist=datetime(year,month,1,tzinfo=IST).isoformat(),
        quarterly_review_due=now>=datetime(year,month,1,tzinfo=IST),historical_seed_basis='Retrospectively recovered historical inputs; new outcomes join only from the verified outcome ledger and saved pre-close forecast inputs.',
        slow_refit_status='No component passed all promotion gates. Quarterly expanding fits require a newly reserved future evaluation cohort; the old 2026 test cannot become training and still be called holdout.')
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2,default=str,allow_nan=False),encoding='utf8');temp.replace(STATE)
    return state


class RegimeWorker(threading.Thread):
    def __init__(self):super().__init__(name='progressive-regime-tracker',daemon=True);self.halt=threading.Event()
    def run(self):
        while not self.halt.is_set():
            try:refresh_regime_state()
            except Exception:
                import logging
                logging.getLogger(__name__).exception('Regime tracker refresh failed; prior state retained')
            self.halt.wait(900)
    def stop(self):self.halt.set()


worker=RegimeWorker()
