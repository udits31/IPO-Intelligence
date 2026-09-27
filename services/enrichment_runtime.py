"""Read-only access to enrichment research and dated monitor previews."""
import json
from datetime import datetime
import pandas as pd
from services.saved_preclose_service import ROOT
from services.point_in_time import IST,eligible
from services.enriched_monitor import assemble,evaluate


def market_records(kind,ipo_id):
    path=ROOT/'data/market'/kind/(str(int(ipo_id))+'.jsonl')
    records=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()] if path.exists() else []
    if kind=='corporate_announcements':records=[r for r in records if r.get('schema_version')==2]
    return records


def anchor_metadata(ipo_id,decision):
    path=ROOT/'data/deal_structure_enriched.csv'
    if not path.exists():return None
    frame=pd.read_csv(path,dtype={'ipo_id':str});rows=frame[frame.ipo_id.eq(str(int(ipo_id)))]
    if rows.empty:return None
    row=rows.iloc[0];result={}
    for field in ('anchor_30d_unlock_date','anchor_90d_unlock_date'):
        if pd.notna(row.get(field)) and eligible(dict(source_timestamp=row.get(field+'_source_timestamp'),source_timestamp_verified=row.get(field+'_pit_verified')==True),decision):result[field]=row[field]
    result['calendar_verified']=len(result)==2
    return result


def preview(monitor,ipo_id,decision_timestamp=None):
    decision=decision_timestamp or datetime.now(IST).isoformat()
    rows=monitor.df_ipos[monitor.df_ipos.id.eq(ipo_id)]
    if rows.empty:raise ValueError('Unknown listed company ID')
    item=rows.iloc[0]
    prices=monitor.attribution_engine.get_stock_series(ipo_id,as_of_date=decision[:10])
    prices=prices[pd.to_datetime(prices.date).dt.date<=pd.Timestamp(decision).date()].copy()
    if prices.empty:return dict(company_name=item.company_name,decision=dict(action='DATA_PENDING',reasons=['No completed dated price session.']),promoted_to_main_monitor=False)
    from services.corporate_action_service import reconcile_prices,PriceBasisError
    try:
        prices,issue,basis=reconcile_prices(prices,item,asof=prices.iloc[-1].date,provider_meta=prices.attrs.get('provider_meta'),allow_network=False)
        if basis['status']=='unspecified_legacy_basis':raise PriceBasisError('Historical share-adjustment basis is unverified.')
    except PriceBasisError as exc:
        return dict(company_name=item.company_name,decision=dict(action='DATA_PENDING',reasons=[str(exc)]),promoted_to_main_monitor=False)
    features=assemble(prices,decision_timestamp=decision,issue_price=issue,
        delivery=market_records('delivery_series',ipo_id),filings=market_records('corporate_announcements',ipo_id),
        deals=market_records('bulk_block_deals',ipo_id),anchor=anchor_metadata(ipo_id,decision),strict=True)
    return dict(company_name=item.company_name,decision_timestamp=decision,decision=evaluate(features),features=features,price_basis=basis,
        promoted_to_main_monitor=False,explanation='Research preview. The benchmark has not established an advantage over the existing monitor or always-sell baseline.')


def audit_report():
    path=ROOT/'output/full_enrichment_audit_report_20261025.json'
    if not path.exists():return dict(status='BUILDING',message='Enrichment audit is still running.')
    return json.loads(path.read_text(encoding='utf8'))
