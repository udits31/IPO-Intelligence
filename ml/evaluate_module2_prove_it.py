"""Exact 132 checkpoint/outcome replay; archived EOD is lagged one day."""
import gzip,hashlib,json,re,zipfile
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from services.saved_preclose_service import ROOT
from services.enrichment_runtime import market_records
from services.point_in_time import eligible
from scrapers.exchange_enrichment_scraper import normalize_delivery
from core.prove_it_monitor_engine import features,decision,VERSION
from ml.calibrate_module2_rules import metrics,paired_interval


def raw_tape(ipos):
    symbols={str(r.nse_symbol):int(r.id) for r in ipos.reset_index().itertuples() if isinstance(r.nse_symbol,str)}
    tapes={}
    for path in (ROOT/'data/market/raw').glob('*.json'):
        record=json.loads(path.read_text(encoding='utf8'))
        if record.get('status')!='fetched' or 'generateSecurityWiseHistoricalData' not in record.get('source_url',''):continue
        symbol=record.get('query',{}).get('symbol');id=symbols.get(symbol)
        if id is None:continue
        context={**record,'ipo_id':id,'symbol':symbol}
        tapes.setdefault(id,[]).extend(normalize_delivery(record['payload'],context))
    for id,ipo in ipos.iterrows():
        date=pd.Timestamp(ipo.listing_date);path=ROOT/'data/audit/nse_bhavcopy_udiff'/(date.strftime('%Y%m%d')+'.csv.zip')
        if not path.exists():continue
        with zipfile.ZipFile(path) as archive:table=pd.read_csv(archive.open(archive.namelist()[0]),usecols=['TradDt','TckrSymb','SctySrs','LwPric','HghPric','ClsPric','OpnPric'])
        found=table[table.TckrSymb.eq(ipo.nse_symbol)&table.TradDt.eq(str(date.date()))]
        preferred=found[found.SctySrs.eq('EQ')];found=preferred if len(preferred)==1 else found
        if len(found)==1:
            row=found.iloc[0];tapes.setdefault(int(id),[]).append(dict(ipo_id=int(id),symbol=ipo.nse_symbol,session_date=str(date.date()),low=float(row.LwPric),high=float(row.HghPric),close=float(row.ClsPric),open_price=float(row.OpnPric),deliverable_qty=None,
                source_timestamp=None,source_timestamp_verified=False,source_url='https://nsearchives.nseindia.com/content/cm/'+path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),series=row.SctySrs))
    return tapes


def cached_allotment(ipo_id,ipo):
    path=ROOT/'data/archive/module1_later_holdout_20261024'/(str(ipo_id)+'.json.gz')
    if not path.exists():return None
    value=json.loads(gzip.decompress(path.read_bytes()))
    for source in value.get('sources',[]):
        content=source.get('raw_response_text','');match=re.search(r'Allotment Date</dt>\s*<dd[^>]*>([^<]+)',content,re.I)
        if match:
            date=pd.to_datetime(match[1],errors='coerce')
            lower=pd.Timestamp(ipo.issue_close_date) if pd.notna(ipo.issue_close_date) else pd.Timestamp(ipo.listing_date)-pd.Timedelta(days=10)
            if pd.notna(date) and lower<=date<=pd.Timestamp(ipo.listing_date):return str(date.date())
    return None


def anchor(row,cutoff,assumed):
    value=row.get('allotment_date')
    if pd.isna(value):
        unlock=row.get('anchor_30d_unlock_date')
        if pd.notna(unlock):value=str((pd.Timestamp(unlock)-pd.Timedelta(days=30)).date())
    if pd.isna(value):return None
    valid=eligible(dict(source_timestamp=row.get('allotment_date_source_timestamp'),source_timestamp_verified=row.get('allotment_date_pit_verified')==True),cutoff)
    return dict(allotment_date=value,calendar_usable=bool(valid or assumed),calendar_assumed=bool(assumed and not valid))


def run():
    ledger=pd.read_csv(ROOT/'output/module2_holdout_decision_ledger_governance_20261023.csv',parse_dates=['decision_date','outcome_date'])
    prices=pd.read_csv(ROOT/'data/audit/module2_holdout_prices_2025_2026_nse_full_ca_reconciled.csv',parse_dates=['date'])
    ipos=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    structures=pd.read_csv(ROOT/'data/deal_structure_enriched.csv',dtype={'ipo_id':str}).set_index('ipo_id')
    if len(ledger)!=132 or ledger.ipo_id.nunique()!=132:raise ValueError('Fixed 132 cases changed')
    ledger['issue_price']=ledger.ipo_id.map(ipos.issue_price)
    ledger['peak_close']=[float(prices[prices.ipo_id.eq(r.ipo_id)&prices.date.le(r.decision_date)].close_price_adjusted.max()) for r in ledger.itertuples()]
    tape=raw_tape(ipos.loc[ledger.ipo_id]);rows=[]
    for row in ledger.itertuples():
        id=int(row.ipo_id);group=prices[prices.ipo_id.eq(id)].sort_values('date').reset_index(drop=True)
        if group.iloc[30].date!=row.decision_date or not np.isclose(group.iloc[30].close_price_adjusted,row.decision_close):raise ValueError('31st-close decision protocol changed')
        direction=100*(group.iloc[60].close_price_adjusted/group.iloc[30].close_price_adjusted-1)
        execution=100*(group.iloc[61].close_price_adjusted/group.iloc[31].close_price_adjusted-1)
        if not np.isclose(direction,row.day30_return_pct) or not np.isclose(execution,row.executed_hold_return_pct,atol=.00001):raise ValueError('Frozen direction or next-close execution changed')
        cutoff=str(row.decision_date.date())+'T15:30:00+05:30'
        structure=structures.loc[str(id)].to_dict() if str(id) in structures.index else {}
        if pd.isna(structure.get('allotment_date')):
            recovered=cached_allotment(id,ipos.loc[id])
            if recovered:structure['allotment_date']=recovered
        news=market_records('corporate_announcements',id);delivery=tape.get(id,[])
        f=features(group,issue_price=float(ipos.loc[id].issue_price),decision_timestamp=cutoff,delivery=delivery,filings=news,anchor=anchor(structure,cutoff,True),archive_assumption=True)
        strict=features(group,issue_price=float(ipos.loc[id].issue_price),decision_timestamp=cutoff,delivery=delivery,filings=news,anchor=anchor(structure,cutoff,False),archive_assumption=False)
        d=decision(f);s=decision(strict)
        if d['sell'] is None or s['sell'] is None:raise ValueError('A fixed mature case was silently abstained')
        rows.append(dict(ipo_id=id,company_name=row.company_name,decision_date=str(row.decision_date.date()),actual_positive=direction>0,
            action=d['action'],sell=d['sell'],correct=d['sell']!=(direction>0),strict_action=s['action'],strict_sell=s['sell'],
            net_strategy_return_pct=-.30 if d['sell'] else execution-.30,features=f,gates=d['gates'],missing=d['missing_gates'],failed=d['failed_gates'],
            old_sell=bool(row.sell),old_correct=bool(row.sell)!=(direction>0),strict_features=strict))
    sell=np.array([r['sell'] for r in rows]);strict_sell=np.array([r['strict_sell'] for r in rows])
    preflight={field:sum(pd.notna(r['features'].get(field)) for r in rows) for field in ['day1_low','sma_20','rs_vs_nifty_20d','delivery_accumulation_ratio','filing_sentiment_score','unabsorbed_unlock_window','atr_14_pct']}
    print('MODULE 2 ELIGIBLE FEATURE COUNTS (132 cases; prior-session archive availability assumption)',flush=True);print(pd.Series(preflight).to_string(),flush=True)
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),policy=VERSION,cohort_n=132,
        original_monitor=metrics(ledger,ledger.sell.to_numpy(dtype=bool)),always_sell=metrics(ledger,np.ones(132,dtype=bool)),
        consistent_exit_cost_net_mean_pct=dict(new_prove_it=float(np.mean([r['net_strategy_return_pct'] for r in rows])),old_monitor=float(np.where(ledger.sell,-.30,ledger.executed_hold_return_pct-.30).mean()),always_sell=-.30),
        enriched_v1_correct=63,new_prove_it=metrics(ledger,sell),strict_clock_sensitivity=metrics(ledger,strict_sell),
        paired_change=paired_interval(ledger,sell),preflight=preflight,
        gate_failures={k:sum(r['gates'].get(k) is False for r in rows) for k in rows[0]['gates']},
        missing_gates={k:sum(r['gates'].get(k) is None for r in rows) for k in rows[0]['gates']},
        day1_0945_rule=dict(implemented=True,historically_tested=False,status='Input-guarded first 15 continuous minutes implemented. A 09:45 IPO first-15-minute rule is invalid during special pre-open; normally the continuous session starts at 10:00, so the earliest such completed window is 10:15. Verified intraday bars are unavailable for this replay.',source='https://www.nseindia.com/static/products-services/equity-market-special-pre-open-session'),
        protocol='Unchanged 31st observed-close features and 31->61 direction label; next observed close entry and 30-session execution outcome; 0.30% exit cost.',
        availability='Filings use exact broadcasts <= decision close. Delivery OHLC/volume uses trading date strictly before decision date with next-morning availability assumption. Static allotment calendar is an explicit archived-calendar assumption.',
        live_promoted=False,limitations=['Rules were proposed after inspecting the later cohort, so this is exploratory retrospective evidence.','An archived EOD date is not an as-published vintage guarantee; strict-clock sensitivity is reported separately.','Missing required survival evidence causes defensive review/reduce, not proof the stock will fall.','No verified intraday data means no simulated 09:45 fills.','Individual conditional strategy returns are not a capital-weighted investor portfolio.'])
    report['beats_always_sell_accuracy']=report['new_prove_it']['correct']>report['always_sell']['correct']
    report['beats_always_sell_net_return']=report['new_prove_it']['conditional_strategy_net_mean_pct']>report['always_sell']['conditional_strategy_net_mean_pct']
    path=ROOT/'output/module2_prove_it_benchmark_20261026.json';path.write_text(json.dumps(report,indent=2,default=str,allow_nan=False),encoding='utf8')
    pd.DataFrame([{**r,'features':json.dumps(r['features'],default=str),'strict_features':json.dumps(r['strict_features'],default=str)} for r in rows]).to_csv(path.with_suffix('.csv'),index=False)
    print(json.dumps({k:v for k,v in report.items() if k in ['new_prove_it','always_sell','strict_clock_sensitivity','beats_always_sell_accuracy','beats_always_sell_net_return']}),flush=True)
    return report


if __name__=='__main__':run()
