"""Frozen-model historical replay; never writes to the issued forecast ledger."""
import json
import math
from datetime import datetime
from pathlib import Path
import pandas as pd
from services.saved_preclose_service import all_listed, clock, DB, ROOT

REPORT = ROOT/'output/listed_closing_day_backtest.json'


def eligible_quotes(quotes,asof):
    """Timed quotes may be same-day; day-only rows must be from an earlier day."""
    return [q for q in quotes if ((q.get('gmp_asof_ist') and clock(q['gmp_asof_ist']) and clock(q['gmp_asof_ist'])<=asof) or (not q.get('gmp_asof_ist') and q['gmp_date']<asof.date().isoformat())) and 0<=(asof.date()-pd.Timestamp(q['gmp_date']).date()).days<=7]


def select_book(row):
    cutoff=clock(row.get('cutoff_at_ist'))
    books=[b for b in row.get('subscription_history',[]) if cutoff and clock(b.get('observed_at_ist'))
           and clock(b['observed_at_ist']).date()==cutoff.date() and clock(b['observed_at_ist'])<cutoff
           and all(b.get(k) is not None and math.isfinite(float(b[k])) and float(b[k])>=0 for k in ('qib','hni','retail'))]
    return max(books,key=lambda b:clock(b['observed_at_ist'])) if books else None


def run(cache=None,db_path=DB):
    from scrapers.live_ipo_scraper import load_scraped_cache
    from services.historical_bid_service import resolve_official_issue
    from services.prospective_module1 import bundles,predict,IST
    from services.prediction_results_service import identity_name
    from ml.research_module1_forward_update import SETS,MACRO
    cache=load_scraped_cache() if cache is None else cache
    protocol,model,_=bundles()
    required=set(k for cfg in model['expert_configs'].values() for k in (['gmp_pct'] if cfg['feature_set']=='gmp_only' else SETS[cfg['feature_set']]))
    gmp=pd.read_csv(ROOT/'data/preclose_gmp_history.csv')
    recovered_path=ROOT/'output/listed_recovered_model_inputs.json'
    recovered={r['company_name']:r for r in json.loads(recovered_path.read_text(encoding='utf8'))['records']} if recovered_path.exists() else {}
    market_path=ROOT/'data/telemetry/prospective_macro.csv'
    market=pd.read_csv(market_path,parse_dates=['date']).set_index('date') if market_path.exists() else pd.DataFrame()
    items={item['company_name']:(url,resolve_official_issue(item)) for url,item in cache.items() if item.get('category')=='LISTED'}
    records=[]
    for row in all_listed(cache,db_path)['records']:
        book=select_book(row)
        result=dict(company_name=row['company_name'],cutoff_at_ist=row['cutoff_at_ist'],outcome=row['outcome'],
                    probability=None,correct=None,status='missing_closing_day_pre17_reading',mode='historical_backtest')
        records.append(result)
        if not book: continue
        url,item=items[row['company_name']];asof=clock(book['observed_at_ist'])
        features={ 'log_'+k:math.log1p(max(.01,float(book[v]))) for k,v in [('qib','qib'),('hni','hni'),('retail','retail')] }
        sources={'bids':book}
        recovered_row=recovered.get(row['company_name'],{})
        for field,key in [('log_offer_cap_price','issue_price'),('log_issue_size','issue_size_cr')]:
            if item.get(key) and float(item[key])>0: features[field]=math.log1p(float(item[key]))
        # Daily historical GMP has no intraday publication time: closing-day quotes are excluded.
        quotes=gmp[gmp.company_name.map(identity_name).eq(identity_name(row['company_name'])) & (gmp.gmp_date<asof.date().isoformat())].sort_values('gmp_date')
        if not quotes.empty and item.get('issue_price'):
            quote=quotes.iloc[-1];age=(asof.date()-pd.Timestamp(quote.gmp_date).date()).days
            if age<=7:
                features.update(gmp_pct=float(quote.gmp_rs)/float(item['issue_price'])*100,gmp_quote_age_days=age)
                sources['gmp']=quote.to_dict()
        recovered_quotes=eligible_quotes(recovered_row.get('quotes',[]),asof)
        if recovered_quotes:
            latest=max(q['gmp_date'] for q in recovered_quotes)
            same=[q for q in recovered_quotes if q['gmp_date']==latest]
            # Deterministic source preference set before looking at outcomes.
            quote=sorted(same,key=lambda q:(not bool(q.get('gmp_asof_ist')),'ipoji.com' not in q['url'],q['url'], -pd.Timestamp(q['gmp_asof_ist']).timestamp() if q.get('gmp_asof_ist') else 0))[0]
            features.update(gmp_pct=100*float(quote['gmp_rs'])/float(item['issue_price']),gmp_quote_age_days=(asof.date()-pd.Timestamp(latest).date()).days)
            sources['gmp']=dict(quote,conflicting_same_day_values=sorted(set(q['gmp_rs'] for q in same)))
            previous=[q for q in recovered_quotes if q['url']==quote['url'] and q['gmp_date']<latest]
            if previous:
                previous=max(previous,key=lambda q:(q['gmp_date'],q.get('gmp_asof_ist','')))
                days=(pd.Timestamp(latest)-pd.Timestamp(previous['gmp_date'])).days
                features['gmp_slope_pp_per_day']=100*(quote['gmp_rs']-previous['gmp_rs'])/float(item['issue_price'])/days
        financial=recovered_row.get('financial',{})
        if financial.get('annual_period') and pd.Timestamp(financial['annual_period']).date()<asof.date():
            for field in ('profit_margin_pct','return_on_assets_pct','revenue_growth_asinh','annual_loss','ipo_earnings_yield_pct','log_ipo_price_book'):
                value=financial.get(field)
                if value is not None and math.isfinite(float(value)): features['pub_'+field]=float(value)
            sources['financial']=financial
        # Growth only within one source, using timestamped earlier observations.
        earlier=[b for b in row['subscription_history'] if b.get('source')==book.get('source') and clock(b.get('observed_at_ist')) and clock(b['observed_at_ist'])<asof]
        if earlier:
            first=min(earlier,key=lambda b:clock(b['observed_at_ist']))
            for cat in ('qib','hni','retail'):
                if first.get(cat) is not None: features['log_'+cat+'_growth']=math.log1p(float(book[cat]))-math.log1p(float(first[cat]))
            sources['growth_start']=first
        histories=[b for b in recovered_row.get('bid_history',[]) if pd.Timestamp(b['date']).date()<asof.date()]
        if histories:
            provider=sorted(set(b['url'] for b in histories))[0]
            same=sorted([b for b in histories if b['url']==provider],key=lambda b:b['date'])
            if len(same)>=2:
                for cat in ('qib','hni','retail'):
                    features['log_'+cat+'_growth']=math.log1p(same[-1][cat+'_subscription'])-math.log1p(same[0][cat+'_subscription'])
                sources['growth_history']=same
        if not market.empty:
            prior=market[market.index<pd.Timestamp(asof.date())]
            if len(prior)>=21 and (asof.date()-prior.index[-1].date()).days<=5:
                n=prior.nifty.dropna();v=prior.vix.dropna()
                if len(n)>=21:
                    features.update(macro_nifty_5d=float((n.iloc[-1]/n.iloc[-6]-1)*100),macro_nifty_20d=float((n.iloc[-1]/n.iloc[-21]-1)*100),macro_nifty_vol20=float((n.pct_change()*100).iloc[-20:].std()))
                if len(v)>=6: features.update(macro_vix=float(v.iloc[-1]),macro_vix_change5=float((v.iloc[-1]/v.iloc[-6]-1)*100))
                sources['market_date']=str(prior.index[-1].date())
        # Never copy today's financial panel or GMP into a historical feature vector.
        probability,baseline=predict(features)
        missing=sorted(required-set(features));outcome=row['outcome'];positive=probability>=protocol['model']['cutoff']
        result.update(status='scored' if outcome else 'opening_unverified',probability=probability,
            predicted_positive=positive,correct=positive==bool(outcome['positive']) if outcome else None,
            bid=book,features=features,sources=sources,missing_features=missing,
            quality='provisional_missing_inputs' if missing else 'complete',
            gmp_probability=baseline if 'gmp_pct' in features else None,
            gmp_correct=(baseline>=protocol['gmp_baseline']['cutoff'])==bool(outcome['positive']) if outcome and 'gmp_pct' in features else None,
            training_overlap='possible' if asof.year<=model['training_end_year'] else 'outside_training_years')
        result['gmp_simple_positive']=features.get('gmp_pct',0)>0 if 'gmp_pct' in features else None
        result['gmp_simple_correct']=result['gmp_simple_positive']==bool(outcome['positive']) if outcome and result['gmp_simple_positive'] is not None else None
    scored=[r for r in records if r['correct'] is not None];paired=[r for r in scored if r['gmp_correct'] is not None]
    report=dict(mode='historical_backtest',computed_at_ist=datetime.now(IST).isoformat(),model_sha=protocol['model']['sha256'],protocol_sha=protocol['_sha256'],
        limitation='Retrospective replay of publisher-reported timestamps, recovered later. Frozen model selected after test inspection; not prospective or an untouched holdout. Missing inputs use frozen training imputation. Flat opens count nonpositive.',
        summary=dict(listed_ipos=len(records),predictions=sum(r['probability'] is not None for r in records),evaluated=len(scored),correct=sum(r['correct'] for r in scored),
            accuracy_pct=100*sum(r['correct'] for r in scored)/len(scored) if scored else None,
            always_positive_correct=sum(bool(r['outcome']['positive']) for r in scored),
            always_positive_accuracy_pct=100*sum(bool(r['outcome']['positive']) for r in scored)/len(scored) if scored else None,
            excluded=sum(r['probability'] is None for r in records),gmp_paired_count=len(paired),
            pending_openings=sum(r['probability'] is not None and r['outcome'] is None for r in records),
            gmp_correct=sum(r['gmp_correct'] for r in paired),model_correct_on_gmp_pairs=sum(r['correct'] for r in paired)),records=records)
    report['summary'].update(gmp_accuracy_pct=100*sum(r['gmp_correct'] for r in paired)/len(paired) if paired else None,
        gmp_simple_correct=sum(r['gmp_simple_correct'] for r in paired),
        gmp_simple_accuracy_pct=100*sum(r['gmp_simple_correct'] for r in paired)/len(paired) if paired else None,
        positive_calls=sum(r['predicted_positive'] for r in scored),nonpositive_calls=sum(not r['predicted_positive'] for r in scored))
    REPORT.write_text(json.dumps(report,indent=2,allow_nan=False,default=str),encoding='utf8')
    pd.DataFrame([dict(company=r['company_name'],status=r['status'],probability=r['probability'],correct=r['correct'],bid_time=(r.get('bid') or {}).get('observed_at_ist'),qib=(r.get('bid') or {}).get('qib'),hni=(r.get('bid') or {}).get('hni'),retail=(r.get('bid') or {}).get('retail'),source=(r.get('bid') or {}).get('source'),source_sha256=(r.get('bid') or {}).get('raw_sha256'),actual_open=(r.get('outcome') or {}).get('actual_open'),return_pct=(r.get('outcome') or {}).get('return_pct'),missing_features=', '.join(r.get('missing_features',[]))) for r in records]).to_csv(REPORT.with_suffix('.csv'),index=False)
    return report


def cached(company=None):
    if not REPORT.exists(): return None
    report=json.loads(REPORT.read_text(encoding='utf8'))
    from services.prospective_module1 import bundles
    if report['model_sha']!=bundles()[0]['model']['sha256']: return None
    from services.locked_model_configuration import historical_card
    report=historical_card(report)
    if company:
        return next((dict(r,summary=report['summary'],limitation=report['limitation'],computed_at_ist=report['computed_at_ist'],comparison_records=report['records']) for r in report['records'] if r['company_name']==company),None)
    return report
