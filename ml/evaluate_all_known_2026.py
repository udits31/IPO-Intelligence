"""Full-known-universe 2026 audit. Inference only; no fitting or threshold search."""
import gzip,hashlib,io,json,math,re,zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,time
from pathlib import Path
import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from services.prospective_module1 import bundles,predict,IST
from services.saved_preclose_service import ROOT
from services.historical_bid_service import name_key
from ml.collect_module1_public_financials import extract
from ml.collect_module1_later_holdout import index_urls
from ml.recover_module1_2025_training_gaps import product_dates,history,source_rank
from ml.recover_module1_other_sources import dated_history,ipoji_payload
from ml.recover_listed_model_inputs import timed_gmp

OUT=ROOT/'output/all_known_2026_frozen_evaluation.json'
ARCHIVE=ROOT/'data/telemetry/all_2026_recovery'


def noncorporate(name):
    return bool(re.search(r'\btrust\b|\breit\b|\binvit\b',name,re.I))


def fetch(url):
    ARCHIVE.mkdir(exist_ok=True)
    path=ARCHIVE/(hashlib.sha256(url.encode()).hexdigest()+'.html.gz')
    if path.exists():raw=gzip.decompress(path.read_bytes())
    else:
        response=httpx.get(url,follow_redirects=True,timeout=20,headers={'User-Agent':'Mozilla/5.0'})
        response.raise_for_status();raw=response.content;path.write_bytes(gzip.compress(raw))
    return raw.decode('utf8',errors='replace'),hashlib.sha256(raw).hexdigest()


def recover(item,index,offers,market):
    result=dict(features={},sources=[],status='provisional_missing_dated_book')
    aliases={655:'sbi-mutual-fund-ipo',661:'cordelia-cruises-ipo'}
    urls=sorted(index,key=lambda u:-source_rank(item,u))[:2]
    if item['id'] in aliases:urls.insert(0,'https://www.ipoji.com/ipo/'+aliases[item['id']])
    for url in dict.fromkeys(urls):
        try:
            html,sha=fetch(url)
            # A verified alias is allowed only alongside cap/date/listing checks.
            check=dict(item)
            if item['id']==655:check['company_name']='SBI Funds Management'
            if item['id']==661:check['company_name']='Waterways Leisure Tourism'
            if item['id']==614:check['company_name']='Asset Reconstruction Company'
            dates=product_dates(html,check)
            if not dates:continue
            offer=offers.loc[item['id']] if item['id'] in offers.index else None
            if offer is not None and any(str(offer[k])!=dates[k] for k in ('issue_open_date','issue_close_date')):continue
            sessions=market.index[(market.index>=pd.Timestamp(dates['issue_open_date']))&(market.index<=pd.Timestamp(dates['issue_close_date']))]
            if len(sessions)<2:continue
            day=sessions[-2];asof=datetime.combine(day.date(),time(17),IST)
            soup=BeautifulSoup(html,'html.parser');payload=ipoji_payload({**check,**dates},soup)
            days=history(payload['bids'],dates['issue_open_date'],dates['issue_close_date']) if payload else []
            chosen=next((b for b in days if b['date']==day.date().isoformat()),None)
            result.update(decision_market_date=day.date().isoformat(),offer_dates=dates)
            source=dict(url=url,raw_sha256=sha);result['sources'].append(source)
            if chosen:
                for cat in ('qib','hni','retail'):result['features']['log_'+cat]=float(np.log1p(max(.01,chosen[cat+'_subscription'])))
                first=min(days,key=lambda b:b['date'])
                if first['date']<chosen['date']:
                    for cat in ('qib','hni','retail'):result['features']['log_'+cat+'_growth']=float(np.log1p(chosen[cat+'_subscription'])-np.log1p(first[cat+'_subscription']))
                result['status']='recovered_penultimate_book'
            quotes=timed_gmp(html,asof) or [q for q in dated_history(html,day) if q['gmp_date']<day.date().isoformat()]
            quotes=[q for q in quotes if 0<=(day.date()-pd.Timestamp(q['gmp_date']).date()).days<=7]
            if quotes:
                q=max(quotes,key=lambda q:(q['gmp_date'],q.get('gmp_asof_ist','')))
                result['features'].update(gmp_pct=100*q['gmp_rs']/item['issue_price'],gmp_quote_age_days=(day.date()-pd.Timestamp(q['gmp_date']).date()).days)
                result['gmp_quote']=dict(q,**source)
                prior=[p for p in quotes if p['gmp_date']<q['gmp_date']]
                if prior:
                    p=max(prior,key=lambda p:(p['gmp_date'],p.get('gmp_asof_ist','')))
                    result['features']['gmp_slope_pp_per_day']=100*(q['gmp_rs']-p['gmp_rs'])/item['issue_price']/(pd.Timestamp(q['gmp_date'])-pd.Timestamp(p['gmp_date'])).days
            facts=extract(html,day,item['issue_price'])
            result['annual_period']=facts.get('annual_period')
            for key in ('profit_margin_pct','return_on_assets_pct','revenue_growth_asinh','annual_loss','ipo_earnings_yield_pct','log_ipo_price_book'):
                if facts.get(key) is not None:result['features']['pub_'+key]=facts[key]
            if chosen:break
        except Exception as exc:result.setdefault('errors',[]).append(str(exc)[:180])
    if result.get('decision_market_date'):
        day=pd.Timestamp(result['decision_market_date'])
        slug=re.sub(r'[^a-z0-9]+','-',re.sub(r'\b(?:ltd|limited)\.?','',item['company_name'],flags=re.I).lower()).strip('-')
        aliases_gmp={614:'asset-reconstruction-company-india',655:'sbi-funds-management',661:'cordelia-cruises'}
        for prefix,suffix in [('https://theipolist.in/ipo-details/',''),('https://www.iporise.com/ipo/','-ipo/gmp')]:
            url=prefix+aliases_gmp.get(item['id'],slug)+suffix
            try:
                html,sha=fetch(url);soup=BeautifulSoup(html,'html.parser')
                brand=aliases_gmp.get(item['id'],slug).replace('-',' ')
                heading=' '.join(h.get_text(' ',strip=True) for h in soup.find_all('h1'))
                if name_key(brand) not in name_key(heading):continue
                text=soup.get_text(' ',strip=True)
                # Match the cap price, as well as the issuer, on this GMP page.
                if not re.search(r'(?:Issue price|Price Band|Upper price band)\s*(?:₹\s*)?'+str(int(item['issue_price']))+r'\b',text,re.I) and str(int(item['issue_price'])) not in text:continue
                quotes=[q for q in dated_history(html,day) if 0<(day.date()-pd.Timestamp(q['gmp_date']).date()).days<=7]
                quotes += [q for q in timed_gmp(html,datetime.combine(day.date(),time(17),IST)) if 0<=(day.date()-pd.Timestamp(q['gmp_date']).date()).days<=7]
                if not quotes:continue
                q=max(quotes,key=lambda q:(q['gmp_date'],q.get('gmp_asof_ist','')))
                result['features'].update(gmp_pct=100*q['gmp_rs']/item['issue_price'],gmp_quote_age_days=(day.date()-pd.Timestamp(q['gmp_date']).date()).days)
                result['gmp_quote']=dict(q,url=url,raw_sha256=sha)
                prior=[p for p in quotes if p['gmp_date']<q['gmp_date']]
                if prior:
                    p=max(prior,key=lambda p:(p['gmp_date'],p.get('gmp_asof_ist','')))
                    result['features']['gmp_slope_pp_per_day']=100*(q['gmp_rs']-p['gmp_rs'])/item['issue_price']/(pd.Timestamp(q['gmp_date'])-pd.Timestamp(p['gmp_date'])).days
                break
            except Exception as exc:result.setdefault('gmp_errors',[]).append(str(exc)[:180])
        if 'gmp_pct' not in result['features'] and item['id']==614:
            url='https://www.ipoguru.in/ipo-gmp/asset-reconstruction-coindia-ipo'
            try:
                html,sha=fetch(url)
                soup=BeautifulSoup(html,'html.parser')
                if name_key(item['company_name']) in name_key(soup.get_text(' ',strip=True)):
                    quotes=[q for q in dated_history(html,day) if 0<(day.date()-pd.Timestamp(q['gmp_date']).date()).days<=7]
                    if quotes:
                        q=max(quotes,key=lambda q:q['gmp_date'])
                        result['features'].update(gmp_pct=100*q['gmp_rs']/item['issue_price'],gmp_quote_age_days=(day.date()-pd.Timestamp(q['gmp_date']).date()).days)
                        result['gmp_quote']=dict(q,url=url,raw_sha256=sha)
                        earlier=[p for p in quotes if p['gmp_date']<q['gmp_date']]
                        if earlier:
                            p=max(earlier,key=lambda p:p['gmp_date'])
                            result['features']['gmp_slope_pp_per_day']=100*(q['gmp_rs']-p['gmp_rs'])/item['issue_price']/(pd.Timestamp(q['gmp_date'])-pd.Timestamp(p['gmp_date'])).days
            except Exception as exc:result.setdefault('gmp_errors',[]).append(str(exc)[:180])
    if not result.get('decision_market_date') and item['id'] in offers.index:
        offer=offers.loc[item['id']];sessions=market.index[(market.index>=pd.Timestamp(offer.issue_open_date))&(market.index<=pd.Timestamp(offer.issue_close_date))]
        if len(sessions)>=2:result['decision_market_date']=sessions[-2].date().isoformat()
    return result


def metrics(rows):
    resolved=[r for r in rows if r.get('correct') is not None]
    paired=[r for r in resolved if r.get('gmp_correct') is not None]
    return dict(records=len(rows),evaluated=len(resolved),correct=sum(r['correct'] for r in resolved),accuracy_pct=100*sum(r['correct'] for r in resolved)/len(resolved) if resolved else None,
        always_positive_correct=sum(r['actual_positive'] for r in resolved),always_positive_accuracy_pct=100*sum(r['actual_positive'] for r in resolved)/len(resolved) if resolved else None,
        gmp_paired=len(paired),gmp_correct=sum(r['gmp_correct'] for r in paired),model_correct_on_gmp_pairs=sum(r['correct'] for r in paired))


def run():
    from ml.research_module1_forward_update import SETS
    protocol,model,_=bundles();before=hashlib.sha256((ROOT/protocol['model']['path']).read_bytes()).hexdigest()
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv');canonical=canonical[pd.to_datetime(canonical.listing_date).dt.year.eq(2026)]
    assert not set(canonical.id)&set(model['training_ids'])
    frame=pd.read_csv(ROOT/'output/module1_forward_coverage_20261024_features.csv').set_index('ipo_id')
    training=frame.loc[model['training_ids']]
    assert training.year.le(2025).all()
    outcome=pd.read_csv(ROOT/'data/audit/module1_exchange_listing_open_audit.csv').set_index('ipo_id')
    offers=pd.read_csv(ROOT/'data/nse_verified_offer_dates.csv').set_index('ipo_id')
    market=pd.read_csv(ROOT/'output/module1_forward_update_20261024_market.csv',parse_dates=['date']).set_index('date')
    required=sorted(set(k for cfg in model['expert_configs'].values() for k in (['gmp_pct'] if cfg['feature_set']=='gmp_only' else SETS[cfg['feature_set']])))
    index=index_urls('ipoji_sitemap','https://www.ipoji.com/sitemap-ipo.xml','/ipo/')
    targets=[r.to_dict() for _,r in canonical.iterrows() if r.id not in frame.index and not noncorporate(r.company_name)]
    with ThreadPoolExecutor(4) as pool:
        recovered=dict(zip([int(i['id']) for i in targets],pool.map(lambda i:recover(i,index,offers,market),targets)))
    records=[]
    for item in canonical.to_dict('records'):
        id=int(item['id']);row=dict(ipo_id=id,company_name=item['company_name'],listing_date=item['listing_date'],scope='canonical_2026',stage='penultimate_session_end',probability=None,correct=None)
        records.append(row)
        if noncorporate(item['company_name']):row['status']='out_of_scope_noncorporate_trust';continue
        if id in frame.index:
            original=frame.loc[id];features={k:float(original[k]) for k in required if k in original and pd.notna(original[k])}
            decision=original.decision_market_date;row['status']='original_69_cohort';row['sources']=[dict(url=original.get('bid_source_url'),gmp_url=original.get('gmp_source_url'))]
        else:
            evidence=recovered[id];features=evidence['features'];decision=evidence.get('decision_market_date');row.update(status=evidence['status'],recovery=evidence)
            for key,source in [('log_offer_cap_price','issue_price'),('log_issue_size','issue_size_cr')]:
                if item.get(source) and math.isfinite(float(item[source])) and float(item[source])>0:features[key]=float(np.log1p(item[source]))
            features['pub_financial_sector']=float(item['sector']=='Financial Services')
            if decision:
                day=pd.Timestamp(decision);prior=market[market.index<day]
                n=prior.NIFTY50.dropna();v=prior.INDIAVIX.dropna()
                if len(n)>=21 and (day-n.index[-1]).days<=5:
                    features.update(macro_nifty_5d=float((n.iloc[-1]/n.iloc[-6]-1)*100),macro_nifty_20d=float((n.iloc[-1]/n.iloc[-21]-1)*100),macro_nifty_vol20=float((n.pct_change()*100).iloc[-20:].std()))
                if len(v)>=6 and (day-v.index[-1]).days<=5:features.update(macro_vix=float(v.iloc[-1]),macro_vix_change5=float((v.iloc[-1]/v.iloc[-6]-1)*100))
        # Frozen imputation allows scoring missing facts, explicitly separated in metrics.
        p,gmp=predict(features);row.update(probability=p,predicted_positive=p>=.5,decision_date=str(decision),features=features,missing_features=sorted(set(required)-set(features)),gmp_probability=gmp if 'gmp_pct' in features else None)
        actual=outcome.loc[id] if id in outcome.index else None
        if actual is not None and actual.status=='verified_daily_exchange_archive':
            opening=float(actual.exchange_open);row.update(actual_open=opening,issue_price=float(actual.issue_price),outcome_source=actual.source_url)
        elif id==609 and id in offers.index:
            from ml.audit_module2_nse_closes import exchange_rows,CACHE
            symbol=offers.loc[id].nse_issue_symbol;date=pd.Timestamp(item['listing_date']).strftime('%Y%m%d');_,meta=exchange_rows(date,{symbol})
            if meta['status']=='verified_archive':
                raw=(CACHE/(date+'.csv.zip')).read_bytes()
                with zipfile.ZipFile(io.BytesIO(raw)) as z:table=pd.read_csv(z.open(z.namelist()[0]))
                matched=table[table.TckrSymb.eq(symbol)&table.SctySrs.isin(['EQ','BE'])]
                if len(matched)==1 and name_key(item['company_name']) in name_key(str(matched.iloc[0].get('FinInstrmNm',''))):
                    row.update(actual_open=float(matched.iloc[0].OpnPric),issue_price=float(item['issue_price']),outcome_source=meta['url'],outcome_sha256=meta['sha256'])
        if row.get('actual_open') is not None:
            y=row['actual_open']>row['issue_price'];row.update(actual_positive=y,return_pct=100*(row['actual_open']/row['issue_price']-1),correct=(p>=.5)==y,gmp_correct=(gmp>=protocol['gmp_baseline']['cutoff'])==y if 'gmp_pct' in features else None)
    recent=json.loads((ROOT/'output/listed_closing_day_backtest.json').read_text())['records']
    earliest_decision=min(pd.Timestamp(r['decision_date']) for r in records if r.get('decision_date') and r['decision_date']!='None')
    assert pd.to_datetime(training.listing_date).max()<earliest_decision
    names={name_key(r['company_name']) for r in records}
    for recent_row in recent:
        if name_key(recent_row['company_name']) in names:raise ValueError('Overlapping issuer in separate caches requires identity reconciliation')
        o=recent_row.get('outcome');records.append(dict(ipo_id=None,company_name=recent_row['company_name'],scope='recent_cache_2026',stage='closing_day_pre17',status='recovered_closing_day',probability=recent_row['probability'],predicted_positive=recent_row['predicted_positive'],correct=recent_row['correct'],actual_positive=o['positive'] if o else None,actual_open=o['actual_open'] if o else None,issue_price=o['issue_price'] if o else None,return_pct=o['return_pct'] if o else None,missing_features=recent_row['missing_features'],gmp_correct=recent_row['gmp_correct'],features=recent_row['features']))
    assert before==hashlib.sha256((ROOT/protocol['model']['path']).read_bytes()).hexdigest()
    canonical_rows=[r for r in records if r['scope']=='canonical_2026'];corporate=[r for r in canonical_rows if r['status']!='out_of_scope_noncorporate_trust']
    result=dict(model_sha256=before,training_end_year=2025,training_test_overlap=0,no_retraining=True,threshold=.5,
        coverage=dict(canonical_2026=len(canonical_rows),corporate=len(corporate),noncorporate_excluded=len(canonical_rows)-len(corporate),recent_additional=len(recent),total_known=len(records)),
        original_69=metrics([r for r in canonical_rows if r['status']=='original_69_cohort']),expanded_canonical=metrics(corporate),
        recovered_penultimate=metrics([r for r in corporate if r['status']=='recovered_penultimate_book']),
        missing_book_provisional=metrics([r for r in corporate if r['status']=='provisional_missing_dated_book']),
        recent_closing_day=metrics([r for r in records if r['scope']=='recent_cache_2026']),all_known_pooled_diagnostic=metrics([r for r in records if r['status']!='out_of_scope_noncorporate_trust']),
        limitations=['This is all locally known 2026 records, not an exchange-certified complete market census.','The original 69 were repeatedly inspected during research; this is not a fresh blind test.','Corporate and noncorporate securities are reported separately.','Closing-day and penultimate-day evaluations are separate; the pooled figure mixes stages and is diagnostic only.','Missing-input predictions are explicitly provisional; no after-cutoff final bids are substituted.'],records=records)
    OUT.write_text(json.dumps(result,indent=2,default=str,allow_nan=False),encoding='utf8')
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('features','recovery','sources')} for r in records]).to_csv(OUT.with_suffix('.csv'),index=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','limitations')},indent=2),flush=True)

if __name__=='__main__':run()
