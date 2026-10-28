"""Four predeclared research ablations. Never edits deployment or raw history."""
import hashlib,json,re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,accuracy_score
from ml import research_module1_forward_update as core
from ml.research_module1_stacking import EXPERTS,logits,stacker
from ml.model_pipeline import summarize
from ml.compare_model_generations_20261028 import guard,paired
from services.historical_bid_service import name_key

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/historical_join_cutoff_test_20261028'
SOURCE=ROOT/'output/model_comparison_20261028/numeric_audited_features.csv'
FACTS=ROOT/'data/audit/ipo_dated_fundamentals_all_20261024.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def eligible_document(record,row):
    if record.get('status')!='dated_document_facts' or name_key(record['company_name'])!=name_key(row.company_name):return False,'identity_or_status'
    doc=pd.to_datetime(record.get('document_date'),errors='coerce');decision=pd.Timestamp(row.decision_market_date)
    if pd.isna(doc) or doc>=decision:return False,'document_not_demonstrably_before_decision'
    period=pd.to_datetime(record.get('facts',{}).get('financial_period_end'),errors='coerce')
    if pd.isna(period) or period>=decision:return False,'financial_period_not_prior'
    return True,None


def nav_candidates(text,decision):
    """Only one-value dated issuer NAV tables, never peer/pro-forma values."""
    out=[]
    heading=r'(?im)^\s*(?:[IVX\dA-Z]+[.)]?\s+)?Net\s+Asset\s+Value[^\n]*'
    date=r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2})'
    for h in re.finditer(heading,text):
        section=text[h.end():h.end()+2000]
        section=re.split(r'(?i)comparison with|listed industry peers|notes\s*:',section)[0]
        # Any consolidated/standalone or pro-forma choice requires review.
        if re.search(r'consolidated.*standalone|assuming conversion|pro.forma',section,re.I|re.S):continue
        for match in re.finditer(r'(?im)^\s*(?:As (?:at|on)\s+)?'+date+r'\s+([\d,]+\.\d+)\s*$',section):
            period=pd.Timestamp(match[1]);value=float(match[2].replace(',',''))
            if period<pd.Timestamp(decision) and value>0:out.append((period,value,match[0].strip()))
    return out


def connect_financials(frame):
    import pymupdf
    records={str(r['ipo_id']):r for r in json.loads(FACTS.read_text())['records']}
    enriched=frame.copy();audit=[];verified={}
    for index,row in frame.iterrows():
        if pd.notna(row.pub_ipo_earnings_yield_pct) and pd.notna(row.pub_log_ipo_price_book):continue
        record=records.get(str(int(row.ipo_id)))
        if not record:continue
        valid,reason=eligible_document(record,row)
        entry=dict(ipo_id=int(row.ipo_id),company=row.company_name,year=int(row.year),
            document_date=record.get('document_date'),decision_date=str(row.decision_market_date.date()),
            source_path=record.get('source_path'),source_sha256=record.get('source_sha256'),changes=[])
        if not valid:audit.append(dict(entry,status='excluded',reason=reason));continue
        path=ROOT/record['source_path']
        if not path.exists():audit.append(dict(entry,status='excluded',reason='source_file_missing'));continue
        if path not in verified:verified[path]=sha(path)==record.get('source_sha256')
        if not verified[path]:audit.append(dict(entry,status='excluded',reason='source_hash_mismatch'));continue
        eps=record['facts'].get('annual_diluted_eps')
        if pd.isna(row.pub_ipo_earnings_yield_pct) and eps is not None and np.isfinite(eps) and row.issue_price>0:
            enriched.loc[index,'pub_ipo_earnings_yield_pct']=100*eps/row.issue_price
            entry['changes'].append(dict(field='pub_ipo_earnings_yield_pct',value=100*eps/row.issue_price,
                annual_eps=eps,period=record['facts']['financial_period_end'],basis='dated_native_RHP_annual_diluted_EPS_divided_by_offer_price'))
        if pd.isna(row.pub_log_ipo_price_book):
            cited=[e['page'] for e in record.get('evidence',[]) if e.get('kind')=='annual_eps']
            candidates=[]
            if cited:
                with pymupdf.open(path) as pdf:
                    # At most five pages around the already-cited valuation section.
                    for page in range(max(1,cited[0]-1),min(len(pdf),cited[0]+3)+1):
                        text=pdf[page-1].get_text(sort=True)
                        candidates.extend((date,v,quote,page,hashlib.sha256(text.encode()).hexdigest()) for date,v,quote in nav_candidates(text,row.decision_market_date))
            if candidates:
                latest=max(r[0] for r in candidates);same=[r for r in candidates if r[0]==latest]
                distinct={r[1] for r in same}
                if len(distinct)==1:
                    date,nav,quote,page,page_sha=same[0]
                    enriched.loc[index,'pub_log_ipo_price_book']=np.log1p(row.issue_price/nav)
                    entry['changes'].append(dict(field='pub_log_ipo_price_book',value=float(np.log1p(row.issue_price/nav)),
                        nav=nav,period=str(date.date()),page=page,page_text_sha256=page_sha,excerpt=quote,basis='dated_issuer_NAV_table_offer_price_divided_by_NAV'))
        audit.append(dict(entry,status='connected' if entry['changes'] else 'no_unambiguous_additional_fact'))
    return enriched,audit


def choose_threshold(validation,selection_year=2023):
    if selection_year not in (2023,2024):raise ValueError('Threshold selection must precede 2025 confirmation and 2026 testing')
    if not validation.year.eq(selection_year).all():raise ValueError(f'Threshold selection is restricted to {selection_year} out-of-time predictions')
    scores=[]
    for threshold in np.arange(.35,.851,.025):
        threshold=round(float(threshold),3)
        pred=validation.p.to_numpy()>=threshold;y=validation.target_binary.to_numpy()
        scores.append(dict(threshold=round(float(threshold),3),balanced_accuracy=float(balanced_accuracy_score(y,pred)),accuracy=float(accuracy_score(y,pred))))
    return sorted(scores,key=lambda s:(-s['balanced_accuracy'],-s['accuracy'],abs(s['threshold']-.5)))[0]


def fit_candidate(train,test,configs,train_end_year=2024,selection_year=2023):
    if train_end_year not in (2024,2025) or selection_year!=train_end_year-1:raise ValueError('Unsupported chronological validation protocol')
    oof=[];trace=[]
    for year in range(2022,train_end_year+1):
        held=train[train.year.eq(year)].copy();fit=train[train.year.lt(year)&train.listing_date.lt(held.decision_market_date.min())]
        assert set(fit.ipo_id).isdisjoint(held.ipo_id) and fit.listing_date.max()<held.decision_market_date.min()
        p=[]
        for name,c in configs.items():
            model=core.estimator(c).fit(core.features(fit,c),fit.target_binary)
            p.append(model.predict_proba(core.features(held,c))[:,1])
        entry=held[['ipo_id','year','target_binary']].copy()
        for name,v in zip(EXPERTS,p):entry[name]=v
        oof.append(entry);trace.append(dict(held_year=year,train_ids=fit.ipo_id.tolist(),held_ids=held.ipo_id.tolist()))
    oof=pd.concat(oof,ignore_index=True);validation=[]
    for year in range(2023,train_end_year+1):
        fit=oof[oof.year.lt(year)];held=oof[oof.year.eq(year)].copy()
        known=train[train.ipo_id.isin(fit.ipo_id)]
        if known.listing_date.max()>=train[train.ipo_id.isin(held.ipo_id)].decision_market_date.min():raise ValueError('Meta validation outcome timing leakage')
        m=stacker().fit(logits(fit[EXPERTS]),fit.target_binary.astype(int))
        held['p']=m.predict_proba(logits(held[EXPERTS]))[:,1];validation.append(held[['ipo_id','year','target_binary','p']])
    validation=pd.concat(validation,ignore_index=True)
    threshold=choose_threshold(validation[validation.year.eq(selection_year)],selection_year)
    confirmation=validation[validation.year.eq(train_end_year)]
    confirmation_metrics=summarize(confirmation.target_binary.astype(int),confirmation.p,threshold['threshold'])
    meta=stacker().fit(logits(oof[EXPERTS]),oof.target_binary.astype(int));models={};p=[]
    for name,c in configs.items():
        m=core.estimator(c).fit(core.features(train,c),train.target_binary);models[name]=m
        p.append(m.predict_proba(core.features(test,c))[:,1])
    return meta.predict_proba(logits(np.column_stack(p)))[:,1],threshold,confirmation_metrics,validation,dict(models=models,meta=meta,trace=trace)


def threshold_metrics(rows,p,threshold):
    m=summarize(rows.target_binary.astype(int),p,threshold);m['correct']=int(((p>=threshold)==rows.target_binary.to_numpy()).sum());return m


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    protocol=dict(train_years=[2021,2022,2023,2024],threshold_selection_year=2023,confirmation_year=2024,
        test_years=[2025,2026],selection_objective='Balanced accuracy, then accuracy, then closest to 0.5',
        threshold_grid=[round(float(v),3) for v in np.arange(.35,.851,.025)],no_test_threshold_tuning=True,
        feature_join_policy='Fill missing values only; identity/hash verified native source, document date strictly before decision; no post-offer NAV or peers',
        nav_page_budget='Only up to five native-text pages around an existing cited EPS/valuation page. No OCR or downloads.',
        live_promotion=False,limitations='Previously inspected retrospective test cohorts; chronological fitting does not turn them into fresh holdouts.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    frame=pd.read_csv(SOURCE);frame=frame[frame.cohort.eq('historical')].copy()
    frame['ipo_id']=pd.to_numeric(frame.ipo_id).astype(int)
    for k in ['listing_date','decision_market_date']:frame[k]=pd.to_datetime(frame[k])
    enriched,audit=connect_financials(frame)
    (OUT/'financial_join_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    enriched.to_csv(OUT/'enriched_derived_features.csv',index=False)
    core.SETS['gmp_only']=['gmp_pct'];all_configs=core.candidates()+[dict(name='gmp_only_control',family='logistic',C=.5,class_weight=None,transform='raw',feature_set='gmp_only')]
    configs={name:next(c for c in all_configs if c['name']==name) for name in EXPERTS}
    report=dict(protocol=protocol,coverage={},variants={},source_sha256=sha(SOURCE),facts_sha256=sha(FACTS),live_models_changed=False)
    prediction_rows=[]
    for name,data in [('baseline',frame),('financial_join',enriched)]:
        train=data[data.year.le(2024)].copy().reset_index(drop=True);test=data[data.year.ge(2025)].copy().reset_index(drop=True);guard(train,test)
        report['coverage'][name]={k:dict(training_observed=int(train[k].notna().sum()),training_n=len(train),test_observed=int(test[k].notna().sum()),test_n=len(test)) for k in ['pub_ipo_earnings_yield_pct','pub_log_ipo_price_book']}
        p,cutoff,confirmation,validation,bundle=fit_candidate(train,test,configs)
        validation.to_csv(OUT/(name+'_pre2025_validation.csv'),index=False)
        joblib.dump(dict(**bundle,training_ids=train.ipo_id.tolist(),training_end_year=2024,threshold=cutoff['threshold'],research_only=True),OUT/(name+'_research.joblib'))
        for suffix,threshold in [('fixed50',.5),('validated_cutoff',cutoff['threshold'])]:
            key=name+'_'+suffix;test[key]=p
            scopes={'clean_day2':test[test.verified_stage.eq('day2')],'all173':test}
            report['variants'][key]=dict(threshold=threshold,threshold_selected_on_2023=cutoff,confirmation_2024=confirmation,
                results={cohort:{'combined':threshold_metrics(rows,rows[key],threshold),**{str(int(year)):threshold_metrics(g,g[key],threshold) for year,g in rows.groupby('year')}} for cohort,rows in scopes.items()})
        keep=test[['ipo_id','company_name','year','verified_stage','target_binary','target_return']].copy()
        keep[name+'_score']=p;keep[name+'_cutoff']=cutoff['threshold'];prediction_rows.append(keep)
    joined=prediction_rows[0].merge(prediction_rows[1][['ipo_id','financial_join_score','financial_join_cutoff']],on='ipo_id',validate='one_to_one')
    joined.to_csv(OUT/'paired_predictions.csv',index=False)
    report['join_counts']=dict(connected_rows=sum(a['status']=='connected' for a in audit),excluded_rows=sum(a['status']=='excluded' for a in audit),
        reasons={reason:sum(a.get('reason')==reason for a in audit) for reason in sorted({a['reason'] for a in audit if a.get('reason')})})
    report['source_documents']={r['source_path']:r['source_sha256'] for r in audit if r['changes']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    lines=['# Historical financial joins and decision cutoff test','', 'Training: 2021–2024. Cutoff selection: 2023 out-of-time scores. Confirmation: 2024. Test: 2025–2026. No live model promotion.','',
        '| Variant | Cutoff | Combined clean Day-2 correct | 2025 | 2026 | Flat/negative caught | Positive caught |','|---|---:|---:|---:|---:|---:|---:|']
    for name,v in report['variants'].items():
        d=v['results']['clean_day2'];m=d['combined'];a=d['2025'];b=d['2026']
        lines.append(f"| {name} | {v['threshold']:.3f} | {m['correct']}/{m['sample_size']} ({m['classifier_accuracy_pct']:.1f}%) | {a['classifier_accuracy_pct']:.1f}% | {b['classifier_accuracy_pct']:.1f}% | {m['tn']}/{m['nonpositive_listings']} | {m['tp']}/{m['positive_listings']} |")
    lines+=['','## Coverage','',json.dumps(report['coverage'],indent=2),'','## Exclusions','',json.dumps(report['join_counts'],indent=2),'',protocol['limitations'],
        'Document date is retrospective availability evidence, not independently certified publication time. A later fetched file is not rejected merely because it was downloaded in 2026; a document dated after the decision is excluded.',
        'Financial features are missing where the source does not establish a unique period/basis. No test labels determine joins, models or thresholds.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['variants','source_documents']},indent=2))
    for k,v in report['variants'].items():print(k,v['threshold'],v['results']['clean_day2']['combined'])


if __name__=='__main__':run()
