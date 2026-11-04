"""Frozen weights, versioned input policy, and a separate capital overlay."""
import hashlib,json,math,sqlite3
from functools import lru_cache
import numpy as np
import pandas as pd
from services.saved_preclose_service import ROOT
from services.historical_bid_service import name_key
from services.retail_hni_economics import institutional_bouncer

VERSION='M1_V1_FROZEN_WITH_RECOVERED_INPUTS'


def cache_live_offer(item,ipo_id,asof):
    """Extract offer fields from the already identity-checked live financial page."""
    directory=ROOT/'data/telemetry/prospective_financials'
    meta=directory/(ipo_id+'_'+asof.strftime('%Y%m%d')+'.json')
    raw=directory/(ipo_id+'_'+asof.strftime('%Y%m%d')+'.html.gz')
    if not meta.exists() or not raw.exists():return
    import gzip,re
    fact=json.loads(meta.read_text());content=gzip.decompress(raw.read_bytes())
    captured=pd.Timestamp(fact.get('captured_at_ist'))
    if pd.isna(captured) or captured>pd.Timestamp(asof) or hashlib.sha256(content).hexdigest()!=fact.get('raw_sha256'):return
    from ml.extract_cached_deal_structure import parse
    from ml.stage_alignment_bridge import number
    structure=parse(content.decode('utf8',errors='replace'),item['issue_price']);reservation={}
    match=re.search(r'<script[^>]*id=["\']ipo-subscription-info-for-chart["\'][^>]*>(.*?)</script>',content.decode('utf8',errors='replace'),re.S)
    if match:
        for entry in json.loads(match[1]):
            if entry.get('key','').lower()=='shares offered / reserved':
                reservation={c:dict(shares=number(entry['value'].get(k))) for c,k in [('QIB','QIB'),('HNI','NII'),('Retail','RII')]}
    reservation.setdefault('Retail',{})['pct']=structure.get('retail_quota_pct') if structure.get('quota_weights_exact') is True else None
    r=dict(company_name=item['company_name'],fetch_timestamp=fact['captured_at_ist'],source_url=fact.get('source_url'),sha256=fact['raw_sha256'],
        offer=dict(ofs_ratio=structure.get('ofs_ratio')),lot_size_shares=structure.get('lot_size_shares'),reservation=reservation,
        identity_checks=dict(cap_reference=float(item['issue_price'])),identity_verified=True)
    out=ROOT/'data/telemetry/prospective_offer_facts';out.mkdir(exist_ok=True)
    (out/(name_key(item['company_name'])+'.json')).write_text(json.dumps(r))


def recovered_record(company):
    path=ROOT/'data/audit/ipopremium_recovered_20261026.jsonl'
    if not path.exists():return None
    found=[r for r in _records(path.stat().st_mtime_ns) if r.get('identity_verified') and name_key(r['company_name'])==name_key(company)]
    return found[0] if len(found)==1 else None


@lru_cache(maxsize=2)
def _records(vintage):
    return [json.loads(s) for s in (ROOT/'data/audit/ipopremium_recovered_20261026.jsonl').read_text(encoding='utf8').splitlines()]


def apply_live_recovered(features,sources,item,asof):
    """Only a genuinely earlier fetch can serve a live decision; never backdate."""
    r=recovered_record(item['company_name'])
    if not r:return
    fetched=pd.Timestamp(r['fetch_timestamp'])
    if fetched>pd.Timestamp(asof):return
    price=float(item.get('issue_price') or 0)
    if not np.isclose(price,r['identity_checks']['cap_reference'],atol=.01):return
    archive=ROOT/r['source_file']
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest()!=r['sha256']:return
    for field in ('profit_margin_pct','return_on_assets_pct','revenue_growth_asinh','annual_loss','ipo_earnings_yield_pct','log_ipo_price_book'):
        value=r['financial'].get(field)
        if value is not None and math.isfinite(float(value)) and 'pub_'+field not in features:features['pub_'+field]=float(value)
    sources['recovered_static']={'url':r['source_url'],'sha256':r['sha256'],'fetched_at':r['fetch_timestamp'],'configuration':VERSION}


def layer2(company,features,probability,*,historical=False,asof=None):
    r=recovered_record(company)
    if not historical:
        live=ROOT/'data/telemetry/prospective_offer_facts'/(name_key(company)+'.json')
        if live.exists():r=json.loads(live.read_text())
    if r and not historical and (asof is None or pd.Timestamp(r['fetch_timestamp'])>pd.Timestamp(asof)):r=None
    if r and not historical:
        price=features.get('log_offer_cap_price')
        if price is None or not np.isclose(np.expm1(price),r['identity_checks']['cap_reference'],atol=.01):r=None
    reserve=(r or {}).get('reservation',{})
    retail_quota=reserve.get('Retail',{}).get('pct')
    # Exact reported bid denominators reconstruct the total of these three pools.
    multiples={c:float(np.expm1(features['log_'+c])) if features.get('log_'+c) is not None and pd.notna(features.get('log_'+c)) else None for c in ('qib','hni','retail')}
    denoms=[reserve.get(c,{}).get('shares') for c in ('QIB','HNI','Retail')]
    total=None
    if all(v is not None and v>0 for v in denoms) and all(v is not None for v in multiples.values()):
        total=sum(v*multiples[c] for v,c in zip(denoms,('qib','hni','retail')))/sum(denoms)
    p=features.get('log_offer_cap_price');price=float(np.expm1(p)) if p is not None and pd.notna(p) else None
    return institutional_bouncer(p_win=probability,gmp_pct=features.get('gmp_pct'),qib=multiples['qib'],nii=multiples['hni'],retail=multiples['retail'],
        ofs_ratio=(r or {}).get('offer',{}).get('ofs_ratio'),retail_quota_pct=retail_quota,total_mult=total,
        original_lot_size=(r or {}).get('lot_size_shares'),offer_cap_price=price,bnii_mult=features.get('context_bnii_mult'))


def historical_card(report):
    """New replay view, never an edit of a saved forecast or previous audit."""
    path=ROOT/'data/audit/module1_stage_aligned_inputs_20261026.csv'
    if not path.exists():return report
    from services.prospective_module1 import predict
    from ml.research_module1_forward_update import SETS
    from services.prospective_module1 import bundles
    required={k for config in bundles()[1]['expert_configs'].values() for k in SETS.get(config['feature_set'],['gmp_pct'])}
    data=pd.read_csv(path);rows={name_key(r.company_name):r for r in data[data.cohort.eq('recent')].itertuples()}
    report=json.loads(json.dumps(report))
    for record in report['records']:
        row=rows.get(name_key(record['company_name']))
        if row is None or record.get('probability') is None:continue
        if not record.get('outcome'):
            with sqlite3.connect('file:'+str(ROOT/'data/telemetry/snapshots.db')+'?mode=ro',uri=True) as conn:
                conn.row_factory=sqlite3.Row
                opening=conn.execute('SELECT * FROM listing_outcome_bse_v1 WHERE company_name=?',(record['company_name'],)).fetchone()
            if opening:
                opening=dict(opening);price=float(opening['issue_price']);actual=float(opening['exchange_open_price'])
                record['outcome']={**opening,'actual_open':actual,'return_pct':100*(actual/price-1),'positive':actual>price}
                record['status']='scored'
        features={k:float(getattr(row,k)) for k in required if hasattr(row,k) and pd.notna(getattr(row,k))}
        p,g=predict(features)
        record.update(features=features,probability=p,gmp_probability=g,predicted_positive=p>=.5,configuration=VERSION,
                      correct=(p>=.5)==bool(record['outcome']['positive']) if record.get('outcome') else None,
                      capital_plan=layer2(record['company_name'],features,p,historical=True),
                      missing_features=sorted(required-set(features)))
        if record.get('outcome'):
            from services.prospective_module1 import bundles
            record['gmp_correct']=(g>=bundles()[0]['gmp_baseline']['cutoff'])==bool(record['outcome']['positive'])
    resolved=[r for r in report['records'] if r.get('correct') is not None]
    paired=[r for r in resolved if r.get('gmp_correct') is not None]
    report['summary'].update(evaluated=len(resolved),correct=sum(r['correct'] for r in resolved),
        accuracy_pct=100*sum(r['correct'] for r in resolved)/len(resolved) if resolved else None,
        model_correct_on_gmp_pairs=sum(r['correct'] for r in resolved if r.get('gmp_correct') is not None),
        positive_calls=sum(r['predicted_positive'] for r in resolved),nonpositive_calls=sum(not r['predicted_positive'] for r in resolved),
        pending_openings=sum(r.get('probability') is not None and not r.get('outcome') for r in report['records']),
        always_positive_correct=sum(bool(r['outcome']['positive']) for r in resolved),
        always_positive_accuracy_pct=100*sum(bool(r['outcome']['positive']) for r in resolved)/len(resolved) if resolved else None,
        gmp_paired_count=len(paired),gmp_correct=sum(r['gmp_correct'] for r in paired),
        gmp_accuracy_pct=100*sum(r['gmp_correct'] for r in paired)/len(paired) if paired else None)
    report['configuration']=VERSION
    return report
