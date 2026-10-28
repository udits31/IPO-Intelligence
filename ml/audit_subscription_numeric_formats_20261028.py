"""Repair derived features only when preserved source bytes bind the old value."""
import gzip,hashlib,html,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from services.subscription_numbers import subscription_multiple
from ml.recover_progressive_totals import cached_pages

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/audit/module1_stage_aligned_inputs_20261026.csv'
OUT=ROOT/'output/model_comparison_20261028'


def run():
    frame=pd.read_csv(SOURCE);records=[];seen=set();paths=[]
    for directory in ['module1_later_holdout_20261024','module1_other_sources_20261024','module1_public_financials_20261024','final_lock_day2_20261026','module1_2025_training_gaps_20261024']:
        paths.extend((ROOT/'data/archive'/directory).glob('*.json.gz'))
    for path in paths:
        key=path.name.split('.')[0]
        rows=frame[frame.issuer_key.astype(str).eq(key)]
        if rows.empty:continue
        try:payload=json.loads(gzip.decompress(path.read_bytes()))
        except (ValueError,OSError):continue
        for content,url in cached_pages(payload):
            raw_sha=hashlib.sha256(content.encode()).hexdigest()
            match=re.search(r'<script[^>]*id=["\']ipo-subscription-info-for-chart["\'][^>]*>(.*?)</script>',content,re.S)
            if not match:continue
            try:history=json.loads(html.unescape(match[1]))
            except ValueError:continue
            for entry in history:
                day=re.search(r'Day\s*(\d+)\s*-\s*(\d{2}-\d{2}-\d{2,4})',entry.get('key',''))
                if not day:continue
                date=pd.to_datetime(day[2],format='%d-%m-%y' if len(day[2])==8 else '%d-%m-%Y').date()
                for cat,label in [('qib','QIB'),('hni','NII'),('retail','RII')]:
                    raw=str(entry.get('value',{}).get(label,'')).strip().strip('()')
                    if not re.fullmatch(r'\d+,\d{1,2}',raw):continue
                    old=float(raw.replace(',',''));new=subscription_multiple(raw)
                    unique=(key,raw_sha,str(date),cat)
                    if unique in seen:continue
                    seen.add(unique)
                    record=dict(issuer_key=key,source_url=url,source_file=str(path.relative_to(ROOT)),source_sha256=raw_sha,
                        day=int(day[1]),observed_on=str(date),category=cat,raw_value=raw,old_value=old,new_value=new,changed_fields=[])
                    for index,row in rows.iterrows():
                        bound=(str(row.get('bid_source_sha256'))==raw_sha or str(row.get('source_url'))==str(url) or str(row.get('bid_source_url'))==str(url))
                        if not bound:continue
                        for column in [cat+'_subscription','day2_'+cat,'cutoff_'+cat]:
                            if column not in frame or pd.isna(row.get(column)) or not np.isclose(row[column],old):continue
                            if column.startswith('day2_') and int(day[1])!=2:continue
                            if column==cat+'_subscription' and pd.Timestamp(row.decision_market_date).date()!=date:continue
                            frame.loc[index,column]=new;record['changed_fields'].append(column)
                        log='log_'+cat
                        if cat+'_subscription' in record['changed_fields'] and np.isclose(row.get(log,np.nan),np.log1p(max(old,.01))):
                            frame.loc[index,log]=np.log1p(max(new,.01));record['changed_fields'].append(log)
                            # Recompute the same-source trajectory only if the
                            # previous stored value matches the old calculation.
                            prior=[]
                            for h in history:
                                if re.match(r'Day\s*1\s*-',h.get('key','')):
                                    v=subscription_multiple(h.get('value',{}).get(label))
                                    if v is not None:prior.append(v)
                            growth=log+'_growth'
                            if len(prior)==1 and np.isclose(row.get(growth,np.nan),np.log1p(old)-np.log1p(prior[0]),atol=.02):
                                frame.loc[index,growth]=np.log1p(new)-np.log1p(prior[0]);record['changed_fields'].append(growth)
                    records.append(record)
    OUT.mkdir(exist_ok=True,parents=True)
    derived=OUT/'numeric_audited_features.csv';frame.to_csv(derived,index=False)
    report=dict(original_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        parsed_pages=len(paths),decimal_comma_records=records,
        changed_records=sum(bool(r['changed_fields']) for r in records),raw_history_changed=False,
        derived_source=str(derived.relative_to(ROOT)))
    (OUT/'numeric_format_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2));return derived


if __name__=='__main__':run()
