"""Independent bid-source cross-check for every fixed holdout issuer."""
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor,as_completed

import pandas as pd
import requests

from ml.collect_module1_later_holdout import ROOT,CACHE as ORIGINAL
from ml.fetch_daywise_subscriptions import find_daywise

CACHE=ROOT/'data/archive/module1_later_bid_crosscheck_20261024'
OUT=ROOT/'data/audit/module1_later_bid_crosscheck_20261024.csv'


class RecordingSession(requests.Session):
    def __init__(self):
        super().__init__(); self.responses=[]

    def get(self,*args,**kwargs):
        response=super().get(*args,**kwargs)
        self.responses.append(dict(url=response.url,status=response.status_code,
            sha256=hashlib.sha256(response.content).hexdigest(),raw_response_text=response.text))
        return response


def check(row):
    path=CACHE/f'{row.ipo_id}.json.gz'
    if path.exists():
        with gzip.open(path,'rt',encoding='utf8') as stream:
            return json.load(stream)
    with gzip.open(ORIGINAL/f'{row.ipo_id}.json.gz','rt',encoding='utf8') as stream:
        prior=json.load(stream)
    calendar=pd.DatetimeIndex([r['date'] for r in prior['bid_history']])
    session=RecordingSession()
    result,status=find_daywise(pd.Series(row._asdict()),session,calendar)
    saved=dict(ipo_id=int(row.ipo_id),result=result,status=status,responses=session.responses)
    with gzip.open(path,'wt',encoding='utf8') as stream:
        json.dump(saved,stream)
    return saved


def run():
    CACHE.mkdir(parents=True,exist_ok=True)
    original=pd.read_csv(ROOT/'output/module1_2025_2026_paired_holdout_20261024_predictions.csv').set_index('ipo_id')
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(check,row) for row in original.reset_index().itertuples(index=False)]
        for future in as_completed(futures):
            observed=future.result(); ipo_id=observed['ipo_id']; prior=original.loc[ipo_id]
            row=dict(ipo_id=ipo_id,company_name=prior.company_name,status=observed['status']['reason'])
            if observed['result']:
                result=observed['result']
                row.update(source_url=result['source_url'],source_sha256=result['source_content_sha256'],
                           evidence_class=result['evidence_class'])
                for category in ('qib','hni','retail'):
                    field=category+'_subscription'
                    row[field]=result[field]
                    row['original_'+field]=float(prior[field])
                conflict=any(abs(row[c+'_subscription']-row['original_'+c+'_subscription'])>
                             max(.25,.2*max(row[c+'_subscription'],row['original_'+c+'_subscription']))
                             for c in ('qib','hni','retail'))
                row['status']='conflicting_sources' if conflict else 'corroborated'
            rows.append(row)
            if len(rows)%20==0:
                print(f'checked={len(rows)}/155 verified_tables={sum(r["status"] in ("corroborated","conflicting_sources") for r in rows)}',flush=True)
    pd.DataFrame(rows).sort_values('ipo_id').to_csv(OUT,index=False)
    print(pd.Series([r['status'] for r in rows]).value_counts().to_dict(),flush=True)


if __name__=='__main__':
    run()
