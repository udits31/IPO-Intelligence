"""Recover public daywise bid tables for one prespecified temporal challenger."""
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor,as_completed

import numpy as np
import pandas as pd
import requests

from ml.complete_module1_benchmark import prepare,ROOT
from ml.fetch_daywise_subscriptions import table_days,HEADERS

CACHE=ROOT/'data/archive/module1_bid_history_20261024'
OUT=ROOT/'data/audit/module1_bid_growth_20261024.csv'


def collect(row):
    path=CACHE/f'{row.ipo_id}.json.gz'
    if path.exists():
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            return json.load(stream)
    result=dict(ipo_id=int(row.ipo_id),source_url=row.source_url,status='no_supported_public_day_table',
        log_qib_growth=None,log_hni_growth=None,log_retail_growth=None)
    try:
        if not row.source_url.startswith('https://ipowatch.in/'):
            result['status']='unsupported_source_table_format'
        else:
            response=requests.get(row.source_url,headers=HEADERS,timeout=15);response.raise_for_status()
            days=table_days(response.text)
            result.update(raw_response_text=response.text,source_sha256=hashlib.sha256(response.content).hexdigest())
            if days and len(days)>=3:
                last=max(days);prior=days[last-1];first=days[min(days)]
                if all(abs(prior[k]-getattr(row,k+'_subscription'))<=.02 for k in ('qib','hni','retail')):
                    result['status']='matched_penultimate_table'
                    for key in ('qib','hni','retail'):
                        result['log_'+key+'_growth']=float(np.log1p(prior[key])-np.log1p(first[key]))
                        result['first_'+key]=first[key]
                        # Explicit diagnostic-only fields, excluded from features.
                        result['postcutoff_final_'+key]=days[last][key]
                    result['prior_day_number']=last-1
                else:
                    result['status']='penultimate_table_conflicts_with_frozen_inputs'
    except (requests.RequestException,ValueError,KeyError) as exc:
        result.update(status='fetch_error',error=str(exc)[:180])
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        json.dump(result,stream)
    return result


def run():
    CACHE.mkdir(parents=True,exist_ok=True)
    early,_,_=prepare();results=[]
    def save():
        pd.DataFrame([{k:v for k,v in r.items() if k!='raw_response_text'} for r in sorted(results,key=lambda r:r['ipo_id'])]).to_csv(OUT,index=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(collect,row) for row in early.itertuples()]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%20==0 or len(results)==len(early):
                save();print(json.dumps(dict(bid_history_done=len(results),total=len(early),
                    usable=sum(r['status']=='matched_penultimate_table' for r in results))),flush=True)
    save()


if __name__=='__main__':
    run()
