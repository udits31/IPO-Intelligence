"""Bounded public-page retry for exact missing issuers, with source hashes."""
import gzip,hashlib,json,time
from datetime import datetime,timezone
import pandas as pd
from services.saved_preclose_service import ROOT
from ml.stage_alignment_bridge import parse_ipoji,complete
from ml.collect_module1_later_holdout import index_urls
from ml.recover_module1_2025_training_gaps import source_rank
from ml.recover_module1_other_sources import fetch
from services.historical_bid_service import name_key

def run():
    gaps=pd.read_csv(ROOT/'output/module1_remaining_day2_category_gaps_20261026.csv',dtype={'issuer_key':str})
    urls=index_urls('ipoji_sitemap','https://www.ipoji.com/sitemap-ipo.xml','/ipo/')
    directory=ROOT/'data/archive/final_lock_day2_20261026';directory.mkdir(exist_ok=True)
    results=[]
    for r in gaps.to_dict('records'):
        path=directory/(r['issuer_key']+'.json.gz')
        if path.exists():
            results.append(json.loads(gzip.decompress(path.read_bytes()))['audit']);continue
        url=max(urls,key=lambda u:source_rank(r,u));time.sleep(1)
        audit=dict(issuer_key=r['issuer_key'],company_name=r['company_name'],source_url=url,attempted_at=datetime.now(timezone.utc).isoformat())
        sources=[]
        try:
            response=fetch(url);text=response.text;sha=hashlib.sha256(response.content).hexdigest()
            identity=name_key(r['company_name']) in name_key(__import__('re').sub('<[^>]+>',' ',text))
            days=[h for h in parse_ipoji(text) if h['day']==2 and complete(h)] if identity else []
            audit.update(status='day2_recovered_candidate' if days else 'identity_or_table_not_recovered',http_status=response.status_code,sha256=sha,day2_rows=days)
            sources.append(dict(url=url,raw_response_text=text))
        except Exception as exc:audit.update(status='fetch_failed',error=str(exc)[:250])
        path.write_bytes(gzip.compress(json.dumps(dict(audit=audit,sources=sources)).encode()))
        results.append(audit);print(json.dumps(audit),flush=True)
    (ROOT/'output/final_lock_day2_recovery_attempts_20261026.json').write_text(json.dumps(results,indent=2))
    # Only bounded searched sources are known. Never assert all public archives were exhausted.
    return results
if __name__=='__main__':run()
