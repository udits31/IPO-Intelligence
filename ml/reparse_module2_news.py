"""Reparse original real RSS bytes with issuer-collision guards; no new fetch."""
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from ml.collect_module2_dated_news import OUTPUT as ORIGINAL, ROOT, parse_news

OUTPUT=ROOT/'data/audit/module2_dated_news_issuer_verified_20261021.jsonl'
REPORT=ROOT/'output/module2_news_issuer_recheck_20261021.json'


def main():
    old=[json.loads(line) for line in ORIGINAL.read_text(encoding='utf-8').splitlines()]
    rows,removed=[],[]
    for record in old:
        revised={**record,'parser_version':'nse_distinct_issuer_prefix_v1_1'}
        if record['status']=='verified_rss_response':
            with gzip.open(ROOT/record['archive_path'],'rb') as stream:
                raw=stream.read()
            if hashlib.sha256(raw).hexdigest()!=record['raw_sha256']:
                raise ValueError('The original RSS source bytes changed')
            articles,rejected,count=parse_news(raw,record['company_name'],pd.Timestamp(record['cutoff_ist']).to_pydatetime())
            revised.update(articles=articles,rejected=rejected,raw_items=count)
            kept={a['title'] for a in articles}
            removed.extend({'ipo_id':record['ipo_id'],'company_name':record['company_name'],'title':a['title'],
                            'reason':'Not eligible under the current distinct-issuer/source filters'}
                           for a in record['articles'] if a['title'] not in kept)
        rows.append(revised)
    OUTPUT.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
    report={'cases':len(rows),'original_nonempty_cases':sum(bool(r['articles']) for r in old),
            'verified_issuer_nonempty_cases':sum(bool(r['articles']) for r in rows),
            'original_articles':sum(len(r['articles']) for r in old),
            'verified_issuer_articles':sum(len(r['articles']) for r in rows),'removed_headlines':removed,
            'input_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ORIGINAL,ROOT/'data/news_issuer_identity_master.csv',ROOT/'services/news_identity.py')},
            'output_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            'limitations':['Identity master is current, not an original historical snapshot; used only for entity disambiguation.',
                           'Conservative matching can reject ambiguous genuine headlines; no absent news is fabricated.']}
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='removed_headlines'},indent=2))


if __name__=='__main__':
    main()
