"""Download official NSE issuer names/symbols for news disambiguation only."""
import csv
import hashlib
import io
import json
from datetime import datetime,timezone
from pathlib import Path

import requests

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'data/news_issuer_identity_master.csv'
SOURCE='https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv'


def main():
    response=requests.get(SOURCE,headers={'User-Agent':'Mozilla/5.0'},timeout=25)
    response.raise_for_status()
    table=list(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))))
    if not table or any(not r.get('SYMBOL') or not r.get('NAME OF COMPANY') for r in table):
        raise ValueError('Official NSE issuer master schema incomplete')
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=['symbol','company_name'],lineterminator='\n')
    writer.writeheader()
    writer.writerows({'symbol':r['SYMBOL'].strip(),'company_name':r['NAME OF COMPANY'].strip()} for r in table)
    data=buffer.getvalue().encode('utf-8')
    OUTPUT.write_bytes(data)
    record={'source_url':SOURCE,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_sha256':hashlib.sha256(response.content).hexdigest(),
            'identity_sha256':hashlib.sha256(data).hexdigest(),'issuer_count':len(table),
            'use':'Issuer identity resolution only, not future financial or price features.',
            'limitation':'Current issuer-name master, not a historical as-published security master.'}
    OUTPUT.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
