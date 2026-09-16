"""Collect real dated headlines for the fixed, exchange-price monitor replay.

No outcomes select searches. Every eligible price case is queried, including
empty/failed feeds. Raw RSS is archived and hashed. Google RSS is a secondary,
retrospectively retrieved index, not immutable original-publisher evidence.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime,timezone,timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from services.news_service import _quote_directory_title, _relevant_to_company

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/archive/module2_dated_news'
OUTPUT=ROOT/'data/audit/module2_dated_news_20261021.jsonl'
REPORT=ROOT/'output/module2_dated_news_collection_20261021.json'
IST=ZoneInfo('Asia/Kolkata')
LEDGERS=[ROOT/f'output/module2_{name}_decision_ledger_goal_corporate_actions_20261021.csv'
         for name in ('development','holdout')]


def parse_news(raw,company,cutoff):
    root=ET.fromstring(raw)
    if root.tag.lower()!='rss' or root.find('channel') is None:
        raise ValueError('Historical source did not return an RSS channel')
    articles, rejected, seen=[],{},set()
    for item in root.findall('./channel/item'):
        title=(item.findtext('title') or '').rsplit(' - ',1)[0].strip()
        link=item.findtext('link') or ''
        published=item.findtext('pubDate') or ''
        reason=None
        try:
            stamp=parsedate_to_datetime(published)
            if stamp.tzinfo is None:
                raise ValueError('Undated timezone')
            stamp=stamp.astimezone(IST)
            if stamp>cutoff:
                reason='after_decision'
            elif cutoff-stamp>timedelta(days=7):
                reason='outside_seven_day_window'
            elif stamp.date()==cutoff.date():
                # Old RSS timestamps may have date-only publisher precision.
                # Do not assume its current-day clock preceded the decision.
                reason='same_day_original_clock_unverified'
        except (ValueError,TypeError,OverflowError):
            reason='invalid_publication_time'
        if not reason and (not link.startswith('https://') or not title):
            reason='missing_title_or_link'
        if not reason and (_quote_directory_title(title) or not _relevant_to_company(title,company)):
            reason='not_issuer_event_news'
        if not reason and title.casefold() in seen:
            reason='duplicate_title'
        if reason:
            rejected[reason]=rejected.get(reason,0)+1
            continue
        seen.add(title.casefold())
        publisher=item.findtext('source') or ''
        articles.append({'title':title,'published_at':published,'url':link,
                         'publisher':publisher,'source':publisher,
                         'publisher_url':item.find('source').get('url') if item.find('source') is not None else None})
    articles.sort(key=lambda x:parsedate_to_datetime(x['published_at']),reverse=True)
    return articles[:6],rejected,len(root.findall('./channel/item'))


def search_name(company):
    words=[word for word in re.findall(r'[A-Za-z0-9]+',company)
           if word.casefold() not in {'ltd','limited','inc','plc','the'}]
    return ' '.join(words[:2])


def collect(limit=0):
    cases=pd.concat(pd.read_csv(path) for path in LEDGERS).sort_values('ipo_id')
    if cases.ipo_id.duplicated().any():
        raise ValueError('Each price case must have exactly one prediction checkpoint')
    if limit:
        # Deterministic coverage pilot, never selected on outcome direction.
        cases=cases.groupby('listing_year',group_keys=False).head(limit)
    RAW.mkdir(parents=True,exist_ok=True)
    rows=[]
    session=requests.Session()
    for n,case in enumerate(cases.itertuples(),1):
        cutoff=datetime.fromisoformat(str(case.decision_date)+'T15:30:00').replace(tzinfo=IST)
        first=cutoff-timedelta(days=7)
        query=f'"{search_name(case.company_name)}" after:{first:%Y-%m-%d} before:{cutoff+timedelta(days=1):%Y-%m-%d}'
        url='https://news.google.com/rss/search?'+urlencode({'q':query,'hl':'en-IN','gl':'IN','ceid':'IN:en'})
        path=RAW/f'{case.ipo_id}_{case.decision_date}.xml.gz'
        meta=path.with_suffix('.meta.json')
        result={'ipo_id':int(case.ipo_id),'company_name':case.company_name,
                'decision_date':case.decision_date,'cutoff_ist':cutoff.isoformat(),
                'listing_year':int(case.listing_year),'source_url':url,'query':query,
                'status':'unavailable','raw_sha256':None,'articles':[],
                'retrieved_at_utc':None,'archive_path':str(path.relative_to(ROOT)),
                'timestamp_policy':'Only preceding-date headlines within seven days; same-day clocks unverified.'}
        try:
            if path.exists() and meta.exists():
                record=json.loads(meta.read_text(encoding='utf-8'))
                if record['source_url']!=url:
                    raise ValueError('Cached search does not match the fixed protocol')
                with gzip.open(path,'rb') as stream:
                    raw=stream.read()
                result['retrieved_at_utc']=record['retrieved_at_utc']
            else:
                response=session.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=25)
                response.raise_for_status()
                raw=response.content
                parse_news(raw,case.company_name,cutoff)
                result['retrieved_at_utc']=datetime.now(timezone.utc).isoformat()
                with gzip.open(path,'wb') as stream:
                    stream.write(raw)
                meta.write_text(json.dumps({'source_url':url,'retrieved_at_utc':result['retrieved_at_utc']})+'\n',encoding='utf-8')
                time.sleep(.5)
            articles,rejected,raw_count=parse_news(raw,case.company_name,cutoff)
            result.update(status='verified_rss_response',raw_sha256=hashlib.sha256(raw).hexdigest(),
                          articles=articles,rejected=rejected,raw_items=raw_count)
        except (requests.RequestException,ValueError,ET.ParseError,OSError) as exc:
            result['error']=str(exc)[:200]
            if isinstance(exc,requests.HTTPError) and exc.response.status_code in (403,429):
                rows.append(result)
                break  # Do not bypass access restrictions or rate limits.
        rows.append(result)
        if n%20==0 or n==len(cases):
            OUTPUT.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
            print(f'Dated news {n}/{len(cases)}: {sum(bool(r["articles"]) for r in rows)} cases with eligible articles',flush=True)
    OUTPUT.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
    data=pd.DataFrame(rows)
    report={'planned_cases':len(cases),'attempted_cases':len(rows),
            'verified_responses':sum(r['status']=='verified_rss_response' for r in rows),
            'cases_with_articles':sum(bool(r['articles']) for r in rows),
            'eligible_articles':sum(len(r['articles']) for r in rows),
            'by_year':{str(y):{'cases':len(g),'nonempty':int(g.articles.map(bool).sum())}
                       for y,g in data.groupby('listing_year')},
            'input_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in LEDGERS},
            'output_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            'limitations':['Secondary retrospective search index; original publisher clocks and immutable text not independently verified.',
               'Empty search results are not proof that no company news existed.',
               'Same-day headlines excluded conservatively; no post-decision or future news is included.',
               'No sentiment label, price, news article or IPO training row was synthesized.']}
    REPORT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-per-year',type=int,default=0)
    collect(parser.parse_args().pilot_per_year)
