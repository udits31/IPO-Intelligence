"""Recover clocked stage cohorts from existing local archives only. No OCR/downloads."""
import gzip
import hashlib
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from ml.test_historical_joins_and_cutoff_20261028 import ROOT,SOURCE,sha
from ml.stage_alignment_bridge import parse_ipoji,complete
from ml.recover_progressive_totals import cached_pages
from ml.recover_module1_other_sources import ipoji_payload
from bs4 import BeautifulSoup

OUT=ROOT/'output/module1_stage_matched_20261028'


def stage_for(day,ts,opened,closed):
    ts=pd.Timestamp(ts)
    if ts.tzinfo is None:return None
    ts=ts.tz_convert('Asia/Kolkata')
    if not opened<=ts.date()<=closed:return None
    minute=ts.hour*60+ts.minute
    if ts.date()==closed:
        if 600<=minute<1020:return 'CLOSING_PRE17'
        return None
    if day not in (1,2):return None
    if day==1 and ts.date()!=opened:return None
    if day==2 and ts.date()<=opened:return None
    if 600<=minute<1020:return f'DAY{day}_INTRADAY'
    if minute>=1020:return f'DAY{day}_EOD'
    return None


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    frame=pd.read_csv(SOURCE);frame=frame[frame.cohort.eq('historical')].copy()
    frame['ipo_id']=pd.to_numeric(frame.ipo_id).astype(int)
    lookup=frame.set_index('ipo_id');rows=[];quotes=[];manifest={};rejected=defaultdict(int);identities={}
    directories=['module1_later_holdout_20261024','module1_other_sources_20261024','module1_public_financials_20261024','final_lock_day2_20261026']
    for directory in directories:
        for path in sorted((ROOT/'data'/'archive'/directory).glob('*.json.gz')):
            try:key=int(path.name.split('.')[0])
            except ValueError:continue
            if key not in lookup.index:continue
            try:value=json.loads(gzip.decompress(path.read_bytes()))
            except (OSError,ValueError):rejected['invalid_archive']+=1;continue
            if directory=='final_lock_day2_20261026' and not value.get('audit',{}).get('identity_offer_dates_and_cap_verified'):continue
            original=lookup.loc[key]
            opened=pd.to_datetime(original.issue_open_date,errors='coerce');closed=pd.to_datetime(original.issue_close_date,errors='coerce')
            if pd.isna(opened) or pd.isna(closed):rejected['missing_offer_dates']+=1;continue
            for html,url in cached_pages(value):
                history=parse_ipoji(html)
                if not history:continue
                content_sha=hashlib.sha256(html.encode()).hexdigest()
                identity_key=(key,content_sha)
                if identity_key not in identities:
                    identities[identity_key]=ipoji_payload(dict(company_name=original.company_name,issue_price=original.issue_price,issue_open_date=opened,issue_close_date=closed),BeautifulSoup(html,'html.parser'))
                if not identities[identity_key]:rejected['issuer_dates_cap_identity_not_verified']+=1;continue
                for quote in identities[identity_key].get('gmp',[]):
                    try:
                        qt=pd.Timestamp(quote['dateTime']);value=float(quote['value'])
                        if qt.tzinfo is None or not np.isfinite(value):continue
                        quotes.append(dict(ipo_id=key,timestamp=qt.tz_convert('Asia/Kolkata').isoformat(),gmp_rs=value,source_url=url,source_sha256=content_sha,archive=str(path.relative_to(ROOT)),time_basis='publisher_reported_graph_clock'))
                    except (KeyError,TypeError,ValueError):continue
                for h in history:
                    if not complete(h):rejected['incomplete_categories']+=1;continue
                    ts=pd.Timestamp(h['timestamp']);stage=stage_for(h['day'],ts,opened.date(),closed.date())
                    if stage is None:rejected['outside_stage_or_postclose']+=1;continue
                    if pd.Timestamp(original.listing_date).date()<=ts.date():rejected['not_prelisting']+=1;continue
                    record=original.to_dict();record.update(ipo_id=key,stage=stage,stage_day=h['day'],stage_timestamp=ts.isoformat(),
                        stage_date=str(ts.date()),stage_source_url=url,stage_source_sha256=hashlib.sha256(html.encode()).hexdigest(),
                        stage_archive=str(path.relative_to(ROOT)),availability_basis='publisher_reported_table_recovered_later')
                    for cat in ['qib','hni','retail']:record['stage_'+cat]=h[cat]
                    record['stage_total']=h.get('total');rows.append(record)
                    manifest[str(path.relative_to(ROOT))]=sha(path)
    print(json.dumps(dict(recovered_rows=len(rows),rejections=dict(rejected))),flush=True)
    allrows=pd.DataFrame(rows)
    accepted=[];conflicts=[]
    for (id,stage),g in allrows.groupby(['ipo_id','stage']):
        stamp=g.stage_timestamp.max();latest=g[g.stage_timestamp.eq(stamp)]
        cats=['stage_qib','stage_hni','stage_retail']
        if not np.allclose(latest[cats].to_numpy(float),latest[cats].iloc[0].to_numpy(float),atol=.02,rtol=.001):
            conflicts.append(dict(ipo_id=int(id),stage=stage,timestamp=stamp));continue
        accepted.append(latest.iloc[0])
    result=pd.DataFrame(accepted).reset_index(drop=True)
    result.to_csv(OUT/'clocked_stage_rows.csv',index=False)
    pd.DataFrame(quotes).drop_duplicates().to_csv(OUT/'clocked_gmp_quotes.csv',index=False)
    report=dict(source_sha256=sha(SOURCE),manifest=manifest,archive_count=len(manifest),raw_clocked_rows=len(allrows),distinct_stage_rows=len(result),
        coverage={stage:{str(int(year)):len(g) for year,g in part.groupby('year')} for stage,part in result.groupby('stage')},
        excluded=dict(rejected),conflicts=conflicts,
        limitations=['Source-displayed clocks recovered retrospectively; no claim that these were immutable live captures.','No final-day EOD row is treated as pre-17:00. Date-only Day3 totals are not clocked pre-close observations.','Latest eligible observation per issuer/stage, chosen without outcomes.'])
    (OUT/'coverage.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='manifest'},indent=2))


if __name__=='__main__':run()
