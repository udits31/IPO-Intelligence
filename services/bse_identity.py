"""Exact BSE scrip-to-symbol identity from a hash-checked exchange archive."""
import hashlib
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'data/audit/module1_exchange_listing_open_audit.csv'


def archived_bse_identity(code,listed,ipo_id):
    if not AUDIT.exists():
        return None
    records=pd.read_csv(AUDIT)
    rows=records[records.ipo_id.eq(ipo_id) & records.exchange.eq('BSE') &
                 records.status.eq('verified_daily_exchange_archive')]
    if len(rows)!=1:
        return None
    row=rows.iloc[0]
    if int(float(row.bse_scrip_code))!=int(code) or pd.Timestamp(row.listing_date)!=pd.Timestamp(listed):
        raise ValueError('BSE listing identity conflicts with its archived exchange record')
    path=(ROOT/row.raw_archive_path).resolve()
    path.relative_to((ROOT/'data/archive').resolve())
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row.archive_sha256:
        raise ValueError('BSE identity archive hash mismatch')
    if pd.Timestamp(listed)<pd.Timestamp('2024-07-08'):
        return None  # Legacy SC_NAME is a display name, not a verified ticker.
    data=pd.read_csv(path)
    actual=data[pd.to_numeric(data.FinInstrmId,errors='coerce').eq(int(code))]
    if len(actual)!=1 or actual.iloc[0].Src!='BSE' or pd.Timestamp(actual.iloc[0].TradDt)!=pd.Timestamp(listed):
        raise ValueError('BSE archive code/date/source mismatch')
    symbol=str(actual.iloc[0].TckrSymb).strip()
    return {'code':str(int(code)),'symbol':symbol,'source_url':row.source_url,
            'raw_archive_path':row.raw_archive_path,'raw_sha256':row.archive_sha256}
