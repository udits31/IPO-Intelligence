"""Shared provenance and cutoff checks for enrichment research."""
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import math

IST=ZoneInfo('Asia/Kolkata')


def timestamp(value):
    try:
        t=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        return t.astimezone(IST) if t.tzinfo else None
    except (ValueError,TypeError):return None


def eligible(record,decision_timestamp):
    source=timestamp(record.get('source_timestamp'))
    cutoff=timestamp(decision_timestamp)
    if source and record.get('source_timestamp_verified') is True:
        return bool(cutoff and source<=cutoff)
    # A successful public fetch bounds availability for later decisions only.
    observed=timestamp(record.get('available_by_timestamp'))
    return bool(observed and cutoff and observed<=cutoff and record.get('observation_verified') is True)


def finite(value):
    try:return float(value) if math.isfinite(float(value)) else None
    except (ValueError,TypeError):return None


def evidence(value,*,source_url,sha256,source_timestamp=None,source_timestamp_verified=False,original_fetch_timestamp=None,basis='local_cache_extraction'):
    return dict(value=value,source_url=source_url,sha256=sha256,source_timestamp=source_timestamp,
        source_timestamp_verified=source_timestamp_verified,fetch_timestamp=datetime.now(timezone.utc).isoformat(),
        original_fetch_timestamp=original_fetch_timestamp,provenance_basis=basis)
