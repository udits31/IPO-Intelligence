"""Point-in-time checks shared by the live frozen-model input assembler."""
import gzip
import hashlib
import math
import pandas as pd

VERSION='module1-input-contract-20261028-v2'


def valid_categories(book):
    try:return all(math.isfinite(float(book[k])) and float(book[k])>=0 for k in ('qib_x','nii_x','retail_x'))
    except (KeyError,TypeError,ValueError):return False


def growth_reference(entries,current):
    """Use the prior published book day, never the analysis clock or itself."""
    current_time,book=current
    candidates=[(time,row) for time,row in entries if row.get('source_family')==book.get('source_family')
        and row.get('id')!=book.get('id') and time.date()<current_time.date()]
    return max(candidates,key=lambda pair:pair[0]) if candidates else None


def financial_cache_eligible(fact,archive,asof):
    try:
        captured=pd.Timestamp(fact.get('captured_at_ist'))
        period=pd.Timestamp(fact.get('annual_period'))
        if pd.isna(captured) or captured.tzinfo is None or captured>pd.Timestamp(asof):return False
        if pd.isna(period) or period.date()>=asof.date():return False
        return hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest()==fact.get('raw_sha256')
    except (OSError,ValueError,TypeError):return False


def stage_contract(book_time,asof,cutoff):
    if book_time is None:return dict(training_stage='penultimate_session',live_stage='no_verified_book',comparable=False)
    stage='closing_day' if cutoff and book_time.date()==cutoff.date() else 'earlier_bidding_session'
    return dict(training_stage='penultimate_session',live_stage=stage,
        book_age_days=(asof.date()-book_time.date()).days,
        comparable=False,
        reason='Exact training-stage equivalence is not established by a generic live snapshot; continuous forecasts are logged for prospective scoring.')
