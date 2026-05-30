"""Availability semantics: event clocks, observations, and explicit assumptions."""
from datetime import datetime,time,timedelta
from services.point_in_time import IST,timestamp


def availability(*,kind,decision_timestamp,source_timestamp=None,fetch_timestamp=None,
                 issue_open_date=None,trading_date=None,allow_historical_assumptions=False):
    cutoff=timestamp(decision_timestamp)
    if cutoff is None:raise ValueError('Timezone-aware decision timestamp required')
    source=timestamp(source_timestamp);effective=source;basis='verified_event_clock' if source else 'unknown'
    assumed=False
    if source is None and allow_historical_assumptions:
        if kind=='PROSPECTUS_STATIC' and issue_open_date:
            effective=datetime.combine(datetime.fromisoformat(str(issue_open_date)[:10]).date(),time(9),IST)
            basis='assumed_original_offer_terms_public_before_open';assumed=True
        elif kind=='EXCHANGE_EOD_TAPE' and trading_date:
            # A same-day trading date cannot prove publication by 15:30.
            # Archive-vintage sensitivity uses the next calendar morning.
            effective=datetime.combine(datetime.fromisoformat(str(trading_date)[:10]).date()+timedelta(days=1),time(9),IST)
            basis='historical_archive_available_next_morning_assumption';assumed=True
    if kind=='CAUSAL_PRIOR_TRACK_RECORD':
        # Caller must separately enforce every source outcome predates decision.
        effective=cutoff;basis='derived_from_prior_verified_outcomes';assumed=False
    return dict(effective_available_ts=effective.isoformat() if effective else None,
        source_timestamp=source_timestamp,fetch_timestamp=fetch_timestamp,
        eligible=bool(effective and effective<=cutoff),availability_basis=basis,availability_assumed=assumed)
