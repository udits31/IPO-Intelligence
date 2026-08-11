"""M2 R1: existing price decisions plus an eligible adverse-filing override."""
import math
from services.point_in_time import eligible

VERSION='M2_R1_WITH_DATED_FILINGS'


def filing_governance(records,decision_timestamp):
    admitted=[r for r in records if eligible(r,decision_timestamp)]
    scores=[]
    for r in admitted:
        try:
            value=float(r.get('filing_sentiment_score'))
            if math.isfinite(value):scores.append(value)
        except (TypeError,ValueError):pass
    score=sum(scores)/len(scores) if scores else None
    hard=any(r.get('hard_governance_red_flag') is True for r in admitted)
    return dict(policy=VERSION,admitted_filings=len(admitted),filing_sentiment_score=score,
        hard_governance_red_flag=hard,override_sell=hard or (score is not None and score<=-.30),
        evidence=[{k:r.get(k) for k in ('source_url','sha256','source_timestamp','broadcast_datetime_ist','subject')} for r in admitted])
