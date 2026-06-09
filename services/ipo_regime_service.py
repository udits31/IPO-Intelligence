"""Causal, issuer-level regime tracker; future outcomes never enter a score."""
import numpy as np
import pandas as pd


def rolling_regime(history,decision_date,k=10,exclude_issuer=None):
    frame=history.copy();cutoff=pd.Timestamp(decision_date).tz_localize(None).normalize()
    frame['listing_date']=pd.to_datetime(frame.listing_date).dt.tz_localize(None)
    frame=frame[frame.listing_date.lt(cutoff)]
    if exclude_issuer is not None:frame=frame[frame.issuer_key.ne(str(exclude_issuer))]
    if frame.issuer_key.duplicated().any():raise ValueError('One verified opening per issuer required')
    recent=frame.sort_values(['listing_date','issuer_key']).tail(k)
    quotes=recent[recent.gmp_pct.notna()&recent.gmp_preclose_verified.eq(True)]
    slippage=float(np.median(quotes.actual_return_pct-quotes.gmp_pct)) if len(quotes) else None
    return dict(rolling_10_hit_rate=float(recent.actual_return_pct.gt(0).mean()) if len(recent) else None,
        rolling_10_gmp_slippage=slippage,prior_listings=len(recent),paired_gmp_listings=len(quotes),
        prior_issuer_keys=recent.issuer_key.tolist(),latest_prior_listing=str(recent.listing_date.max().date()) if len(recent) else None)


def adjusted_gmp(raw,slippage):
    return float(raw)+min(0.,.5*float(slippage)) if raw is not None and pd.notna(raw) and slippage is not None and pd.notna(slippage) else raw
