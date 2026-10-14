"""Runtime share-basis checks use real archived Yahoo and NSE observations."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.attribution_engine import parse_yahoo_chart
from services.corporate_action_service import PriceBasisError, reconcile_prices
from services.module2_dynamic_monitor import issuer_matches
from services.module2_dynamic_monitor import evaluate_dynamic_listing
from services.bse_identity import archived_bse_identity
from core.attribution_engine import AttributionEngine
from services.corporate_action_service import official_actions
from ml.evaluate_module2_prospective import comparable_outcome_closes

ROOT=Path(__file__).resolve().parents[1]
ARCHIVES={
    659:'c6b6cd06cbddbd01b961eba043c1f47d233f8ee871e791062a251b19fcc72598',
    661:'212b7a864d51a995cfded3a3705f4135d1cceebbe550b8f6d51d41263feacb0a',
}


def observed(ipo_id):
    path=ROOT/'data/archive/yahoo_chart'/f'{ARCHIVES[ipo_id]}.json.gz'
    if not path.exists():
        pytest.skip('Actual Yahoo source response not installed')
    with gzip.open(path,'rb') as stream:
        raw=stream.read()
    assert hashlib.sha256(raw).hexdigest()==ARCHIVES[ipo_id]
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id').loc[ipo_id]
    data=parse_yahoo_chart(json.loads(raw),canonical.nse_symbol+'.NS')
    data=data.rename(columns={'close':'close_price_adjusted'})
    data['price_source']='Yahoo Finance (NSE)'
    data['price_basis']='Unadjusted daily close'
    return data,canonical,data.attrs['provider_meta']


@pytest.mark.parametrize('ipo_id,date,factor',[(659,'2026-09-28',2.),(661,'2026-08-25',10.)])
def test_explicit_real_provider_events_match_official_ratios(ipo_id,date,factor):
    data,ipo,meta=observed(ipo_id)
    assert meta['splits']==[{'ex_date':date,'share_factor':factor,'numerator':factor,'denominator':1.}]
    data=data[data.date.le('2026-10-20')]
    result,issue,evidence=reconcile_prices(data,ipo,asof='2026-10-20',provider_meta=meta)
    assert evidence['share_factor']==factor
    assert issue==ipo.issue_price/factor
    assert evidence['event_authority']=='NSE_action_records_cross_checked'
    # Yahoo history is already adjusted: a second adjustment would be wrong.
    np.testing.assert_array_equal(result.close_price_adjusted,data.close_price_adjusted)
    assert result.iloc[-1].close_price_adjusted==data.iloc[-1].close_price_adjusted


def test_pre_split_replay_undoes_only_future_provider_adjustments():
    data,ipo,meta=observed(661)
    data=data[data.date.le('2026-08-12')]
    result,issue,evidence=reconcile_prices(data,ipo,asof='2026-08-12',provider_meta=meta)
    assert issue==808. and evidence['share_factor']==1.
    assert np.isclose(result.iloc[-1].close_price_adjusted,825.50,atol=.001)
    np.testing.assert_allclose(result.close_price_adjusted,data.close_price_adjusted*10)


def test_actual_nse_and_yahoo_have_comparable_return_bases():
    yahoo,ipo,meta=observed(661)
    raw=ROOT/'data/audit/module2_holdout_prices_2025_2026_nse_full.csv'
    if not raw.exists():
        pytest.skip('Actual NSE histories not installed')
    nse=pd.read_csv(raw,parse_dates=['date'])
    nse=nse[nse.ipo_id.eq(661) & nse.date.le('2026-09-24')]
    yahoo=yahoo[yahoo.date.le('2026-09-24')]
    a,issue,_=reconcile_prices(nse,ipo,asof='2026-09-24')
    b,other_issue,_=reconcile_prices(yahoo,ipo,asof='2026-09-24',provider_meta=meta)
    paired=a[['date','close_price_adjusted']].merge(b[['date','close_price_adjusted']],on='date')
    np.testing.assert_allclose(paired.close_price_adjusted_x,paired.close_price_adjusted_y,atol=.001)
    assert issue==other_issue==80.8


def test_missing_provider_event_is_not_silently_treated_as_no_split():
    data,ipo,meta=observed(661)
    with pytest.raises(PriceBasisError,match='conflict'):
        reconcile_prices(data,ipo,asof='2026-10-20',provider_meta={**meta,'splits':[]})


def test_dynamic_listing_requires_distinct_issuer_not_shared_brand():
    assert issuer_matches('National Securities Depository Ltd.','National Securities Depository Limited')
    assert not issuer_matches('Kalpataru Ltd.','Kalpataru Projects International Limited')
    assert not issuer_matches('Indian Railway Finance Corporation','Indian Railways')


def test_prospective_outcomes_use_actual_dated_share_entitlements():
    raw=ROOT/'data/audit/module2_holdout_prices_2025_2026_nse_full.csv'
    corrected=raw.with_name(raw.stem+'_ca_reconciled.csv')
    if not raw.exists() or not corrected.exists():
        pytest.skip('Actual NSE raw/reconciled histories not installed')
    source=pd.read_csv(raw)
    checked=pd.read_csv(corrected)
    for ipo_id,symbol in [(659,'AASTHA'),(661,'CORDELIA')]:
        prices=source[source.ipo_id.eq(ipo_id)].sort_values('date')
        prices=prices.rename(columns={'close_price_adjusted':'nse_close'})
        decision=prices.iloc[30].date
        events,_=official_actions(symbol,decision,prices.iloc[61].date,allow_network=False)
        actual=comparable_outcome_closes(prices.iloc[30:62],events,decision)
        reference=checked[checked.ipo_id.eq(ipo_id)].sort_values('date').iloc[30:62]
        np.testing.assert_allclose(actual.to_numpy(),reference.close_price_adjusted.to_numpy())


def test_bse_route_uses_official_symbol_and_the_shared_monitor(monkeypatch):
    identity=archived_bse_identity('544467','2025-08-06',571)
    assert identity['symbol']=='NSDL'  # Not the non-existent Yahoo ticker 544467.BO.
    paths={
        'NSDL.BO':'70432116df4ba38aaaa9d0640b430df0428c7cda9ca31bc31922872aadd69b7b',
        '^NSEI':'dbe3ec40d647ff1864bb8a1689b60fbda07cdad17535ea524a04945694a20125',
    }
    actual={}
    for symbol,digest in paths.items():
        path=ROOT/'data/archive/yahoo_chart'/f'{digest}.json.gz'
        if not path.exists():
            pytest.skip('Actual BSE/Nifty source snapshots not installed')
        with gzip.open(path,'rb') as stream:
            raw=stream.read()
        assert hashlib.sha256(raw).hexdigest()==digest
        actual[symbol]=parse_yahoo_chart(json.loads(raw),symbol)
        actual[symbol].attrs['provider_meta'].update(raw_sha256=digest,raw_archive_path=str(path))
    monkeypatch.setattr(AttributionEngine,'_fetch_yahoo_series',lambda symbol,**kwargs:actual[symbol].copy())
    monkeypatch.setattr(AttributionEngine,'_get_sector_series',lambda sector:(actual['^NSEI'].copy(),'Nifty 50 broad-market proxy'))
    entry=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id',drop=False).loc[571]
    data={key:None if pd.isna(value) else value for key,value in entry.to_dict().items()}
    result=evaluate_dynamic_listing(data,target=entry.company_name,canonical_entry=entry,as_of_date='2026-10-20')
    assert result['monitor_policy']=='shared_core_monitor'
    assert result['exchange']=='BSE' and result['bse_code']=='544467'
    assert result['nse_symbol'] is None
    assert result['price_basis_adjustment']['event_authority']=='secondary_provider_event_records'
    assert result['price_only_verdict'] and result['verdict']
    assert result['current_price_adjusted']>0
