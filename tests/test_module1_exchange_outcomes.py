"""Use real raw NSE/BSE archives; no generated IPO or price training rows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.audit_module1_listing_opens import OUTPUT, ROOT, verified_outcomes
from services import return_forecast_service as service


@pytest.fixture(scope='module')
def audited():
    if not OUTPUT.exists():
        pytest.skip('Actual exchange opening audit is not installed')
    return verified_outcomes()


def test_every_original_training_ipo_has_an_original_exchange_record(audited):
    training=audited[audited.year.between(2021,2024)]
    assert len(training)==252
    assert not training.ipo_id.duplicated().any()
    assert set(training.exchange)=={'NSE','BSE'}
    for ipo_id,symbol in [(283,'GLS'),(285,'ZOMATO'),(315,'AHL'),(347,'RUCHI')]:
        row=training.set_index('ipo_id').loc[ipo_id]
        assert row.symbol==symbol
        evidence=json.loads(row.symbol_change_evidence)
        assert evidence and row.listing_date < pd.Timestamp(evidence[0]['effective'])
    protean=training.set_index('ipo_id').loc[369]
    assert protean.exchange=='BSE' and protean.exchange_open==792.


def test_rescoring_does_not_rewrite_saved_predictions_or_claim_gmp_edge():
    path=ROOT/'output/module1_exchange_outcome_replay_20261021_frozen_forecasts_ledger.csv'
    if not path.exists():
        pytest.skip('Exchange-source rescoring is not installed')
    old=pd.read_csv(ROOT/'output/module1_return_prediction_ledger.csv',float_precision='round_trip').sort_values(['track','ipo_id'])
    new=pd.read_csv(path,float_precision='round_trip').sort_values(['track','ipo_id'])
    assert list(old.ipo_id)==list(new.ipo_id)
    for column in ['prediction','lower','upper','probability','gmp_pct','ranking_eligible']:
        np.testing.assert_array_equal(old[column],new[column])
    report=json.loads(service.OUTCOME_AUDIT.read_text(encoding='utf-8'))
    assert report['model_promoted'] is False
    assert not report['original_return_forecasts_rescored']['bids_plus_gmp']['ranking_2024']['better_than_gmp_established']


def test_app_displays_only_a_matching_source_audit(monkeypatch,tmp_path):
    if not service.OUTCOME_AUDIT.exists():
        pytest.skip('Exchange-source rescoring is not installed')
    report_copy=tmp_path/'benchmark.json'
    original=service.REPORT.read_text(encoding='utf-8')
    report_copy.write_text(original,encoding='utf-8')
    monkeypatch.setattr(service,'REPORT',report_copy)
    current=service.return_benchmark()['exchange_outcome_audit']
    assert current['status']=='verified_source_recheck'
    assert current['forecasts_refitted'] is False
    report_copy.write_text(original+'\n',encoding='utf-8')
    stale=service.return_benchmark()['exchange_outcome_audit']
    assert stale['status']=='outdated' and 'tracks' not in stale
