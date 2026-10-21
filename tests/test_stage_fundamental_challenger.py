"""No future earnings, OFS misreading or cross-stage coefficients."""
import json
import hashlib
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from ml.extract_dated_ipo_fundamentals import eps_rows,pure_offer_ratio
from services.dated_ipo_fundamentals import at_decision,load_registry,REGISTRY
from services.module1_stage_challenger import score_stage_challenger
from ml.train_stage_fundamental_challenger import OUT


def test_offer_for_sale_proceeds_sentence_does_not_mean_entire_offer_is_ofs():
    assert pure_offer_ratio('We will not receive any proceeds from the Offer for Sale.') is None
    assert pure_offer_ratio('The Offer is entirely an Offer for Sale of shares.')==1.
    assert pure_offer_ratio('Our Company will not receive any proceeds from the Offer.')==1.
    assert pure_offer_ratio('The Issue consists of solely a fresh issue.')==0.


def test_annual_eps_is_not_the_unannualised_six_month_number():
    text='''Basic and Diluted Earnings Per Equity Share
    Financial Year ended Basic EPS Diluted EPS Weight
    March 31, 2024 11.00 11.00 3
    March 31, 2023 8.52 8.52 2
    March 31, 2022 3.94 3.94 1
    September 30, 2024 8.75 8.75 -
    '''
    rows=eps_rows(text)
    assert rows[0]['diluted_eps']==11.
    assert rows[1]['diluted_eps']==8.52


def test_existing_facts_are_rejected_if_dated_after_decision():
    payload=json.loads(REGISTRY.read_text())
    record=next(row for row in payload['records'] if 'annual_diluted_eps' in row.get('facts',{}))
    assert at_decision(record,record['document_date'],100.)['financial_observed'] is False
    future=pd.Timestamp(record['document_date'])+pd.Timedelta(days=1)
    result=at_decision(record,future,100.)
    assert result['financial_observed'] is True
    assert np.isclose(result['earnings_yield_pct'],record['facts']['annual_diluted_eps'])
    assert at_decision(None,future,100.)['earnings_yield_pct'] is None


def test_modified_eps_cannot_pass_using_an_unchanged_pdf_hash(tmp_path):
    payload=json.loads(REGISTRY.read_text())
    record=deepcopy(next(row for row in payload['records'] if 'annual_diluted_eps' in row.get('facts',{})))
    record['facts']['annual_diluted_eps']+=100.
    path=tmp_path/'tampered.json'
    path.write_text(json.dumps({'records':[record]}))
    with pytest.raises(ValueError,match='EPS does not reproduce'):
        load_registry(path)


def test_arbitrary_intraday_stage_cannot_borrow_a_final_book_model():
    with pytest.raises(ValueError,match='No trained model'):
        score_stage_challenger(ipo_id=1,stage='opening_day_11am',
            decision_at='2026-10-23T11:00:00+05:30',source_at='2026-10-23T10:50:00+05:30',
            qib=1.,retail=1.,hni=1.)


def test_challenger_retains_all_cases_and_correct_baseline_denominator():
    report=json.loads(OUT.read_text())
    ledger=pd.read_csv(OUT.with_name(OUT.stem+'_predictions.csv'))
    test=ledger[ledger.year.eq(2024)]
    assert len(test)==75 and test.ipo_id.nunique()==75
    assert report['live_promoted'] is False
    for row in report['trace']:
        assert row['train_latest_outcome']<row['test_first_decision']
    for name,item in report['results'].items():
        correct=test[name].ge(.5).eq(test.target_binary.astype(bool))
        expected=(correct.mean()-test.target_binary.mean())*100
        assert np.isclose(item['vs_always_positive']['change_pp'],expected)
        assert item['metrics']['sample_size']==75


def test_gmp_comparison_uses_identical_subset_for_every_model():
    report=json.loads(OUT.read_text())
    ledger=pd.read_csv(OUT.with_name(OUT.stem+'_predictions.csv'))
    subset=ledger[ledger.year.eq(2024)&ledger.gmp_pct.notna()]
    actual=subset.target_binary.astype(bool)
    comparison=report['paired_gmp']
    assert len(subset)==comparison['n']
    gmp_accuracy=subset.gmp_pct.gt(0).eq(actual).mean()*100
    assert np.isclose(comparison['gmp_accuracy_pct'],gmp_accuracy)
    assert set(comparison['tracks'])==set(report['results'])
    for name,item in comparison['tracks'].items():
        accuracy=subset[name].ge(.5).eq(actual).mean()*100
        assert np.isclose(item['accuracy_pct'],accuracy)
        assert np.isclose(item['vs_gmp']['change_pp'],accuracy-gmp_accuracy)


def test_saved_report_binds_current_training_and_fundamental_inputs():
    from ml.train_stage_fundamental_challenger import ROOT
    report=json.loads(OUT.read_text())
    for relative,expected in report['input_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    assert report['document_coverage']['processed']==441
    assert report['coverage']['confirmation_financial_observed']==56
    ledger=pd.read_csv(OUT.with_name(OUT.stem+'_predictions.csv'))
    assert report['coverage']['confirmation_offer_observed']==int(ledger.loc[ledger.year.eq(2024),'offer_observed'].sum())
