"""Keep six-case diagnostics and the requested 2025 extension reproducible."""
import hashlib
import json
import numpy as np
from ml import audit_module1_six_and_2025 as audit


def report():
    return json.loads(audit.OUT.read_text(encoding='utf-8'))


def test_all_six_frozen_scores_reproduce_without_rewriting_live_bundle():
    result=report()
    bundle=json.loads((audit.ROOT/'ml/model_bundle.json').read_text(encoding='utf-8'))
    assert len(result['six_ipos'])==6
    assert sum(row['correct'] for row in result['six_ipos'])==3
    for row in result['six_ipos']:
        replay=audit.explain(bundle,row['qib_x'],row['retail_x'],row['hni_x'])
        assert np.isclose(replay['probability']*100,row['probability_pct'],atol=1e-9)
        assert row['source_hash_verified']
    assert result['live_model_changed'] is False
    assert result['input_sha256']['ml/model_bundle.json']==hashlib.sha256((audit.ROOT/'ml/model_bundle.json').read_bytes()).hexdigest()


def test_bonfiglioli_is_a_flat_listing_miss_not_a_bad_probability_replay():
    row=next(row for row in report()['six_ipos'] if row['ipo_id']=='chitto_2861')
    assert row['issue_price']==row['listing_open']==586.
    assert row['correct'] is False
    assert row['explanation']['contributions_log_odds']['QIB']>0
    assert row['explanation']['contributions_log_odds']['Retail']<0
    assert row['probability_pct']>90


def test_2025_extension_has_identical_2026_test_and_no_tuned_cutoff():
    study=report()['adding_2025']
    rows=study['comparisons']
    assert study['test_year']==2026 and study['test_n']==75
    assert study['fixed_direction_threshold']==.5
    assert len(study['excluded_2025_noncorporate'])==4
    for label in ('canonical','exchange_verified'):
        pair=[row for row in rows if row['label_source']==label]
        assert [row['training_n'] for row in pair]==[252,356]
        assert all(row['result']['sample_size']==75 for row in pair)
        assert pair[1]['trained_through']==2025
    canonical=[row for row in rows if row['label_source']=='canonical']
    assert canonical[0]['result']['classifier_accuracy_pct']==80.
    assert canonical[1]['result']['classifier_accuracy_pct']==78.7


def test_high_probability_calibration_keeps_failures_in_denominator():
    rows=report()['calibration_2025_2026']
    assert sum(row['n'] for row in rows)==179
    assert sum(row['positive_count'] for row in rows)==120
    assert rows[-1]['n']==61 and rows[-1]['positive_count']==57
