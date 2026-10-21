"""Saved probabilities, not rewritten predictions, determine the teacher table."""
import pytest
from services import prediction_results_service as service


@pytest.mark.parametrize('probability,issue,opening,direction,result', [
    (.6092317744857458, 271, 335, 'Positive', 'Correct'),
    (.26071216465482555, 289, 305, 'Positive', 'Incorrect'),
    (.4, 100, 90, 'Negative', 'Correct'),
    (.8, 100, 90, 'Negative', 'Incorrect'),
    (.4, 100, 100, 'Flat', 'Correct'),
    (.5, 100, 100, 'Flat', 'Incorrect'),
    (.8, 100, None, 'Pending', 'Pending'),
    (.8, 100, 0, 'Pending', 'Pending'),
])
def test_opening_direction(probability, issue, opening, direction, result):
    row = service.result_row(company='Test issuer', probability=probability,
                             issue=issue, opening=opening, evidence_kind='test')
    assert row['actual_direction'] == direction
    assert row['result'] == result
    assert row['actual_return_pct'] == (None if result == 'Pending' else round((opening/issue-1)*100, 3))


@pytest.mark.parametrize('probability', [float('nan'), float('inf'), -.01, 1.01])
def test_invalid_probabilities_rejected(probability):
    with pytest.raises(ValueError):
        service.result_row(company='Test', probability=probability, evidence_kind='test')


def test_latest_only_and_no_old_capture_masquerading_as_preclose(monkeypatch, tmp_path):
    def record(saved, probability):
        return dict(ipo_id='issuer', company_name='Test', recorded_at_ist=saved,
                    source_asof_ist='2026-10-19T16:00:00+05:30',
                    positive_listing_probability=probability, model_version='saved-v1',
                    inputs={'application_cutoff_at_ist':'2026-10-19T17:00:00+05:30'})
    # Deliberately out of order. A later saved call cannot inherit the old input time.
    monkeypatch.setattr(service, 'get_intraday_forecast_scorecard', lambda: {'daily_records':[
        record('2026-10-19T16:00:00+05:30', .1), record('2026-10-20T10:00:00+05:30', .9)]})
    monkeypatch.setattr(service, 'get_prelisting_outcome_scorecard', lambda: {'records':[]})
    monkeypatch.setattr(service, 'verified_listing_outcomes', lambda: {'issuer':{
        'issue_price':100., 'nse_open_price':110.}})
    monkeypatch.setattr(service, 'ROOT', tmp_path)
    result = service.prediction_results()
    assert len(result['recent_records']) == 1
    assert result['recent_records'][0]['positive_probability_pct'] == 90
    assert result['recent_records'][0]['evidence_kind'] == 'Timing not verified'
    assert result['summaries']['saved_preclose']['total'] == 0
    assert result['summaries']['recent_replays']['correct'] == 1


def test_checked_historical_ledger_matches_deployed_model():
    result = service.prediction_results()
    assert result['historical_status'] == 'verified_saved_model_replay'
    rows = result['historical_records']
    assert len(rows) == 179
    assert len({row['company_name'] for row in rows}) == 179
    assert all(row['predicted_at_ist'] is None for row in rows)
    assert all(row['outcome_source_sha256'] for row in rows)
    assert result['summaries']['historical']['correct'] == 136
    assert result['summaries']['historical']['accuracy_pct'] == 76.0


def test_prospective_excludes_training_members_and_late_calls():
    manifest=dict(model_version='saved-v1',trained_at_utc='2026-10-18T00:00:00Z',
                  ids={'12'},names={service.identity_name('Training Company Ltd.')})
    def record(ipo_id, name, saved, probability=.7):
        return dict(ipo_id=ipo_id,company_name=name,model_version='saved-v1',
                    recorded_at_ist=saved,source_asof_ist='2026-10-19T16:00:00+05:30',
                    positive_listing_probability=probability,
                    inputs={'application_cutoff_at_ist':'2026-10-19T17:00:00+05:30','issue_price':100.})
    records=[record('new','New Issuer','2026-10-19T17:01:00+05:30',.1),
             record('new','New Issuer','2026-10-19T16:45:00+05:30'),
             record('new','New Issuer','2026-10-19T16:15:00+05:30',.6),
             record('renamed_id','Training Company Limited','2026-10-19T16:30:00+05:30'),
             record('12','Renamed Training Issuer','2026-10-19T16:30:00+05:30')]
    outcomes={'new':{'issue_price':100.,'nse_open_price':110.,'listing_date':'2026-10-21'}}
    rows=service.prospective_results(records,outcomes,manifest)
    assert len(rows)==1
    assert rows[0]['positive_probability_pct']==70.
    assert rows[0]['used_in_training'] is False
    assert rows[0]['predicted_at_ist']=='2026-10-19T16:45:00+05:30'
    assert rows[0]['result']=='Correct'
    records[1]['model_version']='unknown-bundle'
    records[2]['source_asof_ist']='2026-10-19T18:00:00+05:30'
    assert service.prospective_results(records,outcomes,manifest)==[]


def test_actual_prospective_membership_matches_saved_training_fingerprint():
    manifest=service.training_manifest()
    assert manifest['training_n']==252
    result=service.prediction_results()
    assert result['training_membership_status']=='verified'
    assert result['prospective_records']
    assert all(row['used_in_training'] is False for row in result['prospective_records'])
    assert all(row['evidence_kind']=='Saved before application cutoff' for row in result['prospective_records'])


def test_replay_percentage_uses_all_displayed_verified_openings():
    from services.snapshot_service import get_prelisting_outcome_scorecard
    scorecard=get_prelisting_outcome_scorecard()
    resolved=[row for row in scorecard['records'] if row['direction_correct'] is not None]
    assert scorecard['summary']['openings_verified']==len(resolved)
    assert scorecard['summary']['direction_accuracy_pct']==round(100*sum(row['direction_correct'] for row in resolved)/len(resolved),1)
