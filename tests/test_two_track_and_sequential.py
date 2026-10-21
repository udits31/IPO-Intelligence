import hashlib,json
import numpy as np
import pandas as pd
import pytest
from services.two_track_evidence import build,present,LOCK
from ml.evaluate_module2_sequential_path import first_exit_path,multi_trade_path,dated_filings,replay_calls,ROOT


def test_two_tracks_retain_errors_and_account_for_every_skipped_winner():
    report=build()
    for name,c in report['cohorts'].items():
        m=c['model_direction'];s=c['application_screen']
        assert m['model_direction_correct']+m['model_direction_missed']==c['n']
        assert s['selected_apply_total']+s['skipped_total']+s['data_pending']==c['n']
        assert s['skipped_winners_opportunity_cost']==s['skipped_model_true_positives_vetoed']+s['skipped_model_false_negatives']
        assert all(r['model_direction_correct']==(r['model_predicted_positive']==r['actual_positive']) for r in c['records'])
    assert report['cohorts']['test_69']['model_direction']['model_direction_missed']==13
    assert report['cohorts']['test_69']['application_screen']['selected_apply_losers_admitted']==0
    assert report['cohorts']['recent_23']['model_direction']['raw_positive_calls']==20
    assert report['cohorts']['recent_24_verified']['model_direction']['raw_positive_calls']==21


def test_filter_does_not_rewrite_a_wrong_positive_prediction():
    row=present(.9,False,3)
    assert row['model_outcome']=='Missed' and row['screen_result']=='Loser Avoided'
    assert present(.1,True,3)['screen_result']=='Winner Skipped (Opportunity Cost)'
    assert present(.9,None,1)['model_outcome']=='Pending'


def test_exit_executes_next_close_and_cash_stops_following_prices():
    prices=np.arange(100.,162.);sells=[False]*30;sells[2]=True
    result=first_exit_path(prices,sells)
    assert result['exit_signal_index']==32 and result['execution_index']==33
    assert result['net_return_A_pct']==pytest.approx(100*(prices[33]*.997/prices[31]-1))
    changed=prices.copy();changed[34:]*=10
    assert first_exit_path(changed,sells)['net_return_A_pct']==result['net_return_A_pct']


def test_cost_conventions_and_session_indexing_are_explicit():
    prices=np.arange(100.,162.)
    sold=first_exit_path(prices,[True]+[False]*29)
    assert sold['net_return_A_pct']==pytest.approx(-.3)
    held=first_exit_path(prices,[False]*30)
    assert held['net_return_A_pct']==pytest.approx(100*(prices[61]/prices[31]-1))
    assert held['net_return_B_pct']==pytest.approx(100*(prices[61]*.997/prices[31]-1))
    literal=first_exit_path(prices,[True]+[False]*29,reference_index=30,terminal_index=60)
    assert literal['net_return_A_pct']!=pytest.approx(-.3)


def test_reentry_charges_every_actual_transition():
    prices=np.ones(62)*100;sells=[True,False,True]+[True]*27
    result=multi_trade_path(prices,sells)
    assert result['transition_count']==3
    assert [t['side'] for t in result['trades']]==['SELL','BUY','SELL']
    assert result['net_return_A_pct']==pytest.approx(100*(.997**3-1))


def test_filings_require_both_verified_availability_and_broadcast_clock():
    cutoff='2026-01-02T15:30:00+05:30'
    row=dict(source_timestamp_verified=True,source_timestamp=cutoff,broadcast_datetime_ist='2026-01-02T15:31:00+05:30')
    assert dated_filings([row],cutoff)==[]
    assert dated_filings([{**row,'broadcast_datetime_ist':cutoff}],cutoff)


def test_real_price_prefix_and_future_filings_do_not_change_checkpoint_call():
    prices=pd.read_csv(ROOT/'data/audit/module2_holdout_prices_2025_2026_nse_full_ca_reconciled.csv',parse_dates=['date'])
    group=prices[prices.ipo_id.eq(501)].sort_values('date').reset_index(drop=True)
    ipo=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id').loc[501].to_dict();ipo['id']=501
    before=replay_calls(group,ipo,[])[0]
    changed=group.copy();changed.loc[31:,'close_price_adjusted']*=4
    future=dict(source_timestamp_verified=True,source_timestamp='2027-01-01T15:00:00+05:30',broadcast_datetime_ist='2027-01-01T15:00:00+05:30',filing_sentiment_score=-1.)
    after=replay_calls(changed,ipo,[future])[0]
    assert before==after


def test_export_reproduces_static_anchor_without_modifying_locked_evidence():
    path=ROOT/'output/module2_sequential_path_evaluation_20261026.json'
    if not path.exists():pytest.skip('Run the dated daily-path replay first')
    r=json.loads(path.read_text());locked=json.loads(LOCK.read_text())
    assert r['daily_decision_rows']==132*30
    assert r['locked_evidence_sha256']==hashlib.sha256(LOCK.read_bytes()).hexdigest()
    w=r['windows']['LOCKED_EXECUTION_ALIGNED'];f=w['held_at_31_fates']
    assert (f['initial_held_n'],f['held_eventual_winners']['n'],f['held_eventual_losers']['n'])==(102,44,58)
    expected=locked['module2']['results']['M2_R1_WITH_DATED_FILINGS']['conditional_strategy_net_mean_pct']
    assert w['policies']['M2_R1_STATIC_31']['A']['mean_net_return_pct']==pytest.approx(expected)
