"""Error accounting on the existing exchange-source cohort, without synthetic rows."""
import numpy as np

from ml.calibrate_module2_rules import DEFAULT, assemble, decisions
from ml.diagnose_module2_errors import classify, describe
from ml.replay_module2_news_overlay import SOURCES


def test_rule_reason_partition_exactly_matches_original_engine_replay():
    for name in ('development', 'holdout'):
        frame = assemble(*SOURCES[name])
        reasons = classify(frame)
        np.testing.assert_array_equal(reasons != 'hold', decisions(frame, DEFAULT))
        result = describe(frame)
        assert sum(r['n'] for r in result['reason_groups']) == len(frame)
        assert sum(r['correct'] for r in result['reason_groups']) == result['metrics']['correct']
        assert sum(r['n'] for r in result['years'].values()) == len(frame)


def test_all_later_errors_are_accounted_for_without_excluding_flat_returns():
    frame = assemble(*SOURCES['holdout'])
    result = describe(frame)
    assert result['metrics']['n'] == 132
    assert result['false_sell_recoveries'] == 25
    assert result['false_hold_declines'] == 41
    assert result['total_errors'] == 66
    assert result['errors_with_abs_future_return_at_most_5pct'] == 18
    assert result['errors_with_abs_future_return_above_10pct'] == 26
    assert result['metrics']['correct'] + result['total_errors'] == 132
