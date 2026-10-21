from services.listed_backtest_service import select_book


def book(time,qib=2):
    return dict(observed_at_ist=time,qib=qib,hni=3,retail=4)


def test_latest_closing_day_reading_strictly_before_five():
    early=book('2026-10-19T16:44:00+05:30')
    row=dict(cutoff_at_ist='2026-10-19T17:00:00+05:30',subscription_history=[
        book('2026-10-16T16:59:00+05:30'),early,
        book('2026-10-19T17:00:00+05:30'),book('2026-10-19T19:15:00+05:30')])
    assert select_book(row)==early
    row['outcome']={'positive':False}
    assert select_book(row)==early


def test_no_earlier_day_or_day_only_substitution():
    assert select_book(dict(cutoff_at_ist='2026-10-19T17:00:00+05:30',subscription_history=[
        book('2026-10-16T16:59:00+05:30'),book(None)])) is None


def test_missing_and_negative_bid_rejected_zero_is_valid():
    valid=book('2026-10-19T16:00:00+05:30',0)
    row=dict(cutoff_at_ist='2026-10-19T17:00:00+05:30',subscription_history=[
        valid,book('2026-10-19T16:30:00+05:30',None),book('2026-10-19T16:45:00+05:30',-1)])
    assert select_book(row)==valid
