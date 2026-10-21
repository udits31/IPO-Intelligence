from ml.recover_listed_model_inputs import timed_gmp
from ml.collect_module1_public_financials import extract
from services.saved_preclose_service import clock
from services.listed_backtest_service import eligible_quotes


def test_no_later_quote_or_undated_same_day_reaches_features():
    quotes=[dict(gmp_date='2026-10-19',gmp_asof_ist='2026-10-19T16:40:00+05:30',gmp_rs=1),
            dict(gmp_date='2026-10-19',gmp_asof_ist='2026-10-19T16:55:00+05:30',gmp_rs=9),
            dict(gmp_date='2026-10-19',gmp_rs=10),dict(gmp_date='2026-10-18',gmp_rs=2)]
    assert [q['gmp_rs'] for q in eligible_quotes(quotes,clock('2026-10-19T16:44:00+05:30'))]==[1,2]


def test_intraday_gmp_keeps_different_values_before_cutoff():
    html='''<table><tr><th>Date &amp; time</th><th>GMP</th></tr>
    <tr><td>19 Oct 2026, 4:55 PM IST</td><td>₹2</td></tr>
    <tr><td>19 Oct 2026, 9:30 AM IST</td><td>₹8</td></tr>
    <tr><td>19 Oct 2026, 5:00 PM IST</td><td>₹1</td></tr></table>'''
    rows=timed_gmp(html,clock('2026-10-19T17:00:00+05:30'))
    assert [r['gmp_rs'] for r in rows]==[2,8]
    assert rows[0]['gmp_asof_ist']=='2026-10-19T16:55:00+05:30'


def test_december_annual_period_accepted_but_june_interim_excluded():
    html='''<table><tr><th></th><th>30-Jun-26</th><th>31-Dec-25</th><th>31-Dec-24</th></tr>
    <tr><td>Revenue</td><td>99</td><td>200</td><td>100</td></tr>
    <tr><td>Profit After Tax</td><td>88</td><td>20</td><td>5</td></tr>
    <tr><td>Assets</td><td>555</td><td>400</td><td>200</td></tr></table>'''
    result=extract(html,'2026-10-19',100)
    assert result['annual_period']=='2025-12-31'
    assert result['profit_margin_pct']==10
    assert result['annual_profit_after_tax']==20
