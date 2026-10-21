from datetime import datetime
from zoneinfo import ZoneInfo
from ml.recover_listed_closing_books import parse

IST=ZoneInfo('Asia/Kolkata')
CUTOFF=datetime(2026,9,28,17,tzinfo=IST)


def test_groww_bid_time_does_not_use_page_update_time():
    html="""<h1>A-One Steels India Ltd IPO</h1><h2>Subscription rate</h2>
    Qualified Institutional Buyers 3.30x Non-Institutional Investor 24.38x
    Retail Individual Investor 8.18x As of 28 Sep'26, 04:55 PM
    <p>Updated 25 Oct 2026</p>"""
    rows=parse(html,'A-One Steels India',CUTOFF,'https://groww.in/ipo/a-one-steels-ipo')
    assert len(rows)==1 and rows[0]['qib']==3.30
    assert rows[0]['source_reported_at_ist']=='2026-09-28T16:55:00+05:30'
    assert not parse(html.replace('04:55 PM','05:00 PM'),'A-One Steels India',CUTOFF,'https://groww.in/ipo/a-one-steels-ipo')
    assert not parse(html,'Other Issuer',CUTOFF,'https://groww.in/ipo/a-one-steels-ipo')


def test_intraday_history_utc_conversion_and_cutoff():
    html="""<h1>A-One Steels India Subscription Status</h1><table>
    <tr><th>Snapshot Time (UTC)</th><th>QIB</th><th>NII</th><th>Retail</th></tr>
    <tr><td>Sep 28, 2026 11:13 AM</td><td>2x</td><td>3x</td><td>4x</td></tr>
    <tr><td>Sep 28, 2026 11:30 AM</td><td>8x</td><td>9x</td><td>10x</td></tr>
    <tr><td>Sep 27, 2026 11:13 AM</td><td>1x</td><td>1x</td><td>1x</td></tr></table>"""
    rows=parse(html,'A-One Steels India',CUTOFF,'https://ipometrics.in/ipo/a-one-steels/subscription')
    assert len(rows)==1
    assert rows[0]['source_reported_at_ist']=='2026-09-28T16:43:00+05:30'


def test_iporise_uses_live_table_not_day_one_table():
    html="""<h1>A-One Steels India Subscription</h1><h2>Live Subscription Data</h2>
    Last updated: Sep 28, 2026, 4:57 PM IST
    <table><tr><td>QIB Qualified Institutional Buyers</td><td>100</td><td>310</td><td>3.10×</td></tr>
    <tr><td>NII Non-Institutional Investors</td><td>100</td><td>93</td><td>0.93×</td></tr>
    <tr><td>Retail Retail Investors</td><td>100</td><td>99</td><td>0.99×</td></tr></table>
    <h2>Day 1</h2><table><tr><td>QIB</td><td>100</td><td>0</td><td>0x</td></tr></table>"""
    rows=parse(html,'A-One Steels India',CUTOFF,'https://www.iporise.com/ipo/a-one-steels-ipo/subscription')
    assert len(rows)==1 and rows[0]['qib']==3.1 and rows[0]['hni']==.93
