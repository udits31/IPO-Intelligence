"""Financial features are parsed from real dated, hashed exchange documents."""
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ml.collect_module2_financial_evidence import ROOT,OUTPUT,extract_current_quarter,eligible_results
from ml.calibrate_module2_volatility_rules import observed_volatility,scaled_calls,GRID,select
from ml.calibrate_module2_rules import decisions,DEFAULT
from ml.replay_module2_news_overlay import SOURCES


def observed_result(ipo_id=247):
    if not OUTPUT.exists():
        pytest.skip('Actual NSE result metadata/documents not installed')
    records={r['ipo_id']:r for r in (json.loads(line) for line in OUTPUT.read_text(encoding='utf-8').splitlines())}
    row=records[ipo_id]
    source=row['document_source']
    with gzip.open(ROOT/source['raw_archive_path'],'rb') as stream:
        raw=stream.read()
    assert hashlib.sha256(raw).hexdigest()==source['raw_sha256']
    return raw,row


def test_actual_quarter_values_are_not_annualised_or_invented_growth():
    raw,row=observed_result()
    facts=extract_current_quarter(raw,row['selected_filing'],row['symbol'])
    assert facts['revenue_inr']==4036520000.
    assert facts['profit_inr']==602400000.
    assert np.isclose(facts['profit_margin_pct'],14.923746197219387)
    assert facts['period_end']=='2021-12-31'
    assert facts['published_at_ist']<row['cutoff_ist']
    assert 'growth' not in facts and 'earnings_surprise' not in facts


def test_wrong_issuer_or_currency_cannot_silently_supply_financials():
    raw,row=observed_result()
    with pytest.raises(ValueError,match='issuer symbol'):
        extract_current_quarter(raw,row['selected_filing'],'SUPRIYA')
    assert b'iso4217:INR' in raw
    with pytest.raises(ValueError,match='currency'):
        extract_current_quarter(raw.replace(b'iso4217:INR',b'iso4217:USD'),row['selected_filing'],row['symbol'])


def test_future_publication_and_cumulative_periods_are_rejected():
    _,row=observed_result()
    item=row['selected_filing']
    assert eligible_results([item],row['symbol'],pd.Timestamp('2022-02-14T15:30:00'))
    assert not eligible_results([item],row['symbol'],pd.Timestamp('2022-02-04T15:30:00'))
    assert not eligible_results([{**item,'cumulative':'Cumulative'}],row['symbol'],pd.Timestamp(row['cutoff_ist']))


def test_volatility_scaling_preserves_existing_default_rule_on_actual_cases():
    data=observed_volatility(*SOURCES['development'])
    assert len(data)==244
    reference=data.observed_daily_volatility_20_pct.median()
    np.testing.assert_array_equal(scaled_calls(data,GRID[0],reference),decisions(data,DEFAULT))
    with pytest.raises(ValueError,match='matured'):
        select(data,'2024-12-31')
    known=data[data.outcome_date.le('2024-12-31')]
    chosen,_=select(known,'2024-12-31')
    assert chosen['observed_training_volatility_median_pct']==known.observed_daily_volatility_20_pct.median()
