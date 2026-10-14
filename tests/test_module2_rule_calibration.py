import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.calibrate_module2_rules import (DEV_PRICES,DEV_LEDGER,LATER_PRICES,LATER_LEDGER,
    DEFAULT,GRID,assemble,decisions,select_parameters)
from ml.recover_module2_be_closes import parse_prices

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def actual_development():
    if not DEV_PRICES.exists() or not DEV_LEDGER.exists():
        pytest.skip('Actual exchange replay inputs are not installed')
    return assemble(DEV_PRICES,DEV_LEDGER)


def test_calibration_reproduces_the_current_rule_exactly(actual_development):
    assert len(GRID)==75
    assert DEFAULT in GRID
    assert np.array_equal(decisions(actual_development,DEFAULT),actual_development.sell.to_numpy(dtype=bool))
    assert not actual_development.ipo_id.duplicated().any()


def test_future_labels_cannot_choose_thresholds(actual_development):
    freeze=actual_development.outcome_date.min()-pd.Timedelta(days=1)
    with pytest.raises(ValueError,match='after its freeze date'):
        select_parameters(actual_development,freeze)


def test_no_post2024_ipo_can_choose_thresholds():
    if not LATER_PRICES.exists() or not LATER_LEDGER.exists():
        pytest.skip('Actual later exchange replay is not installed')
    later=assemble(LATER_PRICES,LATER_LEDGER)
    with pytest.raises(ValueError,match='Post-2024 IPOs'):
        select_parameters(later,later.outcome_date.max())


def test_real_be_price_survives_missing_delivery_without_fabricated_fields():
    paths=list((ROOT/'data/archive/module2_nse_be_closes').glob('249_*.json.gz'))
    if not paths:
        pytest.skip('The original HPAL NSE response is not installed')
    payload=json.load(gzip.open(paths[0],'rt'))
    assert all(r.get('COP_DELIV_QTY') is None for r in payload['data'])
    real_dates=pd.to_datetime([r['mTIMESTAMP'] for r in payload['data']],format='%d-%b-%Y')
    rows=parse_prices(payload,ipo_id=249,symbol='HPAL',first=real_dates.min(),last=real_dates.max(),
                      source_url='https://www.nseindia.com/report-detail/eq_security',digest='raw-response-tested')
    assert len(rows)==10
    assert {r['session_date'] for r in rows}=={d.strftime('%Y-%m-%d') for d in real_dates}
    assert all('deliverable_qty' not in r and 'deliverable_pct' not in r for r in rows)
    assert parse_prices(payload,ipo_id=249,symbol='TMB',first=real_dates.min(),last=real_dates.max(),
                        source_url='https://www.nseindia.com/report-detail/eq_security',digest='raw-response-tested')==[]
