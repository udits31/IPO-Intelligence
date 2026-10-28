"""Create a separate, exchange-close Module 2 development replay cohort.

The first 62 observed sessions must match the NSE security-wise archive by
IPO identity and date. No synthetic rows, forward fill, or adjustment factor
is used. Nifty/sector benchmarks retain their original secondary provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/audit/module2_development_prices_2021_2024.csv'
OFFICIAL = ROOT / 'data/audit/module2_nse_delivery_development_full_window.csv'
OUTPUT = ROOT / 'data/audit/module2_development_prices_2021_2024_nse_first62.csv'
REPORT = ROOT / 'output/module2_nse_development_price_reconciliation.json'
REQUIRED_SESSIONS = 62


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile(base: pd.DataFrame, official: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if base.duplicated(['ipo_id', 'date']).any() or official.duplicated(['ipo_id', 'session_date']).any():
        raise ValueError('Duplicate IPO session in source histories')
    first = base.sort_values(['ipo_id', 'date']).groupby('ipo_id', sort=False).head(REQUIRED_SESSIONS).copy()
    first['session_date'] = first.date.dt.strftime('%Y-%m-%d')
    official = official[['ipo_id', 'session_date', 'symbol', 'nse_close']].copy()
    paired = first.merge(official, on=['ipo_id', 'session_date'], how='left', validate='one_to_one')
    paired['source_difference_pct'] = (paired.close_price_adjusted / paired.nse_close - 1).abs() * 100
    eligible = []
    excluded = []
    for ipo_id, group in paired.groupby('ipo_id', sort=True):
        reason = None
        if len(group) != REQUIRED_SESSIONS:
            reason = 'fewer than 62 base sessions'
        elif group.nse_close.isna().any():
            reason = f'{int(group.nse_close.isna().sum())} of 62 NSE closes missing'
        elif (group.nse_close <= 0).any():
            reason = 'nonpositive NSE close'
        elif group.nifty50_close.isna().any():
            reason = 'missing Nifty benchmark close'
        elif group.date.diff().dt.days.iloc[1:].gt(10).any():
            reason = 'large date gap in matched sessions'
        if reason:
            excluded.append({'ipo_id': int(ipo_id), 'reason': reason})
        else:
            eligible.append(int(ipo_id))
    verified = paired[paired.ipo_id.isin(eligible)].copy()
    verified['close_price_adjusted'] = verified.nse_close
    verified['price_basis'] = 'Official NSE security-wise archive unadjusted Close'
    original_columns = list(base.columns)
    verified = verified[original_columns].sort_values(['ipo_id', 'date']).reset_index(drop=True)
    if len(verified) != REQUIRED_SESSIONS * len(eligible):
        raise ValueError('Incomplete official-price output')
    report = {
        'base_ipo_count': int(base.ipo_id.nunique()),
        'nse_ipo_count': int(official.ipo_id.nunique()),
        'eligible_ipo_count': len(eligible),
        'eligible_ipo_ids': eligible,
        'output_rows': len(verified),
        'excluded': excluded,
        'base_vs_nse_days_gt_1pct_in_eligible': int(paired.loc[paired.ipo_id.isin(eligible), 'source_difference_pct'].gt(1).sum()),
        'stock_price_source': 'NSE security-wise archive: https://www.nseindia.com/report-detail/eq_security',
        'benchmark_source_note': 'Sector and Nifty benchmark closes are inherited from the base secondary snapshot, not NSE-certified here.',
        'no_imputation': True,
    }
    return verified, report


def main() -> None:
    prices, report = reconcile(pd.read_csv(BASE, parse_dates=['date']), pd.read_csv(OFFICIAL))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUTPUT, index=False)
    report['input_sha256'] = {str(path.relative_to(ROOT)): digest(path) for path in (BASE, OFFICIAL)}
    report['output_sha256'] = digest(OUTPUT)
    REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('base_ipo_count', 'nse_ipo_count', 'eligible_ipo_count', 'output_rows')}, indent=2))


if __name__ == '__main__':
    main()
