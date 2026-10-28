"""Capture current, internet-backed Module 2 calls before outcomes exist.

The whole verified listed cohort in the configured age window is attempted.
Stale or non-raw prices are rejected by the append-only ledger; no values are
imputed. Re-running on the same IST day does not overwrite the first decision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.module2_decision_ledger import capture_recent_listings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=60)
    args = parser.parse_args()
    report = capture_recent_listings(args.lookback_days)
    for item in report["records"]:
        print(f"{item['ipo_id']} {item['nse_symbol']}: {item['status']}")
    print(f"Attempted {report['attempted']} verified listed IPOs; "
          f"froze {report['recorded']} new decisions")


if __name__ == "__main__":
    main()
