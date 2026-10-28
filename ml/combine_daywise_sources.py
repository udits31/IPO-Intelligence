"""Build a conservative research cohort from two real day-wise archives.

Values are never averaged or invented. IPOWatch is used when both agree;
material disagreements are excluded pending adjudication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("qib", "hni", "retail")


def combine(first: pd.DataFrame, second: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    a = first.set_index("ipo_id", verify_integrity=True)
    b = second.set_index("ipo_id", verify_integrity=True)
    records = []
    rejected = []
    for ipo_id in sorted(set(a.index) | set(b.index)):
        left = a.loc[ipo_id] if ipo_id in a.index else None
        right = b.loc[ipo_id] if ipo_id in b.index else None
        if left is not None and right is not None:
            agreement = {}
            for category in CATEGORIES:
                x = float(left[f"{category}_subscription"])
                y = float(right[f"{category}_subscription"])
                agreement[category] = abs(x - y) <= max(0.25, 0.2 * max(x, y))
            if not all(agreement.values()):
                rejected.append({"ipo_id": int(ipo_id), "disagreed_categories": [
                    category for category, passes in agreement.items() if not passes
                ]})
                continue
        chosen = left if left is not None else right
        records.append({
            "ipo_id": int(ipo_id), "company_name": chosen.company_name,
            "issue_close_date": chosen.issue_close_date,
            "qib_subscription": float(chosen.qib_subscription),
            "hni_subscription": float(chosen.hni_subscription),
            "retail_subscription": float(chosen.retail_subscription),
            "source_url": chosen.source_url,
            "source_published_at": left.source_published_at if left is not None else None,
            "source_tier": "two_source_agreement" if left is not None and right is not None else
                           "ipowatch_only" if left is not None else "equitymaster_only",
            "secondary_source_url": right.source_url if left is not None and right is not None else None,
            "evidence_class": "retrospective_daywise_table_not_timestamped_day_snapshot",
        })
    output = pd.DataFrame(records)
    report = {
        "first_source_rows": int(len(a)), "second_source_rows": int(len(b)),
        "overlap": int(len(set(a.index) & set(b.index))),
        "accepted": int(len(output)),
        "accepted_by_tier": output.source_tier.value_counts().to_dict(),
        "excluded_material_conflicts": len(rejected),
        "conflict_examples": rejected[:25],
        "rule": "Match only the same canonical IPO ID; reject >max(0.25x,20%) disagreement in any category. Prefer original IPOWatch values on agreement; no mean or synthesized reading.",
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/daywise_subscription_research_cohort.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "output/daywise_source_consensus.json")
    args = parser.parse_args()
    output, report = combine(pd.read_csv(ROOT / "data/daywise_subscription_candidates.csv"),
                             pd.read_csv(ROOT / "data/equitymaster_daywise_candidates.csv"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "conflict_examples"}, indent=2))


if __name__ == "__main__":
    main()
