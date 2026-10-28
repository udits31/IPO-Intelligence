"""Audit whether dated NSE credit-rating PDFs can support a Module 2 test.

Keyword mentions are document observations, not validated sentiment labels.
The audit intentionally refuses a model fit when the later-year event group is
too small. No synthetic events, outcomes, or missing-document substitutes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ml.research_module2_exchange_delivery import assemble


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / 'data/audit/module2_nse_rating_documents.jsonl'
OUTPUT = ROOT / 'output/module2_rating_event_coverage.json'


def run() -> dict:
    documents = [json.loads(line) for line in DOCUMENTS.read_text(encoding='utf-8').splitlines()]
    if not documents or any(item['status'] != 'extracted' for item in documents):
        raise ValueError('Rating attachment archive is incomplete')
    cases, cohort = assemble()
    case_by_id = cases.set_index('ipo_id')
    actions = Counter()
    upgraded_ids = set()
    attached_ids = set()
    for item in documents:
        ipo_id = int(item['ipo_id'])
        if item['disseminated_at_ist'] > item['cutoff_ist']:
            raise ValueError('Post-decision attachment in archive')
        if ipo_id not in case_by_id.index:
            raise ValueError(f'Rating document without an aligned price case: {ipo_id}')
        if item['decision_date'] != case_by_id.loc[ipo_id, 'decision_date']:
            raise ValueError(f'Rating document decision date mismatch: {ipo_id}')
        attached_ids.add(ipo_id)
        # The first two pages usually carry the issuer's letter and action.
        # A mention can still refer to context rather than the current action.
        first_text = item['first_two_pages_text'][:2200]
        for name, pattern in [('upgrade_mention', r'upgrad'),
                              ('downgrade_mention', r'downgrad'),
                              ('reaffirm_mention', r'reaffirm'),
                              ('assigned_mention', r'assign')]:
            if re.search(pattern, first_text, re.IGNORECASE):
                actions[name] += 1
                if name == 'upgrade_mention':
                    upgraded_ids.add(ipo_id)
    rows = []
    for year, group in cases.groupby('year'):
        mention = group[group.ipo_id.isin(upgraded_ids)]
        rows.append({'listing_year': int(year), 'cohort_n': len(group),
                     'any_rating_attachment_n': int(group.ipo_id.isin(attached_ids).sum()),
                     'upgrade_mention_n': len(mention),
                     'upgrade_mention_positive_30d_n': int(mention.up.sum()),
                     'upgrade_mention_mean_30d_return_pct': round(float(mention.future_return_pct.mean()), 2)
                     if len(mention) else None})
    later_n = sum(row['upgrade_mention_n'] for row in rows if row['listing_year'] >= 2025)
    report = {
        'source': 'NSE corporate-announcement credit-rating PDF attachments disseminated before exact decision cutoffs',
        'document_archive_sha256': hashlib.sha256(DOCUMENTS.read_bytes()).hexdigest(),
        'documents': len(documents), 'case_cohort': cohort,
        'keyword_document_counts': dict(actions),
        'year_support': rows,
        'minimum_later_event_cases_for_model_test': 10,
        'later_upgrade_mention_cases': later_n,
        'sufficient_for_event_model_test': later_n >= 10,
        'interpretation': 'Keyword mention is not rating direction; later-year support is too small for a credible predictive test.',
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
