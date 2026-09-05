"""Prioritise actual financial tables without trusting PDF text as figures.

PDF extraction order need not match visual order: a title may be emitted after
the complete table. Search the full page for table evidence. The downstream
parser still checks visible basis, dates, units and independently verified
quarter values, followed by full-page visual review.
"""
from __future__ import annotations

import re


def candidate_pages(document, basis):
    matching, tables = [], []
    for index, page in enumerate(document):
        full = re.sub(r'[^a-z0-9]', '', page.get_text('text', sort=True).casefold())
        if 'revenuefromoperations' not in full or 'quarter' not in full or 'profit' not in full:
            continue
        tables.append(index + 1)
        target = ('consolidated' in full and 'standalone' not in full
                  if basis.casefold() == 'consolidated' else
                  'standalone' in full and 'consolidated' not in full)
        if target:
            matching.append(index + 1)
    # Anonymous/garbled titles are not assigned a basis here. Both table
    # candidates may be tried, but only the independent full parser can accept
    # one. No numeric feature is sourced from this page-prioritisation step.
    # A notes page may mention the requested basis while the true table title
    # is garbled in its hidden text layer. Prioritise, but never discard, other
    # table-like pages merely because their extracted title is anonymous.
    priority = matching + [page for page in tables if page not in matching]
    # A PDF can mix text-based notes with an image-only result table. A notes
    # hit is not evidence that other pages contain no table: retain the bounded
    # full-page fallback after priorities. The caller stops at a valid table.
    return priority + [page for page in range(1, min(len(document), 32) + 1)
                       if page not in priority]
