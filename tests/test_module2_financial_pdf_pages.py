"""Select table pages in actual filed PDFs, not generated layout fixtures."""
import json

import pytest

from ml.financial_pdf_pages import candidate_pages
from ml.recheck_module2_financial_comparatives import MANIFEST, ROOT

fitz = pytest.importorskip('fitz')


def document(ipo_id):
    if not MANIFEST.exists():
        pytest.skip('Original financial-source documents are not installed')
    record = next(json.loads(s) for s in MANIFEST.read_text(encoding='utf-8').splitlines()
                  if json.loads(s)['ipo_id'] == ipo_id)
    return fitz.open(ROOT / record['source_pdf_path'])


def test_tenneco_table_title_emitted_late_still_selects_consolidated_page():
    with document(515) as pdf:
        assert candidate_pages(pdf, 'Consolidated')[0] == 4
        assert candidate_pages(pdf, 'Non-Consolidated')[0] == 9


def test_sudeep_consolidated_and_standalone_tables_are_not_interchanged():
    with document(511) as pdf:
        # The separate consolidated segment table (page 14) is a legitimate
        # extra candidate, not an accepted group earnings feature. The full
        # downstream parser still rejects its different title/row basis.
        assert candidate_pages(pdf, 'Consolidated')[:2] == [11, 14]
        assert candidate_pages(pdf, 'Non-Consolidated')[0] == 4


def test_anonymous_ksh_title_can_prioritise_a_page_but_cannot_create_features():
    with document(502) as pdf:
        assert candidate_pages(pdf, 'Non-Consolidated')[0] == 2


def test_park_table_not_discarded_because_notes_have_clearer_basis_label():
    with document(504) as pdf:
        pages = candidate_pages(pdf, 'Consolidated')
        assert 5 in pages
        assert set(pages) == set(range(1, len(pdf) + 1))
