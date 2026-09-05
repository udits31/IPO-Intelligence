"""Exact embedded-PDF text with spatial boxes; no fabricated OCR confidence."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path

import fitz

METHOD = 'embedded_pdf_text'


def native_tokens(page):
    lines = defaultdict(list)
    for word in page.get_text('words', sort=True):
        if str(word[4]).strip():
            lines[(word[5], word[6])].append(word)
    result = []
    for words in lines.values():
        words.sort(key=lambda word: word[0])
        fragments = []
        for word in words:
            if fragments:
                previous = fragments[-1][-1]
                height = max(previous[3] - previous[1], word[3] - word[1])
                if word[0] - previous[2] <= max(10., height * 1.3):
                    fragments[-1].append(word)
                    continue
            fragments.append([word])
        for fragment in fragments:
            text = ' '.join(str(word[4]) for word in fragment).strip()
            x0, y0 = min(word[0] for word in fragment) * 2, min(word[1] for word in fragment) * 2
            x1, y1 = max(word[2] for word in fragment) * 2, max(word[3] for word in fragment) * 2
            result.append({'text': text, 'confidence': None, 'source_kind': METHOD,
                           'box': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]})
    return sorted(result, key=lambda token: (token['box'][0][1], token['box'][0][0]))


def verify_native_tokens(payload, pdf):
    if payload.get('extraction_method') != METHOD:
        raise ValueError('Native source method is not identified')
    if payload.get('text_extractor_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Native text-extractor version changed; rebuild its source cache')
    with fitz.open(pdf) as document:
        expected = native_tokens(document[payload['page_number'] - 1])
    actual = payload['tokens']
    if len(actual) != len(expected):
        raise ValueError('Native tokens do not reproduce the original PDF text')
    for token, source in zip(actual, expected):
        if (token['text'] != source['text'] or token.get('confidence') is not None or
                token.get('source_kind') != METHOD or
                len(token['box']) != 4 or any(len(point) != 2 for point in token['box']) or
                any(not math.isclose(a, b, abs_tol=.001) for p, q in zip(token['box'], source['box']) for a, b in zip(p, q))):
            raise ValueError('Native token text/coordinates differ from the original PDF')

