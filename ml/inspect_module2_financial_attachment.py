"""Read an archived original NSE financial attachment, preserving its bytes."""
import gzip
import hashlib
import io
import zipfile
from pathlib import Path

import pdfplumber

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/archive/module2_financials/83755c30a89f56df4df43cfb080cb489ab4773fcacd179cb39aeee178961b332.gz'


def main():
    with gzip.open(SOURCE,'rb') as stream:
        raw=stream.read()
    if hashlib.sha256(raw).hexdigest()!=SOURCE.stem:
        raise ValueError('Original exchange attachment archive hash mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        files=[item for item in archive.infolist() if item.filename.lower().endswith('.pdf')]
        if len(files)!=1 or files[0].file_size>20_000_000:
            raise ValueError('Expected one bounded original financial PDF')
        content=archive.read(files[0])
    digest=hashlib.sha256(content).hexdigest()
    path=SOURCE.parent/'documents'/f'{digest}.pdf'
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        path.write_bytes(content)  # Extract original source, not authored PDF content.
    print('Original PDF',path,digest)
    with pdfplumber.open(io.BytesIO(content)) as document:
        print('Pages',len(document.pages))
        for page_index,page in enumerate(document.pages):
            text=page.extract_text() or ''
            if any(word in text.casefold() for word in ('revenue from','statement of unaudited','statement of un-audited')):
                print('PAGE',page_index+1,text[:9000])
    return path


if __name__=='__main__':
    main()
