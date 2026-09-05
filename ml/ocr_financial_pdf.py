"""Local, auditable OCR of a real financial PDF page; not model training.

Run with .venv-financial-ocr/Scripts/python.exe. Raw OCR is NOT automatically
accepted as a financial feature: table dates, basis, units and numeric source
checks must pass separately before any prediction experiment can use it.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import pymupdf
from rapidocr import RapidOCR

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'data/archive/module2_financials/documents/013048e3e504243457231a20bd9b2dd6fae7dc14b9235995709d7a4e6880db0f.pdf'


def run(path,page_number,*,engine=None,quiet=False,artifact_tag=None):
    path=Path(path).resolve()
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if len(path.stem)==64 and path.stem!=digest:
        raise ValueError('Source PDF hash conflicts with its archive identity')
    with pymupdf.open(path) as document:
        page=document[page_number-1]
        # Preserve the full page, including headers, units, period labels and
        # footnotes. No manual inpainting or generated text is used.
        image=page.get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False)
        suffix='_'+artifact_tag if artifact_tag else ''
        image_dir=ROOT/'data/archive/module2_financials/ocr_renders' if artifact_tag else ROOT/'tmp/pdfs'
        png=image_dir/f'{digest}_page{page_number}{suffix}_ocr.png'
        png.parent.mkdir(parents=True,exist_ok=True)
        image.save(png)
    engine=engine or RapidOCR(params={'EngineConfig.onnxruntime.intra_op_num_threads':2,
                           'EngineConfig.onnxruntime.inter_op_num_threads':2,
                           'Global.log_level':'warning'})
    result=engine(str(png))
    tokens=[{'text':text,'confidence':float(score),'box':box.tolist()}
            for text,score,box in zip(result.txts or [],result.scores or [],result.boxes if result.boxes is not None else [])]
    import rapidocr
    models=Path(rapidocr.__file__).parent/'models'
    report={'source_pdf_path':path.relative_to(ROOT).as_posix(),'source_pdf_sha256':digest,
        'page_number':page_number,'page_render_path':png.relative_to(ROOT).as_posix(),
        'render_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),'render_width':image.width,'render_height':image.height,
        'versions':{name:importlib.metadata.version(name) for name in ('rapidocr','onnxruntime','pymupdf')},
        'ocr_model_sha256':{model.name:hashlib.sha256(model.read_bytes()).hexdigest() for model in models.glob('*.onnx')},
        'feature_status':'unvalidated_raw_source_extraction','tokens':tokens}
    output=ROOT/'data/audit/financial_pdf_ocr'/f'{digest}_page{page_number}{suffix}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if not quiet:
        print('Saved raw source OCR:',output,'tokens:',len(tokens),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf',type=Path,default=DEFAULT)
    parser.add_argument('--page',type=int,default=7)
    args=parser.parse_args()
    run(args.pdf,args.page)
