"""Parse subscription multiples without confusing decimal commas with grouping."""
import math
import re


def subscription_multiple(value):
    text=str(value).strip()
    text=re.sub(r'\s*(?:x|times|×)\s*$','',text,flags=re.I).strip()
    parcel=re.fullmatch(r'[\d,]+\s+\(([^()]+)\)',text)
    if parcel:text=parcel[1]
    text=text.strip('()').strip()
    if re.fullmatch(r'\d+,\d{1,2}',text):
        text=text.replace(',','.')
    elif ',' in text:
        if not re.fullmatch(r'(?:\d{1,3}(?:,\d{3})+|\d{1,3}(?:,\d{2})*,\d{3})(?:\.\d+)?',text):return None
        text=text.replace(',','')
    if not re.fullmatch(r'\d+(?:\.\d+)?',text):return None
    result=float(text)
    return result if math.isfinite(result) and result>=0 else None
