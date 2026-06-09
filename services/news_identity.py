"""Conservative issuer matching, including same-brand listed-company clashes.

NSE names/symbols establish identity only. They supply no forecast feature.
An industry phrase is not an issuer name; a shared bare brand is ambiguous.
"""
import csv
import hashlib
import io
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/'data/news_issuer_identity_master.csv'
IDENTITY_POLICY_VERSION='nse_distinct_issuer_prefix_v1_1'
IDENTITY_CODE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LEGAL={'ltd','limited','inc','plc','corp','corporation','co','company','the','india','and','of'}
GENERIC_PREFIX={'indian','national','international','global','one','five','first','standard','popular','capital','credit'}
COMMON_TICKERS=GENERIC_PREFIX|{'idea','route','power','oil','prime','central','unity','world','fine'}
BUSINESS_CONTEXT={'share','shares','stock','stocks','ipo','listing','nse','bse','profit','earnings','revenue',
    'ebitda','quarter','quarterly','results','dividend','tax','subsidiary','acquires','acquisition','order','orders',
    'contract','contracts','fund','funds','business','board','investor','investors','expansion','rating','brokerage',
    'analyst','ceo','director','crore','rs','cfo','merger'}


def tokens(text):
    return tuple(w for w in re.findall(r'[a-z0-9]+',str(text).casefold()) if w not in LEGAL)


@lru_cache(maxsize=4)
def _registry(master_stamp,canonical_stamp):
    identities={}
    try:
        raw=MASTER.read_bytes()
        meta=json.loads(MASTER.with_suffix('.json').read_text(encoding='utf-8'))
        if hashlib.sha256(raw).hexdigest()!=meta['identity_sha256']:
            raise ValueError('Issuer master hash mismatch')
        for row in csv.DictReader(io.StringIO(raw.decode('utf-8'))):
            identities.setdefault(tokens(row['company_name']),set()).add(row['symbol'].casefold())
    except (OSError,ValueError,KeyError):
        pass
    # Existing canonical names retain renamed issuers' historical full names;
    # a ticker is accepted only for this exact canonical name, never by fuzzy
    # substring matching an unrelated issuer.
    try:
        with (ROOT/'data/canonical_listed_ipos.csv').open(encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                key=tokens(row['company_name'])
                symbol=row.get('nse_symbol','').strip().casefold()
                identities.setdefault(key,set())
                if symbol:
                    identities[key].add(symbol)
    except (OSError,ValueError,KeyError):
        pass
    return identities


def registry():
    def stamp(path):
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0
    return _registry(stamp(MASTER),stamp(ROOT/'data/canonical_listed_ipos.csv'))


def identity_provenance():
    try:
        metadata=json.loads(MASTER.with_suffix('.json').read_text(encoding='utf-8'))
        master_sha=metadata.get('identity_sha256')
    except (OSError,ValueError):
        master_sha=None
    return {'identity_policy_version':IDENTITY_POLICY_VERSION,'identity_code_sha256':IDENTITY_CODE_SHA256,
            'issuer_master_sha256':master_sha}


def _word(required,actual):
    return required==actual or actual==required+'s' or required==actual+'s'


def phrase_in(phrase,words):
    return any(all(_word(a,b) for a,b in zip(phrase,words[i:i+len(phrase)]))
               for i in range(len(words)-len(phrase)+1))


def relevant_headline(title,company):
    name=tokens(company)
    words=tokens(title)
    if not name or not words:
        return False
    identities=registry()
    minimum=len(name) if name[0] in GENERIC_PREFIX else min(2,len(name))
    # Expand to the first distinguishing prefix: e.g. Anand Rathi *Wealth*
    # versus Anand Rathi *Share and Stock Brokers*.
    while minimum<=len(name):
        prefix=name[:minimum]
        conflicts=[other for other in identities if other!=name and len(other)>=minimum
                   and all(_word(a,b) for a,b in zip(prefix,other[:minimum]))]
        if not conflicts:
            break
        minimum+=1
    explicit_name=bool(re.search(
        r'\b'+r'\W+'.join(re.escape(w) for w in name)+r'\W+(?:ltd|limited|corp|corporation)\b',title,re.I))
    if (minimum<=len(name) and phrase_in(name[:minimum],words)
            and (len(name)>1 or set(words)&BUSINESS_CONTEXT)):
        return True
    if explicit_name:
        return True
    for symbol in identities.get(name,()):
        if len(symbol)<4 or not symbol.isalnum():
            continue
        if symbol in words:
            # A ticker equal to an ambiguous shared brand needs explicit
            # exchange syntax, not merely the same bare word in a headline.
            shared_brand=any(other!=name and other and _word(name[0],other[0]) for other in identities)
            if symbol in COMMON_TICKERS or (symbol==name[0] and shared_brand):
                if not re.search(r'\bNSE\s*:\s*'+re.escape(symbol)+r'\b',title,re.I):
                    continue
            if not set(words)&BUSINESS_CONTEXT:
                continue
            return True
    return False
