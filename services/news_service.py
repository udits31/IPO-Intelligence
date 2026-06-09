import os
import json
import hashlib
import copy
import re
import tempfile
import threading
import time
from pathlib import Path
import urllib.parse
import xml.etree.ElementTree as ET
import httpx
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any
from zoneinfo import ZoneInfo

NEWS_TTL_SECONDS = 600
NEWS_RETRY_SECONDS = 60
NEWS_CACHE_PATH = Path(__file__).resolve().parents[1] / 'data/telemetry/news_cache.json'
NEWS_SCORER_VERSION = 'keyword_financial_context_v2_1'
NEWS_SCORER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_NEWS_CACHE = None
_NEWS_LOCK = threading.RLock()
_COMPANY_LOCKS = {}


def _quote_directory_title(title: str) -> bool:
    """Exclude evergreen quote/chart pages, not real price-move reporting."""
    text = re.sub(r'[^a-z0-9]+', ' ', title.casefold()).strip()
    return bool(
        re.search(r'(?:interactive stock chart|stock share price nse bse forecast news and live quotes)$', text)
        or re.search(r'stock price news quote (?:and )?history$', text)
        or re.search(r'share price(?: today)?(?: nse bse)?(?: charts)?$', text)
        or re.search(r'(?:share price today price action and valuation|key financial ratios valuation profitability more)$', text)
        or re.search(r'\bis rated (?:strong )?(?:buy|sell|hold)$',text)
    )


def _eligible_snapshot(snapshot):
    """Apply current eligibility even to a fresh cache from an older process."""
    result = copy.deepcopy(snapshot)
    recent = []
    for article in result.get('articles', []):
        try:
            published = parsedate_to_datetime(article['published_at']).astimezone(timezone.utc)
            if (0 <= (datetime.now(timezone.utc)-published).total_seconds() <= 7*86400
                    and not _quote_directory_title(article.get('title', ''))
                    and _relevant_to_company(article.get('title',''),result.get('company_name',''))):
                recent.append(article)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    result['articles'] = recent
    return result


def _news_cache():
    global _NEWS_CACHE
    with _NEWS_LOCK:
        if _NEWS_CACHE is None:
            try:
                payload = json.loads(NEWS_CACHE_PATH.read_text(encoding='utf-8'))
                _NEWS_CACHE = payload if isinstance(payload, dict) else {}
            except (OSError, ValueError):
                _NEWS_CACHE = {}
        return _NEWS_CACHE


def _save_news_cache():
    # Atomic replacement keeps a restart from reading a partly written cache.
    with _NEWS_LOCK:
        NEWS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=NEWS_CACHE_PATH.parent,
                                             prefix='news-', suffix='.tmp', delete=False) as stream:
                temporary = stream.name
                json.dump(_news_cache(), stream)
            os.replace(temporary, NEWS_CACHE_PATH)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


def get_news_snapshot(company_name: str, *, force: bool = False) -> Dict[str, Any]:
    key = re.sub(r'\s+', ' ', company_name.strip().casefold())
    if not key:
        return {'articles': [], 'freshness': 'unavailable', 'error': 'Company is required'}
    with _NEWS_LOCK:
        lock = _COMPANY_LOCKS.setdefault(key, threading.Lock())
    with lock:
        now = time.time()
        cached = _news_cache().get(key, {})
        last_success = cached.get('last_success_epoch', 0)
        if not force and last_success and not cached.get('error') and now - last_success < NEWS_TTL_SECONDS:
            return _eligible_snapshot(cached)
        if not force and cached.get('error') and now - cached.get('last_attempt_epoch', 0) < NEWS_RETRY_SECONDS:
            return _eligible_snapshot(cached)
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            articles = _download_company_news(company_name, limit=12)
            snapshot = {'company_name': company_name, 'articles': articles, 'freshness': 'fresh',
                        'last_success_epoch': now, 'last_attempt_epoch': now,
                        'checked_at': stamp, 'last_success_at': stamp, 'error': None}
        except Exception as exc:
            # Preserve last good data and its original timestamp, not a fake
            # successful "no news" result on timeout, 429, or malformed RSS.
            snapshot = {**cached, 'company_name': company_name,
                        'articles': cached.get('articles', []), 'last_attempt_epoch': now,
                        'checked_at': stamp, 'freshness': 'stale' if cached.get('articles') else 'unavailable',
                        'error': f'{type(exc).__name__}: news source unavailable'}
        # Cached headlines have the same seven-day eligibility limit as RSS.
        snapshot = _eligible_snapshot(snapshot)
        with _NEWS_LOCK:
            _news_cache()[key] = snapshot
            try:
                _save_news_cache()
            except OSError:
                pass  # The successful in-memory fetch remains usable.
        return copy.deepcopy(snapshot)


def news_refresh_status() -> Dict[str, Any]:
    with _NEWS_LOCK:
        rows = list(_news_cache().values())
        now = time.time()
        return {'companies_cached': len(rows),
                'fresh_companies': sum(bool(r.get('last_success_epoch')) and
                    now-r['last_success_epoch'] < NEWS_TTL_SECONDS and not r.get('error') for r in rows),
                'ttl_seconds': NEWS_TTL_SECONDS,
                'last_checked_at': max((r.get('checked_at', '') for r in rows), default=None)}

def _relevant_to_company(title: str, company_name: str) -> bool:
    from services.news_identity import relevant_headline
    return relevant_headline(title,company_name)

def fetch_company_news(company_name: str, limit: int = 5) -> List[Dict[str, str]]:
    return get_news_snapshot(company_name)['articles'][:limit]


def _download_company_news(company_name: str, limit: int = 5) -> List[Dict[str, str]]:
    query = f'"{company_name}" stock'
    encoded = urllib.parse.quote(query)
    url = 'https://news.google.com/rss/search?q=' + encoded + '&hl=en-IN&gl=IN&ceid=IN:en'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        resp = httpx.get(url, headers=headers, timeout=3.0, follow_redirects=True)
        resp.raise_for_status()
            
        root = ET.fromstring(resp.text)
        if root.tag.lower() != 'rss':
            raise ValueError('News source returned a non-RSS document')
        items = []
        now = datetime.now(timezone.utc)
        for item in root.findall('.//item'):
            title = item.findtext('title') or ''
            link = item.findtext('link') or ''
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
            try:
                published = parsedate_to_datetime(pub_date).astimezone(timezone.utc)
            except (TypeError, ValueError, OverflowError):
                continue
            if published > now or now - published > timedelta(days=7):
                continue
            source_el = item.find('source')
            source_name = source_el.text if source_el is not None else 'Financial News'
            
            clean_title = title.split(' - ')[0] if ' - ' in title else title
            if not _relevant_to_company(clean_title, company_name) or _quote_directory_title(clean_title):
                continue
            
            items.append({
                'title': clean_title,
                'publisher': source_name,
                'source': source_name,
                'published_at': pub_date,
                'published_date': pub_date,
                'url': link,
                'link': link
            })
        items.sort(key=lambda a: parsedate_to_datetime(a['published_at']), reverse=True)
        seen, unique = set(), []
        for article in items:
            key = article['title'].casefold()
            if key not in seen:
                unique.append(article)
                seen.add(key)
        return unique[:limit]
    except Exception:
        raise

class NewsService:
    def __init__(self):
        # Mentions of profit/revenue alone do not establish improvement.
        self.positive_keywords = ['growth','surge','surges','rally','rallies','win','wins','won','soar','soars',
            'bullish','expansion','strong','beat','beats','partnership','rise','rises','jump','jumps','gains',
            'secures','bags','bagged']
        self.negative_keywords = ['plunge','plunges','loss','losses','down','slump','slumps','crash','investigation',
            'fraud','probe','resigns','headwinds','deficit','drop','drops','weak','fell','fall','falls',
            'selloff','penalty','decline','declines','cancellation','cancelled','downgrade']

    def analyze_sentiment(self, text: str) -> float:
        words=re.findall(r'[a-z]+',text.casefold())
        used=set()
        pos_count=neg_count=0
        metrics={'profit','profits','pat','revenue','revenues','earnings','sales','income','margin','margins',
                 'ebitda','growth','loss','losses','order','orders'}
        up={'rise','rises','rising','rose','increase','increases','increased','growth','grows','grew','jump',
            'jumps','jumped','surge','surges','surged','improves','improved','higher','up','doubles','doubled',
            'expands','expanded','widens','widened','win','wins','won','secures','bags','bagged','strong'}
        down={'fall','falls','falling','fell','decline','declines','declined','drop','drops','dropped',
              'decrease','decreases','decreased','lower','down','narrows','narrowed','shrinks','shrank',
              'weak','slows','slowing','cancelled','cancellation','terminated'}
        flat={'flat','unchanged','stable'}
        for i in range(len(words)-1):
            if words[i]=='strong' and words[i+1] in {'buy','sell'}:
                pos_count+=words[i+1]=='buy'
                neg_count+=words[i+1]=='sell'
                used.update((i,i+1))
        for i,word in enumerate(words):
            if word not in metrics or i in used:
                continue
            nearby=[j for j in range(max(0,i-4),min(len(words),i+5)) if j!=i and j not in used
                    and words[j] in up|down|flat]
            if not nearby:
                continue
            j=min(nearby,key=lambda j:(abs(j-i),j<i))
            direction=0 if words[j] in flat else (1 if words[j] in up else -1)
            if word in {'loss','losses'}:
                direction=-direction  # A narrower loss is better, not worse.
            if any(w in {'no','not','never'} for w in words[max(0,j-2):j]):
                direction=0  # "not declining" does not establish growth.
            pos_count+=direction>0
            neg_count+=direction<0
            used.update((i,j))
        for i,word in enumerate(words):
            if i in used:
                continue
            negated=any(w in {'no','not','denies','denied','without'} for w in words[max(0,i-3):i])
            if negated:
                continue
            pos_count+=word in self.positive_keywords
            neg_count+=word in self.negative_keywords
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return round((pos_count - neg_count) / total, 2)

    @staticmethod
    def scorer_provenance():
        from services.news_identity import identity_provenance
        return {'scorer_version':NEWS_SCORER_VERSION,'scorer_sha256':NEWS_SCORER_SHA256,**identity_provenance()}

    def summarize_historical_snapshot(self, company_name: str, as_of_date: str,
                                      snapshot: Dict[str,Any]) -> Dict[str,Any]:
        """Replay a dated source snapshot, never fetch today's news for the past.

        Raw RSS identity/hash is checked by the archive reader before calling
        this method. These guards also reject a mismatched issuer/checkpoint
        and filter unknown, late or unrelated headlines. A successful empty
        index is not proof that the issuer had no news.
        """
        cutoff=datetime.fromisoformat(str(as_of_date)[:10]+'T15:30:00').replace(tzinfo=ZoneInfo('Asia/Kolkata'))
        if (str(snapshot.get('company_name','')).strip().casefold()!=company_name.strip().casefold()
                or str(snapshot.get('decision_date'))!=cutoff.date().isoformat()
                or snapshot.get('cutoff_ist')!=cutoff.isoformat()):
            raise ValueError('Historical news issuer or decision cutoff mismatch')
        if snapshot.get('status')!='verified_rss_response':
            return {'articles':[],'sentiment_score':None,'status':'historical_news_source_unavailable',
                    'freshness':'unavailable','source_url':snapshot.get('source_url')}
        if not re.fullmatch(r'[0-9a-f]{64}',str(snapshot.get('raw_sha256') or '')):
            raise ValueError('Historical news has no archived source hash')
        kept,seen,rejected=[],set(),0
        for article in snapshot.get('articles',[]):
            try:
                published=parsedate_to_datetime(article['published_at'])
                if published.tzinfo is None:
                    raise ValueError('Missing publication timezone')
                title=str(article['title'])
                if (published>cutoff or cutoff-published>timedelta(days=7)
                        or published.astimezone(cutoff.tzinfo).date()==cutoff.date()
                        or _quote_directory_title(title) or not _relevant_to_company(title,company_name)
                        or not str(article.get('url','')).startswith('https://')
                        or title.casefold() in seen):
                    rejected+=1
                    continue
                seen.add(title.casefold())
                kept.append(copy.deepcopy(article))
            except (ValueError,TypeError,KeyError,OverflowError):
                rejected+=1
        kept.sort(key=lambda a:parsedate_to_datetime(a['published_at']),reverse=True)
        kept=kept[:6]
        score=round(sum(self.analyze_sentiment(a['title']) for a in kept)/len(kept),2) if kept else None
        return {'company_name':company_name,'articles':kept,'sentiment_score':score,
                'status':'historical_indexed_headlines' if kept else 'historical_no_eligible_indexed_headlines',
                'freshness':'historical_asof','source_url':snapshot.get('source_url'),
                'raw_sha256':snapshot['raw_sha256'],'rejected_by_asof_guard':rejected,
                'checked_at':snapshot.get('retrieved_at_utc'),
                **self.scorer_provenance(),
                'note':'Retrospective RSS index, not immutable publisher text or exhaustive news coverage.'}

    def get_company_news(self, company_name: str, max_results: int = 6) -> Dict[str, Any]:
        articles = fetch_company_news(company_name, limit=max_results)
        cached = _news_cache().get(re.sub(r'\s+', ' ', company_name.strip().casefold()), {})
        freshness = {key: cached.get(key) for key in ('freshness', 'checked_at', 'last_success_at', 'error')}
        if not articles:
            return {
                'company_name': company_name,
                'articles': [],
                'sentiment_score': None,
                'status': 'news_source_unavailable' if cached.get('error') else 'no_recent_timestamped_articles',
                **self.scorer_provenance(),
                **freshness,
            }
        scores = [self.analyze_sentiment(a['title']) for a in articles]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        return {
            'company_name': company_name,
            'articles': articles,
            'sentiment_score': None if cached.get('error') else avg_score,
            'status': 'stale_headlines_source_unavailable' if cached.get('error') else 'observed_recent_headlines',
            **self.scorer_provenance(),
            **freshness,
        }

if __name__ == '__main__':
    service = NewsService()
    print('Testing live news service on Tata Technologies:')
    res = service.get_company_news('Tata Technologies', max_results=3)
    print('Sentiment Score:', res['sentiment_score'])
    for n in res['articles']:
        print('  * [' + n['source'] + '] ' + n['title'] + ' (' + n['published_date'] + ')')
