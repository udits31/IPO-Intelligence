import os
import re
import json
import hashlib
import copy
import logging
import threading
import uuid
import httpx
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date, datetime, time as dt_time
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple

CALENDAR_FETCH_STATUS = {'status': 'not_started'}
_CACHE_WRITE_LOCK=threading.RLock()
_LAST_VALID_CACHE={}

def parse_date_interval(date_str: str, ref_year: int = 2026) -> Tuple[Optional[date], Optional[date]]:
    """
    Parses dynamic date intervals from diverse Indian IPO calendar formats into (open_date, close_date).
    Handles:
      - '28 Aug to 1 Sep, 2026'
      - 'Aug 27-31, 2026'
      - 'Aug 28-Sep 1, 2026'
      - '25 - 29 Sep'
      - '22 - 24 Sep'
      - '17 - 21 Sep'
    """
    if not date_str:
        return None, None
    s = str(date_str).strip()
    iso_dates = re.findall(r'\d{4}-\d{2}-\d{2}', s)
    if len(iso_dates) == 2:
        try:
            return date.fromisoformat(iso_dates[0]), date.fromisoformat(iso_dates[1])
        except ValueError:
            return None, None
    s = s.replace('–', '-').replace('—', '-')
    months = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    # Format 1: 'Aug 27-31, 2026' or 'Aug 28-Sep 1, 2026'
    m_lead = re.search(r'([A-Za-z]{3})\s*(\d{1,2})\s*(?:-|to)\s*(?:([A-Za-z]{3})\s*)?(\d{1,2}),?\s*(\d{4})?', s, re.IGNORECASE)
    if m_lead:
        m1_str, d1, m2_str, d2, y_str = m_lead.groups()
        year = int(y_str) if y_str else ref_year
        m1_num = months.get(m1_str.lower()[:3], 9)
        m2_num = months.get(m2_str.lower()[:3], m1_num) if m2_str else m1_num
        try:
            return date(year, m1_num, int(d1)), date(year, m2_num, int(d2))
        except Exception:
            pass

    # Format 2: '28 Aug to 1 Sep, 2026'
    m1 = re.search(r'(\d{1,2})\s*([A-Za-z]{3})?\s*(?:to|-)\s*(\d{1,2})\s*([A-Za-z]{3}),?\s*(\d{4})?', s, re.IGNORECASE)
    if m1:
        d1, m1_str, d2, m2_str, y_str = m1.groups()
        year = int(y_str) if y_str else ref_year
        m2_num = months.get(m2_str.lower()[:3], 9)
        m1_num = months.get(m1_str.lower()[:3], m2_num) if m1_str else m2_num
        try:
            return date(year, m1_num, int(d1)), date(year, m2_num, int(d2))
        except Exception:
            pass

    # Format 3: '17 - 21 Sep' or '25 - 29 Sep'
    m2 = re.search(r'(\d{1,2})\s*-\s*(\d{1,2})\s+([A-Za-z]{3})', s, re.IGNORECASE)
    if m2:
        d1, d2, m_str = m2.groups()
        m_num = months.get(m_str.lower()[:3], 9)
        try:
            return date(ref_year, m_num, int(d1)), date(ref_year, m_num, int(d2))
        except Exception:
            pass
            
    return None, None

def classify_ipo_stage(item: Dict[str, Any], as_of_date: Optional[date] = None) -> Tuple[str, str]:
    """
    Deterministically computes (category, status) based on temporal calendar arithmetic against as_of_date.
    Category is strictly one of: 'OPEN', 'UPCOMING', 'CLOSED', 'LISTED'.
    """
    live_clock = datetime.now(ZoneInfo('Asia/Kolkata')) if as_of_date is None else None
    if as_of_date is None:
        as_of_date = live_clock.date()
            
    # 1. Listed check
    if item.get('is_listed') or 'Listed' in str(item.get('status', '')) or 'Listed' in str(item.get('date_range', '')):
        listing_date = item.get('listing_date')
        status = f"Listed ({listing_date})" if listing_date else (item.get('status') if 'Listed' in str(item.get('status')) else 'LISTED')
        return 'LISTED', status
        
    # Check if explicit listing_date <= as_of_date
    if item.get('listing_date'):
        try:
            l_dt = pd.to_datetime(item['listing_date']).date()
            if l_dt <= as_of_date:
                return 'LISTED', f"Listed ({l_dt.strftime('%b %d, %Y')})"
        except Exception:
            pass

    # 2. Date interval check (open and close dates)
    d_open, d_close = parse_date_interval(item.get('date_range') or item.get('status') or '')
    if d_open and d_close:
        if as_of_date > d_close:
            return 'CLOSED', 'CLOSED / AWAITING ALLOTMENT'
        elif d_open <= as_of_date <= d_close:
            if live_clock is not None and as_of_date == d_close and live_clock.time() >= dt_time(17, 0):
                return 'CLOSED', 'CLOSED / CLOSING BOOK PENDING'
            return 'OPEN', 'OPEN / ACTIVE BIDDING'
        elif as_of_date < d_open:
            return 'UPCOMING', 'UPCOMING / FORTHCOMING'
            
    # 3. Fallbacks
    raw_status = str(item.get('status', '')).upper()
    if 'OPEN' in raw_status or 'ACTIVE' in raw_status:
        return 'UPCOMING', 'UPCOMING / FORTHCOMING'
    elif 'CLOSE' in raw_status:
        return 'CLOSED', 'CLOSED / AWAITING ALLOTMENT'
    return item.get('category', 'UPCOMING'), item.get('status', 'UPCOMING / FORTHCOMING')

CACHE_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'live_scraped_cache.json')
SUBSCRIPTION_SOURCE = 'chittorgarh_subscription_page_v1'
NSE_SUBSCRIPTION_SOURCE = 'nse_ipo_category_v1'
SUBSCRIPTION_KEYS = ('qib_subscription', 'retail_subscription', 'hni_subscription')


def _subscription_window_open() -> bool:
    now = datetime.now(ZoneInfo('Asia/Kolkata'))
    return now.weekday() < 5 and dt_time(10, 0) <= now.time() <= dt_time(17, 0)


def _subscription_url(ipo_url: str) -> str:
    return ipo_url.replace('/ipo/', '/ipo_subscription/', 1)


def parse_subscription_table(html: str) -> Optional[Dict[str, float]]:
    """Only accept three observed bid multiples from a labelled subscription table."""
    soup = BeautifulSoup(html, 'html.parser')
    labels = {
        'qualified institutional': 'qib_subscription',
        'qib': 'qib_subscription',
        'non institutional': 'hni_subscription',
        'nii': 'hni_subscription',
        'retail individual': 'retail_subscription',
        'retail': 'retail_subscription',
    }
    for table in soup.find_all('table'):
        rows = [[cell.get_text(' ', strip=True) for cell in tr.find_all(['th', 'td'])]
                for tr in table.find_all('tr')]
        if not rows or len(rows[0]) != 2:
            continue
        header = [value.lower() for value in rows[0]]
        if not ('category' in header[0] and 'subscription' in header[1] and
                ('times' in header[1] or 'x' in header[1])):
            continue
        values = {}
        for row in rows[1:]:
            if len(row) != 2:
                continue
            key = labels.get(re.sub(r'\s+', ' ', row[0].lower()).strip())
            match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*x?\s*', row[1], re.IGNORECASE)
            if key and match:
                values[key] = float(match.group(1))
        if values:
            return values
    return None


def fetch_subscription_evidence(ipo_url: str) -> Dict[str, Any]:
    """Fetch once and return both the parsed observation and audit evidence.

    fetched_at is our clock, never a claimed source publication timestamp.
    A page without an unambiguous bid table is an unavailable observation.
    """
    url = _subscription_url(ipo_url)
    evidence: Dict[str, Any] = {
        'subscription_url': url, 'fetched_at_ist': None, 'http_status': None,
        'status': 'invalid_url' if url == ipo_url else 'fetch_error',
        'values': None, 'raw_sha256': None, 'raw_html': None, 'error': None,
    }
    if url == ipo_url:
        evidence['error'] = 'IPO URL has no /ipo/ segment'
        return evidence
    try:
        response = httpx.get(url, headers={'User-Agent': 'Mozilla/5.0'}, follow_redirects=True, timeout=12.0)
        evidence['fetched_at_ist'] = datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds')
        evidence['http_status'] = getattr(response, 'status_code', 200)
        raw = response.text
        evidence['raw_html'] = raw
        evidence['raw_sha256'] = hashlib.sha256(raw.encode('utf-8')).hexdigest()
        response.raise_for_status()
        values = parse_subscription_table(raw)
        evidence['source_displayed_at_ist'] = parse_subscription_published_at(raw)
        evidence['values'] = values
        evidence['status'] = ('verified' if all(key in values for key in SUBSCRIPTION_KEYS)
                              else 'partial_subscription_table') if values else 'no_subscription_table'
    except Exception as exc:
        evidence['status'] = 'fetch_error'
        evidence['error'] = f'{type(exc).__name__}: {exc}'[:500]
    return evidence


def parse_subscription_published_at(html: str) -> Optional[str]:
    """Read the subscription FAQ's own as-of time, not a GMP/page clock."""
    text = BeautifulSoup(html, 'html.parser').get_text(' ', strip=True)
    match = re.search(r'IPO is subscribed\s+[\d.]+\s*x?\s+(?:times\s+)?as of\s+'
                      r'([A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)', text, re.I)
    if not match:
        return None
    raw = re.sub(r'\s+', ' ', match[1]).strip()
    for fmt in ('%b %d, %Y %H:%M', '%b %d, %Y %H:%M:%S', '%b %d, %Y %I:%M %p', '%b %d, %Y %I:%M:%S %p'):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds')
        except ValueError:
            pass
    return None


def fetch_chittorgarh_subscriptions(ipo_url: str) -> Optional[Dict[str, Any]]:
    """Compatibility wrapper for the live cache; no false 0x subscriptions."""
    evidence = fetch_subscription_evidence(ipo_url)
    if evidence['status'] != 'verified':
        return None
    return {
        **evidence['values'],
        'subscription_observed': True,
        'subscription_source': SUBSCRIPTION_SOURCE,
        'subscription_observed_at': evidence.get('source_displayed_at_ist') or evidence['fetched_at_ist'],
        'subscription_source_displayed_at': evidence.get('source_displayed_at_ist'),
        'subscription_fetched_at': evidence['fetched_at_ist'],
        'subscription_url': evidence['subscription_url'],
    }


def refresh_cached_subscriptions(ipo_url: str, row: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
    """Refresh an open book during bidding hours; retain a dated last observation otherwise."""
    item = dict(row)
    category = str(item.get('category', '')).upper()
    # The application forecast is frozen at the offer cutoff. Listing outcomes
    # have their own exchange-price collector; later bids cannot improve it.
    if category in ('CLOSED', 'LISTED'):
        return item
    if category == 'UPCOMING' or (category == 'OPEN' and not _subscription_window_open()):
        return item
    # The scheduled exchange collector owns this book. A later secondary-site
    # refresh must never replace an independently timestamped NSE reading.
    if item.get('subscription_source') == NSE_SUBSCRIPTION_SOURCE:
        if category == 'OPEN':
            from services.nse_subscription_service import fetch_nse_subscription_evidence
            clock = datetime.now(ZoneInfo('Asia/Kolkata'))
            last_fetch = pd.to_datetime(item.get('subscription_fetched_at'), errors='coerce', utc=True)
            if not force and pd.notna(last_fetch) and (pd.Timestamp(clock) - last_fetch).total_seconds() < 60:
                return item
            _, closing_date = parse_date_interval(item.get('date_range'), clock.year)
            if closing_date is None:
                return item
            try:
                evidence = fetch_nse_subscription_evidence(
                    item.get('company_name') or '', closing_date,
                    nse_symbol=item.get('nse_symbol'),
                    symbol_verified=item.get('nse_symbol_source') == 'nse_current_issue',
                    nse_series=item.get('nse_series'))
                published = pd.to_datetime(evidence.get('source_displayed_at_ist'), errors='coerce', utc=True)
                previous = pd.to_datetime(item.get('subscription_observed_at'), errors='coerce', utc=True)
                values = evidence.get('values') or {}
                if (evidence.get('status') == 'verified' and pd.notna(published)
                        and published <= pd.Timestamp(clock)
                        and (pd.isna(previous) or published >= previous)
                        and all(values.get(key) is not None for key in SUBSCRIPTION_KEYS)):
                    item.update(values)
                    item.update({
                        'subscription_observed': True, 'subscription_observation_status': 'verified',
                        'subscription_source_displayed_at': evidence['source_displayed_at_ist'],
                        'subscription_observed_at': evidence['source_displayed_at_ist'],
                        'subscription_fetched_at': evidence['fetched_at_ist'],
                        'subscription_raw_sha256': evidence.get('raw_sha256'),
                        'subscription_url': evidence.get('subscription_url'),
                        'subscription_source_family': 'nse_exchange',
                        'subscription_is_final': False, 'subscription_capture_phase': 'pre_deadline',
                    })
            except Exception:
                pass  # Preserve the last complete published book on feed failure.
            return item
        return item
    if category not in ('OPEN', 'CLOSED', 'LISTED'):
        return item
    if item.get('subscription_source') == SUBSCRIPTION_SOURCE and item.get('subscription_observed'):
        if category != 'OPEN' and item.get('subscription_is_final'):
            return item
        observed_at = pd.to_datetime(item.get('subscription_observed_at'), errors='coerce', utc=True)
        if not force and pd.notna(observed_at) and (pd.Timestamp.now(tz='UTC') - observed_at).total_seconds() < 300:
            return item
    try:
        observed = fetch_chittorgarh_subscriptions(ipo_url)
        if observed:
            item.update(observed)
            # A secondary site's undated table is not a certified final book.
            item['subscription_is_final'] = False
    except Exception:
        pass
    return item

def reconcile_cache_stages(cache: Dict[str, Any], as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Reconciles all cached offerings to ensure their category ('OPEN', 'UPCOMING', 'CLOSED', 'LISTED')
    is 100% mathematically consistent with the market calendar date (as of today, Sep 22, 2026).
    """
    updated = {}
    for url, item in cache.items():
        copy_item = dict(item)
        # All old cache subscriptions came from the detail-page allocation
        # table, not a subscription observation. Never serve them as bids.
        if copy_item.get('subscription_source') not in (SUBSCRIPTION_SOURCE, NSE_SUBSCRIPTION_SOURCE):
            for key in SUBSCRIPTION_KEYS:
                copy_item[key] = None
            copy_item['subscription_observed'] = False
            copy_item['subscription_observed_at'] = None
        # Versions before 5.2 wrote 20/25 as generic GMP placeholders. Those
        # values are not observations and must not be exposed as market data.
        if copy_item.get('gmp_source') is None and copy_item.get('gmp_pct') in (20, 20.0, 25, 25.0):
            copy_item['gmp_pct'] = None
            copy_item['gmp_source'] = 'unavailable'
        if copy_item.get('gmp_source') in ('registry_snapshot', 'verified_archive'):
            copy_item['gmp_pct'] = None
            copy_item['gmp_rs'] = None
            copy_item['gmp_source'] = 'unavailable'
        if copy_item.get('gmp_pct') == 0 and copy_item.get('gmp_rs') in (None, 0, 0.0):
            copy_item['gmp_pct'] = None
            copy_item['gmp_rs'] = None
            copy_item['gmp_source'] = 'unavailable'
        cat, status = classify_ipo_stage(copy_item, as_of_date=as_of_date)
        copy_item['category'] = cat
        copy_item['status'] = status
        if cat == 'LISTED':
            copy_item['is_listed'] = True
        elif cat in ('OPEN', 'UPCOMING', 'CLOSED') and not copy_item.get('listing_date'):
            copy_item['is_listed'] = False
        updated[url] = copy_item
    return updated

def _read_cache_payload():
    """Keep a validated in-process fallback; never serve a partial JSON write."""
    try:
        with open(CACHE_FILE_PATH,encoding='utf8') as stream:
            payload=json.load(stream)
        if not isinstance(payload,dict) or not all(isinstance(v,dict) for v in payload.values()):
            raise ValueError('Invalid IPO cache shape')
        _LAST_VALID_CACHE[CACHE_FILE_PATH]=copy.deepcopy(payload)
        return payload
    except (OSError,ValueError):
        logging.getLogger(__name__).warning('IPO cache unreadable; retaining the last validated in-process view')
        return copy.deepcopy(_LAST_VALID_CACHE.get(CACHE_FILE_PATH,{}))


def load_scraped_cache() -> Dict[str, Any]:
    """Loads the persistent 2025-2026 live-scraped IPO cache from disk with automatic calendar reconciliation."""
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with _CACHE_WRITE_LOCK:
                raw_cache = _read_cache_payload()
                # Legacy headline records were seeded from code constants, not
                # fetched pages. Keep the file for recovery but never serve
                # unproven records as live market data.
                observed = {url: row for url, row in raw_cache.items() if row.get('source_observed') is True}
                # Read-only, source-verified calendar overlay also feeds the
                # subscription harvester; no extra network requests on cache reads.
                overlay_path=os.path.join(os.path.dirname(CACHE_FILE_PATH),'telemetry','prospective_calendar_overlay.json')
                if os.path.exists(overlay_path):
                    try:
                        import gzip,hashlib
                        with open(overlay_path,encoding='utf8') as overlay_file:
                            overlays=json.load(overlay_file)
                        for url,row in observed.items():
                            overlay=overlays.get(url)
                            if not row.get('date_range') and overlay and overlay.get('listing_date')==row.get('listing_date'):
                                if abs(float(overlay['issue_price'])-float(row.get('issue_price') or 0))>.01:
                                    continue
                                with open(overlay['raw_archive'],'rb') as raw_file:
                                    raw=gzip.decompress(raw_file.read())
                                if hashlib.sha256(raw).hexdigest()==overlay['evidence']['raw_sha256']:
                                    row.update(date_range=overlay['date_range'],calendar_evidence=overlay['evidence'])
                    except (OSError,ValueError,KeyError,TypeError):
                        pass
                return reconcile_cache_stages(observed)
        except Exception:
            return {}
    return {}

def save_scraped_cache(cache_data: Dict[str, Any]) -> None:
    """Saves the live-scraped cache atomically to disk."""
    try:
        with _CACHE_WRITE_LOCK:
            os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
            # Each writer needs its own temporary file. Sharing '.tmp' lets a
            # second thread keep writing after the first renames that inode.
            legacy={url:row for url,row in _read_cache_payload().items()
                    if row.get('source_observed') is not True} if os.path.exists(CACHE_FILE_PATH) else {}
            payload={**legacy,**cache_data}
            temp_path=CACHE_FILE_PATH+'.'+uuid.uuid4().hex+'.tmp'
            try:
                with open(temp_path,'w',encoding='utf8') as stream:
                    json.dump(payload,stream,indent=2,ensure_ascii=False)
                os.replace(temp_path,CACHE_FILE_PATH)
                _LAST_VALID_CACHE[CACHE_FILE_PATH]=copy.deepcopy(payload)
            finally:
                if os.path.exists(temp_path):os.unlink(temp_path)
    except Exception:
        pass


def refresh_open_ipo_subscriptions() -> int:
    """Update active IPO bids in the background while the source window is open."""
    if not _subscription_window_open():
        return 0
    cache = load_scraped_cache()
    updated = 0
    for url, item in cache.items():
        if item.get('category') != 'OPEN':
            continue
        fresh = refresh_cached_subscriptions(url, item)
        if fresh != item:
            cache[url] = fresh
            updated += 1
    if updated:
        save_scraped_cache(cache)
    return updated

def sync_chittorgarh_calendar_delta(force_refresh: bool = False,
                                    refresh_subscriptions: bool = True) -> List[Dict[str, Any]]:
    """
    Fetches the live Mainboard IPO calendar from Chittorgarh dashboard.
    Maintains the local disk cache (data/live_scraped_cache.json) and avoids re-scraping existing IPOs.
    """
    cache = load_scraped_cache()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    url = 'https://www.chittorgarh.com/ipo/ipo_dashboard.asp'
    
    calendar_items = []
    CALENDAR_FETCH_STATUS.update(status='refreshing',attempted_at_ist=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),error=None)
    
    # Ingest the live dashboard. Legacy hardcoded headline records are not seeded.
    try:
        r = httpx.get(url, headers=headers, follow_redirects=True, timeout=12.0)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            tables = soup.find_all('table')
            if tables:
                for tr in tables[0].find_all('tr')[1:]:
                    a = tr.find('a')
                    if not a:
                        continue
                    name = a.get_text(strip=True)
                    href = a.get('href', '')
                    if not href.startswith('http'):
                        href = 'https://www.chittorgarh.com' + href
                    raw_text = tr.get_text(strip=True)
                    
                    m_dates = re.search(r'(\d{1,2})\s*-\s*(\d{1,2}\s+[A-Za-z]{3})', raw_text)
                    date_str = m_dates.group(0) if m_dates else ''
                    
                    # Status categorization via dynamic calendar math against current market date
                    category, status = classify_ipo_stage({'date_range': date_str, 'status': raw_text})
                        
                    cached = cache.get(href, {})
                    
                    item = {
                        'company_name': name,
                        'url': href,
                        'date_range': date_str,
                        'status': status,
                        'category': category,
                        'issue_price': cached.get('issue_price'),
                        'issue_size_cr': cached.get('issue_size_cr'),
                        'sector': cached.get('sector', 'Technology' if any(w in name.lower() for w in ['tech', 'digital', 'software', 'snapdeal', 'moneyview']) else 'Industrials'),
                        'gmp_pct': cached.get('gmp_pct'),
                        'gmp_source': cached.get('gmp_source', 'unavailable'),
                        'rhp_pdf_url': cached.get('rhp_pdf_url'),
                        'drhp_pdf_url': cached.get('drhp_pdf_url')
                    }
                    calendar_items.append(item)
                    
                    # If this is a new URL, scrape its full page details once to populate immutable fundamentals & RHP link
                    if href not in cache or force_refresh:
                        try:
                            parsed = parse_chittorgarh_html(href)
                            parsed_cat, parsed_status = classify_ipo_stage({**parsed, 'date_range': date_str})
                            cache[href] = {
                                **cached,
                                **parsed,
                                'source_observed': True,
                                'status': parsed_status,
                                'category': parsed_cat,
                                'date_range': date_str,
                                'first_discovered_at': cached.get('first_discovered_at') or pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            if cached.get('subscription_source') in (SUBSCRIPTION_SOURCE, NSE_SUBSCRIPTION_SOURCE):
                                for key in (*SUBSCRIPTION_KEYS, 'subscription_observed', 'subscription_source',
                                            'subscription_observed_at', 'subscription_url', 'subscription_is_final',
                                            'subscription_source_family', 'subscription_source_displayed_at',
                                            'subscription_fetched_at', 'subscription_cross_check',
                                            'subscription_observation_status', 'nse_symbol_source'):
                                    cache[href][key] = cached.get(key)
                                if cached.get('nse_symbol_source') == 'nse_current_issue':
                                    cache[href]['nse_symbol'] = cached.get('nse_symbol')
                                    cache[href]['nse_series'] = cached.get('nse_series')
                            item.update({key: parsed.get(key) for key in (
                                'issue_price', 'issue_size_cr', 'sector', 'rhp_pdf_url',
                                'drhp_pdf_url', 'listing_date', 'nse_symbol', 'bse_code',
                            )})
                            item['category'] = parsed_cat
                            item['status'] = parsed_status
                        except Exception:
                            # A dashboard row alone does not verify the detail
                            # page's fundamentals; retry on the next refresh.
                            pass
                    if href in cache and refresh_subscriptions:
                        cache[href] = refresh_cached_subscriptions(href, cache[href], force=force_refresh)
                            
                cache = reconcile_cache_stages(cache)
                save_scraped_cache(cache)
                CALENDAR_FETCH_STATUS.update(status='fresh',source_rows=len(calendar_items),last_success_at_ist=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),error=None)
            else:
                CALENDAR_FETCH_STATUS.update(status='cached_fallback',error='Calendar source has no recognised tables')
        else:
            CALENDAR_FETCH_STATUS.update(status='cached_fallback',error=f'Calendar HTTP {r.status_code}')
    except Exception as exc:
        CALENDAR_FETCH_STATUS.update(status='cached_fallback',error=f'{type(exc).__name__}: {exc}'[:350])
        
    # Fallback to returning cached list if network fetch failed
    if not calendar_items:
        for k, v in cache.items():
            calendar_items.append({
                'company_name': v.get('company_name', 'IPO'),
                'url': k,
                'date_range': v.get('date_range', ''),
                'status': v.get('status', 'UPCOMING'),
                'category': v.get('category', 'UPCOMING'),
                'issue_price': v.get('issue_price'),
                'issue_size_cr': v.get('issue_size_cr'),
                'sector': v.get('sector', 'Industrials'),
                'gmp_pct': v.get('gmp_pct'),
                'gmp_source': v.get('gmp_source', 'unavailable'),
                'rhp_pdf_url': v.get('rhp_pdf_url'),
                'drhp_pdf_url': v.get('drhp_pdf_url')
            })

    # Enrich calendar items and cache with live empirical GMP
    try:
        from services.gmp_service import gmp_service
        for item in calendar_items:
            gr = gmp_service.resolve_gmp(
                item['company_name'],
                chittorgarh_url=item.get('url'),
                issue_price=float(item['issue_price']) if item.get('issue_price') else None,
                is_listed=item.get('category') == 'LISTED',
            )
            if gr.get('gmp_pct') is not None:
                item['gmp_pct'] = gr['gmp_pct']
                item['gmp_rs'] = gr.get('gmp_rs')
                item['gmp_source'] = gr.get('gmp_source', 'live_investorgain')
                item['gmp_observed_at'] = gr.get('gmp_observed_at')
                item['gmp_is_stale'] = bool(gr.get('gmp_is_stale'))
                if gr.get('price') and not item.get('issue_price'):
                    item['issue_price'] = gr['price']
                if gr.get('issue_size_cr') and not item.get('issue_size_cr'):
                    item['issue_size_cr'] = gr['issue_size_cr']
                if item.get('url') and item['url'] in cache:
                    cache[item['url']]['gmp_pct'] = gr['gmp_pct']
                    cache[item['url']]['gmp_rs'] = gr.get('gmp_rs')
                    cache[item['url']]['gmp_source'] = gr.get('gmp_source', 'live_investorgain')
                    cache[item['url']]['gmp_observed_at'] = gr.get('gmp_observed_at')
                    cache[item['url']]['gmp_is_stale'] = bool(gr.get('gmp_is_stale'))
            else:
                item['gmp_pct'] = None
                item['gmp_rs'] = None
                item['gmp_source'] = 'unavailable'
                if item.get('url') and item['url'] in cache:
                    cache[item['url']]['gmp_pct'] = None
                    cache[item['url']]['gmp_rs'] = None
                    cache[item['url']]['gmp_source'] = 'unavailable'
        save_scraped_cache(cache)
    except Exception:
        pass
            
    return calendar_items

MACRO_CACHE_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'live_macro_cache.json')


def _read_macro_cache() -> Dict[str, Any]:
    try:
        with open(MACRO_CACHE_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_macro_cache(payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(MACRO_CACHE_FILE_PATH), exist_ok=True)
        temp_path = MACRO_CACHE_FILE_PATH + '.' + uuid.uuid4().hex + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_path, MACRO_CACHE_FILE_PATH)
    except Exception:
        pass


def _parse_nse_market_timestamp(value: Any) -> datetime:
    """Treat the exchange's quote time, never our fetch time, as market as-of."""
    if not value:
        raise ValueError('NSE index response has no exchange timestamp')
    return datetime.strptime(str(value), '%d-%b-%Y %H:%M').replace(
        tzinfo=ZoneInfo('Asia/Kolkata'))


def fetch_live_macro_regime() -> Dict[str, Any]:
    """Fetch official NSE spot values and Yahoo history, falling back only to last-known data."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36',
        'Accept': 'application/json,text/plain,*/*',
        'Referer': 'https://www.nseindia.com/market-data/live-equity-market',
    }
    requested_at = datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=10.0) as client:
            nse_response = client.get('https://www.nseindia.com/api/allIndices')
            nse_response.raise_for_status()
            nse_payload = nse_response.json()
            index_rows = nse_payload.get('data', [])
            source_asof = _parse_nse_market_timestamp(nse_payload.get('timestamp'))
            nifty_row = next((r for r in index_rows if r.get('index') == 'NIFTY 50'), None)
            vix_row = next((r for r in index_rows if r.get('index') == 'INDIA VIX'), None)
            if not nifty_row or not vix_row:
                raise ValueError('NSE response did not contain NIFTY 50 and INDIA VIX')

            curr_nifty = float(nifty_row['last'])
            curr_vix = round(float(vix_row['last']), 2)
            nifty_10d_ret = None
            history_source = 'unavailable'
            try:
                yahoo = client.get('https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?range=1mo&interval=1d')
                yahoo.raise_for_status()
                closes = yahoo.json()['chart']['result'][0]['indicators']['quote'][0]['close']
                closes = [float(v) for v in closes if v is not None]
                if len(closes) >= 10:
                    nifty_10d_ret = round(((closes[-1] - closes[-10]) / closes[-10]) * 100.0, 2)
                    history_source = 'yahoo_finance'
            except Exception:
                cached = _read_macro_cache()
                if cached.get('nifty_10d_return') is not None:
                    nifty_10d_ret = float(cached['nifty_10d_return'])
                    history_source = 'last_known_cache'

        regime_label = 'Stable' if curr_vix < 16.0 else ('Moderate Volatility' if curr_vix < 22.0 else 'High Volatility')
        result = {
            'india_vix': curr_vix,
            'vix_change_pct': round(float(vix_row.get('percentChange', 0.0)), 2),
            'nifty_close': round(curr_nifty, 2),
            'nifty_day_change_pct': round(float(nifty_row.get('percentChange', 0.0)), 2),
            'nifty_10d_return': nifty_10d_ret,
            'regime_label': regime_label,
            'source': 'nse_india',
            'history_source': history_source,
            'is_fallback': False,
            'timestamp': source_asof.isoformat(),
            'retrieved_at_ist': requested_at,
            'is_market_session_current': source_asof.date() == datetime.now(
                ZoneInfo('Asia/Kolkata')).date(),
        }
        _save_macro_cache(result)
        return result
    except Exception as exc:
        cached = _read_macro_cache()
        if cached:
            cached.update({
                'source': 'last_known_cache',
                'is_fallback': True,
                'source_error': str(exc),
                'requested_at': requested_at,
            })
            return cached
        return {
            'india_vix': None,
            'vix_change_pct': None,
            'nifty_close': None,
            'nifty_day_change_pct': None,
            'nifty_10d_return': None,
            'regime_label': 'Live data unavailable',
            'source': 'unavailable',
            'history_source': 'unavailable',
            'is_fallback': True,
            'source_error': str(exc),
            'timestamp': None,
            'requested_at': requested_at,
        }

def find_chittorgarh_ipo_url(company_name: str) -> Optional[str]:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    endpoints = [
        'https://www.chittorgarh.com/report/ipo-in-india-list-main-board-sme/82/main-board/',
        'https://www.chittorgarh.com/report/ipo-in-india-list-main-board-sme/82/sme/'
    ]
    
    clean_query = company_name.lower().replace('ltd', '').replace('limited', '').replace('ipo', '').strip()
    
    for ep in endpoints:
        try:
            r = httpx.get(ep, headers=headers, follow_redirects=True, timeout=8.0)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                for a in soup.find_all('a', href=True):
                    text = a.get_text(strip=True).lower()
                    href = a['href']
                    if '/ipo/' in href and clean_query in text:
                        if not href.startswith('http'):
                            href = 'https://www.chittorgarh.com' + href
                        return href
        except Exception:
            continue
    return None

def parse_chittorgarh_html(url: str) -> Dict[str, Any]:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    r = httpx.get(url, headers=headers, follow_redirects=True, timeout=10.0)
    soup = BeautifulSoup(r.text, 'html.parser')
    
    h1 = soup.find('h1')
    name = h1.get_text(strip=True).replace(' IPO Details', '').replace(' IPO', '').strip() if h1 else 'Target IPO'
    
    details = {}
    for table in soup.find_all('table'):
        for tr in table.find_all('tr'):
            tds = tr.find_all(['td', 'th'])
            if len(tds) == 2:
                k = tds[0].get_text(strip=True).replace('\u20b9', 'Rs ')
                v = tds[1].get_text(strip=True).replace('\u20b9', 'Rs ')
                details[k] = v

    # Extract Price Band
    price_str = details.get('Price Band', details.get('Issue Price', ''))
    price_matches = re.findall(r'[0-9]+', price_str)
    upper_price = float(price_matches[-1]) if price_matches else None
    
    page_text = ' '.join(soup.text.split()).encode('ascii', 'ignore').decode('ascii')
    
    # Extract Issue Size in Cr
    tot_sz = None
    m_tot = re.search(r'Total Issue Size[^\d]*[\d,]+\s*shares[^\(]*\(\s*agg\.\s*up\s*to\s*[\D]*([\d,]+(?:\.\d+)?)\s*Cr', page_text, re.IGNORECASE)
    if m_tot:
        tot_sz = float(m_tot.group(1).replace(',', ''))
    else:
        m_tot2 = re.search(r'Issue Size[^\d]*[\D]*([\d,]+(?:\.\d+)?)\s*Cr', page_text, re.IGNORECASE)
        if m_tot2:
            tot_sz = float(m_tot2.group(1).replace(',', ''))
        elif 'Total Issue Size' in details:
            m_d = re.search(r'([\d,]+(?:\.\d+)?)\s*Cr', details['Total Issue Size'])
            if m_d: tot_sz = float(m_d.group(1).replace(',', ''))

    issue_size_cr = tot_sz
    
    # Extract Offer for Sale in Cr
    ofs_sz = None
    m_ofs = re.search(r'Offer for Sale[^\(]*\(\s*agg\.\s*up\s*to\s*[\D]*([\d,]+(?:\.\d+)?)\s*Cr', page_text, re.IGNORECASE)
    if m_ofs:
        ofs_sz = float(m_ofs.group(1).replace(',', ''))
    elif 'Offer for Sale' in details:
        m_d = re.search(r'([\d,]+(?:\.\d+)?)\s*Cr', details['Offer for Sale'])
        if m_d: ofs_sz = float(m_d.group(1).replace(',', ''))
        
    # Extract Fresh Issue in Cr
    fresh_sz = None
    m_fr = re.search(r'Fresh Issue[^\(]*\(\s*agg\.\s*up\s*to\s*[\D]*([\d,]+(?:\.\d+)?)\s*Cr', page_text, re.IGNORECASE)
    if m_fr:
        fresh_sz = float(m_fr.group(1).replace(',', ''))
    elif 'Fresh Issue' in details:
        m_d = re.search(r'([\d,]+(?:\.\d+)?)\s*Cr', details['Fresh Issue'])
        if m_d: fresh_sz = float(m_d.group(1).replace(',', ''))
        
    # Calculate exact OFS ratio
    if ofs_sz is not None and issue_size_cr is not None and issue_size_cr > 0:
        ofs_ratio = round(ofs_sz / issue_size_cr, 3)
    elif fresh_sz is not None and issue_size_cr is not None and issue_size_cr > 0:
        ofs_ratio = round(max(0.0, issue_size_cr - fresh_sz) / issue_size_cr, 3)
    else:
        ofs_ratio = None
    
    # The IPO details page describes allocation quotas, not bid demand.
    # Subscription multiples are fetched separately from /ipo_subscription/.

    # Sector heuristic
    sector = 'Industrials'
    name_l = name.lower()
    if any(w in name_l for w in ['dairy', 'food', 'milk', 'agro', 'beverage', 'fmcg', 'retail', 'sugar', 'consumer', 'biscuit', 'grain', 'oil', 'tea', 'coffee']):
        sector = 'Consumer Staples'
    elif any(w in name_l for w in ['tech', 'software', 'digital', 'info', 'data', 'cloud', 'system', 'cyber', 'communication', 'telecom', 'network', 'paytm', 'one 97', 'media', 'internet', 'ai', 'analytics']):
        sector = 'Technology'
    elif any(w in name_l for w in ['energy', 'power', 'solar', 'green', 'renew', 'gas', 'oil']):
        sector = 'Energy'
    elif any(w in name_l for w in ['health', 'pharma', 'lab', 'bio', 'hospital', 'life', 'medic', 'clinic', 'diag', 'therapeutics']):
        sector = 'Healthcare'
    elif any(w in name_l for w in ['bank', 'finance', 'fin', 'capital', 'insurance', 'credit', 'wealth', 'housing', 'lending', 'broker', 'fund', 'pay', 'fintech']):
        sector = 'Financials'
    elif any(w in name_l for w in ['motor', 'auto', 'ev', 'mobility', 'vehicle']):
        sector = 'Automotive'

    # Extract Listing Date & Trading Status
    listing_date_str = None
    listing_iso = None
    is_listed = False
    listing_open_price = None
    listing_high_price = None
    listing_day_close_price = None
    nse_symbol = None
    bse_code = None
    
    for tr in soup.find_all('tr'):
        txt = ' '.join(tr.text.split())
        if 'Listed on' in txt or 'Listing Date' in txt:
            m = re.search(r'(?:Listed on|Listing Date)\s*(?:[A-Za-z]+,\s*)?([A-Za-z]+\s*\d{1,2},\s*\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})', txt, re.IGNORECASE)
            if m:
                listing_date_str = m.group(1).strip()
                try:
                    dt = pd.to_datetime(listing_date_str)
                    listing_iso = dt.strftime('%Y-%m-%d')
                    if dt <= pd.Timestamp.now():
                        is_listed = True
                except Exception:
                    pass
                    
        # Scrape BSE code & NSE symbol from "BSE Script Code / NSE Symbol" row
        # Row format: "BSE Script Code / NSE Symbol 544868 / MILKYMIST"
        if 'BSE Script Code' in txt or 'NSE Symbol' in txt or ('NSE' in txt and '/' in txt and is_listed):
            # Extract BSE numeric code (5-6 digits)
            bse_m = re.search(r'\b(5\d{5})\b', txt)
            if bse_m:
                bse_code = bse_m.group(1).strip()
            # Extract NSE ticker symbol from the last token after slash
            parts = [p.strip() for p in txt.split('/')]
            if len(parts) >= 2:
                last_segment = parts[-1]
                tokens = [t for t in re.findall(r'[A-Z0-9]{3,14}', last_segment.upper()) if not t.isdigit() and t not in ('NSE', 'BSE', 'IPO', 'LTD', 'SYMBOL', 'CODE', 'SCRIPT')]
                if tokens:
                    nse_symbol = tokens[0]
            if not nse_symbol:
                m = re.search(r'NSE\s*Symbol\s*[\d/]*\s*([A-Z]{3,14})\b', txt.upper())
                if m:
                    sym = m.group(1).strip()
                    if sym not in ('NSE', 'BSE', 'IPO', 'LTD', 'AND', 'THE', 'CODE', 'SCRIPT', 'SYMBOL'):
                        nse_symbol = sym
                    
        # Scrape listing open, high, and day close price from Chittorgarh "Price Details" table
        if txt.startswith('Open') and is_listed:
            nums = re.findall(r'[\d,]+\.?\d*', txt)
            if nums:
                try:
                    listing_open_price = float(nums[-1].replace(',', '')) if len(nums) > 1 else float(nums[0].replace(',', ''))
                except Exception:
                    pass
        elif txt.startswith('High') and is_listed:
            nums = re.findall(r'[\d,]+\.?\d*', txt)
            if nums:
                try:
                    listing_high_price = float(nums[-1].replace(',', ''))
                except Exception:
                    pass
        elif (txt.startswith('Last Trade') or txt.startswith('Close') or txt.startswith('Day Close')) and is_listed:
            nums = re.findall(r'[\d,]+\.?\d*', txt)
            if nums:
                try:
                    # Last number in row is the NSE closing price (or BSE if single exchange)
                    listing_day_close_price = float(nums[-1].replace(',', ''))
                except Exception:
                    pass
                    
    # Extract official RHP / DRHP PDF URLs
    drhp_pdf_url = None
    rhp_pdf_url = None
    for a in soup.find_all('a', href=True):
        txt_a = a.get_text(strip=True).upper()
        href_a = a['href']
        if href_a.endswith('.pdf') or 'bseindia.com' in href_a or 'nseindia.com' in href_a or 'sebi.gov.in' in href_a or 'rhp' in href_a.lower() or 'drhp' in href_a.lower():
            if 'DRHP' in txt_a or 'DRHP' in href_a:
                if not drhp_pdf_url:
                    drhp_pdf_url = href_a
            elif 'RHP' in txt_a or 'RHP' in href_a:
                if not rhp_pdf_url:
                    rhp_pdf_url = href_a
                    
    # GMP is unofficial and time-sensitive. Use it only when this fetched page
    # explicitly publishes a value; never inject a generic percentage.
    gmp_pct = None
    gmp_source = 'unavailable'
    gmp_percent = re.search(r'(?:GMP|Grey Market Premium)[^%]{0,80}(-?\d+(?:\.\d+)?)\s*%', page_text, re.IGNORECASE)
    gmp_rupees = re.search(r'(?:GMP|Grey Market Premium)[^\d-]{0,80}(?:Rs\.?\s*)?(-?\d+(?:\.\d+)?)', page_text, re.IGNORECASE)
    if gmp_percent:
        gmp_pct = float(gmp_percent.group(1))
        gmp_source = 'scraped_page_percentage'
    elif gmp_rupees and upper_price:
        gmp_pct = round(float(gmp_rupees.group(1)) / float(upper_price) * 100.0, 2)
        gmp_source = 'scraped_page_rupees'

    return {
        'company_name': name,
        'sector': sector,
        'issue_price': upper_price,
        'issue_size_cr': issue_size_cr,
        'promoter_ofs_ratio': ofs_ratio,
        'qib_subscription': None,
        'retail_subscription': None,
        'hni_subscription': None,
        'subscription_observed': False,
        'gmp_pct': gmp_pct,
        'gmp_source': gmp_source,
        'risk_factor_count': None,
        'objects_capex_ratio': None,
        'listing_date': listing_iso,
        'listing_open_price': listing_open_price,
        'listing_high_price': listing_high_price,
        'listing_day_close_price': listing_day_close_price,
        'nse_symbol': nse_symbol,
        'bse_code': bse_code,
        'is_listed': is_listed,
        'rhp_pdf_url': rhp_pdf_url,
        'drhp_pdf_url': drhp_pdf_url
    }

def scrape_chittorgarh_url(url_or_name: str) -> Dict[str, Any]:
    macro = fetch_live_macro_regime()
    
    # 0. Check Persistent Live Scraped Cache First (Sub-2ms response)
    cache = load_scraped_cache()
    if url_or_name in cache:
        cached_item = cache[url_or_name]
        return {
            'success': True,
            'source': 'Live Calendar Offering',
            'data': {**cached_item, **macro, 'url': url_or_name}
        }
    for k, v in cache.items():
        if v.get('company_name', '').lower() in url_or_name.lower() or url_or_name.lower() in v.get('company_name', '').lower():
            return {
                'success': True,
                'source': 'Live Calendar Offering',
                'data': {**v, **macro, 'url': k}
            }
            
    # Only fetched pages or canonical exchange records can satisfy research.
    # Legacy code-constant headline entries are deliberately not consulted.
    # If it is already a direct URL
    if url_or_name.startswith('http'):
        try:
            parsed = parse_chittorgarh_html(url_or_name)
            category, status = classify_ipo_stage(parsed)
            parsed.update({'category': category, 'status': status})
            parsed = refresh_cached_subscriptions(url_or_name, parsed)
            # Update cache
            cache[url_or_name] = {
                **parsed,
                'source_observed': True,
                'first_discovered_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            save_scraped_cache(cache)
            return {
                'success': True,
                'source': 'Live Issue Scrape',
                'data': {**parsed, **macro, 'url': url_or_name}
            }
        except Exception as e:
            pass

    # 3. Dynamic Search on Chittorgarh
    found_url = find_chittorgarh_ipo_url(url_or_name)
    if found_url:
        try:
            parsed = parse_chittorgarh_html(found_url)
            category, status = classify_ipo_stage(parsed)
            parsed.update({'category': category, 'status': status})
            parsed = refresh_cached_subscriptions(found_url, parsed)
            cache[found_url] = {
                **parsed,
                'source_observed': True,
                'first_discovered_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            save_scraped_cache(cache)
            return {
                'success': True,
                'source': 'Live Issue Scrape',
                'data': {**parsed, **macro, 'url': found_url}
            }
        except Exception as e:
            pass

    # 4. Check Unified Entity Resolver if available
    try:
        from core.unified_resolver import UnifiedIPOResolver
        u_res = UnifiedIPOResolver().resolve(url_or_name)
        if u_res.get('success'):
            return {
                'success': True,
                'source': u_res.get('source', 'Unified Entity Resolver'),
                'data': {**u_res['data'], **macro}
            }
    except Exception:
        pass

    # 5. Return explicit unverified profile without hallucinated financials
    return {
        'success': False,
        'source': 'Unverified Input Target',
        'data': {
            'company_name': url_or_name if url_or_name else 'Target IPO',
            'sector': 'Unclassified',
            'qib_subscription': None,
            'retail_subscription': None,
            'hni_subscription': None,
            'issue_size_cr': 0.0,
            'issue_price': 0.0,
            'gmp_pct': None,
            'gmp_source': 'unavailable',
            'promoter_ofs_ratio': None,
            'risk_factor_count': None,
            'objects_capex_ratio': None,
            **macro
        }
    }
