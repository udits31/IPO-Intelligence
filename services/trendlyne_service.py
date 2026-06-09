import logging
import time
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
import httpx
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_TRENDLYNE_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 120

TRENDLYNE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://trendlyne.com/',
    'X-Requested-With': 'XMLHttpRequest',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
}

class TrendlyneService:
    def __init__(self, timeout: float = 6.0):
        self.timeout = timeout
        self.api_url = 'https://trendlyne.com/member/api/ac_snames/stock/'

    def fetch_live_stock_data(self, query: str) -> Optional[Dict[str, Any]]:
        clean_query = query.strip()
        if not clean_query:
            return None

        cache_key = clean_query.upper()
        now = time.time()
        cached = _TRENDLYNE_CACHE.get(cache_key)
        if cached and (now - cached['timestamp'] < CACHE_TTL_SECONDS):
            return cached['data'].copy() if cached['data'] else None

        try:
            with httpx.Client(headers=TRENDLYNE_HEADERS, timeout=self.timeout) as client:
                resp = client.get(self.api_url, params={'term': clean_query})
                if resp.status_code != 200:
                    logger.warning("Trendlyne API returned status %s for query '%s'", resp.status_code, clean_query)
                    return None
                
                results = resp.json() if resp.status_code == 200 else []
                has_valid_item = isinstance(results, list) and any(
                    isinstance(r, dict) and (r.get('currentPrice') is not None or r.get('stock_code'))
                    for r in results
                )

                if not has_valid_item:
                    # Try query normalization without corporate suffixes (e.g. 'National Stock Exchange of India' -> 'National Stock Exchange')
                    clean_alt = re.sub(r'\b(of india|limited|ltd|corporation|corp|co)\b', '', clean_query, flags=re.I).strip()
                    clean_alt = re.sub(r'\s+', ' ', clean_alt)
                    if clean_alt and clean_alt.lower() != clean_query.lower():
                        resp_alt = client.get(self.api_url, params={'term': clean_alt})
                        if resp_alt.status_code == 200:
                            alt_results = resp_alt.json()
                            if isinstance(alt_results, list) and any(
                                isinstance(r, dict) and (r.get('currentPrice') is not None or r.get('stock_code'))
                                for r in alt_results
                            ):
                                results = alt_results

                if not results or not isinstance(results, list):
                    logger.info("Trendlyne API returned no results for '%s'", clean_query)
                    _TRENDLYNE_CACHE[cache_key] = {'data': None, 'timestamp': now}
                    return None

                item = None
                for res in results:
                    if isinstance(res, dict) and (res.get('currentPrice') is not None or res.get('stock_code')):
                        item = res
                        break

                if not item:
                    _TRENDLYNE_CACHE[cache_key] = {'data': None, 'timestamp': now}
                    return None

                label = item.get('label', '')
                clean_name = re.sub(r'\s*\([A-Z0-9_\-]+\)\s*$', '', label).strip() if label else item.get('stock_code', clean_query)

                urls = item.get('urls', [])
                url_dict = {}
                for entry in urls:
                    if isinstance(entry, list) and len(entry) >= 2:
                        url_dict[str(entry[0]).lower()] = entry[1]

                current_p = None
                try:
                    raw_p = item.get('currentPrice')
                    if raw_p is not None:
                        current_p = round(float(raw_p), 2)
                except (ValueError, TypeError):
                    current_p = None

                day_chg = None
                try:
                    raw_chg = item.get('day_change')
                    if raw_chg is not None:
                        day_chg = round(float(raw_chg), 2)
                except (ValueError, TypeError):
                    day_chg = None

                day_chg_pct = None
                try:
                    raw_pct = item.get('day_changeP')
                    if raw_pct is not None:
                        day_chg_pct = round(float(raw_pct), 2)
                except (ValueError, TypeError):
                    day_chg_pct = None

                day_high = None
                try:
                    raw_hi = item.get('day_high')
                    if raw_hi is not None:
                        day_high = round(float(raw_hi), 2)
                except (ValueError, TypeError):
                    day_high = None

                day_low = None
                try:
                    raw_lo = item.get('day_low')
                    if raw_lo is not None:
                        day_low = round(float(raw_lo), 2)
                except (ValueError, TypeError):
                    day_low = None

                data = {
                    'stock_code': item.get('stock_code') or item.get('NSEcode') or '',
                    'company_name': clean_name,
                    'nse_symbol': item.get('NSEcode') or item.get('stock_code') or '',
                    'bse_scrip_code': str(item.get('BSEcode', '')) if item.get('BSEcode') else '',
                    'current_price': current_p,
                    'day_change': day_chg,
                    'day_change_pct': day_chg_pct,
                    'day_high': day_high,
                    'day_low': day_low,
                    'sector': item.get('sectorName'),
                    'industry': (item.get('industry') or {}).get('industry_name', '') if isinstance(item.get('industry'), dict) else '',
                    'isin': item.get('ISIN') or '',
                    'trendlyne_url': item.get('nexturl') or url_dict.get('overview', ''),
                    'technicals_url': url_dict.get('technicals', ''),
                    'delivery_url': url_dict.get('delivery', ''),
                    'forecaster_url': url_dict.get('forecaster', ''),
                    'source': 'Trendlyne autocomplete',
                    'source_family': 'trendlyne',
                    'fetched_at_ist': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds'),
                    'source_quote_at': None,
                    'quote_timestamp_status': 'not_provided_by_source',
                }

                _TRENDLYNE_CACHE[cache_key] = {'data': data, 'timestamp': now}
                if data['stock_code']:
                    _TRENDLYNE_CACHE[data['stock_code'].upper()] = {'data': data, 'timestamp': now}
                return data

        except Exception as exc:
            logger.warning("Error fetching Trendlyne live data for '%s': %s", clean_query, exc)
            return None

trendlyne_service = TrendlyneService()
