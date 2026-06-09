"""
Live & Historical Grey Market Premium (GMP) Service
===================================================
Fetches unofficial Grey Market Premium (GMP) from the InvestorGain /
Chittorgarh source family. These are not exchange quotes or independent feeds.
Implements a 15-minute file-backed cache to minimize upstream requests.
"""

import os
import re
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

CACHE_FILE = 'data/live_gmp_cache.json'
CACHE_TTL_SECONDS = 900  # 15 minutes

def _clean_token(name: str) -> str:
    if not name:
        return ''
    n = name.lower()
    for drop in ['ltd', 'limited', 'india', 'ipo', 'enterprises', 'industries', 'technologies', 'services', 'corp', 'corporation', 'bse', 'sme', 'nse', 'u', 'o', 'c', 'allotted', '()']:
        n = re.sub(rf'\b{drop}\b', '', n)
    return re.sub(r'[^a-z0-9]', '', n)

class GMPService:
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self.cache: Dict[str, Any] = self._load_cache()
        self.last_fetch_status = 'not_attempted'
        self.last_attempt_at = 0.0

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                # Migrate legacy ambiguity: older versions stored the source's
                # "-- (0.00%)" sentinel as a genuine zero and cached manually
                # keyed archive values without a verifiable observation URL.
                for row in payload.get('live_table', {}).values():
                    if row.get('gmp_pct') == 0 and row.get('gmp_rs') == 0 and not row.get('gmp_observation_status'):
                        row['gmp_pct'] = None
                        row['gmp_rs'] = None
                        row['gmp_observation_status'] = 'unavailable'
                payload['entity_cache'] = {
                    key: row for key, row in payload.get('entity_cache', {}).items()
                    if row.get('gmp_source') != 'verified_archive'
                    and not (row.get('gmp_pct') == 0 and row.get('gmp_rs') == 0 and not row.get('gmp_observation_status'))
                }
                return payload
            except Exception as e:
                logger.warning(f"Error loading GMP cache: {e}")
        return {'last_updated': 0, 'live_table': {}, 'entity_cache': {}}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving GMP cache: {e}")

    def fetch_live_table(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        now = time.time()
        last_updated = self.cache.get('last_updated', 0)
        if self.last_fetch_status == 'failed_using_cache' and now - self.last_attempt_at < 60:
            return self.cache.get('live_table', {})
        
        if not force_refresh and (now - last_updated < CACHE_TTL_SECONDS) and self.cache.get('live_table'):
            self.last_fetch_status = 'fresh_cache'
            return self.cache['live_table']

        logger.info("Fetching fresh live GMP table from InvestorGain...")
        self.last_attempt_at = now
        records: Dict[str, Dict[str, Any]] = {}

        try:
            # httpx follows the provider's current redirect and works with the
            # host's normal TLS/network stack. curl_cffi was failing on Windows
            # with a socket-permission error even when the site was reachable.
            import httpx
            r = httpx.get(
                'https://www.investorgain.com/report/live-ipo-gmp/331/',
                timeout=15.0,
                follow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36'},
            )
            
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                table = soup.find('table')
                if table:
                    rows = table.find_all('tr')
                    for row in rows[1:]:
                        cols = row.find_all(['th', 'td'])
                        if len(cols) >= 5:
                            raw_name = cols[0].get_text(separator=' ', strip=True)
                            gmp_text = cols[1].get_text(separator=' ', strip=True)
                            price_text = cols[4].get_text(separator=' ', strip=True) if len(cols) > 4 else ''
                            size_text = cols[5].get_text(separator=' ', strip=True) if len(cols) > 5 else ''

                            # Parse GMP: e.g. "35 (26.12%)" or "12 (9.23%)" or "-- (0.00%)"
                            gmp_rs = None
                            gmp_pct = None
                            
                            m = re.search(r'(-?\d+(?:\.\d+)?)\s*\(\s*(-?\d+(?:\.\d+)?)\s*%\s*\)', gmp_text)
                            if m:
                                gmp_rs = float(m.group(1))
                                gmp_pct = float(m.group(2))
                            # "-- (0.00%)" means the publisher has no quote. It
                            # is missing data, not an observed zero premium.

                            # Parse Price
                            price = None
                            pm = re.search(r'(\d+(?:\.\d+)?)', price_text.replace(',', ''))
                            if pm:
                                price = float(pm.group(1))

                            # Parse Size
                            size_cr = None
                            sm = re.search(r'(\d+(?:\.\d+)?)\s*Cr', size_text)
                            if sm:
                                size_cr = float(sm.group(1))

                            clean_key = _clean_token(raw_name)
                            if clean_key:
                                records[clean_key] = {
                                    'raw_name': raw_name,
                                    'gmp_rs': gmp_rs,
                                    'gmp_pct': gmp_pct,
                                    'price': price,
                                    'issue_size_cr': size_cr,
                                    'gmp_observation_status': 'observed' if gmp_pct is not None else 'unavailable',
                                    'updated_at': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds')
                                }

                self.cache['live_table'] = records
                self.cache['last_updated'] = now
                self.cache['last_success_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                self._save_cache()
                self.last_fetch_status = 'fresh'
                logger.info(f"Successfully cached {len(records)} live GMP records.")
        except Exception as e:
            self.last_fetch_status = 'failed_using_cache'
            logger.error(f"Failed to fetch live GMP table: {e}")

        return self.cache.get('live_table', {})

    def resolve_gmp(self, company_name: str, chittorgarh_url: Optional[str] = None, issue_price: Optional[float] = None, is_listed: bool = False) -> Dict[str, Any]:
        """
        Multi-tier empirical GMP resolver:
        1. Current live table fuzzy match
        2. Dedicated InvestorGain chr-gmp endpoint by Chittorgarh ID
        3. Final historical observation cache, only after exchange listing
        """
        if not company_name:
            return {'gmp_rs': None, 'gmp_pct': None, 'gmp_source': 'unavailable'}

        clean_q = _clean_token(company_name)
        entity_cache = self.cache.setdefault('entity_cache', {})

        # Active/upcoming GMP is mutable. Never serve its entity cache before
        # refreshing the current table. A 15-minute cache created by a
        # successful fetch is considered current; a failed fetch is not.
        live_table = self.fetch_live_table()
        live_table_is_current = self.last_fetch_status in ('fresh', 'fresh_cache') and (
            time.time() - float(self.cache.get('last_updated') or 0) < CACHE_TTL_SECONDS
        )

        # 1. Match against the current live table.
        # Direct key match
        if live_table_is_current and clean_q in live_table:
            match = live_table[clean_q]
            if match.get('gmp_pct') is None or (
                match.get('gmp_pct') == 0 and match.get('gmp_rs') == 0 and not match.get('gmp_observation_status')
            ):
                return {'gmp_rs': None, 'gmp_pct': None, 'gmp_source': 'unavailable'}
            res = {
                'gmp_rs': match['gmp_rs'],
                'gmp_pct': match['gmp_pct'],
                'gmp_source': 'live_investorgain',
                'gmp_source_family': 'chittorgarh_investorgain',
                'gmp_source_url': 'https://www.investorgain.com/report/live-ipo-gmp/331/',
                'gmp_official': False,
                'gmp_observed_at': match.get('updated_at'),
                'gmp_is_stale': False,
                'gmp_observation_status': 'observed',
                'price': match.get('price'),
                'issue_size_cr': match.get('issue_size_cr')
            }
            entity_cache[clean_q] = res
            self._save_cache()
            return res

        # Substring / token matching
        for k, v in live_table.items() if live_table_is_current else []:
            if len(clean_q) >= 4 and (clean_q in k or k in clean_q):
                if v.get('gmp_pct') is None or (
                    v.get('gmp_pct') == 0 and v.get('gmp_rs') == 0 and not v.get('gmp_observation_status')
                ):
                    continue
                res = {
                    'gmp_rs': v['gmp_rs'],
                    'gmp_pct': v['gmp_pct'],
                    'gmp_source': 'live_investorgain',
                    'gmp_source_family': 'chittorgarh_investorgain',
                    'gmp_source_url': 'https://www.investorgain.com/report/live-ipo-gmp/331/',
                    'gmp_official': False,
                    'gmp_observed_at': v.get('updated_at'),
                    'gmp_is_stale': False,
                    'gmp_observation_status': 'observed',
                    'price': v.get('price'),
                    'issue_size_cr': v.get('issue_size_cr')
                }
                entity_cache[clean_q] = res
                self._save_cache()
                return res

        # 2. Check the dedicated live page when the main table was reachable
        # but name matching did not find a row. Avoid hammering the provider
        # with per-company calls during a broader outage.
        if chittorgarh_url and live_table_is_current:
            m = re.search(r'/ipo/([a-z0-9\-]+)/(\d+)/?', chittorgarh_url)
            if m:
                slug, cid = m.group(1), m.group(2)
                res = self._fetch_dedicated_gmp(slug, cid, issue_price)
                if res.get('gmp_pct') is not None:
                    entity_cache[clean_q] = res
                    self._save_cache()
                    return res

        # 3. Once listed, the last observed pre-listing GMP becomes historical
        # context and no longer needs to refresh. Never use this path for an
        # open, upcoming, or closed-but-not-listed issue.
        if is_listed and clean_q in entity_cache:
            cached = entity_cache[clean_q]
            if cached.get('gmp_pct') is not None and not (
                cached.get('gmp_pct') == 0 and cached.get('gmp_rs') == 0 and not cached.get('gmp_observation_status')
            ):
                observed_at = cached.get('gmp_observed_at')
                if not observed_at and self.cache.get('last_updated'):
                    observed_at = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.cache['last_updated']))
                return {
                    **cached,
                    'gmp_source': 'historical_final_gmp',
                    'gmp_observed_at': observed_at,
                    'gmp_is_stale': False,
                    'gmp_observation_status': 'historical',
                }

        return {'gmp_rs': None, 'gmp_pct': None, 'gmp_source': 'unavailable'}

    def _fetch_dedicated_gmp(self, slug: str, cid: str, issue_price: Optional[float] = None) -> Dict[str, Any]:
        try:
            import httpx
            url = f'https://www.investorgain.com/chr-gmp/{slug}/{cid}'
            r = httpx.get(
                url,
                timeout=12.0,
                follow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36'},
            )
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                tbl = soup.find('table')
                if tbl:
                    rows = tbl.find_all('tr')
                    if len(rows) > 1:
                        # First data row has the final listing day GMP
                        cells = rows[1].find_all(['td', 'th'])
                        if len(cells) >= 2:
                            gmp_col = cells[1].get_text(separator=' ', strip=True)
                            if '--' in gmp_col:
                                return {'gmp_rs': None, 'gmp_pct': None, 'gmp_source': 'unavailable'}
                            # Parse e.g. "30 (25.21%)"
                            m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:\(\s*(-?\d+(?:\.\d+)?)\s*%\s*\))?', gmp_col)
                            if m:
                                gmp_rs = float(m.group(1))
                                if m.group(2):
                                    gmp_pct = float(m.group(2))
                                elif issue_price and issue_price > 0:
                                    gmp_pct = round((gmp_rs / issue_price) * 100.0, 2)
                                else:
                                    gmp_pct = None
                                return {
                                    'gmp_rs': gmp_rs,
                                    'gmp_pct': gmp_pct,
                                    'gmp_source': 'live_investorgain_detail',
                                    'gmp_source_family': 'chittorgarh_investorgain',
                                    'gmp_source_url': url,
                                    'gmp_official': False,
                                    'gmp_observed_at': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds'),
                                    'gmp_is_stale': False,
                                    'gmp_observation_status': 'observed',
                                }
        except Exception as e:
            logger.debug(f"Failed to fetch dedicated GMP for {slug}/{cid}: {e}")
        return {'gmp_rs': None, 'gmp_pct': None, 'gmp_source': 'unavailable'}

# Global Singleton Instance
gmp_service = GMPService()
