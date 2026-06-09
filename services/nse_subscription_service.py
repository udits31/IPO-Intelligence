"""Read-only NSE IPO category observations with strict issue-identity checks.

NSE labels these as bid positions, not certified final subscriptions. The
publisher's update time is kept separate from our fetch time, and no absent
category is converted to zero.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import httpx

IST = ZoneInfo('Asia/Kolkata')
BASE = 'https://www.nseindia.com'
SOURCE_FAMILY = 'nse_exchange'
SOURCE_NAME = 'nse_ipo_category_v1'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36',
    'Accept': 'application/json,text/plain,*/*',
    'Referer': BASE + '/market-data/all-upcoming-issues-ipo',
}


def _name(value: str) -> str:
    # Legal issuer names commonly spell '&' as 'and'. Keep every other name
    # token and the exact closing-date constraint; never use fuzzy prefixes.
    value=value.lower().replace('&',' and ')
    return re.sub(r'[^a-z0-9]+', '', re.sub(r'\b(?:ipo|limited|ltd)\b', '', value))


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if 0 <= result < 1_000_000 else None
    except (TypeError, ValueError):
        return None


def _category_values(rows: list[dict]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for row in rows:
        label = str(row.get('category') or '').lower()
        value = _number(row.get('noOfTotalMeant'))
        if value is None:
            continue
        if 'qualified institutional buyers' in label and row.get('srNo') == '1':
            values['qib_subscription'] = value
        elif label.strip() == 'non institutional investors' and row.get('srNo') == '2':
            values['hni_subscription'] = value
        elif 'retail individual investors' in label and row.get('srNo') == '3':
            values['retail_subscription'] = value
        elif str(row.get('srNo','')).startswith('2') and ('10' in label or 'ten' in label):
            if ('above' in label or 'more than' in label) and not ('up to' in label or 'upto' in label):
                values['bnii_subscription']=value
            elif 'up to' in label or 'upto' in label:
                values['snii_subscription']=value
    return values


def fetch_nse_subscription_evidence(company_name: str, closing_date, *,
                                    nse_symbol: Optional[str] = None,
                                    symbol_verified: bool = False,
                                    nse_series: Optional[str] = None) -> Dict[str, Any]:
    """Return a separately sourced bid-position snapshot; never infer a symbol."""
    result: Dict[str, Any] = {
        'status': 'identity_unavailable', 'source_family': SOURCE_FAMILY,
        'source_name': SOURCE_NAME, 'values': None, 'error': None,
    }
    now = datetime.now(IST)
    result['fetched_at_ist'] = now.isoformat(timespec='seconds')
    try:
        with httpx.Client(headers=HEADERS, timeout=12.0, follow_redirects=True) as client:
            client.get(BASE + '/')
            listing_response = client.get(BASE + '/api/ipo-current-issue')
            listing_response.raise_for_status()
            listings = listing_response.json()
            if not isinstance(listings, list):
                raise ValueError('NSE current-issue response was not a list')
            matches = [issue for issue in listings if isinstance(issue, dict)
                       and _name(str(issue.get('companyName') or '')) == _name(company_name)
                       and (closing_date is None or issue.get('issueEndDate') == closing_date.strftime('%d-%b-%Y'))]
            if len(matches) == 1:
                issue = matches[0]
                symbol = str(issue.get('symbol') or '').strip().upper()
                series = str(issue.get('series') or 'EQ').strip().upper()
            elif symbol_verified and nse_symbol and re.fullmatch(r'[A-Z0-9&-]{2,30}', nse_symbol.upper()):
                # Current-issue listing can disappear after close. Only reuse a
                # symbol that we previously resolved from this exact NSE issue.
                symbol = nse_symbol.upper()
                series = nse_series if nse_series in ('EQ', 'SME') else 'EQ'
            else:
                result['error'] = 'No unique exact NSE issue identity and closing date match'
                return result
            if not re.fullmatch(r'[A-Z0-9&-]{2,30}', symbol):
                result['error'] = 'Invalid NSE symbol in issue listing'
                return result
            url = BASE + '/api/ipo-active-category?symbol=' + symbol
            response = client.get(url)
            result['http_status'] = response.status_code
            response.raise_for_status()
            raw = response.text
            payload = response.json()
        if not isinstance(payload, dict) or (payload.get('symbol') and str(payload['symbol']).upper() != symbol):
            result['status'] = 'identity_mismatch'
            result['error'] = 'NSE category response symbol did not match issue identity'
            return result
        result.update(raw_html=raw, raw_sha256=hashlib.sha256(raw.encode('utf-8')).hexdigest(),
                      nse_symbol=symbol, series=series,
                      subscription_url=BASE + '/market-data/issue-information?series=' + series + '&symbol=' + symbol)
        if str(payload.get('updateTime') or '').strip().lower() in ('', 'null', 'updated as on null'):
            result.update(status='awaiting_publication', error=None)
            return result
        updated = datetime.strptime(str(payload.get('updateTime') or ''),
                                    'Updated as on %d-%b-%Y %H:%M:%S').replace(tzinfo=IST)
        checked_at = datetime.now(IST)
        result['fetched_at_ist'] = checked_at.isoformat(timespec='seconds')
        historical_close = (
            closing_date is not None and updated.date() == closing_date
            and updated.hour >= 17 and checked_at.date() > closing_date
        )
        if ((updated - checked_at).total_seconds() > 60 or
                ((checked_at - updated).total_seconds() > 86400 and not historical_close)):
            result['status'] = 'stale_source'
            result['error'] = 'NSE category update time is future-dated or over one day old'
            return result
        values = _category_values(payload.get('dataList') or [])
        result.update({
            'status': 'verified' if all(k in values for k in ('qib_subscription','hni_subscription','retail_subscription')) else ('partial_subscription_table' if values else 'no_subscription_table'),
            'values': values or None, 'nse_symbol': symbol, 'series': series,
            'source_displayed_at_ist': updated.isoformat(timespec='seconds'),
            'historical_close': historical_close,
            'subscription_url': BASE + '/market-data/issue-information?series=' + series + '&symbol=' + symbol,
            'raw_html': raw, 'raw_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
            'parser_version': 'nse_category_json_v1',
        })
    except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result['status'] = 'fetch_error'
        result['error'] = f'{type(exc).__name__}: {exc}'[:500]
    return result
