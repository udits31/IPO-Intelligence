"""Build one lifecycle-consistent calendar without duplicate issuer stages."""

import re
from datetime import date


def identity(item: dict) -> str:
    url = item.get('url') or ''
    match = re.search(r'/ipo/[^/]+/(\d+)', url)
    return 'chitto_' + match[1] if match else re.sub(r'[^a-z0-9]', '', item.get('company_name', '').lower())


def reconcile_calendar(items: list[dict], listed: list[dict], today: date) -> tuple[list, list]:
    confirmed = {}
    for item in listed:
        try:
            listing = date.fromisoformat(item.get('listing_date') or '')
        except ValueError:
            continue
        if listing <= today:
            confirmed[identity(item)] = item
    active = {}
    for item in items:
        key = identity(item)
        if key not in confirmed:
            active[key] = item
    return list(active.values()), list(confirmed.values())
