"""Recover dated public bid tables and replay the model on pre-close inputs."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from db import ledger as sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime, time as wall_time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from scrapers.live_ipo_scraper import parse_date_interval
from services.snapshot_service import DB_PATH, _scheduled_predictor, extract_ipo_id


ROOT = Path(__file__).resolve().parents[1]
IST = ZoneInfo('Asia/Kolkata')
INDEX = 'https://www.moneycontrol.com/ipo/closed-ipos/'
_index_cache: tuple[float, dict] = (0, {})
_timed_index_cache: tuple[float, dict] = (0, {})
RECOVERY_PROTOCOL = 'precutoff_timestamp_v3'


def _issuer_matches(soup: BeautifulSoup, company: str, *, short_heading: bool = False) -> bool:
    heading = soup.find('h1')
    actual = name_key(heading.get_text(' ', strip=True)) if heading else ''
    expected = name_key(company)
    if not expected:
        return False
    if actual == expected or actual.startswith(expected + ' '):
        return True
    # Some publishers use the unique brand as the heading. Require the full
    # issuer name in the page body as well, e.g. Zetwerk Manufacturing Business.
    return bool(short_heading and len(actual.split()) == 1 and len(actual) >= 5
                and actual == expected.split()[0]
                and expected in name_key(soup.get_text(' ', strip=True)))


def _ratio(raw: str) -> float | None:
    match = re.match(r'^\s*(\d[\d,]*(?:\.\d+)?)\s*(?:x|times|🔥|$)', raw, re.I)
    if not match:
        return None
    from services.subscription_numbers import subscription_multiple
    return subscription_multiple(match[1])


def parse_timestamped_snapshot(html: str, company: str, opening: date,
                               closing: date, source_url: str) -> list[dict]:
    """Keep the subscription table's own time, never a GMP or article time."""
    soup = BeautifulSoup(html, 'html.parser')
    if not _issuer_matches(soup, company, short_heading=True):
        return []
    host = (urlparse(source_url).hostname or '').removeprefix('www.')
    observed, table, basis = None, None, None
    if host == 'theipolist.in':
        heading = next((h for h in soup.find_all(['h2', 'h3'])
                        if h.get_text(' ', strip=True) == 'IPO Subscription Status'), None)
        section = heading.find_parent('section') if heading else None
        if section is None:
            return []
        stamp = re.search(r"Last updated:\s*(\d{1,2}\s+[A-Za-z]+'\d{2}\s+\d{1,2}:\d{2}\s*[AP]M)",
                          section.get_text(' ', strip=True), re.I)
        if stamp:
            try:
                observed = datetime.strptime(re.sub(r'\s+([AP]M)$', r'\1', stamp[1]),
                                             "%d %B'%y %I:%M%p").replace(tzinfo=IST)
            except ValueError:
                return []
        table = section.find('table')
        basis = 'publisher_subscription_table_update'
        # Check this issue's actual timetable, independently of the update time.
        for label, expected in [('Issue open date', opening), ('Issue close date', closing)]:
            field = next((h for h in soup.find_all(['h2', 'h3', 'h4'])
                          if h.get_text(' ', strip=True) == label), None)
            text = field.parent.get_text(' ', strip=True).replace(label, '').strip() if field else ''
            parsed = pd.to_datetime(text, errors='coerce', dayfirst=True)
            if pd.isna(parsed) or parsed.date() != expected:
                return []
    elif host == 'niftytrader.in':
        heading = next((h for h in soup.find_all(['h2', 'h3'])
                        if 'IPO Subscriptions Status' in h.get_text(' ', strip=True)), None)
        if heading is None:
            return []
        table = heading.find_next('table')
        fragments = []
        for node in heading.next_elements:
            if node is table:
                break
            if isinstance(node, str):
                fragments.append(str(node))
        stamp = re.search(r'by\s+([A-Za-z]+\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*[AP]M)',
                          ' '.join(fragments), re.I)
        if stamp:
            try:
                observed = datetime.strptime(stamp[1], '%B %d, %Y %I:%M:%S %p').replace(tzinfo=IST)
            except ValueError:
                return []
        basis = 'publisher_bid_asof_time'
    if table is None or observed is None:
        return []
    starts = datetime.combine(opening, wall_time(10), tzinfo=IST)
    cutoff = datetime.combine(closing, wall_time(17), tzinfo=IST)
    if not starts <= observed < cutoff:
        return []
    values = {}
    for tr in table.find_all('tr'):
        cells = [cell.get_text(' ', strip=True) for cell in tr.find_all(['td', 'th'])]
        if len(cells) < 2:
            continue
        label = cells[0].lower()
        key = ('qib' if re.match(r'qib\b|qibs\b', label) else
               'hni' if re.match(r'hni\b|hnis\b|nii\b', label) else
               'retail' if re.match(r'retail\b', label) else None)
        value = _ratio(cells[1])
        if key and value is not None:
            values[key] = value
    if len(values) != 3:
        return []
    return [{'observed_on': observed.date().isoformat(),
             'source_reported_at_ist': observed.isoformat(), 'time_basis': basis,
             'timezone_basis': 'publisher India market clock interpreted as IST', **values}]


def name_key(name: str) -> str:
    name = re.sub(r"['’]s\b", 's', name, flags=re.I)
    name = re.sub(r'\(snapdeal\)', '', name, flags=re.I)
    tokens = re.findall(r'[a-z0-9]+', name.lower())
    return ' '.join('business' if word == 'businesses' else word for word in tokens
                    if word not in {'limited', 'ltd', 'ipo', 'mainline', 'mainboard', 'and'})


def parse_article_snapshot(html: str, company: str, opening: date,
                           closing: date, source_url: str) -> list[dict]:
    """Recover explicitly timed article observations, retaining quality warnings.

    Publication time is never substituted for bid time. Contradictory times and
    late publications remain visible in the reconstruction's evidence flags.
    """
    soup = BeautifulSoup(html, 'html.parser')
    if not _issuer_matches(soup, company):
        return []
    host = (urlparse(source_url).hostname or '').removeprefix('www.')
    if host not in {'choiceindia.com', '5paisa.com', 'rajkotcityguide.com'}:
        return []
    metadata = {}
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.get_text())
            entries = data if isinstance(data, list) else data.get('@graph', [data])
            for entry in entries:
                if entry.get('@type') in ('NewsArticle', 'Article', 'BlogPosting'):
                    metadata.update({k: entry[k] for k in ('datePublished', 'dateModified') if entry.get(k)})
        except (ValueError, TypeError, AttributeError):
            continue
    main = soup.find('main') or soup.find('article') or soup
    text = main.get_text(' ', strip=True)
    observed, values, warnings = None, {}, []
    if host == 'choiceindia.com':
        stamp = re.search(r'\btill\s+(?:(\d{1,2}-[A-Za-z]{3}-\d{4})\s+)?'
                          r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([ap]m)', text, re.I)
        if not stamp:
            return []
        day = datetime.strptime(stamp[1], '%d-%b-%Y').date() if stamp[1] else closing
        hour = int(stamp[2])
        if hour <= 12:
            hour = hour % 12 + (12 if stamp[5].lower() == 'pm' else 0)
        observed = datetime.combine(day, wall_time(hour, int(stamp[3]), int(stamp[4] or 0)), IST)
        for table in main.find_all('table'):
            for tr in table.find_all('tr'):
                cells = [x.get_text(' ', strip=True) for x in tr.find_all(['th', 'td'])]
                if len(cells) < 3:
                    continue
                label = cells[1].lower()
                key = ('qib' if 'qualified institutional' in label else 'hni' if 'non-institutional' in label
                       else 'retail' if 'retail individual' in label else None)
                value = _ratio(cells[2])
                if key and value is not None:
                    values[key] = value
    elif host == '5paisa.com':
        # Scope the time to the opening subscription paragraph, not the GMP,
        # a different day's table, or a templated later detailed-table heading.
        intro = next((p.get_text(' ', strip=True) for p in main.find_all('p')
                      if re.search(r'as of\s+\d{1,2}:\d{2}\s*[AP]M\s+on', p.get_text(), re.I)), '')
        stamp = re.search(r'as of\s+(\d{1,2}:\d{2}\s*[AP]M)\s+on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})', intro, re.I)
        if not stamp:
            return []
        observed = datetime.strptime(stamp[2] + ' ' + stamp[1], '%B %d, %Y %I:%M %p').replace(tzinfo=IST)
        for table in main.find_all('table'):
            rows = [[x.get_text(' ', strip=True) for x in tr.find_all(['th', 'td'])] for tr in table.find_all('tr')]
            if not rows or rows[0][0].lower() != 'date':
                continue
            keys = {}
            for i, label in enumerate(rows[0]):
                key = 'qib' if label == 'QIB' else 'hni' if label == 'NII' else 'retail' if label in ('Retail', 'Individual') else None
                if key:
                    keys[key] = i
            for row in rows[1:]:
                if not row or observed.strftime('%B') not in row[0] or str(observed.day) not in row[0]:
                    continue
                values = {key: _ratio(row[i]) for key, i in keys.items() if i < len(row)}
        if re.search(r'as of\s+5:05\s*PM', text, re.I):
            warnings.append('article_also_labels_detailed_table_17_05_conflicting_with_intro')
    else:
        stamp = re.search(r'By\s+(\d{1,2}:\d{2}\s*[AP]M)\s+on\s+(?:[A-Za-z]+,\s*)?'
                          r'([A-Za-z]+\s+\d{1,2},\s+\d{4})', text, re.I)
        if not stamp:
            return []
        observed = datetime.strptime(stamp[2] + ' ' + stamp[1], '%B %d, %Y %I:%M %p').replace(tzinfo=IST)
        for key, label in [('qib', 'Qualified Institutional'), ('hni', 'Non-Institutional'), ('retail', 'Retail Individual')]:
            match = re.search(r'\|\s*' + label + r'[^|]*\|[^|]*\|[^|]*\|\s*(\d+(?:\.\d+)?)x', text, re.I)
            if match:
                values[key] = float(match[1])
        warnings.append('single_small_news_publisher_not_independently_corroborated')
    if (observed is None or not datetime.combine(opening, wall_time(10), IST) <= observed
            < datetime.combine(closing, wall_time(17), IST)
            or set(values) != {'qib', 'hni', 'retail'} or any(v is None for v in values.values())):
        return []
    publication = {}
    for key, raw in metadata.items():
        try:
            moment = datetime.fromisoformat(raw.replace('Z', '+00:00'))
            if moment.tzinfo:
                publication[key] = moment.astimezone(IST).isoformat()
        except ValueError:
            pass
    published = publication.get('datePublished')
    modified = publication.get('dateModified')
    if published and datetime.fromisoformat(published) < observed:
        # Updated articles may postdate their original publication; retain the
        # modification clock and require that it also covers the observation.
        if not modified or datetime.fromisoformat(modified) < observed:
            warnings.append('publisher_publication_and_modification_precede_claimed_bid_time')
    if published and datetime.fromisoformat(published) >= datetime.combine(closing, wall_time(17), IST):
        warnings.append('article_first_published_after_application_cutoff')
    return [{'observed_on': observed.date().isoformat(), 'source_reported_at_ist': observed.isoformat(),
             'time_basis': 'publisher_article_explicit_bid_asof',
             'timezone_basis': 'publisher India market clock interpreted as IST',
             'publication_times_ist': publication, 'evidence_warnings': warnings, **values}]


def parse_daywise(html: str, company: str, opening: date, closing: date) -> list[dict]:
    """Read header positions, never assume that retail and NII have the same order."""
    soup = BeautifulSoup(html, 'html.parser')
    heading = soup.find('h1')
    heading_key = name_key(heading.get_text(' ', strip=True)) if heading else ''
    target_key = name_key(company)
    if not target_key or not (heading_key == target_key or heading_key.startswith(target_key + ' ')):
        return []
    found = {}
    for table in soup.find_all('table'):
        rows = [[cell.get_text(' ', strip=True) for cell in row.find_all(['th', 'td'])]
                for row in table.find_all('tr')]
        columns = None
        for row in rows:
            labels = [cell.lower() for cell in row]
            mapped = {}
            for index, label in enumerate(labels):
                category = ('qib' if re.match(r'qib\b|qibs\b', label) else
                            'hni' if re.match(r'nii\b|niis\b|hni\b', label) else
                            'retail' if re.match(r'rii\b|riis\b|retail\b', label) else None)
                if category:
                    mapped[category] = index
            if set(mapped) == {'qib', 'hni', 'retail'}:
                columns = mapped
                continue
            if columns is None or not row:
                continue
            match = re.search(r'\d{4}-\d{2}-\d{2}|\d{1,2}[- ]+[A-Za-z]{3,9}[- ,]+\d{4}|\d{1,2}-\d{1,2}-\d{4}', row[0])
            if not match:
                continue
            parsed = pd.to_datetime(match.group(), errors='coerce',
                                    format='%Y-%m-%d' if re.fullmatch(r'\d{4}-\d{2}-\d{2}', match.group()) else 'mixed',
                                    dayfirst=True)
            if pd.isna(parsed) or not opening <= parsed.date() < closing:
                continue  # Closing-day totals have no trustworthy pre-17:00 publication time.
            values = {}
            for category, index in columns.items():
                if index >= len(row):
                    break
                number = re.fullmatch(r'\s*(\d[\d,]*(?:\.\d+)?)\s*(?:x|times)?\s*', row[index], re.I)
                if number:
                    from services.subscription_numbers import subscription_multiple
                    value = subscription_multiple(number[1])
                    if value is not None:
                        values[category] = value
            if len(values) == 3:
                found[parsed.date().isoformat()] = {'observed_on': parsed.date().isoformat(), **values}
    return sorted(found.values(), key=lambda row: row['observed_on'])


def _fetch(url: str) -> tuple[str, str]:
    response = httpx.get(url, timeout=20, follow_redirects=True,
                         headers={'User-Agent': 'Mozilla/5.0 (compatible; IPOBidResearch/1.0)'})
    response.raise_for_status()
    return str(response.url), response.text


def discover_sources(company: str) -> list[str]:
    global _index_cache, _timed_index_cache
    registry = json.loads((ROOT / 'data/historical_bid_sources.json').read_text(encoding='utf-8'))
    urls = [url for name, sources in registry.items() if name_key(name) == name_key(company) for url in sources]
    if time.monotonic() - _index_cache[0] > 600:
        try:
            _, html = _fetch(INDEX)
            soup = BeautifulSoup(html, 'html.parser')
            links = {}
            for link in soup.find_all('a', href=True):
                url = urljoin(INDEX, link['href'])
                if '-ipodetail' in url and urlparse(url).hostname == 'www.moneycontrol.com':
                    links[name_key(link.get_text(' ', strip=True))] = url
            _index_cache = (time.monotonic(), links)
        except httpx.HTTPError:
            pass
    if name_key(company) in _index_cache[1]:
        urls.append(_index_cache[1][name_key(company)])
    if time.monotonic() - _timed_index_cache[0] > 600:
        try:
            index_url = 'https://theipolist.in/ipo-calendar'
            _, html = _fetch(index_url)
            soup = BeautifulSoup(html, 'html.parser')
            links = {}
            for link in soup.find_all('a', href=True):
                url = urljoin(index_url, link['href'])
                if urlparse(url).hostname == 'theipolist.in' and '/ipo-details/' in url:
                    links[name_key(link.get_text(' ', strip=True))] = url
            _timed_index_cache = (time.monotonic(), links)
        except httpx.HTTPError:
            pass
    if name_key(company) in _timed_index_cache[1]:
        urls.append(_timed_index_cache[1][name_key(company)])
    return list(dict.fromkeys(urls))


def _init(conn):
    conn.execute('PRAGMA busy_timeout=10000')
    conn.execute('''CREATE TABLE IF NOT EXISTS historical_bid_page_v1 (
        id INTEGER PRIMARY KEY, ipo_id TEXT NOT NULL, source_url TEXT NOT NULL,
        fetched_at_ist TEXT NOT NULL, raw_sha256 TEXT NOT NULL, raw_html_gzip BLOB NOT NULL,
        UNIQUE(ipo_id, source_url, raw_sha256))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS historical_bid_reconstruction_v1 (
        id INTEGER PRIMARY KEY, ipo_id TEXT NOT NULL, company_name TEXT NOT NULL,
        observed_on TEXT NOT NULL, computed_at_ist TEXT NOT NULL,
        model_version TEXT NOT NULL, model_sha256 TEXT NOT NULL,
        positive_listing_probability REAL NOT NULL, evidence_json TEXT NOT NULL,
        input_json TEXT NOT NULL, UNIQUE(ipo_id, observed_on, model_sha256, evidence_json))''')
    conn.commit()


def resolve_official_issue(item: dict, *, path: Path | None = None) -> dict:
    """Overlay unique NSE past-issue identity/date evidence, never invented dates."""
    path = path or ROOT / 'data/nse_public_past_issues_raw.json'
    if not path.exists():
        return dict(item)
    raw = path.read_bytes()
    matches = [r for r in json.loads(raw) if name_key(r.get('company') or r.get('companyName') or '')
               == name_key(item.get('company_name') or '')]
    if len(matches) != 1:
        return dict(item)
    row = matches[0]
    try:
        opening = datetime.strptime(row['ipoStartDate'], '%d-%b-%Y').date()
        closing = datetime.strptime(row['ipoEndDate'], '%d-%b-%Y').date()
        listing = datetime.strptime(row['listingDate'], '%d-%b-%Y').date()
        price = float(row['issuePrice'])
    except (KeyError, ValueError, TypeError):
        return dict(item)
    if not opening <= closing < listing or price <= 0:
        return dict(item)
    if item.get('issue_price') is not None and abs(float(item['issue_price']) - price) > .01:
        return {**item, 'official_identity_error': 'NSE issue-price conflict'}
    return {**item, 'date_range': f'{opening.isoformat()} to {closing.isoformat()}',
            'listing_date': listing.isoformat(), 'issue_price': price, 'nse_symbol': row['symbol'],
            'official_issuer_name': row.get('company') or row.get('companyName'),
            'official_offer_evidence': {'url': 'https://www.nseindia.com/api/public-past-issues',
                                      'raw_sha256': hashlib.sha256(raw).hexdigest()}}


def evidence_severity(inputs: dict) -> int:
    warnings = inputs.get('evidence_warnings') or []
    if any(w != 'single_small_news_publisher_not_independently_corroborated' for w in warnings):
        return 2
    return 1 if warnings else 0


def get_historical_bid_forecast(company: str, *, db_path: str = DB_PATH) -> dict | None:
    try:
        with closing(sqlite3.connect(db_path, timeout=10)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''SELECT * FROM historical_bid_reconstruction_v1
                WHERE LOWER(company_name) = LOWER(?) ORDER BY observed_on DESC, id DESC''', (company,)).fetchall()
        snapshots = []
        for raw in rows:
            row = dict(raw)
            row['sources'] = json.loads(row.pop('evidence_json'))
            row['inputs'] = json.loads(row.pop('input_json'))
            row['source_reported_at_ist'] = row['inputs'].get('source_reported_at_ist')
            row['reconstructed'] = True
            row['eligible_for_prospective_accuracy'] = False
            snapshots.append(row)
        # Observation time, not insertion/fetch time, determines the latest bids.
        snapshots.sort(key=lambda row: (row['observed_on'], -evidence_severity(row['inputs']),
                       row['source_reported_at_ist'] or '', row['id']), reverse=True)
        history, seen = [], set()
        for row in snapshots:
            if row['observed_on'] not in seen:
                history.append(row)
                seen.add(row['observed_on'])
        if not history:
            return None
        return {**history[0], 'history': history, 'snapshot_history': snapshots}
    except sqlite3.OperationalError:
        return None


def recover_historical_bid_forecast(item: dict, *, db_path: str = DB_PATH,
                                    force: bool = False) -> dict | None:
    item = resolve_official_issue(item)
    company = item.get('company_name') or ''
    cached = get_historical_bid_forecast(company, db_path=db_path)
    if item.get('official_identity_error'):
        return cached
    now = datetime.now(IST)
    opening, cutoff = parse_date_interval(item.get('date_range') or '', now.year)
    if opening is None or cutoff is None or cutoff > now.date():
        return cached
    if cached and not force and cached['inputs'].get('recovery_protocol') == RECOVERY_PROTOCOL:
        completed = (cached['observed_on'] == cutoff.isoformat() and cached.get('source_reported_at_ist')
                     and not evidence_severity(cached['inputs']))
        recent = (now - datetime.fromisoformat(cached['computed_at_ist'])).total_seconds() < 600
        if completed or recent:
            return cached
    urls = discover_sources(company)
    if not urls:
        return cached
    ipo_id = extract_ipo_id(item)
    pages, errors = [], []

    def fetch_source(url):
        try:
            final_url, html = _fetch(url)
            rows = (parse_daywise(html, company, opening, cutoff)
                    + parse_timestamped_snapshot(html, company, opening, cutoff, final_url)
                    + parse_article_snapshot(html, company, opening, cutoff, final_url))
            if not rows:
                return None
            return {'url': final_url, 'html': html, 'rows': rows,
                    'sha256': hashlib.sha256(html.encode()).hexdigest(),
                    'fetched_at_ist': datetime.now(IST).isoformat(),
                    'publisher': urlparse(final_url).hostname.removeprefix('www.')}
        except httpx.HTTPError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        pages = [page for page in pool.map(fetch_source, urls) if page]
    predictor = _scheduled_predictor()
    model_hash = hashlib.sha256(json.dumps(predictor.model_bundle, sort_keys=True).encode()).hexdigest()
    with closing(sqlite3.connect(db_path, timeout=10)) as conn:
        _init(conn)
        for page in pages:
            conn.execute('''INSERT OR IGNORE INTO historical_bid_page_v1
                (ipo_id,source_url,fetched_at_ist,raw_sha256,raw_html_gzip) VALUES (?,?,?,?,?)''',
                (ipo_id, page['url'], page['fetched_at_ist'], page['sha256'], gzip.compress(page['html'].encode())))
        observations = sorted({(row['observed_on'], row.get('source_reported_at_ist', ''))
                               for page in pages for row in page['rows']})
        for observed, reported_at in observations:
            # A 14:30 reading and a 16:44 reading are different observations,
            # not conflicting votes about the same bid book.
            candidates = [(page, row) for page in pages for row in page['rows']
                          if (row['observed_on'], row.get('source_reported_at_ist', '')) == (observed, reported_at)]
            # Preserve a source reading. Conflicts are not averaged into a new number.
            agreed = []
            for reference, values in candidates:
                matches = [(page, row) for page, row in candidates if all(
                    abs(values[key] - row[key]) <= max(0.03, 0.02 * max(values[key], row[key]))
                    for key in ('qib', 'hni', 'retail'))]
                if len(matches) > len(agreed):
                    agreed = matches
            if len(candidates) > 1 and len(agreed) < 2:
                errors.append({'observed_on': observed, 'source_reported_at_ist': reported_at,
                               'reason': 'historical_sources_disagree'})
                continue
            page, values = agreed[0]
            sources = [{'url': p['url'], 'publisher': p['publisher'], 'raw_sha256': p['sha256'],
                        'fetched_at_ist': p['fetched_at_ist']}
                       for p, _ in agreed]
            inputs = {'qib_subscription': values['qib'], 'hni_subscription': values['hni'],
                      'retail_subscription': values['retail'], 'observed_on': observed,
                      'evidence_class': ('source_reported_historical_article' if values.get('time_basis') == 'publisher_article_explicit_bid_asof'
                                         else 'source_reported_historical_timestamped_table' if reported_at
                                         else 'source_reported_historical_daywise_table'),
                      'source_reported_at_ist': reported_at or None,
                      'time_basis': values.get('time_basis', 'publisher_daywise_date_only'),
                      'timezone_basis': values.get('timezone_basis'),
                      'publication_times_ist': values.get('publication_times_ist', {}),
                      'evidence_warnings': values.get('evidence_warnings', []),
                      'clean_timestamp_evidence': not values.get('evidence_warnings'),
                      'official_offer_evidence': item.get('official_offer_evidence'),
                      'recovery_protocol': RECOVERY_PROTOCOL,
                      'application_closing_date': cutoff.isoformat()}
            score = predictor.score_interim_bid_snapshot(qib_sub=values['qib'], hni_sub=values['hni'], retail_sub=values['retail'])
            conn.execute('''INSERT OR IGNORE INTO historical_bid_reconstruction_v1
                (ipo_id,company_name,observed_on,computed_at_ist,model_version,model_sha256,
                 positive_listing_probability,evidence_json,input_json) VALUES (?,?,?,?,?,?,?,?,?)''',
                (ipo_id, company, observed, datetime.now(IST).isoformat(timespec='seconds'), score['model_version'], model_hash,
                 score['positive_listing_probability'], json.dumps(sources, sort_keys=True), json.dumps(inputs, sort_keys=True)))
        conn.commit()
    result = get_historical_bid_forecast(company, db_path=db_path)
    if result:
        result['source_conflicts'] = errors
    return result
