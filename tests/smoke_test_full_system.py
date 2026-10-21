import httpx

API_BASE = 'http://127.0.0.1:8000/api/v1'
FRONTEND_BASE = 'http://127.0.0.1:3000'

client = httpx.Client(timeout=15.0)
print('======================================================================')
print('          END-TO-END SYSTEM INTEGRATION SMOKE TEST (v5.0)')
print('======================================================================')

# 1. Health
r1 = client.get(API_BASE + '/health')
assert r1.status_code == 200
macro = r1.json()['live_macro_regime']
print('1. [PASS] FastAPI Backend Health: Status: OK')
print('   -> Live India VIX:', macro['india_vix'], '| Nifty 50:', macro['nifty_close'], '(' + str(macro['nifty_10d_return']) + '% 10d)')

# 2. Upcoming
r2 = client.get(API_BASE + '/module1/live-upcoming')
assert r2.status_code == 200
upcoming = r2.json()['upcoming_ipos']
print('2. [PASS] Live 2024-2026 IPO Registry:', len(upcoming), 'Headline IPOs Loaded')

# 3. Scrape and evaluate
r3 = client.post(API_BASE + '/module1/scrape-and-evaluate', json={'url_or_name': 'Waaree Energies', 'fetch_news': True})
assert r3.status_code == 200
eval_res = r3.json()['evaluation']
print('3. [PASS] Live Scraper & Hybrid ML (Waaree Energies):')
print('   -> Verdict:', eval_res['verdict'], '(' + eval_res['conviction'] + ')')
print('   -> Hybrid ML Win Probability:', eval_res['hybrid_ml_scores']['hybrid_positive_listing_prob'], '%')
print('   -> bHNI Sizing Tier:', eval_res['bHNI_sizing_tier'])
print('   -> Top Historical Analogs:', len(eval_res['similarity_baseline']['top_analogs']), 'matches')
print('   -> Live News Headlines Fetched:', len(eval_res['qualitative_news_context']))

# 4. Module 2
r4 = client.get(API_BASE + '/module2/holdings/364?as_of_date=2024-03-15')
assert r4.status_code == 200
h_res = r4.json()
print('4. [PASS] Module 2 Hold/Sell Monitor (Tata Tech):')
print('   -> Verdict:', h_res['verdict'], '(Confidence:', h_res['confidence'] + ')')
print('   -> 30d Lock-In Expiry Date:', h_res['lockin_status']['lockin_30d_date'])
print('   -> 30d Relative Spread vs Sector:', str(h_res['attribution_30d']['spread_vs_sector_pct']) + '%')

# 5. Frontend
r5 = client.get(FRONTEND_BASE)
assert r5.status_code == 200
assert 'OpenBell' in r5.text or 'IPO' in r5.text
print('5. [PASS] Next.js Institutional Dashboard: Rendered on ' + FRONTEND_BASE + ' (200 OK)')

print('======================================================================')
print('      ALL 5 END-TO-END PIPELINE MODULES 100% OPERATIONAL & VERIFIED')
print('======================================================================')
