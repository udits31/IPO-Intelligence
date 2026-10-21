import httpx

client = httpx.Client(base_url='http://127.0.0.1:8000', timeout=15.0)

print('=== 1. HEALTH CHECK ===')
r = client.get('/api/v1/health')
print('Health:', r.status_code, r.json().get('status'))

print('\n=== 2. SEARCH BY SYMBOL / CODE ===')
for q in ['STARHEALTH', '543412', 'ZOMATO', '543320', 'PAYTM']:
    r = client.get(f'/api/v1/ipos?search={q}')
    ipos = r.json().get('ipos', [])
    if ipos:
        first = ipos[0]
        print(f'Query "{q}" -> {first["company_name"]} (NSE: {first["nse_symbol"]}, BSE: {first["bse_scrip_code"]}) | Issue: Rs {first["issue_price"]} | Size: Rs {first["issue_size_cr"]} Cr | Sector: {first["sector"]}')
    else:
        print(f'Query "{q}" -> Not found')

print('\n=== 3. MODULE 1 EVALUATION (GROUND TRUTH CHECK) ===')
for target in ['Star Health & Allied Insurance Co.Ltd.', 'STARHEALTH', '543412', 'ZOMATO', 'Ather Energy Ltd.']:
    r = client.post('/api/v1/module1/scrape-and-evaluate', json={'url_or_name': target, 'fetch_news': False})
    d = r.json()
    data = d.get('scraped_data', {})
    eval_res = d.get('evaluation', {})
    print(f'Target "{target}" =>')
    print(f'   Company: {data.get("company_name")}')
    print(f'   Price: Rs {data.get("issue_price")} | Size: Rs {data.get("issue_size_cr")} Cr | Sector: {data.get("sector")}')
    print(f'   Subscriptions: QIB {data.get("qib_subscription")}x | Retail {data.get("retail_subscription")}x | HNI {data.get("hni_subscription")}x')
    print(f'   Verdict: {eval_res.get("verdict")} | Win Prob: {eval_res.get("hybrid_ml_scores", {}).get("hybrid_positive_listing_prob")}% | Pred Ret: {eval_res.get("hybrid_ml_scores", {}).get("predicted_listing_return_pct")}%')
