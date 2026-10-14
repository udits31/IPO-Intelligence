import os
import pytest
import httpx

API_BASE = 'http://127.0.0.1:8000/api/v1'

test_cases = [
    {
        'query': 'tata tech',
        'expected_name': 'Tata Technologies Ltd.',
        'expected_symbol': 'TATATECH',
    },
    {
        'query': 'Zomato',
        'expected_name': 'Zomato Ltd.',
        'expected_symbol': 'ETERNAL',
    },
    {
        'query': 'Nykaa',
        'expected_name': 'FSN E-Commerce Ventures Ltd.',
        'expected_symbol': 'NYKAA',
    },
    {
        'query': 'Delhivery',
        'expected_name': 'Delhivery Ltd.',
        'expected_symbol': 'DELHIVERY',
    },
    {
        'query': 'Ola Electric',
        'expected_name': 'Ola Electric Mobility Ltd.',
        'expected_symbol': 'OLAELEC',
    }
]

@pytest.mark.live
@pytest.mark.skipif(os.getenv("OPENBELL_LIVE_TESTS") != "1", reason="Opt-in external-provider smoke test; set OPENBELL_LIVE_TESTS=1")
def test_live_resolution():
    client = httpx.Client(timeout=15.0)
    print("======================================================================")
    print("       MODULE 2 TRENDLYNE RESOLUTION & DATA INTEGRITY TESTS")
    print("======================================================================")
    
    for tc in test_cases:
        q = tc['query']
        resp = client.get(f'{API_BASE}/module2/holdings?target={q}')
        assert resp.status_code == 200, f"Failed for {q}: status {resp.status_code}"
        data = resp.json()
        
        assert data.get('company_name') == tc['expected_name'], f"Name mismatch for {q}: got {data.get('company_name')}, expected {tc['expected_name']}"
        
        tl = data.get('trendlyne_data')
        assert tl is not None, f"Trendlyne data missing for {q}"
        assert tl.get('current_price') is not None, f"Trendlyne CMP missing for {q}"
        assert data.get('price_source') != 'Trendlyne Live (NSE)', f"Undated autocomplete quote became decision price: {data.get('price_source')}"
        assert data.get('price_is_live') is not True
        assert tl.get('source_quote_at') is None
        
        # Check decision integrity
        cmp = float(data['current_price_adjusted'])
        issue_p = float(data.get('price_waterfall', {}).get('issue_price', cmp))
        if cmp >= issue_p:
            assert 'below' not in data.get('explanation', '').lower() or 'below issue price' not in data.get('explanation', '').lower(), f"Contradictory explanation when profitable: {data.get('explanation')}"
        
        print(f" [PASS] {q:15} -> {data['company_name']:30} | CMP: Rs.{cmp:<7} | Source: {data['price_source']} | Verdict: {data['verdict']}")
    
    # Test Search Autocomplete augmentation
    search_resp = client.get(f'{API_BASE}/module2/search?q=tatatech')
    assert search_resp.status_code == 200
    results = search_resp.json().get('results', [])
    assert len(results) > 0
    print(f" [PASS] Module 2 Search Autocomplete: Found {len(results)} items for 'tatatech'")
    
    print("======================================================================")
    print("       ALL MODULE 2 TRENDLYNE INTEGRITY TESTS PASSED SUCCESSFULLY     ")
    print("======================================================================")

if __name__ == "__main__":
    test_live_resolution()
