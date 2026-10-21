import json
from concurrent.futures import ThreadPoolExecutor
from scrapers import live_ipo_scraper as scraper


def test_concurrent_writers_produce_one_complete_cache_and_preserve_legacy(tmp_path,monkeypatch):
    path=tmp_path/'calendar.json'
    monkeypatch.setattr(scraper,'CACHE_FILE_PATH',str(path))
    path.write_text(json.dumps({'legacy':dict(company_name='Unverified',source_observed=False)}))
    def write(i):
        row=dict(company_name='Observed',source_observed=True,category='OPEN',
                 date_range='2026-10-28 to 2026-11-02',writer=i,body=str(i)*10000)
        scraper.save_scraped_cache({'observed':row})
        # A read during other writes must still parse a whole JSON document.
        parsed=json.loads(path.read_text(encoding='utf8'))
        assert 'legacy' in parsed and parsed['observed']['body']==str(parsed['observed']['writer'])*10000
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(write,range(40)))
    assert set(scraper.load_scraped_cache())=={'observed'}
    assert list(tmp_path.glob('*.tmp'))==[]


def test_broken_read_retains_only_previously_validated_cache_without_rewriting(tmp_path,monkeypatch):
    path=tmp_path/'calendar.json';monkeypatch.setattr(scraper,'CACHE_FILE_PATH',str(path))
    row=dict(company_name='Observed',source_observed=True,category='OPEN',date_range='2026-10-28 to 2026-11-02',
             gmp_pct=None,subscription_observed_at='2026-10-30T13:30:00+05:30')
    scraper.save_scraped_cache({'observed':row});good=scraper.load_scraped_cache()
    path.write_text('{broken',encoding='utf8')
    assert scraper.load_scraped_cache()==good
    assert path.read_text()=='{broken'
    assert scraper.load_scraped_cache()['observed']['gmp_pct'] is None
