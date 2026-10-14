"""Other-source recovery, temporal leakage and paired-comparison contracts."""
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from ml.recover_module1_other_sources import (dated_history,verify_top10,verify_ipowala,
    verify_top10stock,verify_a2z,title_matches,CACHE,AUDIT)
from ml.module1_other_source_benchmark import OUT,ROOT,VARIANTS,TEMPORAL,temporal_features


def test_gmp_column_is_not_issue_price_and_missing_is_not_zero():
    content='''<table><tr><th>Date</th><th>IPO Price</th><th>GMP</th><th>Est. Listing Price</th></tr>
    <tr><td>23 Jan 2024</td><td>41</td><td>20 Rs</td><td>61</td></tr>
    <tr><td>22 Jan 2024</td><td>41</td><td>--</td><td>41</td></tr>
    <tr><td>21 Jan 2024</td><td>41</td><td>0</td><td>41</td></tr></table>'''
    assert dated_history(content,'2024-01-25')==[dict(gmp_date='2024-01-21',gmp_rs=0.),dict(gmp_date='2024-01-23',gmp_rs=20.)]


def test_yearless_history_handles_year_boundary_without_backdating_a_year():
    content='<table><tr><th>Date</th><th>GMP</th></tr><tr><td>30 Dec</td><td>5</td></tr></table>'
    assert dated_history(content,'2024-01-03')==[dict(gmp_date='2023-12-30',gmp_rs=5.)]


def test_two_digit_year_and_later_revision_are_handled_conservatively():
    content='''<table><tr><th>GMP Date</th><th>IPO Price</th><th>GMP (Price)</th><th>Last Updated</th></tr>
    <tr><td>01 Dec 24</td><td>441</td><td>₹0</td><td>2024-12-01 06:31:00 PM</td></tr>
    <tr><td>30 Nov 24</td><td>441</td><td>₹20</td><td>2024-12-06 09:00:00 AM</td></tr></table>'''
    assert dated_history(content,'2024-12-03')==[dict(gmp_date='2024-12-01',gmp_rs=0.)]


def test_explicit_published_ranges_keep_bounds_and_label_the_midpoint():
    content='''<table><tr><th>Date</th><th>GMP (in INR)</th></tr>
    <tr><td>3 Mar 2021</td><td>Rs 430-440</td></tr>
    <tr><td>2 Mar 2021</td><td>400–410</td></tr><tr><td>1 Mar 2021</td><td>--</td></tr></table>'''
    rows=dated_history(content,'2021-03-05')
    assert rows[1]==dict(gmp_date='2021-03-03',gmp_rs=435.,gmp_low_rs=430.,gmp_high_rs=440.,gmp_basis='published_range_midpoint')
    assert len(rows)==2 and rows[0]['gmp_rs']==405.


def test_verified_ticker_alias_requires_matching_offer_dates_and_cap():
    item=dict(company_name='Brainbees Solutions Ltd.',nse_symbol='FIRSTCRY',
        issue_open_date='2024-08-06',issue_close_date='2024-08-08',issue_price=465.)
    body='''<h1>FirstCry IPO GMP</h1><div class="entry-content">IPO opens on 06-August-2024.
    It will close on 08-August-2024. Price is 440-465 Rs Per Share.</div>'''
    assert verify_ipowala(item,BeautifulSoup(body,'html.parser'))
    assert not verify_ipowala({**item,'issue_price':500.},BeautifulSoup(body,'html.parser'))
    assert title_matches(item,'Brainbees Solutions ( FirstCry ) IPO Date, Price, GMP')


def test_related_issuer_in_body_cannot_validate_wrong_headline():
    item=dict(company_name='Example Systems Ltd.',issue_open_date='2024-01-01',issue_close_date='2024-01-03',issue_price=100.)
    body='''<h1>Other Company IPO GMP</h1><p>Related: Example Systems Limited</p>
    <table><tr><td>IPO Open Date</td><td>1 Jan 2024</td></tr><tr><td>IPO Close Date</td><td>3 Jan 2024</td></tr></table>'''
    assert not verify_top10(item,BeautifulSoup(body,'html.parser'))


def test_older_page_identity_can_use_listing_calendar_and_explicit_lot_cost():
    item=dict(company_name='Stove Kraft Ltd.',issue_open_date='2021-01-25',issue_close_date='2021-01-28',
        listing_date='2021-02-05',issue_price=385.)
    page='''<h1>Stove Kraft IPO GMP Today</h1><table>
    <tr><td>Lot Size:</td><td>Minimum 38 Shares & Maximum 494 Shares</td></tr>
    <tr><td>Minimum Amount:</td><td>₹14630/-</td></tr>
    <tr><td>Listing Date:</td><td>Feb 5, 2021</td></tr></table>'''
    assert verify_top10(item,BeautifulSoup(page,'html.parser'))
    assert not verify_top10({**item,'listing_date':'2021-02-06'},BeautifulSoup(page,'html.parser'))
    assert not verify_top10({**item,'issue_price':500.},BeautifulSoup(page,'html.parser'))


def test_ipowala_identity_dates_and_cap_all_required():
    item=dict(company_name='Medi Assist Healthcare Services Ltd.',issue_open_date='2024-01-15',issue_close_date='2024-01-17',issue_price=418.)
    body='''<h1>Medi Assist Healthcare IPO GMP, Grey Market Premium Today</h1><div class="entry-content">
    The Medi Assist Healthcare has begun on 15-January-2024. It will close on 17-January-2024.
    The company set its IPO price is 397-418 Rs Per Share.</div>'''
    assert verify_ipowala(item,BeautifulSoup(body,'html.parser'))
    assert not verify_ipowala({**item,'issue_price':500.},BeautifulSoup(body,'html.parser'))


def test_no_final_book_columns_in_any_model_feature_set():
    for features in VARIANTS.values():
        assert not any('final' in f or 'target' in f or 'exchange_open' in f for f in features)


def test_temporal_slope_uses_only_previous_same_publisher_quotes():
    frame=pd.DataFrame([dict(ipo_id=1,gmp_pct=20.,gmp_date='2024-01-02',decision_market_date=pd.Timestamp('2024-01-03'),
        gmp_source_url='https://publisher.example/a',log_offer_cap_price=np.log1p(100.))])
    quotes=pd.DataFrame([dict(ipo_id=1,gmp_date=date,gmp_rs=value,source_url=source) for date,value,source in [
        ('2024-01-01',10,'https://publisher.example/a'),('2024-01-01',90,'https://other.example/b'),
        ('2024-01-03',500,'https://publisher.example/a')]])
    growth=pd.DataFrame([dict(ipo_id=1,status='penultimate_table_conflicts_with_frozen_inputs',
        log_qib_growth=99.,log_hni_growth=99.,log_retail_growth=99.,postcutoff_final_qib=999.)])
    row=temporal_features(frame,quotes,growth).iloc[0]
    assert row.gmp_quote_age_days==1
    assert np.isclose(row.gmp_slope_pp_per_day,10.)
    assert pd.isna(row.log_qib_growth)


def test_all_63_were_attempted_and_accepted_quotes_reproduce_public_pages():
    audit=json.loads(AUDIT.read_text());assert audit['completed']==audit['attempted']==63
    offers=pd.read_csv(ROOT/'data/nse_verified_offer_dates.csv').set_index('ipo_id')
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    for row in audit['sources']:
        with gzip.open(CACHE/f"{row['ipo_id']}.json.gz",'rt',encoding='utf-8') as stream:
            cached=json.load(stream)
        assert {'ipoji','ipowala','top10stockbroker','top10stock','a2zipo'}<=set(s['provider'] for s in cached['sources'])
        item=offers.loc[row['ipo_id']].to_dict()
        item.update(company_name=canonical.loc[row['ipo_id']].company_name,issue_price=canonical.loc[row['ipo_id']].price_band_upper,
            nse_symbol=str(canonical.loc[row['ipo_id']].nse_symbol))
        for source in cached['sources']:
            if not source.get('quotes'):
                continue
            content=source['raw_response_text'];soup=BeautifulSoup(content,'html.parser')
            checker={'top10stockbroker':verify_top10,'ipowala':verify_ipowala,'top10stock':verify_top10stock,'a2zipo':verify_a2z,
                'investoracademy':verify_top10,'adigitalblogger':verify_top10}[source['provider']]
            identity_soup=BeautifulSoup(source['identity_response_text'],'html.parser') if source['provider']=='a2zipo' else soup
            assert checker(item,identity_soup)
            assert hashlib.sha256(content.encode('utf-8')).hexdigest()==source['source_sha256']
            assert dated_history(content,item['issue_close_date'])==source['quotes']


def test_all_predictions_kept_on_same_39_and_51_cases_and_metrics_recompute():
    report=json.loads(OUT.read_text());ledger=pd.read_csv(OUT.with_name(OUT.stem+'_predictions.csv'))
    assert report['results']['original_39']['n']==39 and report['results']['previous_51']['n']==51
    previous=json.loads((ROOT/'output/module1_completion_20261024.json').read_text())
    assert set(report['results']['original_39']['ipo_ids'])==set(previous['results']['original_primary']['ipo_ids'])
    assert set(report['results']['previous_51']['ipo_ids'])==set(previous['results']['expanded_secondary']['ipo_ids'])
    test=ledger[ledger.year.eq(2024)]
    for population,result in report['results'].items():
        rows=test[test.ipo_id.isin(result['ipo_ids'])]
        assert len(rows)==result['n']==rows.ipo_id.nunique()
        for track in VARIANTS:
            assert result['direction'][track]['metrics']['classifier_accuracy_pct']==round(
                rows['p_'+track].ge(.5).eq(rows.target_binary.astype(bool)).mean()*100,1)
        for name,metrics in result['returns'].items():
            assert np.isclose(metrics['mae_pp'],np.abs(rows.target_return-rows[name]).mean())
    for trace in report['trace']:
        assert not set(trace['train_ids'])&set(trace['test_ids'])
        assert trace['latest_training_outcome']<trace['first_test_decision']
    assert pd.to_datetime(ledger.gmp_date).lt(pd.to_datetime(ledger.decision_market_date)).all()
    assert not report['live_promoted'] and not report['module2_changed'] and not report['paper_changed']
    for relative,expected in report['input_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    assert hashlib.sha256((ROOT/report['research_artifact']).read_bytes()).hexdigest()==report['research_artifact_sha256']
    selected=min([t for t in VARIANTS if t!='gmp_fitted'],key=lambda t:(report['development_direction'][t]['brier_score'],t))
    assert selected==report['selected_direction']
    ranks=report['development_ranking']
    eligible=[n for n,item in ranks.items() if item['panels']>=5 and item['pair_count']>=10]
    def key(name):
        complexity=0 if name=='gmp_pct' else len(VARIANTS[name[2:].rsplit('_',1)[0]])
        return (-(ranks[name]['pairwise_accuracy_pct'] or 0),complexity,name)
    assert report['selected_ranking']==(min(eligible,key=key) if eligible else 'gmp_pct')
    errors=pd.read_csv(OUT.with_name(OUT.stem+'_errors.csv'))
    assert set(errors.ipo_id)==set(test.ipo_id)
    gaps=pd.read_csv(OUT.with_name(OUT.stem+'_63_gap_results.csv'))
    assert len(gaps)==gaps.ipo_id.nunique()==63
    assert int(gaps.newly_recovered.sum())==report['coverage']['recovered_n']
    assert gaps.loc[gaps.gmp_pct.isna(),'gmp_gap_reason'].notna().all()
