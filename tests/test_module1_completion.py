"""Module 1 completion: frozen cohorts, as-of sources, ranking and coverage."""
import hashlib
import gzip
import json

import numpy as np
import pandas as pd

from ml.complete_module1_benchmark import OUT,ROOT,TRACKS,SectorRelativeFacts,attach_quotes
from ml.module1_gap_recovery import history,verified_post
from ml.module1_gap_recovery import CACHE,AUDIT as RECOVERY_AUDIT
from ml.extract_dated_ipo_fundamentals import document_date


def test_document_title_date_precedes_incorporation_narrative_date():
    assert document_date('''RED HERRING PROSPECTUS
    March 9, 2021
    Please read Section 32
    Our company has a certificate of incorporation dated January 29, 2009.''')=='2021-03-09'
    assert document_date('Our company was incorporated pursuant to a certificate dated May 14, 2007.') is None
    assert document_date('Dated: September 2, 2024')=='2024-09-02'


def test_header_normalization_never_turns_unavailable_gmp_into_zero():
    html='''<table><tr><th>GMP Date</th><th>GMP (Rs)</th></tr>
    <tr><td>12 March 2024</td><td>--</td></tr>
    <tr><td>13 March 2024</td><td>0</td></tr>
    <tr><td>14 March 2024</td><td>5%</td></tr></table>'''
    assert history(html,'2024-03-14')==[dict(gmp_date='2024-03-13',gmp_rs=0.)]


def test_issuer_name_alone_cannot_verify_a_history():
    item=dict(company_name='Example Systems Ltd.',issue_open_date='2024-03-11',issue_close_date='2024-03-13',issue_price=100)
    post=dict(date='2024-03-01',content={'rendered':'Example Systems Limited'})
    assert not verified_post(item,'Example Systems IPO GMP',post)
    post['content']['rendered']+='<table><tr><td>IPO Close Date:</td><td>13 March 2024</td></tr></table>'
    assert verified_post(item,'Example Systems IPO GMP',post)
    post['date']='2026-03-01'
    assert not verified_post(item,'Example Systems IPO GMP',post)


def test_punctuation_in_an_exact_issuer_title_does_not_break_identity():
    item=dict(company_name='J.G.Chemicals Ltd.',issue_open_date='2024-03-05',issue_close_date='2024-03-07',issue_price=221)
    post=dict(date='2024-03-01',content={'rendered':
        'JG Chemicals Limited<table><tr><td>IPO Close Date:</td><td>7 March 2024</td></tr></table>'})
    assert verified_post(item,'JG Chemicals IPO GMP',post)


def test_quote_gate_rejects_same_day_stale_and_conflicting_values():
    early=pd.DataFrame([dict(ipo_id=1,decision_market_date='2024-03-13',gmp_pct=np.nan,gmp_date=None,gmp_source_url=None)])
    canonical=pd.DataFrame([dict(id=1,price_band_upper=100)]).set_index('id')
    quotes=pd.DataFrame([dict(ipo_id=1,gmp_date=date,gmp_rs=value,source_url='https://publisher.example/history')
        for date,value in [('2024-03-01',20),('2024-03-13',90),('2024-03-12',10),('2024-03-12',11),('2024-03-11',5)]])
    result=attach_quotes(early,quotes,canonical).iloc[0]
    assert result.gmp_pct==5. and result.gmp_date=='2024-03-11'
    assert result.gmp_conflict_days==1


def test_sector_transform_does_not_learn_from_test_rows_or_labels():
    train=pd.DataFrame(dict(financial_sector=[0.]*6+[1.]*6,
        earnings_yield_pct=[1,2,3,4,5,6,10,20,30,40,50,60],eps_growth_asinh=[1.]*12))
    a=SectorRelativeFacts().fit(train,np.zeros(12))
    b=SectorRelativeFacts().fit(train,np.ones(12))
    assert a.statistics_==b.statistics_
    stats=json.dumps(a.statistics_,sort_keys=True)
    test=train.iloc[:1].copy();test.loc[:,'earnings_yield_pct']=1e12
    a.transform(test)
    assert json.dumps(a.statistics_,sort_keys=True)==stats
    assert a.transform(test).earnings_yield_pct.iloc[0]<10


def test_complete_evidence_and_frozen_primary_denominator():
    report=json.loads(OUT.read_text())
    ledger=pd.read_csv(OUT.with_name(OUT.stem+'_predictions.csv'))
    test=ledger[ledger.year.eq(2024)]
    assert report['results']['original_primary']['n']==39
    assert report['results']['expanded_secondary']['n']==len(test)==test.ipo_id.nunique()
    for subset,result in report['results'].items():
        rows=test[test.original_gmp_covered] if subset=='original_primary' else test
        assert set(rows.ipo_id)==set(result['ipo_ids'])
        for track in TRACKS:
            m=result['direction'][track]['metrics']
            assert m['classifier_accuracy_pct']==round(rows['p_'+track].ge(.5).eq(rows.target_binary.astype(bool)).mean()*100,1)
        for name,metrics in result['returns'].items():
            assert np.isclose(metrics['mae_pp'],np.abs(rows[name]-rows.target_return).mean())
        for name,ranking in result['ranking'].items():
            for choice in ranking['choices']:
                panel=rows[pd.to_datetime(rows.decision_market_date).eq(pd.Timestamp(choice['decision_date']))]
                assert set(panel.ipo_id)==set(choice['eligible_ids'])
                assert choice['n']==len(panel)>=2
    for trace in report['trace']:
        assert not set(trace['train_ids'])&set(trace['test_ids'])
        assert trace['latest_training_outcome']<trace['first_test_decision']
    assert pd.to_datetime(ledger.gmp_date).lt(pd.to_datetime(ledger.decision_market_date)).all()
    assert not report['live_promoted'] and not report['paper_changed'] and not report['module2_changed']


def test_candidates_selected_on_development_only_and_sources_bound():
    report=json.loads(OUT.read_text())
    sd=min([t for t in TRACKS if t!='gmp_fitted'],key=lambda t:(report['direction_development'][t]['brier_score'],t))
    sr=min([n for n in report['return_development'] if 'gmp_fitted' not in n],key=lambda n:(report['return_development'][n]['mae_pp'],n))
    assert sd==report['selected_direction'] and sr==report['selected_return']
    for relative,expected in report['input_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    coverage=pd.read_csv(OUT.with_name(OUT.stem+'_coverage.csv'))
    assert len(coverage)==coverage.ipo_id.nunique()==report['coverage']['universe_n']==441
    assert int(coverage.gmp_recovered.fillna(False).sum())==report['coverage']['recovered_paired_n']


def test_recovered_quotes_reproduce_retained_publisher_responses():
    from bs4 import BeautifulSoup
    audit=json.loads(RECOVERY_AUDIT.read_text())
    offers=pd.read_csv(ROOT/'data/nse_verified_offer_dates.csv').set_index('ipo_id')
    canonical=pd.read_csv(ROOT/'data/canonical_listed_ipos.csv').set_index('id')
    for source in audit['sources']:
        if source['status']!='matched_daily_history':
            continue
        with gzip.open(CACHE/f"{source['ipo_id']}.json.gz",'rt',encoding='utf-8') as stream:
            retained=json.load(stream)
        raw=retained['raw_response_text']
        assert hashlib.sha256(raw.encode('utf-8')).hexdigest()==retained['source_sha256']
        post=json.loads(raw);item=offers.loc[source['ipo_id']].to_dict()
        item['company_name']=canonical.loc[source['ipo_id']].company_name
        item['issue_price']=canonical.loc[source['ipo_id']].issue_price
        title=BeautifulSoup(post['title']['rendered'],'html.parser').get_text(' ',strip=True)
        assert verified_post(item,title,post)
        assert history(post['content']['rendered'],item['issue_close_date'])==retained['rows']
