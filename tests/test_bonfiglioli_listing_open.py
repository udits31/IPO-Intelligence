"""Regression: an archived, flat listing must not remain pending."""
import sqlite3
from datetime import date
from copy import deepcopy

from scrapers.live_ipo_scraper import load_scraped_cache
from services.module2_recent_exchange import _identity_matches
from services.listing_outcome_service import record_verified_listing_outcomes, verified_listing_outcomes
from services.prediction_results_service import result_row


def test_exchange_abbreviation_is_bounded_to_the_actual_issuer():
    assert _identity_matches('Bonfiglioli Transmissions', 'BONFIGLIOLI TRANS LIMITED')
    assert not _identity_matches('Bonfiglioli Transport', 'BONFIGLIOLI TRANS LIMITED')
    assert not _identity_matches('Other Transmissions', 'BONFIGLIOLI TRANS LIMITED')
    assert not _identity_matches('Bonfiglioli Transmissions', 'BONFIGLIOLI TEXTILES LIMITED')


def test_real_nse_open_is_recorded_without_rewriting_prediction(tmp_path):
    cache=deepcopy({url:item for url,item in load_scraped_cache().items() if '/2861/' in url})
    assert len(cache)==1
    db=tmp_path/'outcomes.db'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE automatic_decision_v1 (ipo_id TEXT, decision_kind TEXT)')
        connection.execute('CREATE TABLE prelisting_outcome_forecast_v1 (ipo_id TEXT)')
        connection.execute("INSERT INTO prelisting_outcome_forecast_v1 VALUES ('chitto_2861')")
    response=record_verified_listing_outcomes(cache=cache,as_of=date(2026,10,23),db_path=db)
    assert response['saved']==['chitto_2861']
    outcome=verified_listing_outcomes(db)['chitto_2861']
    assert outcome['issue_price']==outcome['nse_open_price']==586.
    assert outcome['nse_symbol']=='BONFIG'
    assert len(outcome['archive_sha256'])==64
    row=result_row(company=outcome['company_name'],probability=.9120698868302227,
                   issue=outcome['issue_price'],opening=outcome['nse_open_price'],evidence_kind='After-bidding replay')
    assert row['actual_direction']=='Flat'
    assert row['actual_return_pct']==0
    assert row['result']=='Incorrect'
    assert record_verified_listing_outcomes(cache=cache,as_of=date(2026,10,23),db_path=db)['saved']==[]
