import zipfile
from datetime import date

from ml.compare_preclose_listing_opens import compare_row, metrics


def case(tmp_path, opening=100):
    with zipfile.ZipFile(tmp_path/'20261005.csv.zip','w') as archive:
        archive.writestr('bhav.csv', 'TradDt,TckrSymb,SctySrs,FinInstrmNm,ISIN,OpnPric,ClsPric\n'
                         f'2026-10-05,RUNWALENTR,EQ,RUNWAL ENTERPRISES LTD,INE804W01029,{opening},999\n')
    return {'company_name':'Runwal Enterprises','nse_symbol':'RUNWALENTR',
            'issue_price':100,'listing_date':'2026-10-05','positive_listing_probability':.39,
            'closing_day_timestamp_found':True,'source_reported_at_ist':'2026-09-29T14:24:00+05:30',
            'inputs':{'evidence_warnings':[]}}


def test_flat_is_nonpositive_and_open_not_day_close(tmp_path):
    row = compare_row(case(tmp_path), as_of=date(2026,10,21), archives=tmp_path)
    assert row['actual_open_price'] == 100  # Never use the deliberately different 999 close.
    assert row['actual_flat'] is True
    assert row['direction_correct'] is True
    assert row['band_result'] == 'RIGHT'
    assert row['eligible_for_primary_comparison'] is True


def test_future_and_wrong_issuer_are_not_scored(tmp_path):
    item = case(tmp_path)
    assert 'direction_correct' not in compare_row(item,as_of=date(2026,10,4),archives=tmp_path)
    wrong = compare_row({**item,'company_name':'Wrong Issuer'},as_of=date(2026,10,21),archives=tmp_path)
    assert wrong['outcome_status'] == 'identity_conflict'


def test_warning_cases_excluded_and_mixed_is_abstention(tmp_path):
    item = case(tmp_path,110)
    result = compare_row({**item,'positive_listing_probability':.55,
        'inputs':{'evidence_warnings':['article_first_published_after_application_cutoff']}},
        as_of=date(2026,10,21),archives=tmp_path)
    assert result['direction_correct'] is True
    assert result['band_result'] == 'ABSTAINED'
    assert result['eligible_for_primary_comparison'] is False
    score = metrics([result])
    assert score['n'] == 1 and score['mixed_n'] == 1 and score['decisive_n'] == 0
