import numpy as np
import pandas as pd
from ml.train_module2_next_session import trading, continuous_trading, historical_news_at, split_data


def test_trading_uses_next_open_return_and_charges_round_trip_cost():
    frame=pd.DataFrame({'outcome_date':pd.to_datetime(['2026-01-05','2026-01-05']),
                        'executable_return_pct':[1.,-10.]})
    s=trading(frame,[1,0],35)
    assert s['trades']==1
    assert np.isclose(s['mean_net_return_per_opportunity_pct'],.325)
    assert np.isclose(s['equal_weight_daily_compound_return_pct'],.325)
    assert trading(frame,[0,0],35)['equal_weight_daily_compound_return_pct']==0


def test_news_excludes_future_same_day_and_stale_articles():
    articles=[(pd.Timestamp(d),1.) for d in ['2025-01-01T10:00Z','2025-01-08T10:00Z',
        '2025-01-09T03:00Z','2025-01-10T10:00Z']]
    admitted=historical_news_at(articles,'2025-01-09T18:00:00+05:30')
    assert len(admitted)==1 and admitted[0][0]==pd.Timestamp('2025-01-08T10:00Z')


def test_continuous_hold_pays_only_entry_and_terminal_exit():
    f=pd.DataFrame({'ipo_id':[1,1,1],'outcome_date':pd.date_range('2025-01-01',periods=3),
                    'next_open':[100.,100.,100.],'next_close':[100.,100.,100.]})
    s=continuous_trading(f,[1,1,1],35)
    assert s['buys']==s['sells']==1
    assert np.isclose(s['mean_issuer_net_window_return_pct'],100*((1-.00175)**2-1))


def test_split_purges_immature_outcomes_and_keeps_issuers_disjoint():
    f=pd.DataFrame({'ipo_id':[1,2,3,4], 'listing_year':[2023,2023,2024,2025],
        'decision_date':pd.to_datetime(['2023-12-01','2023-12-31','2024-01-01','2025-01-01']),
        'outcome_date':pd.to_datetime(['2023-12-02','2024-01-02','2024-01-02','2025-01-02'])})
    train,val,final,test=split_data(f)
    assert train.ipo_id.tolist()==[1]
    assert not set(final.ipo_id)&set(test.ipo_id)
    assert final.outcome_date.max()<test.decision_date.min()
