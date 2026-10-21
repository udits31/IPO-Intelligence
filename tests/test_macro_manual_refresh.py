from fastapi import Response


def test_every_market_refresh_polls_source_and_keeps_exchange_time(monkeypatch):
    from api import main
    calls=[]
    def fetch():
        calls.append(True)
        return {'timestamp':'2026-10-28T15:30:00+05:30',
                'retrieved_at_ist':f'2026-10-28T19:00:0{len(calls)}+05:30'}
    monkeypatch.setattr(main,'fetch_live_macro_regime',fetch)
    first_response=Response();second_response=Response()
    first=main.get_live_macro(first_response);second=main.get_live_macro(second_response)
    assert len(calls)==2
    assert first['timestamp']==second['timestamp']
    assert first['retrieved_at_ist']!=second['retrieved_at_ist']
    assert first_response.headers['cache-control']=='no-store, max-age=0'
    assert second_response.headers['cache-control']=='no-store, max-age=0'
