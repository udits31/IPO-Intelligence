"""Exercise the real route body without starting background API collectors."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import pandas as pd
import services.module2_recent_exchange as recent


def test_reliance_symbol_cannot_resolve_to_reliance_nippon(monkeypatch):
    source=Path('api/main.py').read_text(encoding='utf8')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_holding_by_id_or_name')
    node.decorator_list=[]
    def wrong_issuer(*args,**kwargs):raise AssertionError('Resolved a different company')
    ns={'Optional':Optional,'pd':pd,
        'df_canonical':pd.DataFrame([{'id':1,'company_name':'Reliance Nippon Life Asset Management Ltd.',
                                    'nse_symbol':'NAM-INDIA','bse_scrip_code':'540767'}]),
        'trendlyne_service':SimpleNamespace(fetch_live_stock_data=lambda q:{'stock_code':'RELIANCE','company_name':'Reliance Industries Ltd.'}),
        '_canonical_monitor_result':wrong_issuer}
    monkeypatch.setattr(recent,'recent_exchange_monitors',lambda **kw:({},{}))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'api/main.py','exec'),ns)
    r=ns[node.name]('RELIANCE',as_of_date='2026-10-28')
    assert r['nse_symbol']=='RELIANCE'
    assert r['company_name']=='Reliance Industries Ltd.'
    assert r['status']=='OUTSIDE_IPO_MONITOR_UNIVERSE'
    assert 'verdict' not in r
