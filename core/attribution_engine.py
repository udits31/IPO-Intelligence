import pandas as pd
import numpy as np
import requests
import time
import gzip
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional

_YAHOO_CACHE: Dict[str, Any] = {}
YAHOO_ARCHIVES=Path(__file__).resolve().parents[1]/'data/archive/yahoo_chart'


def parse_yahoo_chart(payload,expected_symbol):
    results=payload.get('chart',{}).get('result') or []
    if not results:
        return pd.DataFrame(columns=['date','close'])
    if len(results)!=1 or results[0].get('meta',{}).get('symbol')!=expected_symbol:
        raise ValueError('Yahoo price response symbol conflicts with the requested security')
    result=results[0]
    meta=result.get('meta',{})
    if expected_symbol.endswith(('.NS','.BO')) and meta.get('currency')!='INR':
        raise ValueError('India stock price response has an unexpected currency')
    timestamps=result.get('timestamp') or []
    closes=(((result.get('indicators') or {}).get('quote') or [{}])[0].get('close') or [])
    if len(timestamps)!=len(closes):
        raise ValueError('Yahoo close/date arrays do not align')
    rows=[{'date':pd.Timestamp(ts,unit='s',tz='UTC').tz_convert('Asia/Kolkata').normalize().tz_localize(None),
           'close':float(close)} for ts,close in zip(timestamps,closes)
          if close is not None and pd.notna(close) and np.isfinite(float(close)) and float(close)>0]
    data=pd.DataFrame(rows,columns=['date','close']).sort_values('date').reset_index(drop=True)
    if data.date.duplicated().any():
        raise ValueError('Duplicate daily stock observations in Yahoo response')
    changes=[]
    for event in (result.get('events',{}).get('splits') or {}).values():
        date=pd.Timestamp(event['date'],unit='s',tz='UTC').tz_convert('Asia/Kolkata').normalize().tz_localize(None)
        numerator,denominator=float(event['numerator']),float(event['denominator'])
        if not np.isfinite([numerator,denominator]).all() or min(numerator,denominator)<=0:
            raise ValueError('Invalid explicit Yahoo split ratio')
        if not data.empty and date<=data.date.max():
            changes.append({'ex_date':date.strftime('%Y-%m-%d'),'share_factor':numerator/denominator,
                            'numerator':numerator,'denominator':denominator})
    if len({e['ex_date'] for e in changes})!=len(changes):
        raise ValueError('Ambiguous same-day Yahoo share changes')
    data.attrs['provider_meta']={'symbol':expected_symbol,'historical_basis':'split_adjusted_quote_close',
                                 'splits':sorted(changes,key=lambda e:e['ex_date']),
                                 'company_name':meta.get('longName') or meta.get('shortName'),
                                 'dividends_included':False}
    return data

SECTOR_BENCHMARKS = {
    'Basic Materials': [('^CNXMETAL', 'Nifty Metal index')],
    'Communication Services': [('^CNXMEDIA', 'Nifty Media index')],
    'Consumer Cyclical': [('^CNXAUTO', 'Nifty Auto index'),
                          ('AUTOBEES.NS', 'Nifty Auto ETF proxy')],
    'Consumer Defensive': [('^CNXFMCG', 'Nifty FMCG index'),
                            ('FMCGIETF.NS', 'Nifty FMCG ETF proxy')],
    'Consumer Staples': [('^CNXFMCG', 'Nifty FMCG index'),
                         ('FMCGIETF.NS', 'Nifty FMCG ETF proxy')],
    'Energy': [('^CNXENERGY', 'Nifty Energy index')],
    'Financial Services': [('^NSEBANK', 'Nifty Bank index')],
    'Healthcare': [('^CNXPHARMA', 'Nifty Pharma index')],
    # Yahoo's ^CNXINFRA feed currently returns a single observation. Keep the
    # official index first, then use a liquid ETF tracking the same index.
    'Industrials': [
        ('^CNXINFRA', 'Nifty Infrastructure index'),
        ('INFRABEES.NS', 'Nifty Infrastructure ETF proxy'),
    ],
    'Real Estate': [('^CNXREALTY', 'Nifty Realty index')],
    'Technology': [('^CNXIT', 'Nifty IT index')],
    'Utilities': [('^CNXENERGY', 'Nifty Energy index')],
}


def _series_return(series: pd.DataFrame, start_date: Any, end_date: Any = None) -> Optional[float]:
    """Return benchmark performance from the close before start through end."""
    if series is None or len(series) < 2:
        return None
    data = series.dropna(subset=['date', 'close']).sort_values('date')
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize() if end_date is not None else data['date'].max()
    prior = data[data['date'] < start]
    current = data[data['date'] <= end]
    if prior.empty or current.empty:
        return None
    base = float(prior.iloc[-1]['close'])
    latest = float(current.iloc[-1]['close'])
    if base <= 0:
        return None
    return round(((latest - base) / base) * 100.0, 2)


def fetch_market_benchmark_returns(sector: str, start_date: Any, end_date: Any = None) -> Dict[str, Any]:
    """Fetch real sector and Nifty returns for a listing-period comparison."""
    nifty = AttributionEngine._fetch_yahoo_series('^NSEI')
    sector_series, source = AttributionEngine._get_sector_series(sector)
    return {
        'sector_return_pct': _series_return(sector_series, start_date, end_date),
        'nifty_return_pct': _series_return(nifty, start_date, end_date),
        'benchmark_source': source,
    }

class AttributionEngine:
    def __init__(self, prices_path='data/post_listing_prices.csv',
                 ipos_path='data/canonical_listed_ipos.csv', *,
                 prices_frame: Optional[pd.DataFrame] = None,
                 ipos_frame: Optional[pd.DataFrame] = None,
                 prefer_supplied_prices: bool = False):
        self.prices_path = prices_path
        self.ipos_path = ipos_path
        self.df_prices = prices_frame.copy() if prices_frame is not None else pd.read_csv(self.prices_path)
        self.df_prices['date'] = pd.to_datetime(self.df_prices['date'])
        self.df_ipos = ipos_frame.copy() if ipos_frame is not None else pd.read_csv(self.ipos_path)
        self.prefer_supplied_prices = prefer_supplied_prices
        
    def get_stock_series(self, ipo_id: int, as_of_date: Optional[str] = None) -> pd.DataFrame:
        df = self.df_prices[self.df_prices['ipo_id'] == ipo_id].sort_values('date').reset_index(drop=True)
        ipo_rows = self.df_ipos[self.df_ipos['id'] == ipo_id]
        if ipo_rows.empty:
            return df

        if self.prefer_supplied_prices:
            return df

        ipo = ipo_rows.iloc[0]
        symbol = str(ipo.get('nse_symbol', '')).strip()
        if not symbol or symbol.lower() == 'nan':
            return df

        # Local-First Optimization: If querying a historical date that already exists in local CSV,
        # return the local series immediately with zero external network overhead.
        if as_of_date:
            target_dt = pd.to_datetime(as_of_date)
            if not df.empty and target_dt <= df['date'].max():
                return df

        try:
            stock = self._fetch_yahoo_series(f'{symbol}.NS', ttl_seconds=900,
                                             start_date=ipo.get('listing_date'))
            nifty_raw = self._fetch_yahoo_series('^NSEI', ttl_seconds=900)
            nifty = nifty_raw.rename(columns={'close': 'nifty50_close'})
            sector_raw, benchmark_source = self._get_sector_series(str(ipo.get('sector', '')))
            sector = sector_raw.rename(columns={'close': 'sector_index_close'})
            if stock.empty:
                return df
            if len(sector) < 2:
                sector = nifty.rename(columns={'nifty50_close': 'sector_index_close'}).copy()
                benchmark_source = 'Nifty 50 broad-market proxy'

            live = stock.rename(columns={'close': 'close_price_adjusted'})
            live = live.merge(nifty, on='date', how='left').merge(sector, on='date', how='left')
            live['ipo_id'] = ipo_id
            live['price_source'] = 'Yahoo Finance (NSE)'
            live['price_basis'] = 'Unadjusted daily close'  # The latest quote, not every earlier bar.
            live['benchmark_source'] = benchmark_source
            live.attrs['provider_meta']=dict(stock.attrs.get('provider_meta') or {})
            live['archive_sha256']=stock.attrs.get('provider_meta',{}).get('raw_sha256')

            # Yahoo quote.close history already adjusts for splits, even when
            # dividend-adjusted Adj Close is not used. The issue-price basis is
            # reconciled below; never splice an unrelated adjusted CSV into it.
            return live.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
        except Exception:
            # Fall back cleanly to local dataset on network timeout or rate-limit
            return df

    @staticmethod
    def _fetch_yahoo_series(symbol: str, ttl_seconds: int = 900,
                            start_date: Any = None) -> pd.DataFrame:
        now = time.time()
        cache_key = (symbol, str(start_date) if start_date is not None else None)
        cached = _YAHOO_CACHE.get(cache_key)
        if cached and now - cached['fetched_at'] < ttl_seconds:
            return cached['data'].copy()

        if start_date is None:
            params = {'range': '6mo', 'interval': '1d','events':'div,splits'}
        else:
            start = pd.to_datetime(start_date).tz_localize('UTC')
            params = {'period1': int(start.timestamp()) - 86400,
                      'period2': int(now) + 86400, 'interval': '1d','events':'div,splits'}
        response = requests.get(
            f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}',
            params=params,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'},
            timeout=5.0,
        )
        response.raise_for_status()
        data=parse_yahoo_chart(response.json(),symbol)
        if data.empty:
            return data
        digest=hashlib.sha256(response.content).hexdigest()
        YAHOO_ARCHIVES.mkdir(parents=True,exist_ok=True)
        archive=YAHOO_ARCHIVES/f'{digest}.json.gz'
        if not archive.exists():
            with gzip.open(archive,'wb') as stream:
                stream.write(response.content)
        data.attrs['provider_meta'].update(raw_sha256=digest,source_url=response.url,
            raw_archive_path=str(archive),retrieved_at_utc=datetime.now(ZoneInfo('UTC')).isoformat())
        if start_date is not None:
            data = data[data['date'] >= pd.to_datetime(start_date).normalize()].reset_index(drop=True)
        _YAHOO_CACHE[cache_key] = {'fetched_at': now, 'data': data.copy()}
        return data

    @staticmethod
    def _get_sector_series(sector: str) -> tuple[pd.DataFrame, str]:
        candidates = SECTOR_BENCHMARKS.get(sector, [])
        try:
            reference = AttributionEngine._fetch_yahoo_series('^NSEI', ttl_seconds=900)
        except Exception:
            return pd.DataFrame(columns=['date', 'close']), 'Nifty 50 broad-market proxy'
        recent_dates = reference.sort_values('date')['date'].tail(31)
        for symbol, label in candidates:
            try:
                data = AttributionEngine._fetch_yahoo_series(symbol, ttl_seconds=900)
                # An isolated point is not a usable 30-session benchmark.
                observed = data['date'].isin(recent_dates).sum()
                if len(recent_dates) >= 31 and observed >= 28 and (pd.Timestamp.now().normalize() - data['date'].max()).days <= 10:
                    return data, label
            except Exception:
                continue
        return pd.DataFrame(columns=['date', 'close']), 'Nifty 50 broad-market proxy'

    def compute_attribution(self, ipo_id: int, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        df_stock = self.get_stock_series(ipo_id, as_of_date=as_of_date)
        if len(df_stock) == 0:
            return {'error': 'No price history found for ipo_id ' + str(ipo_id)}
            
        ipo_info = self.df_ipos[self.df_ipos['id'] == ipo_id].iloc[0]
        company_name = ipo_info['company_name']
        sector = ipo_info['sector']
        listing_date = pd.to_datetime(ipo_info['listing_date'])
        lockin_30d = pd.to_datetime(ipo_info['anchor_lockin_50pct_date'])
        lockin_90d = pd.to_datetime(ipo_info['anchor_lockin_full_date'])
        
        if as_of_date:
            target_dt = pd.to_datetime(as_of_date)
            df_curr = df_stock[df_stock['date'] <= target_dt]
        else:
            # Never use a future or still-forming daily candle as a close.
            now_ist = datetime.now(ZoneInfo('Asia/Kolkata'))
            last_date = pd.Timestamp(now_ist.date())
            if (now_ist.hour, now_ist.minute) < (15, 30):
                df_curr = df_stock[df_stock['date'] < last_date]
            else:
                df_curr = df_stock[df_stock['date'] <= last_date]
            
        if len(df_curr) == 0:
            return {'error': 'No price records on or before ' + str(as_of_date)}

        # Reconcile to the same per-share units before returns or risk rules.
        # The newest actual quote is unchanged; original offer price remains
        # separately available for an honest, labelled waterfall.
        from services.corporate_action_service import reconcile_prices,PriceBasisError
        original_issue=float(ipo_info['issue_price'])
        try:
            df_curr,comparable_issue,basis_adjustment=reconcile_prices(df_curr,ipo_info,
                asof=df_curr.iloc[-1]['date'],provider_meta=df_stock.attrs.get('provider_meta'),
                allow_network=as_of_date is None)
        except PriceBasisError as exc:
            reference=df_curr.iloc[-1]
            return {'status':'PRICE_BASIS_UNVERIFIED','company_name':company_name,
                    'evaluation_date':reference['date'].strftime('%Y-%m-%d'),
                    'current_price_adjusted':float(reference['close_price_adjusted']),
                    'price_source':str(reference.get('price_source','Local market history')),
                    'message':str(exc)}
        if (not as_of_date and not self.prefer_supplied_prices
                and basis_adjustment['status']=='unspecified_legacy_basis'):
            reference=df_curr.iloc[-1]
            return {'status':'PRICE_BASIS_UNVERIFIED','company_name':company_name,
                    'evaluation_date':reference['date'].strftime('%Y-%m-%d'),
                    'current_price_adjusted':float(reference['close_price_adjusted']),
                    'price_source':str(reference.get('price_source','Local market history')),
                    'message':'The retained price history has no verified share-adjustment basis. Current advice is paused; the reference quote is still shown.'}
        ipo_info=ipo_info.copy()
        ipo_info['issue_price']=comparable_issue
            
        latest_idx = len(df_curr) - 1
        curr_row = df_curr.iloc[latest_idx]
        curr_date = curr_row['date']
        curr_stock_p = curr_row['close_price_adjusted']
        curr_sec_p = curr_row['sector_index_close']
        curr_nifty_p = curr_row['nifty50_close']
        benchmark_is_market_proxy = 'broad-market proxy' in str(curr_row.get('benchmark_source', '')).lower()

        tl_data = None
        if not as_of_date:
            try:
                from services.trendlyne_service import trendlyne_service
                sym = str(ipo_info.get('nse_symbol', '')).strip()
                if sym and sym.lower() not in {'nan','none'}:
                    tl_data = trendlyne_service.fetch_live_stock_data(sym)
                if not tl_data and company_name:
                    tl_data = trendlyne_service.fetch_live_stock_data(company_name)
                # Autocomplete provides no source quote timestamp. It may be
                # shown as context, but cannot replace a dated daily close in
                # a historical return or monitor decision.
            except Exception:
                tl_data = None
        
        horizons = [10, 30, 90]
        attribution_results = {}
        
        for h in horizons:
            if latest_idx >= h:
                past_row = df_curr.iloc[latest_idx - h]
                past_date = past_row['date']
                stock_ret = ((curr_stock_p - past_row['close_price_adjusted']) / past_row['close_price_adjusted']) * 100.0
                
                sec_ret = None
                if not benchmark_is_market_proxy and pd.notna(curr_sec_p) and pd.notna(past_row['sector_index_close']) and past_row['sector_index_close'] > 0:
                    sec_ret = ((curr_sec_p - past_row['sector_index_close']) / past_row['sector_index_close']) * 100.0
                    
                nifty_ret = None
                if pd.notna(curr_nifty_p) and pd.notna(past_row['nifty50_close']) and past_row['nifty50_close'] > 0:
                    nifty_ret = ((curr_nifty_p - past_row['nifty50_close']) / past_row['nifty50_close']) * 100.0
                    
                spread_sec = stock_ret - sec_ret if sec_ret is not None else None
                spread_nifty = stock_ret - nifty_ret if nifty_ret is not None else None
                
                if sec_ret is None:
                    cat = 'INSUFFICIENT_BENCHMARK'
                    desc = 'Sector benchmark history is unavailable for this interval.'
                elif stock_ret < 0 and sec_ret < 0 and abs(spread_sec) <= 4.0:
                    cat = 'SECTOR_PULLBACK'
                    desc = 'Sector-wide move (macro/industry trend affecting entire sector).'
                elif stock_ret <= -5.0 and spread_sec < -6.0:
                    cat = 'STOCK_SPECIFIC_LAG'
                    desc = 'Stock-specific problem (significantly underperforming sector peers).'
                elif nifty_ret is not None and nifty_ret < -5.0 and stock_ret < 0 and spread_nifty >= -3.0:
                    cat = 'BROAD_MARKET_CORRECTION'
                    desc = 'Broad market sell-off dragging the stock along with benchmark.'
                elif spread_sec >= 5.0:
                    cat = 'OUTPERFORMING_SECTOR'
                    desc = 'Company outperforming industry benchmark with strong relative alpha.'
                else:
                    cat = 'NEUTRAL_MOVE'
                    desc = 'Normal trading volatility within expected peer boundaries.'
                    
                attribution_results[str(h) + 'd'] = {
                    'window_days': h,
                    'start_date': past_date.strftime('%Y-%m-%d'),
                    'end_date': curr_date.strftime('%Y-%m-%d'),
                    'stock_return_pct': round(stock_ret, 2),
                    'sector_return_pct': round(sec_ret, 2) if sec_ret is not None else None,
                    'nifty_return_pct': round(nifty_ret, 2) if nifty_ret is not None else None,
                    'spread_vs_sector_pct': round(spread_sec, 2) if spread_sec is not None else None,
                    'spread_vs_nifty_pct': round(spread_nifty, 2) if spread_nifty is not None else None,
                    'category': cat,
                    'description': desc
                }
            else:
                attribution_results[str(h) + 'd'] = None

        # Young listings still need a meaningful comparison before ten sessions
        # have elapsed. Use every completed trading interval available instead of
        # rendering an empty relative-performance panel.
        attribution_results['since_listing'] = None
        if latest_idx >= 1:
            past_row = df_curr.iloc[0]
            stock_ret = ((curr_stock_p - past_row['close_price_adjusted']) / past_row['close_price_adjusted']) * 100.0
            sec_ret = None
            if not benchmark_is_market_proxy and pd.notna(curr_sec_p) and pd.notna(past_row['sector_index_close']) and past_row['sector_index_close'] > 0:
                sec_ret = ((curr_sec_p - past_row['sector_index_close']) / past_row['sector_index_close']) * 100.0
            nifty_ret = None
            if pd.notna(curr_nifty_p) and pd.notna(past_row['nifty50_close']) and past_row['nifty50_close'] > 0:
                nifty_ret = ((curr_nifty_p - past_row['nifty50_close']) / past_row['nifty50_close']) * 100.0
            spread_sec = stock_ret - sec_ret if sec_ret is not None else None
            if sec_ret is None:
                category = 'INSUFFICIENT_BENCHMARK'
                description = 'Sector benchmark history is still building.'
            elif stock_ret < 0 and sec_ret < 0 and abs(spread_sec) <= 4.0:
                category = 'SECTOR_PULLBACK'
                description = 'Sector-wide move rather than company-specific weakness.'
            elif stock_ret <= -5.0 and spread_sec < -6.0:
                category = 'STOCK_SPECIFIC_LAG'
                description = 'Stock is materially underperforming its sector benchmark.'
            elif spread_sec >= 5.0:
                category = 'OUTPERFORMING_SECTOR'
                description = 'Stock is outperforming its sector benchmark.'
            else:
                category = 'NEUTRAL_MOVE'
                description = 'Stock is trading within normal sector-relative bounds.'
            attribution_results['since_listing'] = {
                'label': f'Since listing · {latest_idx + 1} sessions',
                'window_days': latest_idx,
                'start_date': past_row['date'].strftime('%Y-%m-%d'),
                'end_date': curr_date.strftime('%Y-%m-%d'),
                'stock_return_pct': round(stock_ret, 2),
                'sector_return_pct': round(sec_ret, 2) if sec_ret is not None else None,
                'nifty_return_pct': round(nifty_ret, 2) if nifty_ret is not None else None,
                'spread_vs_sector_pct': round(spread_sec, 2) if spread_sec is not None else None,
                'category': category,
                'description': description,
            }
                
        is_past_30d = curr_date >= lockin_30d
        is_past_90d = curr_date >= lockin_90d
        
        lockin_status = {
            'dates_verified': (str(ipo_info.get('anchor_lockin_dates_verified', '')).lower() == 'true'
                               and str(ipo_info.get('anchor_lockin_source_url', '')).startswith('https://')),
            'source_url': str(ipo_info.get('anchor_lockin_source_url', '')),
            'date_basis': str(ipo_info.get('anchor_lockin_date_basis', 'legacy_calendar_unverified')),
            'lockin_30d_date': lockin_30d.strftime('%Y-%m-%d'),
            'lockin_90d_date': lockin_90d.strftime('%Y-%m-%d'),
            'is_30d_unlocked': bool(is_past_30d),
            'is_90d_unlocked': bool(is_past_90d),
            'days_to_30d_lockin': (lockin_30d - curr_date).days if not is_past_30d else 0,
            'days_to_90d_lockin': (lockin_90d - curr_date).days if not is_past_90d else 0
        }
        
        issue_p = float(ipo_info['issue_price']) if 'issue_price' in ipo_info and pd.notna(ipo_info['issue_price']) else float(curr_stock_p)
        day1_close = float(df_curr.iloc[0]['close_price_adjusted'])
        day30_close = float(df_curr.iloc[30]['close_price_adjusted']) if len(df_curr) > 30 else None
        day90_close = float(df_curr.iloc[90]['close_price_adjusted']) if len(df_curr) > 90 else None
        peak_p = float(df_curr['close_price_adjusted'].max())
        cum_ret = round(((curr_stock_p - issue_p) / issue_p) * 100.0, 2)
        dd_peak = round(((curr_stock_p - peak_p) / peak_p) * 100.0, 2)
        
        # Recent 10 trading days trajectory
        trajectory = []
        for _, tr_row in df_curr.tail(10).iterrows():
            trajectory.append({
                'date': tr_row['date'].strftime('%Y-%m-%d'),
                'price': round(float(tr_row['close_price_adjusted']), 2)
            })
        
        price_waterfall = {
            'issue_price': round(issue_p, 2),
            'original_issue_price':original_issue,
            'issue_price_is_share_adjusted':basis_adjustment.get('share_factor') not in (None,1.),
            'day1_close': round(day1_close, 2),
            'day1_return_pct': round(((day1_close - issue_p) / issue_p) * 100.0, 2),
            'day30_close': round(day30_close, 2) if day30_close else None,
            'day90_close': round(day90_close, 2) if day90_close else None,
            'current_price': round(curr_stock_p, 2),
            'peak_high_price': round(peak_p, 2),
            'drawdown_from_peak_pct': dd_peak,
            'cumulative_gain_pct': cum_ret
        }
        
        price_source = str(curr_row.get('price_source', 'Local market history'))
        price_basis = curr_row.get('price_basis')
        if pd.isna(price_basis):
            price_basis = None
        benchmark_source = str(curr_row.get('benchmark_source', f'{sector} sector history'))
        return {
            'ipo_id': ipo_id,
            'company_name': company_name,
            'nse_symbol': (str(ipo_info.get('nse_symbol')).strip().upper()
                           if pd.notna(ipo_info.get('nse_symbol')) and str(ipo_info.get('nse_symbol')).lower()!='none' else None),
            'sector': sector,
            'listing_date': listing_date.strftime('%Y-%m-%d'),
            'evaluation_date': curr_date.strftime('%Y-%m-%d'),
            'price_source': price_source,
            'price_basis': price_basis,
            'price_basis_adjustment':basis_adjustment,
            'history_price_basis':'As-of split/bonus-comparable closes; cash dividends excluded',
            'price_archive_sha256': curr_row.get('archive_sha256'),
            'price_is_live': False,
            'price_age_calendar_days': None if as_of_date else (now_ist.date()-curr_date.date()).days,
            'price_is_stale': False if as_of_date else (now_ist.date()-curr_date.date()).days > 5,
            'benchmark_source': benchmark_source,
            'current_price_adjusted': round(curr_stock_p, 2),
            'price_waterfall': price_waterfall,
            'recent_trajectory': trajectory,
            'lockin_status': lockin_status,
            'attribution': attribution_results,
            'trendlyne_data': tl_data,
        }

if __name__ == '__main__':
    engine = AttributionEngine()
    print('Testing AttributionEngine on Tata Tech (ID: 364) as of 2024-03-15:')
    res = engine.compute_attribution(364, as_of_date='2024-03-15')
    print('Company: ' + str(res['company_name']) + ' | Sector: ' + str(res['sector']))
    print('Lock-in Status: ' + str(res['lockin_status']))
    print('30-Day Attribution: ' + str(res['attribution']['30d']))
