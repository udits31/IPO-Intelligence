import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SECTOR_MAP = {
    'Technology': '^CNXIT',
    'Financial Services': '^NSEBANK',
    'Healthcare': '^CNXPHARMA',
    'Consumer Cyclical': '^CNXAUTO',
    'Consumer Defensive': '^CNXFMCG',
    'Industrials': '^CNXINFRA',
    'Basic Materials': '^CNXMETAL',
    'Real Estate': '^CNXREALTY',
    'Energy': '^CNXENERGY',
    'Utilities': '^CNXINFRA',
    'Communication Services': '^CNXIT'
}

TICKER_OVERRIDES = {
    'ZOMATO': 'ETERNAL.NS',
    'POLICYBZR': 'POLICYBZR.NS',
    'PAYTM': 'PAYTM.NS',
    'NYKAA': 'NYKAA.NS',
    'LIC': 'LICI.NS',
    'DELHIVERY': 'DELHIVERY.NS',
    'TATATECH': 'TATATECH.NS',
    'BARBEQUE': 'UFBL.NS',
    'SAH': 'AERONEU.NS',
    'AMIORG': 'ACUTAAS.NS'
}

def get_yfinance_symbol(nse_symbol, bse_code):
    if pd.isna(nse_symbol) or str(nse_symbol).strip() in ['', 'nan', 'None']:
        return str(int(bse_code)) + '.BO'
    sym = str(nse_symbol).strip().upper()
    if sym in TICKER_OVERRIDES:
        return TICKER_OVERRIDES[sym]
    return sym + '.NS'

def ingest_all_prices():
    canonical_path = 'data/canonical_listed_ipos.csv'
    df_ipos = pd.read_csv(canonical_path)
    modern_ipos = df_ipos[df_ipos['is_primary_universe'] == True].copy()
    logger.info('Found ' + str(len(modern_ipos)) + ' primary universe IPOs (2021-2024).')
    
    all_indices = list(set(SECTOR_MAP.values())) + ['^NSEI']
    logger.info('Downloading ' + str(len(all_indices)) + ' benchmark indices...')
    bench_raw = yf.download(all_indices, start='2021-01-01', auto_adjust=True, progress=False)
    bench_data = bench_raw['Close'] if 'Close' in bench_raw else bench_raw
    bench_data = bench_data.ffill().bfill()
    bench_data.index = pd.to_datetime(bench_data.index).strftime('%Y-%m-%d')
    
    records = []
    failed_tickers = []
    
    total = len(modern_ipos)
    for idx, (_, row) in enumerate(modern_ipos.iterrows(), 1):
        ipo_id = int(row['id'])
        company_name = str(row['company_name'])
        nse_sym = row['nse_symbol']
        bse_code = row['bse_scrip_code']
        listing_date = str(row['listing_date'])
        sector = str(row['sector'])
        sector_ticker = SECTOR_MAP.get(sector, '^NSEI')
        
        yf_sym = get_yfinance_symbol(nse_sym, bse_code)
        
        try:
            ticker_obj = yf.Ticker(yf_sym)
            hist = ticker_obj.history(start=listing_date, auto_adjust=True)
            
            if len(hist) == 0 and not yf_sym.endswith('.BO') and pd.notna(bse_code):
                yf_sym = str(int(bse_code)) + '.BO'
                ticker_obj = yf.Ticker(yf_sym)
                hist = ticker_obj.history(start=listing_date, auto_adjust=True)
                
            if len(hist) == 0:
                logger.warning('[' + str(idx) + '/' + str(total) + '] No price data for ' + company_name + ' (' + yf_sym + ')')
                failed_tickers.append((ipo_id, company_name, yf_sym))
                continue
                
            hist.index = pd.to_datetime(hist.index).strftime('%Y-%m-%d')
            
            for date_str, close_val in hist['Close'].items():
                if pd.isna(close_val) or close_val <= 0:
                    continue
                    
                sec_close = None
                nifty_close = None
                
                if date_str in bench_data.index:
                    if sector_ticker in bench_data.columns:
                        val = bench_data.loc[date_str, sector_ticker]
                        if pd.notna(val):
                            sec_close = round(float(val), 4)
                    if '^NSEI' in bench_data.columns:
                        val_n = bench_data.loc[date_str, '^NSEI']
                        if pd.notna(val_n):
                            nifty_close = round(float(val_n), 4)
                            
                records.append({
                    'ipo_id': ipo_id,
                    'date': date_str,
                    'close_price_adjusted': round(float(close_val), 4),
                    'sector_index_close': sec_close,
                    'nifty50_close': nifty_close
                })
                
            if idx % 25 == 0 or idx == total:
                logger.info('[' + str(idx) + '/' + str(total) + '] Processed ' + company_name + ' -> ' + str(len(hist)) + ' trading days. Total records: ' + str(len(records)))
                
        except Exception as e:
            logger.error('Error processing ' + company_name + ' (' + yf_sym + '): ' + str(e))
            failed_tickers.append((ipo_id, company_name, yf_sym))
            
    df_out = pd.DataFrame(records)
    out_path = 'data/post_listing_prices.csv'
    df_out.to_csv(out_path, index=False)
    
    logger.info('=== Ingestion Complete ===')
    logger.info('Total price records: ' + str(len(df_out)))
    logger.info('Successfully covered: ' + str(len(modern_ipos) - len(failed_tickers)) + ' / ' + str(len(modern_ipos)) + ' IPOs')
    if failed_tickers:
        logger.warning('Failed tickers (' + str(len(failed_tickers)) + '): ' + str(failed_tickers))
        
    return df_out, failed_tickers

if __name__ == '__main__':
    ingest_all_prices()
