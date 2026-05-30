# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Any, Optional

class SimilarityEngine:
    def __init__(self, data_path='data/canonical_listed_ipos.csv',
                 sub_path='data/subscription_observations.csv',
                 mkt_path='data/market_observations_nse_verified_2021_2024.csv',
                 drhp_path='data/raw/drhp_features_master.csv'):
        self.data_path = data_path
        self.sub_path = sub_path
        self.mkt_path = mkt_path
        self.drhp_path = drhp_path
        
        self.df_universe = None
        self.feature_matrix = None
        self.scaler = None
        self.knn_model = None
        
        self.feature_names = [
            'log_qib',
            'log_retail',
            'qib_retail_ratio',
            'india_vix',
            'nifty_10d_return',
        ]
        
        self._load_and_fit()

    def _load_and_fit(self):
        df_ipos = pd.read_csv(self.data_path)
        df_sub = pd.read_csv(self.sub_path)
        df_mkt = pd.read_csv(self.mkt_path)
        # The analog display is descriptive, not the promoted classifier.
        # Restrict it to the paper's development era and source-verified
        # offer-date market context; later outcomes must not join the pool.
        modern = df_ipos[df_ipos['is_primary_universe'].eq(True) &
                         pd.to_datetime(df_ipos['listing_date'], errors='coerce').dt.year.between(2021, 2024)].copy()
        
        merged = modern.merge(df_sub[['ipo_id', 'qib_subscription', 'hni_subscription', 'retail_subscription']], 
                              left_on='id', right_on='ipo_id', how='left')
        merged = merged.merge(df_mkt[['ipo_id', 'nifty_10d_return', 'india_vix']],
                              on='ipo_id', how='inner', validate='one_to_one')
        required = ['qib_subscription', 'retail_subscription', 'nifty_10d_return',
                    'india_vix', 'listing_return_pct']
        merged[required] = merged[required].apply(pd.to_numeric, errors='coerce')
        merged = merged.dropna(subset=required)
        merged = merged[merged.qib_subscription.ge(0) & merged.retail_subscription.ge(0)]
        if len(merged) < 15:
            raise ValueError('Too few source-complete historical IPO analogs')
        merged['log_qib'] = np.log1p(merged['qib_subscription'])
        merged['log_retail'] = np.log1p(merged['retail_subscription'])
        merged['qib_retail_ratio'] = (merged['qib_subscription'] + 0.1) / (merged['retail_subscription'] + 0.1)
        
        self.df_universe = merged.reset_index(drop=True)
        
        X = self.df_universe[self.feature_names].values
        self.scaler = StandardScaler()
        self.feature_matrix = self.scaler.fit_transform(X)
        
        self.knn_model = NearestNeighbors(n_neighbors=15, metric='euclidean')
        self.knn_model.fit(self.feature_matrix)

    def evaluate_ipo(self, qib_sub, retail_sub, gmp_pct,
                     vix=None, nifty_10d=None, sector_10d=0.0,
                     promoter_ofs=0.2, k_neighbors=15):
        if vix is None or nifty_10d is None or not np.isfinite([vix, nifty_10d]).all():
            raise ValueError('Observed VIX and Nifty return are required for market-context analogs')
        log_qib = np.log1p(max(qib_sub, 0.01))
        log_retail = np.log1p(max(retail_sub, 0.01))
        qib_ret_ratio = (qib_sub + 0.1) / (retail_sub + 0.1)
        
        raw_vec = np.array([[log_qib, log_retail, qib_ret_ratio, vix, nifty_10d]])
        scaled_vec = self.scaler.transform(raw_vec)
        
        distances, indices = self.knn_model.kneighbors(scaled_vec, n_neighbors=k_neighbors)
        
        analogs = []
        returns = []
        for dist, idx in zip(distances[0], indices[0]):
            row = self.df_universe.iloc[idx]
            ret = float(row['listing_return_pct'])
            returns.append(ret)
            analogs.append({
                'id': int(row['id']),
                'company_name': str(row['company_name']),
                'listing_date': str(row['listing_date']),
                'qib_subscription': float(row['qib_subscription']),
                'retail_subscription': float(row['retail_subscription']),
                'qib_retail_ratio': round(float(row['qib_retail_ratio']), 2),
                'listing_return_pct': round(ret, 2),
                'distance': round(float(dist), 3)
            })
            
        returns = np.array(returns)
        positive_count = int(np.sum(returns > 0))
        win_rate = (positive_count / len(returns)) * 100
        median_return = float(np.median(returns))
        mean_return = float(np.mean(returns))
        worst_drawdown = float(np.min(returns))
        
        if win_rate >= 80 and median_return >= 15.0 and qib_ret_ratio >= 1.5:
            verdict = 'APPLY'
            confidence = 'High'
            risk_level = 'Low'
            sizing_tier = 'Aggressive (Full bHNI Allocation)'
        elif win_rate >= 65 and median_return >= 5.0:
            verdict = 'APPLY'
            confidence = 'Medium'
            risk_level = 'Moderate'
            sizing_tier = 'Moderate bHNI Allocation'
        elif win_rate >= 45 and median_return >= 0.0:
            verdict = 'NEUTRAL / CAUTION'
            confidence = 'Low'
            risk_level = 'Elevated'
            sizing_tier = 'Minimum Application Only'
        else:
            verdict = 'SKIP'
            confidence = 'High' if win_rate < 30 else 'Medium'
            risk_level = 'High Risk'
            sizing_tier = 'Zero Allocation (Avoid)'
            
        informed_text = 'informed-money conviction leading the issue' if qib_ret_ratio >= 1.5 else 'retail-heavy speculative momentum'
        vix_text = 'stable' if vix < 16 else 'elevated volatility'
        
        explanation = (
            verdict + ' (Confidence: ' + confidence + ') - QIB subscription at ' + str(round(qib_sub, 1)) + 'x with a ' +
            'QIB-to-Retail ratio of ' + str(round(qib_ret_ratio, 2)) + ' suggests ' + informed_text + '. ' +
            'GMP is not an analog-distance feature. Market regime: ' + vix_text + ' ' +
            '(India VIX ' + str(round(vix, 1)) + ', Nifty 10d return ' + str(round(nifty_10d, 1)) + '%). ' +
            'Based on ' + str(k_neighbors) + ' historically similar 2021-2024 IPOs, the median listing return was ' +
            str(round(median_return, 1)) + '%, with ' + str(positive_count) + ' of ' + str(k_neighbors) +
            ' listing positive (' + str(round(win_rate, 0)) + '% win rate, worst listing return ' + str(round(worst_drawdown, 1)) + '%).'
        )
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'risk_level': risk_level,
            'bHNI_sizing_tier': sizing_tier,
            'win_rate_pct': round(win_rate, 1),
            'median_return_pct': round(median_return, 2),
            'mean_return_pct': round(mean_return, 2),
            'worst_drawdown_pct': round(worst_drawdown, 2),
            'positive_analogs_count': positive_count,
            'total_analogs_count': k_neighbors,
            'explanation': explanation,
            'historical_analogs': analogs,
            'feature_source': 'NSE-verified 2021–2024 issue dates and observed market closes; no missing-value fills',
        }

if __name__ == '__main__':
    engine = SimilarityEngine()
    print('Testing SimilarityEngine on strong candidate (QIB 42x, Retail 11x, GMP 18%, VIX 14.2):')
    res = engine.evaluate_ipo(qib_sub=42.0, retail_sub=11.0, gmp_pct=18.0, vix=14.2, nifty_10d=0.5)
    print('\nVerdict:', res['verdict'], '(' + res['confidence'] + ')')
    print('Explanation:\n', res['explanation'])
    print('\nTop 5 Analogs:')
    for a in res['historical_analogs'][:5]:
        print('  * ' + a['company_name'] + ' (' + a['listing_date'] + '): QIB ' + str(a['qib_subscription']) + 'x, Retail ' + str(a['retail_subscription']) + 'x -> Return: ' + str(a['listing_return_pct']) + '%')
