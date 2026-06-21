
import pandas as pd
from db.supabase_client import get_supabase

def sync_ipos_columns():
    df = pd.read_csv('data/canonical_listed_ipos.csv')
    print(f'Loaded {len(df)} rows from canonical CSV.')
    
    supabase = get_supabase(use_service_role=True)
    success_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        ipo_id = int(row['id'])
        payload = {
            'is_primary_universe': bool(row['is_primary_universe']),
            'regime_era': str(row['regime_era']),
            'anchor_lockin_50pct_date': str(row['anchor_lockin_50pct_date']),
            'anchor_lockin_full_date': str(row['anchor_lockin_full_date'])
        }
        try:
            supabase.table('ipos').update(payload).eq('id', ipo_id).execute()
            success_count += 1
            if success_count % 100 == 0:
                print(f'  Synced {success_count}/{len(df)} rows...')
        except Exception as e:
            error_count += 1
            if error_count == 1:
                print(f'Error updating row {ipo_id}: {e}')
                print('Please execute db/migrations/001_plan_v5_schema.sql in Supabase SQL Editor first.')
                break
                
    if error_count == 0:
        print(f'SUCCESS: Successfully synced all {success_count} IPO records to Supabase!')

if __name__ == '__main__':
    sync_ipos_columns()
