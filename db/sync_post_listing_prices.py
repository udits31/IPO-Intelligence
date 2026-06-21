import os
import time
import math
import pandas as pd
from db.supabase_client import get_supabase

CSV_PATH = 'data/post_listing_prices.csv'
CHUNK_SIZE = 1000

def clean_val(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v

def sync_prices():
    if not os.path.exists(CSV_PATH):
        print('Error: ' + CSV_PATH + ' does not exist.')
        return
        
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    print('Loaded ' + str(total_rows) + ' records from ' + CSV_PATH)
    
    records = df.to_dict(orient='records')
    # Clean all NaN/inf
    cleaned_records = []
    for r in records:
        cleaned_records.append({k: clean_val(v) for k, v in r.items()})
        
    supabase = get_supabase(use_service_role=True)
    
    print('Starting batch upload to Supabase post_listing_prices table in chunks of ' + str(CHUNK_SIZE) + '...')
    inserted = 0
    start_time = time.time()
    
    for i in range(0, total_rows, CHUNK_SIZE):
        chunk = cleaned_records[i : i + CHUNK_SIZE]
        try:
            supabase.table('post_listing_prices').upsert(chunk, on_conflict='ipo_id,date').execute()
            inserted += len(chunk)
            if inserted % 10000 == 0 or inserted == total_rows:
                elapsed = time.time() - start_time
                print('  Uploaded ' + str(inserted) + ' / ' + str(total_rows) + ' records in ' + str(round(elapsed, 1)) + 's...')
        except Exception as e:
            print('Error at chunk ' + str(i) + ': ' + str(e))
            return
            
    print('SUCCESS: Successfully synced all ' + str(inserted) + ' price records to Supabase!')

if __name__ == '__main__':
    sync_prices()
