"""
Supabase Client Module for IPO Intelligence System
===================================================
Provides database client instances with environment variable loading:
- `get_supabase()`: Returns Service Role client (Full CRUD, bypasses RLS for scrapers/pipelines)
- `get_anon_supabase()`: Returns Anon client (Read-only for frontend testing)
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Ensure .env is loaded
load_dotenv()

SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def get_supabase(use_service_role: bool = True) -> Client:
    """
    Returns an authenticated Supabase client instance.
    
    Args:
        use_service_role (bool): If True, uses SUPABASE_SERVICE_ROLE_KEY to bypass RLS 
                                 for pipeline/scraper tasks. If False, uses SUPABASE_ANON_KEY.
    """
    if not SUPABASE_URL:
        raise ValueError("SUPABASE_URL is not configured in .env file.")
        
    key = SUPABASE_SERVICE_ROLE_KEY if use_service_role else SUPABASE_ANON_KEY
    if not key:
        raise ValueError(f"Supabase key ('{'SERVICE_ROLE' if use_service_role else 'ANON'}') is missing in .env file.")
        
    return create_client(SUPABASE_URL, key)


def verify_connection() -> bool:
    """Verifies connection to Supabase by querying the ipos master table."""
    try:
        client = get_supabase(use_service_role=True)
        res = client.table("ipos").select("count", count="exact").execute()
        logger.info(f"Supabase connection successful! Current `ipos` count: {res.count}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Supabase Database Connection...")
    success = verify_connection()
    if success:
        print("[SUCCESS] Supabase Client initialized and verified successfully!")
    else:
        print("[FAILED] Supabase Connection Failed. Check database tables and credentials.")
