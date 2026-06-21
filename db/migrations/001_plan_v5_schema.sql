-- Migration 001: Plan v5 Schema Updates
-- 1. Alter ipos table
ALTER TABLE ipos ADD COLUMN IF NOT EXISTS is_primary_universe BOOLEAN DEFAULT FALSE;
ALTER TABLE ipos ADD COLUMN IF NOT EXISTS regime_era TEXT DEFAULT '2010_2020_legacy';
ALTER TABLE ipos ADD COLUMN IF NOT EXISTS anchor_lockin_50pct_date DATE;
ALTER TABLE ipos ADD COLUMN IF NOT EXISTS anchor_lockin_full_date DATE;

-- 2. Create regulatory_disclosures table
CREATE TABLE IF NOT EXISTS regulatory_disclosures (
    id BIGSERIAL PRIMARY KEY,
    ipo_id INT REFERENCES ipos(id) ON DELETE CASCADE,
    company_name TEXT NOT NULL,
    nse_symbol TEXT,
    discloser_type TEXT,
    regulation TEXT NOT NULL,
    disclosure_date DATE NOT NULL,
    transaction_type TEXT,
    shares_traded BIGINT,
    pre_holding_pct NUMERIC,
    post_holding_pct NUMERIC,
    pct_change NUMERIC,
    source_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regulatory_disclosures_ipo_id ON regulatory_disclosures(ipo_id);
CREATE INDEX IF NOT EXISTS idx_regulatory_disclosures_date ON regulatory_disclosures(disclosure_date);
CREATE INDEX IF NOT EXISTS idx_regulatory_disclosures_regulation ON regulatory_disclosures(regulation);

-- 3. Create post_listing_prices table
CREATE TABLE IF NOT EXISTS post_listing_prices (
    id BIGSERIAL PRIMARY KEY,
    ipo_id INT REFERENCES ipos(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    close_price_adjusted NUMERIC NOT NULL,
    sector_index_close NUMERIC,
    nifty50_close NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_post_listing_prices_ipo_date UNIQUE (ipo_id, date)
);

CREATE INDEX IF NOT EXISTS idx_post_listing_prices_ipo_date ON post_listing_prices(ipo_id, date);
CREATE INDEX IF NOT EXISTS idx_post_listing_prices_date ON post_listing_prices(date);
