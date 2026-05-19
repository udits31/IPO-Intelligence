import os
import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from core.predictor import HybridIPOPredictor
from core.monitor_engine import HoldSellMonitor
from core.unified_resolver import UnifiedIPOResolver
from core.benchmark_engine import build_walk_forward_benchmark
from scrapers.live_ipo_scraper import (
    CACHE_FILE_PATH,
    fetch_live_macro_regime,
    scrape_chittorgarh_url,
    sync_chittorgarh_calendar_delta,
)
from services.news_service import get_news_snapshot, news_refresh_status
from services.module2_decision_ledger import (
    record_module2_decision, get_module2_event_history, module2_collector_thread,
)
from services.module2_intraday import collector as intraday_collector
from services.trendlyne_service import trendlyne_service
from services.ai_copilot_service import generate_ai_investment_memo, ask_ai_copilot
from services.snapshot_service import (
    get_snapshots_summary,
    get_subscription_capture_rows,
    get_automatic_decision_rows,
    get_intraday_forecast_scorecard,
    get_prelisting_replay_scorecard,
    get_prelisting_outcome_scorecard,
    record_automatic_decision,
    application_forecast_window,
    get_latest_application_forecast,
    get_last_complete_bid_capture,
    harvester_thread,
)

logger = logging.getLogger(__name__)

# Live data is refreshed at process startup. The TTL controls whether the
# potentially expensive detail-page rescrape is needed; the calendar itself is
# reconciled on every startup. Override with LIVE_DATA_TTL_MINUTES in production.
LIVE_DATA_TTL_MINUTES = max(1, int(os.getenv('LIVE_DATA_TTL_MINUTES', '30')))
live_data_status: Dict[str, Any] = {
    'status': 'not_started',
    'last_refresh_at': None,
    'cache_age_seconds': None,
    'items_synced': 0,
    'detail_rescrape': False,
    'error': None,
}


def _number_or_default(value: Any, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _optional_number(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None and not pd.isna(value) else None
    except (TypeError, ValueError):
        return None


def _refresh_live_data_on_startup() -> None:
    """Refresh live inputs before serving requests, without making startup brittle."""
    started = time.monotonic()
    now = pd.Timestamp.now().isoformat()
    cache_age = None
    force_refresh = True
    try:
        if os.path.exists(CACHE_FILE_PATH):
            cache_age = max(0.0, time.time() - os.path.getmtime(CACHE_FILE_PATH))
            force_refresh = cache_age >= LIVE_DATA_TTL_MINUTES * 60

        # This performs the live calendar check and updates the persistent cache.
        # New IPOs are hydrated automatically; stale caches also refresh details.
        items = sync_chittorgarh_calendar_delta(force_refresh=force_refresh)
        macro = fetch_live_macro_regime()
        live_data_status.update({
            'status': 'fresh',
            'last_refresh_at': now,
            'cache_age_seconds': round(cache_age, 1) if cache_age is not None else None,
            'items_synced': len(items),
            'detail_rescrape': force_refresh,
            'macro_timestamp': macro.get('timestamp'),
            'duration_seconds': round(time.monotonic() - started, 2),
            'error': None,
        })
        logger.info(
            'Live startup refresh complete: %s calendar items, detail_rescrape=%s, duration=%ss',
            len(items), force_refresh, live_data_status['duration_seconds']
        )
    except Exception as exc:
        # Preserve the last known local cache and allow the API to serve.
        live_data_status.update({
            'status': 'stale_fallback',
            'last_refresh_at': now,
            'cache_age_seconds': round(cache_age, 1) if cache_age is not None else None,
            'detail_rescrape': force_refresh,
            'duration_seconds': round(time.monotonic() - started, 2),
            'error': str(exc),
        })
        logger.warning('Live startup refresh failed; serving cached/local data: %s', exc)


def _warm_active_news() -> None:
    """Warm stale active-IPO news without delaying startup or bid polling."""
    from scrapers.live_ipo_scraper import load_scraped_cache
    names = list(dict.fromkeys(row['company_name'] for row in load_scraped_cache().values()
                               if row.get('category') in ('OPEN', 'UPCOMING')))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(get_news_snapshot, names))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from db.ledger import cloud_enabled
    cloud = cloud_enabled()
    if cloud:
        from services.cloud_assets import restore
        await run_in_threadpool(restore,force=True,scope='calendar')
        # API functions serve requests only. The separately leased daemon owns
        # scheduled collection, so scale-to-zero does not stop data gathering.
        if os.getenv('OPENBELL_COLLECTORS_ENABLED') != '1':
            yield
            return
    await run_in_threadpool(_refresh_live_data_on_startup)
    try:
        from services.listing_outcome_service import record_verified_listing_outcomes
        await run_in_threadpool(record_verified_listing_outcomes)
    except Exception:
        logger.exception('Listing outcome check failed; live API will continue')
    try:
        from services.sqlite_supabase_sync import sync_and_record_status
        result = await run_in_threadpool(sync_and_record_status)
        if result.get('status') != 'ok':
            logger.warning('Startup Supabase reconciliation failed; local data remains available')
    except Exception:
        logger.exception('Startup Supabase reconciliation failed; local data remains available')
    if not harvester_thread.is_alive():
        harvester_thread.start()
    if not module2_collector_thread.is_alive():
        module2_collector_thread.start()
    if not intraday_collector.is_alive():
        intraday_collector.start()
    threading.Thread(target=_warm_active_news, name='startup-news-refresh', daemon=True).start()
    from services.prospective_module1 import collector as frozen_collector
    if not frozen_collector.is_alive():
        frozen_collector.start()
    from services.progressive_engine_service import worker as regime_worker
    if not regime_worker.is_alive():regime_worker.start()
    from services.data_maintenance import worker as data_worker
    if not data_worker.is_alive():data_worker.start()
    yield
    harvester_thread.stop()
    module2_collector_thread.stop()
    intraday_collector.stop()
    frozen_collector.stop()
    regime_worker.stop()
    data_worker.stop()


app = FastAPI(
    title='IPO Intelligence Engine API',
    description='Two-Module Decision System with AI Copilot & Investment Memo',
    version='5.1.0',
    lifespan=lifespan,
)


@app.middleware('http')
async def refresh_cloud_sources(request,call_next):
    from db.ledger import cloud_enabled
    if cloud_enabled():
        from services.cloud_assets import restore
        try:
            path=request.url.path
            scope='monitor' if path.startswith('/api/v1/module2') else 'research' if path.startswith(('/api/v1/module1','/api/v1/performance','/api/v1/ai')) else 'calendar'
            await run_in_threadpool(restore,scope=scope)
        except Exception:
            logger.exception('Cloud source refresh failed; retaining validated cached evidence')
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Global Engine Instances
predictor = HybridIPOPredictor()
monitor = HoldSellMonitor()
resolver = UnifiedIPOResolver()
df_canonical = pd.read_csv('data/canonical_listed_ipos.csv')

# --- Pydantic Schemas ---
class ManualEvaluateRequest(BaseModel):
    company_name: str = Field(default='Target IPO', description='Company name')
    sector: str = Field(default='Technology', description='Industry sector')
    qib_subscription: float = Field(default=10.0, ge=0.0, description='QIB subscription multiple (x)')
    retail_subscription: float = Field(default=2.0, ge=0.0, description='Retail subscription multiple (x)')
    hni_subscription: float = Field(default=5.0, ge=0.0, description='HNI/NII subscription multiple (x)')
    issue_size_cr: float = Field(default=1000.0, ge=10.0, description='Total issue size in Rs Crores')
    issue_price: float = Field(default=500.0, ge=1.0, description='Issue price per share (Rs)')
    gmp_pct: Optional[float] = Field(default=None, description='Observed Grey Market Premium (%); omit when unavailable')
    promoter_ofs_ratio: float = Field(default=0.25, ge=0.0, le=1.0, description='Offer for sale ratio in DRHP')
    risk_factor_count: float = Field(default=50.0, ge=10.0, le=150.0, description='Risk factor count in DRHP')
    objects_capex_ratio: float = Field(default=0.30, ge=0.0, le=1.0, description='Capex objects ratio in DRHP')
    price_band_spread_pct: float = Field(default=5.0, ge=0.0, le=20.0, description='Price band spread percentage')
    fetch_news: bool = Field(default=True, description='Whether to fetch live news headlines')

class ScrapeAndEvaluateRequest(BaseModel):
    url_or_name: str = Field(description='Chittorgarh IPO URL or Company Name')
    fetch_news: bool = Field(default=True, description='Whether to fetch live news headlines')
    save_forecast: bool = Field(default=True, description='Save an explicit Run analysis action; false for background page refresh')

class AICopilotRequest(BaseModel):
    company_name: str = Field(description='Company Name')
    question: str = Field(description='User question for AI copilot')
    context_data: Dict[str, Any] = Field(default={}, description='Current evaluation payload')

class AIMemoRequest(BaseModel):
    company_name: str
    sector: str = 'General'
    qib_subscription: float = 10.0
    retail_subscription: float = 2.0
    hni_subscription: float = 5.0
    issue_size_cr: float = 1000.0
    issue_price: float = 500.0
    gmp_pct: Optional[float] = None
    win_prob: float = 80.0
    predicted_return_pct: float = 15.0
    sizing_tier: str = 'Moderate Allocation'
    verdict: str = 'APPLY'
    conviction: str = 'High'
    news_items: List[Dict[str, str]] = []
    analogs: List[Dict[str, Any]] = []

# --- API Endpoints ---
@app.get('/api/v1/health')
def health_check():
    from services.data_maintenance import worker as data_worker
    macro = fetch_live_macro_regime()
    from services.prospective_module1 import bundles,collector as frozen_collector,status as frozen_status
    from services.locked_model_configuration import VERSION
    protocol,stack,_=bundles()
    from db.ledger import cloud_enabled,remote_collector_active
    cloud=cloud_enabled()
    external_active=remote_collector_active() if cloud else False
    return {
        'status': 'healthy',
        'version': '5.1.0',
        'storage':{'backend':'supabase_api' if cloud else 'sqlite_with_supabase_mirror','sqlite_used':not cloud},
        'collector_execution':{'mode':'external_daemon' if cloud else 'local_process','external_lease_active':external_active},
        'live_data_refresh': live_data_status,
        'news_refresh': news_refresh_status(),
        'data_maintenance': {k:v for k,v in data_worker.status().items() if k != 'jobs'},
        'models_loaded': ['Frozen five-expert stack (active)',predictor.model_metadata['model_name']+' (legacy context)', 'SimilarityEngine (kNN)', 'AttributionEngine', 'HoldSellMonitor', 'AICopilotService'],
        'module1_model': {'model_name':'Frozen five-expert log-odds stack','configuration':VERSION,
            'artifact_sha256':protocol['model']['sha256'],'training_end_year':stack['training_end_year'],
            'decision_threshold':.5,'probability_calibration':'none'},
        'legacy_context_model': predictor.model_metadata,
        'collectors': {
            'module1_bid_collector_running': harvester_thread.is_alive(),
            'module1_last_poll_slot': harvester_thread._last_slot,
            'frozen_forecast_collector_running':frozen_collector.is_alive(),
            'frozen_forecast_last_cycle':frozen_status,
            'module2_reaction_collector_running': module2_collector_thread.is_alive(),
            'module2_intraday_collector_running': intraday_collector.is_alive(),
            'module2_intraday_last_cycle': intraday_collector.last_cycle,
            'module2_intraday_cycle_in_progress': intraday_collector.running_cycle,
            'module2_last_news_check_ist': module2_collector_thread._last_news_attempt_at.isoformat()
            if module2_collector_thread._last_news_attempt_at else None,
            'module2_last_close_check_ist': module2_collector_thread._last_attempt_at.isoformat()
            if module2_collector_thread._last_attempt_at else None,
        },
        'live_macro_regime': macro
    }

@app.get('/api/v1/macro/live')
def get_live_macro(response: Response):
    response.headers['Cache-Control']='no-store, max-age=0'
    return fetch_live_macro_regime()

@app.get('/api/v1/module1/benchmark')
def get_module1_benchmark(limit: int = Query(default=50, ge=1, le=250)):
    result = build_walk_forward_benchmark()
    return {**result, 'records': result['records'][:limit]}

@app.get('/api/v1/performance/evidence')
def get_performance_evidence():
    from services.performance_service import performance_evidence
    return performance_evidence()


@app.get('/api/v1/performance/model-comparison')
def get_model_comparison():
    from pathlib import Path
    import hashlib
    root=Path(__file__).resolve().parents[1]
    path=root/'output/model_comparison_20261028/report.json'
    if not path.exists():return {'status':'unavailable','message':'The matched model comparison has not been generated.'}
    report=json.loads(path.read_text(encoding='utf8'))
    for relative,expected in report['input_hashes'].items():
        source=root/relative
        if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            return {'status':'outdated','message':'Comparison inputs changed; rerun the matched benchmark before quoting it.'}
    period_path=root/'output/module1_period_stability_20261028/report.json'
    period=None
    if period_path.exists():
        candidate=json.loads(period_path.read_text(encoding='utf8'))
        if all((root/relative).exists() and hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected for relative,expected in candidate['input_hashes'].items()):
            period=candidate
    stage_path=root/'output/module1_stage_matched_20261028/report.json'
    stage=None
    if stage_path.exists():
        candidate=json.loads(stage_path.read_text(encoding='utf8'))
        if all((root/relative).exists() and hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected for relative,expected in candidate['input_hashes'].items()):
            stage=candidate
    from services.model_lineage import active_architecture
    return {'status':'ready',**report,'period_stability':period,'stage_matched':stage,
            'active_architecture':active_architecture()}


@app.get('/api/v1/performance/dashboard')
def get_performance_dashboard():
    from services.performance_dashboard import build_dashboard
    try:return build_dashboard()
    except ValueError as exc:raise HTTPException(status_code=409,detail=str(exc))


@app.get('/api/v1/module1/two-track-evidence')
def get_two_track_evidence():
    from services.two_track_evidence import OUT,LOCK
    import hashlib
    report=json.loads(OUT.read_text(encoding='utf8'))
    if report['locked_evidence_sha256']!=hashlib.sha256(LOCK.read_bytes()).hexdigest():
        raise HTTPException(status_code=409,detail='Two-track evidence needs regeneration for the locked snapshot.')
    return report


@app.get('/api/v1/module2/sequential-evaluation')
def get_sequential_evaluation():
    from ml.evaluate_module2_sequential_path import OUT
    if not OUT.exists():return {'status':'building'}
    report=json.loads(OUT.read_text(encoding='utf8'))
    return {'status':'verified_snapshot',**report}


@app.get('/api/v1/module2/next-session-evaluation')
def get_next_session_evaluation():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'output/module2_next_session_20261029/report.json'
    if not path.exists():return {'status':'building'}
    report=json.loads(path.read_text(encoding='utf8'))
    return {'status':'research_snapshot',**{k:report[k] for k in
        ('protocol','splits','selected','comparisons','live_promoted','reason')}}


@app.get('/api/v1/module2/next-session-improvement')
def get_next_session_improvement():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'output/module2_next_session_improvement_20261029/report.json'
    if not path.exists():return {'status':'building'}
    report=json.loads(path.read_text(encoding='utf8'))
    return {'status':'research_snapshot',**{k:report[k] for k in
        ('protocol','splits','selected','comparisons','coverage','baselines','majority_direction_accuracy_pct',
         'zero_return_mae_pct','promotion_gate_passed','live_promoted','verdict')}}


@app.get('/api/v1/module2/holding-policy-evaluation')
def get_holding_policy_evaluation():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'output/module2_holding_rules_entry_20261029/report.json'
    if not path.exists():return {'status':'building'}
    report=json.loads(path.read_text(encoding='utf8'))
    return {'status':'research_snapshot',**{k:report[k] for k in
        ('selected_policy','comparison','paired_comparisons','costs','by_year','n_issuers','n_decisions',
         'old_exits_held_by_tuned','promotion_gate_passed','live_promoted','interpretation','execution')},
        'earlier_selected_folds':report['earlier_selection']['selection']['folds'],
        'qualified_on_both_earlier_folds':report['earlier_selection']['qualified_in_both_earlier_folds']}


@app.get('/api/v1/module2/multihorizon-evaluation')
def get_multihorizon_evaluation():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'output/module2_multihorizon_20261029/report.json'
    if not path.exists():return {'status':'building'}
    report=json.loads(path.read_text(encoding='utf8'))
    return {'status':'research_snapshot',**{k:report[k] for k in
        ('selected','earlier_selection_qualified','comparison','paired_comparisons','costs','by_year',
         'test_issuers','test_decisions','training_issuers','training_examples','protocol','interpretation',
         'promotion_gate_passed','live_promoted')}}


@app.get('/api/v1/module2/shadow-scorecard')
def get_module2_shadow_scorecard():
    from services.module2_shadow_forecast import scorecard
    return scorecard()


@app.get('/api/v1/telemetry/subscription-training')
def get_subscription_training(date_ist: str, asof: Optional[str]=None):
    from services.subscription_training import training_captures
    try:return training_captures(date_ist,asof=asof)
    except ValueError as exc:raise HTTPException(status_code=400,detail=str(exc))


@app.get('/api/v1/research/enrichment')
def get_enrichment_audit():
    from services.enrichment_runtime import audit_report
    return audit_report()


@app.get('/api/v1/research/progressive')
def get_progressive_research():
    from services.saved_preclose_service import ROOT
    from services.progressive_engine_service import STATE
    paths={'module1':ROOT/'output/module1_progressive_ablation_20261026.json','module2':ROOT/'output/module2_prove_it_benchmark_20261026.json','regime':STATE}
    return {name:json.loads(path.read_text(encoding='utf8')) if path.exists() else {'status':'building'} for name,path in paths.items()}


class CapitalPlanScenario(BaseModel):
    score: float = Field(ge=0,le=1)
    regime_adjusted_gmp_pct: float
    qib: float = Field(ge=0)
    hni: float = Field(ge=0)
    retail: float = Field(ge=0)
    cap_price: float = Field(gt=0)
    lot_size: int = Field(gt=0)
    issue_size_cr: Optional[float] = Field(default=None,ge=0)
    snii: Optional[float] = Field(default=None,ge=0)
    bnii: Optional[float] = Field(default=None,ge=0)
    financing_apr_pct: float = Field(default=0,ge=0)


@app.post('/api/v1/module1/capital-plan')
def calculate_capital_plan(payload: CapitalPlanScenario):
    from services.retail_hni_economics import capital_plan
    try:return capital_plan(**payload.model_dump())
    except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.get('/api/v1/research/module2/enriched-preview/{ipo_id}')
def get_enriched_monitor_preview(ipo_id: int, decision_timestamp: Optional[str]=None):
    from services.enrichment_runtime import preview
    try:return preview(monitor,ipo_id,decision_timestamp)
    except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc


class CapitalScenario(BaseModel):
    category: str
    cap_price: float = Field(gt=0)
    lot_size: int = Field(gt=0)
    expected_return_pct: Optional[float] = None
    subscription_mult: Optional[float] = Field(default=None,ge=0)
    application_lots: Optional[int] = Field(default=None,gt=0)
    valid_applications: Optional[int] = Field(default=None,gt=0)
    category_shares: Optional[int] = Field(default=None,ge=0)
    minimum_allotment_shares: Optional[int] = Field(default=None,gt=0)
    allotment_basis_verified: bool = False
    financing_apr_pct: float = Field(default=0,ge=0)
    blocked_days: int = Field(default=0,ge=0)
    funded_fraction: float = Field(default=0,ge=0,le=1)
    sale_cost_pct: float = Field(default=0,ge=0)
    fixed_cost_rupees: float = Field(default=0,ge=0)
    probability_positive: Optional[float] = Field(default=None,ge=0,le=1)
    precision_hurdle: Optional[float] = Field(default=None,ge=0,le=1)
    allow_multiple_approximation: bool = False


@app.post('/api/v1/module1/capital-scenario')
def calculate_capital_scenario(payload: CapitalScenario):
    from services.retail_hni_economics import assess
    try:return assess(**payload.model_dump())
    except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc


@app.get('/api/v1/module1/prediction-results')
def get_prediction_results():
    from services.prediction_results_service import prediction_results
    return prediction_results()

@app.get('/api/v1/module1/live-upcoming')
def get_live_upcoming_ipos():
    macro = fetch_live_macro_regime()
    items = [item for item in sync_chittorgarh_calendar_delta()
             if item.get('category') in ('OPEN', 'UPCOMING')]
    return {
        'count': len(items),
        'live_macro': macro,
        'upcoming_ipos': items
    }

@app.post('/api/v1/module1/evaluate')
def evaluate_ipo_manual(req: ManualEvaluateRequest):
    macro = fetch_live_macro_regime()
    manual_complete = {'qib_subscription', 'retail_subscription', 'hni_subscription'}.issubset(req.model_fields_set)
    res = predictor.predict(
        company_name=req.company_name,
        qib_sub=req.qib_subscription,
        retail_sub=req.retail_subscription,
        hni_sub=req.hni_subscription,
        issue_size_cr=req.issue_size_cr,
        gmp_pct=req.gmp_pct,
        vix=_number_or_default(macro.get('india_vix'), 15.0),
        nifty_10d=_optional_number(macro.get('nifty_10d_return')),
        promoter_ofs_ratio=req.promoter_ofs_ratio,
        risk_factor_count=req.risk_factor_count,
        objects_capex_ratio=req.objects_capex_ratio,
        price_band_spread_pct=req.price_band_spread_pct,
        category='CLOSED',
        bidding_ended=True,
        subscription_verified=manual_complete,
        fetch_news=req.fetch_news
    )
    return {
        'success': True,
        'company_name': req.company_name,
        'sector': req.sector,
        'input_multiples': {
            'qib': req.qib_subscription if manual_complete else None,
            'retail': req.retail_subscription if manual_complete else None,
            'hni': req.hni_subscription if manual_complete else None,
            'gmp_pct': req.gmp_pct
        },
        'live_market_conditions': macro,
        'evaluation': res
    }

@app.post('/api/v1/module1/scrape-and-evaluate')
def scrape_and_evaluate_ipo(req: ScrapeAndEvaluateRequest):
    resolved = resolver.resolve(req.url_or_name)
    if resolved.get('success'):
        data = resolved['data']
        source = resolved['source']
    else:
        scrape_res = scrape_chittorgarh_url(req.url_or_name)
        if not scrape_res.get('success'):
            raise HTTPException(status_code=404, detail='No verified IPO record was found for this target.')
        data = scrape_res['data']
        source = scrape_res['source']
    
    from services.preclose_context import restore_preclose_gmp, attach_offer_context, saved_book_total
    data = attach_offer_context(restore_preclose_gmp(data))
    macro = fetch_live_macro_regime()
    # Canonical records retain the market backdrop from their own listing date for
    # research, but a decision made today must always be scored against today's
    # verified (or explicitly last-known) market regime.
    historical_market_context = {
        'india_vix': data.get('india_vix'),
        'nifty_10d_return': data.get('nifty_10d_return'),
        'sector_10d_return': data.get('sector_10d_return'),
    }
    macro_current = not bool(macro.get('is_fallback'))
    current_vix = _optional_number(macro.get('india_vix')) if macro_current else None
    current_nifty_10d = _optional_number(macro.get('nifty_10d_return')) if macro_current else None
    observed_gmp = _optional_number(data.get('gmp_pct'))
    display_qib = _optional_number(data.get('qib_subscription'))
    display_retail = _optional_number(data.get('retail_subscription'))
    display_hni = _optional_number(data.get('hni_subscription'))
    subscription_observed = bool(data.get('subscription_observed')) and all(
        value is not None for value in (display_qib, display_retail, display_hni)
    )
    published_book_verified = subscription_observed
    cat = str(data.get('category', 'OPEN')).strip().upper()
    subscription_time = pd.to_datetime(data.get('subscription_observed_at'), errors='coerce', utc=True)
    trusted_subscription_source = data.get('subscription_source') in (
        'chittorgarh_subscription_page_v1', 'nse_ipo_category_v1'
    )
    # Retain a dated, source-attributed reading for display even after the
    # freshness window. It must not silently become a current model input.
    display_has_provenance = trusted_subscription_source and pd.notna(subscription_time) and any(
        value is not None for value in (display_qib, display_retail, display_hni)
    )
    data['subscription_is_stale'] = bool(
        cat == 'OPEN' and (
            pd.isna(subscription_time) or
            (pd.Timestamp.now(tz='UTC') - subscription_time).total_seconds() > 1200
        )
    )
    if data['subscription_is_stale']:
        subscription_observed = False
    # Partial readings are display-only and must come from a dated subscription
    # table, never a legacy allocation table or a missing/undated cache field.
    if not display_has_provenance and not subscription_observed:
        display_qib = display_retail = display_hni = None
    is_listed_deal = bool(data.get('is_listed')) or cat == 'LISTED'
    if is_listed_deal and observed_gmp is not None:
        quote_time = pd.to_datetime(data.get('gmp_observed_at'), errors='coerce')
        listing_time = pd.to_datetime(data.get('listing_date'), errors='coerce')
        # A quote first fetched on/after listing cannot be presented as a
        # verified pre-listing signal, even if the provider labels it final.
        if pd.isna(quote_time) or pd.isna(listing_time) or quote_time.date() >= listing_time.date():
            observed_gmp = None
            data['gmp_pct'] = None
            data['gmp_rs'] = None
            data['gmp_source'] = 'post_listing_quote_excluded'
    bidding_ended = bool(data.get('bidding_ended')) or cat in ['CLOSED', 'LISTED'] or is_listed_deal
    if bidding_ended and trusted_subscription_source and not data.get('subscription_is_final'):
        subscription_observed = False
        if not display_has_provenance:
            display_qib = display_retail = display_hni = None
    data['subscription_display_status'] = (
        'last_published' if display_has_provenance and (
            data['subscription_is_stale'] or (bidding_ended and not data.get('subscription_is_final'))
        ) else 'current' if subscription_observed else
        'partial' if any(value is not None for value in (display_qib, display_retail, display_hni))
        else 'unavailable'
    )
    observed_qib = display_qib if subscription_observed else None
    observed_retail = display_retail if subscription_observed else None
    observed_hni = display_hni if subscription_observed else None

    pred_res = predictor.predict(
        company_name=data['company_name'],
        sector=data.get('sector', 'Technology'),
        qib_sub=observed_qib if observed_qib is not None else 0.0,
        retail_sub=observed_retail if observed_retail is not None else 0.0,
        hni_sub=observed_hni if observed_hni is not None else 0.0,
        issue_size_cr=_number_or_default(data.get('issue_size_cr'), 500.0),
        gmp_pct=observed_gmp,
        vix=current_vix,
        nifty_10d=current_nifty_10d,
        promoter_ofs_ratio=_optional_number(data.get('promoter_ofs_ratio')),
        risk_factor_count=_number_or_default(data.get('risk_factor_count'), 50.0),
        objects_capex_ratio=_number_or_default(data.get('objects_capex_ratio'), 0.30),
        category=cat,
        is_listed=is_listed_deal,
        bidding_ended=bidding_ended,
        subscription_verified=subscription_observed,
        fetch_news=req.fetch_news
    )
    # Run analysis produces and saves a call throughout the offer period.
    # Between sessions or after an empty refresh, use the last published book
    # with its original timestamp. The final day's 17:00 cutoff is absolute.
    decision_now = datetime.now(ZoneInfo('Asia/Kolkata'))
    window = application_forecast_window(data, decision_now)
    forecast_save_error = False
    if window['is_open'] and req.save_forecast:
        try:
            from scrapers.live_ipo_scraper import load_scraped_cache
            current_cache = load_scraped_cache()
            matches = [(url, item) for url, item in current_cache.items()
                       if str(item.get('company_name', '')).casefold() == str(data['company_name']).casefold()]
            capture = None
            complete_book = published_book_verified and display_has_provenance
            if not complete_book:
                capture = get_last_complete_bid_capture(data['company_name'], now_ist=decision_now)
            ipo_url = data.get('url') or (matches[0][0] if len(matches) == 1 else None)
            if capture and not ipo_url:
                ipo_url = capture['ipo_url']
            if ipo_url and (complete_book or capture):
                record_automatic_decision(
                    item=({**data, 'subscription_source_displayed_at': capture.get('source_published_at_ist'),
                           'subscription_publisher_time_unverified': not bool(capture.get('source_published_at_ist'))}
                          if capture else data), ipo_url=ipo_url,
                    slot=decision_now.strftime('%Y-%m-%d_%H%M%S%f') + '_manual',
                    stage='MANUAL_PRE_CUTOFF', now_ist=decision_now,
                    values={'qib_subscription': capture['qib_x'] if capture else display_qib,
                            'retail_subscription': capture['retail_x'] if capture else display_retail,
                            'hni_subscription': capture['nii_x'] if capture else display_hni},
                    source_family=capture['source_family'] if capture else data.get('subscription_source_family')
                    or ('nse_exchange' if data.get('subscription_source') == 'nse_ipo_category_v1'
                        else 'chittorgarh_investorgain'),
                    source_status='verified',
                    source_asof=(capture.get('source_asof_ist') or capture['fetched_at_ist']) if capture else data.get('subscription_source_displayed_at')
                    or data.get('subscription_observed_at'),
                    source_sha256=capture['raw_sha256'] if capture else data.get('subscription_raw_sha256'),
                    post_close=False,
                )
        except Exception:
            forecast_save_error = True
            logger.exception('Could not save manual intraday forecast for %s', data.get('company_name'))
    frozen_forecast = None
    try:
        from services.prospective_module1 import analysis_card
        frozen_forecast = analysis_card(data,data.get('url') or data.get('chittorgarh_url') or '')
    except Exception:
        logger.exception('Frozen model analysis failed for %s', data.get('company_name'))
    application_forecast = get_latest_application_forecast(data['company_name'])
    if bidding_ended:
        saved_inputs = (frozen_forecast or {}).get('inputs') or (application_forecast or {}).get('inputs') or {}
        source_sha = saved_inputs.get('source_sha256') or (((saved_inputs.get('evidence') or {}).get('sources') or {}).get('bids') or {}).get('raw_sha256')
        total = saved_book_total(source_sha)
        if total:
            data['total_subscription'] = total['multiple']
            data['total_subscription_provenance'] = total
            data['total_subscription_status'] = 'saved_pre_close'
    if data.get('category') in ('OPEN','UPCOMING') and frozen_forecast:
        inputs=frozen_forecast.get('inputs') or {}
        sources=(inputs.get('evidence') or {}).get('sources') or {}
        offer=sources.get('offer') or {}
        if offer and not data.get('offer_facts'):
            data['offer_facts']=offer
            data['promoter_ofs_ratio']=(offer.get('offer') or {}).get('ofs_ratio')
            data['retail_quota_pct']=((offer.get('reservation') or {}).get('Retail') or {}).get('pct')
            data['fresh_issue_cr']=(offer.get('offer') or {}).get('fresh_issue_cr')
            data['ofs_issue_cr']=(offer.get('offer') or {}).get('ofs_issue_cr')
        total=sources.get('total_subscription')
        data['total_subscription']=(inputs.get('features') or {}).get('context_total_mult')
        data['total_subscription_provenance']=total
        data['total_subscription_status']='not_started' if data.get('category')=='UPCOMING' else 'last_published' if total else 'not_published'
    if application_forecast:
        application_forecast['is_locked'] = not window['is_open']
    historical_bid_forecast = None
    if application_forecast is None and bidding_ended:
        try:
            from services.historical_bid_service import recover_historical_bid_forecast,get_historical_bid_forecast
            historical_bid_forecast = (get_historical_bid_forecast(data['company_name']) if is_listed_deal
                                       else recover_historical_bid_forecast(data))
        except Exception:
            logger.exception('Historical subscription recovery failed for %s', data.get('company_name'))
    forecast_probability = (application_forecast or {}).get('positive_listing_probability')
    active_probability=(frozen_forecast or {}).get('probability')
    if active_probability is None:active_probability=((frozen_forecast or {}).get('historical_backtest') or {}).get('probability')
    if active_probability is not None:
        from services.forecast_presentation import forecast_identity
        identity_source=frozen_forecast if (frozen_forecast or {}).get('probability') is not None else (frozen_forecast or {}).get('historical_backtest') or {}
        identity=forecast_identity(identity_source)
        pred_res['active_model_configuration']=identity['model_configuration']
        pred_res['active_model_prediction']={'probability':float(active_probability),
            'standard_binary_call':'POSITIVE' if active_probability>=.5 else 'FLAT_OR_NEGATIVE',
            'basis':(identity_source or {}).get('mode'),'legacy_final_book_score_unchanged':True,
            **identity}
        forecast_probability=float(active_probability)
    
    # Generate AI investment memo with real numbers
    from services.forecast_presentation import saved_demand
    memo_demand=saved_demand(frozen_forecast or {})
    historical_test=((frozen_forecast or {}).get('historical_backtest') or {}) if not (frozen_forecast or {}).get('saved_call_found') else {}
    ai_memo = generate_ai_investment_memo(
        company_name=data['company_name'],
        sector=data.get('sector', 'General'),
        qib_sub=memo_demand['qib_subscription'] if memo_demand['qib_subscription'] is not None else observed_qib,
        retail_sub=memo_demand['retail_subscription'] if memo_demand['retail_subscription'] is not None else observed_retail,
        hni_sub=memo_demand['hni_subscription'] if memo_demand['hni_subscription'] is not None else observed_hni,
        issue_size_cr=_number_or_default(data.get('issue_size_cr'), 500.0),
        issue_price=_optional_number(data.get('issue_price')),
        gmp_pct=historical_test.get('features',{}).get('gmp_pct') if historical_test else observed_gmp,
        win_prob=forecast_probability * 100 if forecast_probability is not None
        else pred_res['hybrid_ml_scores']['hybrid_positive_listing_prob'],
        predicted_return_pct=pred_res['hybrid_ml_scores'].get('historical_median_listing_return_pct'),
        sizing_tier=pred_res['bHNI_sizing_tier'],
        verdict=('HIGH' if forecast_probability >= 0.6 else 'LOW' if forecast_probability <= 0.4 else 'MIXED')
        if forecast_probability is not None else pred_res['verdict'],
        conviction='PROVISIONAL' if forecast_probability is not None else pred_res['conviction'],
        news_items=pred_res['qualitative_news_context'],
        analogs=pred_res['similarity_baseline']['top_analogs']
    )
    if active_probability is not None and isinstance(ai_memo,dict):
        ai_memo['forecast_model']=pred_res['active_model_prediction']
        plan=((frozen_forecast or {}).get('inputs') or {}).get('capital_plan') or historical_test.get('capital_plan') or {}
        if plan.get('verdict')=='HARD_SKIP':
            ai_memo.setdefault('executive_summary',{}).update(verdict='SKIP',
                thesis_headline='The opening-direction score does not pass the separate application screen.',
                recommended_action='Application screen: Skip. '+', '.join(str(r).replace('_',' ') for r in plan.get('reasons',[])))
            for strategy in ai_memo.get('bidding_strategy',{}).values():
                strategy['strategy']='Application screen: Skip this setup based on the saved pre-close inputs.'
        elif bidding_ended:
            ai_memo.setdefault('executive_summary',{}).update(
                thesis_headline='Historical benchmark prediction versus the verified listing opening.' if historical_test else 'Saved pre-close opening-direction forecast; the application window has ended.',
                recommended_action='Applications are closed. Review the saved forecast and listing outcome; use Module 2 for post-listing monitoring.' if is_listed_deal else 'Applications are closed. The saved forecast is locked until the listing outcome is verified.')
            for strategy in ai_memo.get('bidding_strategy',{}).values():
                strategy['strategy']='The application window has ended; this is a historical forecast record.'
        model_description=('This saved call used the earlier subscription-only model; it is not a five-expert forecast.'
            if pred_res['active_model_prediction']['model_family']=='subscription_logistic'
            else 'The active frozen five-expert model uses dated bid demand, GMP, available financial facts, offer size and market history. Missing inputs retain frozen training-time handling.')
        ai_memo['bull_catalysts']=[(model_description
            if 'uses only final' in str(point).lower() else point) for point in ai_memo.get('bull_catalysts',[])]
    
    return {
        'success': True,
        'source': source,
        'scraped_data': {
            **data,
            'qib_subscription': display_qib,
            'retail_subscription': display_retail,
            'hni_subscription': display_hni,
            'subscription_observed': subscription_observed,
            'india_vix': current_vix,
            'nifty_10d_return': current_nifty_10d,
            'historical_market_context': historical_market_context,
            'context_imputations': {
                'nifty_10d_return': 'neutral_zero_for_analog_context_only' if current_nifty_10d is None else None,
                'issue_size_cr': 'internal_default_for_non_model_context' if _optional_number(data.get('issue_size_cr')) is None else None,
            },
        },
        'live_market_conditions': macro,
        'evaluation': pred_res,
        'application_forecast': application_forecast,
        'frozen_forecast': frozen_forecast,
        'historical_bid_forecast': historical_bid_forecast,
        'application_forecast_window': window,
        'forecast_save_error': forecast_save_error,
        'ai_investment_memo': ai_memo
    }

@app.get('/api/v1/telemetry/snapshots')
def get_telemetry_snapshots():
    return get_snapshots_summary()


@app.get('/api/v1/telemetry/data-quality')
def get_data_quality():
    from services.data_maintenance import worker
    from services.sqlite_supabase_sync import STATUS_FILE
    result = worker.status()
    result['capture_coverage'] = get_snapshots_summary()['subscription_collector']
    result['database_sync'] = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {'status': 'not_started'}
    return result


@app.get('/api/v1/module1/prospective')
def frozen_module1_results():
    from services.prospective_module1 import scorecard
    return scorecard()


@app.get('/api/v1/module1/listed-saved-calls')
def listed_saved_calls():
    from services.saved_preclose_service import all_listed
    return all_listed()


@app.get('/api/v1/module1/listed-historical-backtest')
def listed_historical_backtest():
    from services.listed_backtest_service import cached
    return cached() or {'records': [], 'summary': {}, 'status': 'not_built'}


@app.get('/api/v1/module1/return-benchmark')
def get_return_benchmark():
    from services.return_forecast_service import return_benchmark, prospective_returns
    from services.snapshot_service import DB_PATH
    try:
        return {**return_benchmark(), 'prospective': prospective_returns(DB_PATH)}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail='Return challenger has not been trained yet')


@app.get('/api/v1/module1/priorities')
def get_return_priorities():
    from services.return_forecast_service import current_priorities
    return current_priorities()

@app.get('/api/v1/telemetry/subscription-captures')
def get_subscription_captures(date_ist: Optional[str] = None, limit: int = Query(500, ge=1, le=5000)):
    if date_ist and not __import__('re').fullmatch(r'\d{4}-\d{2}-\d{2}', date_ist):
        raise HTTPException(status_code=400, detail='date_ist must be YYYY-MM-DD')
    return get_subscription_capture_rows(date_ist=date_ist, limit=limit)

@app.get('/api/v1/telemetry/automatic-decisions')
def get_automatic_decisions(company: Optional[str] = None,
                            limit: int = Query(100, ge=1, le=500)):
    return get_automatic_decision_rows(company=company, limit=limit)


@app.get('/api/v1/telemetry/intraday-forecast-scorecard')
def get_intraday_forecast_results():
    return get_intraday_forecast_scorecard()


@app.get('/api/v1/telemetry/prelisting-replay-scorecard')
def get_prelisting_replay_results():
    return get_prelisting_replay_scorecard()


@app.get('/api/v1/telemetry/prelisting-outcome-scorecard')
def get_prelisting_outcome_results():
    return get_prelisting_outcome_scorecard()


@app.get('/api/v1/telemetry/module2-events')
def get_module2_events(ipo_id: int = Query(..., ge=1), limit: int = Query(30, ge=1, le=100)):
    return get_module2_event_history(ipo_id, limit=limit)


@app.get('/api/v1/module2/call-outcomes')
def get_module2_call_outcomes(ipo_id: int = Query(...,ge=1),horizon: int = Query(1)):
    from services.module2_call_outcomes import call_outcomes
    try:return call_outcomes(ipo_id,horizon=horizon)
    except ValueError as exc:raise HTTPException(status_code=400,detail=str(exc))

@app.post('/api/v1/ai/copilot')
def run_ai_copilot(req: AICopilotRequest):
    answer = ask_ai_copilot(
        company_name=req.company_name,
        question=req.question,
        context_data=req.context_data
    )
    return {
        'success': True,
        'company_name': req.company_name,
        'question': req.question,
        'answer': answer
    }

@app.get('/api/v1/live-ipo-calendar')
def get_live_ipo_calendar(background_tasks: BackgroundTasks, refresh: bool = False):
    """
    Returns the live Mainboard IPO calendar categorized into:
    - OPEN / ACTIVE BIDDING
    - UPCOMING / FORTHCOMING
    - CLOSED / AWAITING ALLOTMENT
    - RECENTLY LISTED
    Backed by local disk cache data/live_scraped_cache.json (zero redundant scraping).
    """
    from scrapers.live_ipo_scraper import sync_chittorgarh_calendar_delta, load_scraped_cache
    cache = load_scraped_cache()
    from services.market_calendar_feed import cached_items,queue_refresh,refresh_status
    items = cached_items(cache)
    if refresh or not items:
        queue_refresh(background_tasks)
    
    # Also include listed modern IPOs from cache for quick Module 2 selection
    listed_items = []
    for k, v in cache.items():
        if v.get('source_observed') is not True:
            continue
        if v.get('is_listed') or v.get('category') == 'LISTED' or 'Listed' in str(v.get('status')):
            from services.preclose_context import restore_preclose_gmp
            v = restore_preclose_gmp({**v, 'url': k})
            cname = v.get('company_name', 'Listed IPO')
            nse = v.get('nse_symbol')
            bse = v.get('bse_code') or v.get('bse_scrip_code')
            if not nse or not bse:
                # lookup in df_canonical if missing
                cm = df_canonical[df_canonical['company_name'].str.lower() == cname.lower()]
                if not cm.empty:
                    if not nse:
                        nse = cm.iloc[0].get('nse_symbol')
                    if not bse:
                        bse = cm.iloc[0].get('bse_scrip_code')
            bse_str = str(int(float(bse))) if bse and not pd.isna(bse) else (str(bse) if bse and not pd.isna(bse) else None)
            listed_items.append({
                'company_name': cname,
                'url': k,
                'date_range': v.get('listing_date') or v.get('date_range') or 'Listed',
                'listing_date': v.get('listing_date') or v.get('date_range') or 'Listed',
                'status': 'LISTED',
                'category': 'LISTED',
                'issue_price': v.get('issue_price'),
                'issue_size_cr': v.get('issue_size_cr'),
                'sector': v.get('sector', 'Industrials'),
                'gmp_pct': v.get('gmp_pct'),
                'gmp_rs': v.get('gmp_rs'),
                'gmp_source': v.get('gmp_source', 'unavailable'),
                'gmp_basis': v.get('gmp_basis'),
                'gmp_preclose_provenance': v.get('gmp_preclose_provenance'),
                'gmp_observed_at': v.get('gmp_observed_at'),
                'gmp_is_stale': False,
                'nse_symbol': nse,
                'bse_code': bse_str,
                'bse_scrip_code': bse_str,
                'rhp_pdf_url': v.get('rhp_pdf_url'),
                'drhp_pdf_url': v.get('drhp_pdf_url')
            })
            
    from services.calendar_view import reconcile_calendar
    items, listed_items = reconcile_calendar(items, listed_items, datetime.now(ZoneInfo('Asia/Kolkata')).date())
    return {
        'total_count': len(items) + len(listed_items),
        'calendar': items,
        'recently_listed': listed_items,
        'cached_total': len(cache),
        'delivery': 'saved_calendar',
        'source_refresh': refresh_status(),
        'timestamp': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds')
    }

@app.post('/api/v1/live-ipo-calendar/sync')
def trigger_calendar_sync():
    from scrapers.live_ipo_scraper import sync_chittorgarh_calendar_delta
    items = sync_chittorgarh_calendar_delta(force_refresh=True)
    return {'status': 'success', 'synced_count': len(items)}

@app.get('/api/v1/module2/search')
def search_module2_holdings(q: str):
    df = df_canonical.copy()
    matches = df[df['company_name'].str.contains(q, case=False, na=False) | df['nse_symbol'].str.contains(q, case=False, na=False) | df['sector'].str.contains(q, case=False, na=False)]
    records = matches[['id', 'company_name', 'nse_symbol', 'sector', 'listing_date', 'issue_price', 'listing_return_pct']].head(15).to_dict(orient='records')
    
    # Check Trendlyne live lookup
    try:
        tl = trendlyne_service.fetch_live_stock_data(q)
        if tl and tl.get('stock_code'):
            already = any(r.get('nse_symbol') == tl['stock_code'] for r in records)
            if not already:
                records.insert(0, {
                    'id': 999900,
                    'company_name': tl['company_name'],
                    'nse_symbol': tl['stock_code'],
                    'sector': tl['sector'],
                    'listing_date': None,
                    'issue_price': None,
                    'listing_return_pct': tl['day_change_pct'],
                    'source': 'Trendlyne autocomplete · quote time unverified'
                })
    except Exception:
        pass

    return {
        'count': len(records),
        'results': records
    }

@app.get('/api/v1/module2/intraday')
def module2_intraday_status(ipo_id: Optional[int] = Query(default=None,ge=1), refresh: bool = False,
                           company: str = ''):
    from services.module2_intraday import tracked_issuers, read, capture
    from datetime import timezone
    universe=tracked_issuers()
    issuer=universe.get(ipo_id)
    if ipo_id is None and company:
        matches=[r for r in universe.values() if r['company_name'].casefold()==company.strip().casefold()]
        if len(matches)==1:
            issuer=matches[0];ipo_id=issuer['ipo_id']
    if issuer is None:
        # An explicitly requested older canonical listing can be monitored too.
        entries=df_canonical[df_canonical.id.eq(ipo_id)]
        if not entries.empty:
            entry=entries.iloc[0];symbol=str(entry.nse_symbol or '')
            import re
            if re.fullmatch(r'[A-Z0-9&-]{2,25}',symbol):
                issuer=dict(ipo_id=ipo_id,company_name=str(entry.company_name),symbol=symbol+'.NS',
                    listing_date=str(entry.listing_date)[:10],issue_price=_optional_number(entry.issue_price))
    if issuer is None:
        return {'ipo_id':ipo_id,'status':'IDENTITY_UNAVAILABLE','signal':'DATA_PENDING',
            'reason':'An exchange-bound market symbol is needed for minute prices.','alerts':[]}
    saved=read(ipo_id)
    observed=saved.get('observed_at_utc')
    expired=not observed or (datetime.now(timezone.utc)-datetime.fromisoformat(observed)).total_seconds()>=55
    if refresh and expired:
        capture(issuer)
    return read(ipo_id)


def _canonical_monitor_result(ipo_id: int, as_of_date: Optional[str]=None):
    """IDs and names use the same policy, including BSE-only canonical issues."""
    result=monitor.evaluate_holding(ipo_id,as_of_date=as_of_date)
    if 'error' not in result:
        return result
    entry=df_canonical[df_canonical.id.eq(ipo_id)].iloc[0]
    data={key:None if pd.isna(value) else value for key,value in entry.to_dict().items()}
    data['is_listed']=True
    from services.module2_dynamic_monitor import evaluate_dynamic_listing
    return evaluate_dynamic_listing(data,target=str(entry.company_name),canonical_entry=entry,as_of_date=as_of_date)


@app.get('/api/v1/module2/holdings/{target:path}')
@app.get('/api/v1/module2/holdings')
def evaluate_holding_by_id_or_name(target: str = '', as_of_date: Optional[str] = None):
    if as_of_date is None:
        from services.market_cache_freshness import invalidate_preclose_market_cache
        invalidate_preclose_market_cache()
    clean_target = target.strip()
    if not clean_target:
        return {'status': 'ERROR', 'message': 'Target company or URL is required.'}
        
    # Check if target is integer ID
    if clean_target.isdigit():
        ipo_id = int(clean_target)
        if ipo_id in set(df_canonical['id'].astype(int)):
            res = _canonical_monitor_result(ipo_id, as_of_date=as_of_date)
        else:
            from services.module2_recent_exchange import recent_exchange_monitors
            recent, _ = recent_exchange_monitors(refresh_today=as_of_date is None)
            res = recent[ipo_id].evaluate_holding(ipo_id, as_of_date=as_of_date) \
                if ipo_id in recent else {'status': 'DATA_UNAVAILABLE',
                                          'message': 'No exchange-verified listed IPO matches this ID.'}
        if as_of_date is None and 'verdict' in res:
            record_module2_decision(res)
        return res

    # Autocomplete supplies identity context, not an authoritative timed quote.
    tl_quote = None
    try:
        tl_quote = trendlyne_service.fetch_live_stock_data(clean_target)
    except Exception:
        tl_quote = None
        
    # Check if target is in canonical master database
    clean = clean_target.lower().replace('.', ' ').replace('-', ' ').replace('/', ' ')
    tokens = [t for t in clean.split() if t not in ('ltd', 'limited', 'of', 'the', 'co', 'corp', 'corporation', 'india', 'ipo')]
    
    match = pd.DataFrame()
    # A verified exact name/ticker takes priority over autocomplete suggestions.
    match=df_canonical[df_canonical['company_name'].str.casefold().eq(clean_target.casefold()) |
        df_canonical['nse_symbol'].str.casefold().eq(clean_target.casefold()) |
        df_canonical['bse_scrip_code'].astype(str).str.replace(r'\.0$','',regex=True).eq(clean_target)]
    if match.empty and tl_quote and tl_quote.get('stock_code'):
        tl_sym = tl_quote['stock_code'].upper()
        match = df_canonical[df_canonical['nse_symbol'].str.upper() == tl_sym]

    # An exact external ticker/name must not fall through to a different
    # canonical issuer which happens to share a word (RELIANCE -> Nippon).
    import re
    def identity_name(value):
        return ' '.join(t for t in re.findall(r'[a-z0-9]+',str(value).casefold())
                        if t not in ('limited','ltd'))
    external_identity = bool(tl_quote and tl_quote.get('stock_code') and
        (clean_target.casefold()==str(tl_quote['stock_code']).casefold() or
         identity_name(clean_target)==identity_name(tl_quote.get('company_name',''))))
    # Exact or substring match
    if len(match) == 0 and not external_identity:
        match = df_canonical[df_canonical['company_name'].str.lower().str.replace('.', ' ', regex=False).str.contains(clean_target.lower(), regex=False, na=False)]
    if len(match) == 0 and not external_identity:
        match = df_canonical[df_canonical['nse_symbol'].str.lower().str.contains(clean_target.lower(), regex=False, na=False)]
        
    # Token intersection match
    if len(match) == 0 and len(tokens) >= 1 and not external_identity:
        match = df_canonical[df_canonical['company_name'].str.lower().str.replace('.', ' ', regex=False).apply(
            lambda name: all(tok in name for tok in tokens)
        )]
        
    # Corrected Known aliases
    if len(match) == 0:
        alias_map = {
            'lic': 342,
            'paytm': 262,
            'zomato': 285,
            'eternal': 285,
            'nykaa': 268,
            'delhivery': 339,
            'tata tech': 364,
            'tatatech': 364,
            'tata technologies': 364,
            'ola': 458,
            'ola electric': 458,
        }
        for alias, cid in alias_map.items():
            if alias == clean_target.lower():
                match = df_canonical[df_canonical['id'] == cid]
                break
        
    canonical_entry = None
    if len(match)>1:
        return {'status':'DATA_IDENTITY_AMBIGUOUS','company_name':clean_target,
                'message':'More than one issuer matches. Select the exact company or exchange symbol.',
                'matches':match[['id','company_name','nse_symbol']].where(pd.notnull(match),None).to_dict('records')}
    if len(match) > 0:
        # Prefer active primary universe if multiple matches
        primary_match = match[match['is_primary_universe'] == True]
        chosen = primary_match.iloc[0] if len(primary_match) > 0 else match.iloc[0]
        ipo_id = int(chosen['id'])
        res = _canonical_monitor_result(ipo_id, as_of_date=as_of_date)
        if 'error' not in res:
            if tl_quote:
                res['trendlyne_data'] = tl_quote
                res['autocomplete_quote_status'] = 'indicative_quote_time_unverified'
            if as_of_date is None:
                record_module2_decision(res)
            return res
        if pd.notnull(chosen.get('listing_date')):
            canonical_entry = chosen
            clean_target = str(chosen['company_name'])

    # New listings are not in the frozen research cohort. Resolve them from
    # archived NSE symbol/name/ISIN rows and run the same Module 2 rules.
    try:
        from services.module2_recent_exchange import recent_exchange_monitors
        recent, _ = recent_exchange_monitors(refresh_today=as_of_date is None)
        recent_matches = [
            (ipo_id, recent_monitor) for ipo_id, recent_monitor in recent.items()
            if clean_target.casefold() in (
                str(recent_monitor.df_ipos.iloc[0].company_name).casefold(),
                str(recent_monitor.df_ipos.iloc[0].nse_symbol).casefold())
        ]
        if len(recent_matches) == 1:
            ipo_id, recent_monitor = recent_matches[0]
            res = recent_monitor.evaluate_holding(ipo_id, as_of_date=as_of_date)
            if as_of_date is None and 'verdict' in res:
                record_module2_decision(res)
            return res
    except Exception:
        logger.exception('Recent exchange-listed Module 2 resolution failed for %s', clean_target)
        
    if external_identity and canonical_entry is None:
        return {'status':'OUTSIDE_IPO_MONITOR_UNIVERSE',
                'company_name':tl_quote.get('company_name',clean_target),
                'nse_symbol':tl_quote['stock_code'],'trendlyne_data':tl_quote,
                'message':'This listed stock is outside the verified IPO monitor universe. Its autocomplete quote has no verified quote time; no IPO hold/sell verdict is assigned.'}

    # Otherwise, target is an unlisted / active bidding IPO (e.g. Ardee Industries or Chittorgarh URL)
    try:
        scrape_res = scrape_chittorgarh_url(clean_target)
        data = scrape_res.get('data', {}) if isinstance(scrape_res, dict) else {}
        if not scrape_res.get('success') and canonical_entry is None:
            return {'status':'DATA_UNAVAILABLE','company_name':clean_target,
                    'message':'The source did not verify this issuer or listing status. No pre-listing status or holding call is inferred.'}
        
        # Merge canonical entry if available
        if canonical_entry is not None:
            data['company_name'] = str(canonical_entry.get('company_name', clean_target))
            data['sector'] = str(canonical_entry.get('sector')) if pd.notnull(canonical_entry.get('sector')) else data.get('sector', 'Unclassified')
            data['issue_price'] = _optional_number(canonical_entry.get('issue_price')) or _optional_number(data.get('issue_price'))
            if pd.notnull(canonical_entry.get('issue_size_cr')):
                data['issue_size_cr'] = float(canonical_entry['issue_size_cr'])
            if pd.notnull(canonical_entry.get('listing_date')):
                data['listing_date'] = str(canonical_entry['listing_date'])
                data['is_listed'] = True
            if pd.notnull(canonical_entry.get('nse_symbol')):
                data['nse_symbol'] = str(canonical_entry['nse_symbol'])
            if pd.notnull(canonical_entry.get('bse_scrip_code')):
                try:
                    data['bse_code'] = str(int(float(canonical_entry['bse_scrip_code'])))
                except:
                    data['bse_code'] = str(canonical_entry['bse_scrip_code'])
            if pd.notnull(canonical_entry.get('listing_open_price')):
                data['listing_open_price'] = float(canonical_entry['listing_open_price'])
            if pd.notnull(canonical_entry.get('listing_day_close_price')):
                data['listing_day_close_price'] = float(canonical_entry['listing_day_close_price'])
            if pd.notnull(canonical_entry.get('listing_day_high_price')):
                data['listing_high_price'] = float(canonical_entry['listing_day_high_price'])

        cname = data.get('company_name', clean_target)
        if data.get('is_listed') and data.get('listing_date'):
            from services.module2_dynamic_monitor import evaluate_dynamic_listing
            result = evaluate_dynamic_listing(data, target=clean_target,
                canonical_entry=canonical_entry, as_of_date=as_of_date)
            if as_of_date is None and 'verdict' in result:
                record_module2_decision(result)
            return result
        return {
            'status': 'PRE_LISTING', 'company_name': cname,
            'listing_date': data.get('listing_date'),
            'message': ('This issue has no verified post-listing session in this source. '
                        'Use Module 1 for the saved pre-close forecast; Module 2 will not '
                        'create a second application verdict from missing bid inputs.'),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        if 'data' in locals() and isinstance(data, dict) and data.get('is_listed'):
            cname = data.get('company_name', clean_target)
            return {
                'status': 'DATA_UNAVAILABLE',
                'company_name': cname,
                'message': f'Surveillance feed for listed IPO "{cname}" encountered a processing error: {str(e)}. Please retry or check exchange quote directly.',
                'listing_date': data.get('listing_date')
            }
        return {
            'status': 'DATA_UNAVAILABLE',
            'company_name': clean_target,
            'message': f'Could not verify listing and dated prices for "{clean_target}". No holding verdict or pre-listing status is inferred from the source failure.'
        }

@app.get('/api/v1/ipos')
def list_canonical_ipos(search: Optional[str] = None, universe_only: bool = True, limit: int = 800):
    df = pd.read_csv('data/canonical_listed_ipos.csv')
    if universe_only:
        df = df[df['is_primary_universe'] == True]
    if search:
        s = search.strip().lower()
        df = df[
            df['company_name'].str.lower().str.contains(s, na=False) |
            df['nse_symbol'].str.lower().str.contains(s, na=False) |
            df['bse_scrip_code'].astype(str).str.contains(s, na=False)
        ]
    
    # Sort strictly descending: newest IPOs first (2026 -> 2025 -> 2024 -> 2021)
    df['_sort_dt'] = pd.to_datetime(df['listing_date'], errors='coerce')
    df = df.sort_values('_sort_dt', ascending=False, na_position='last')
    
    cols = ['id', 'company_name', 'nse_symbol', 'bse_scrip_code', 'sector', 'listing_date', 'issue_price', 'issue_size_cr', 'listing_return_pct', 'is_primary_universe', 'anchor_lockin_50pct_date', 'anchor_lockin_full_date']
    df_subset = df[cols].head(limit).where(pd.notnull(df[cols]), None)
    records = df_subset.to_dict(orient='records')
    for r in records:
        bse = r.get('bse_scrip_code')
        if bse is not None:
            try:
                r['bse_scrip_code'] = str(int(float(bse)))
            except:
                r['bse_scrip_code'] = str(bse)
        else:
            r['bse_scrip_code'] = ''
        # Sanitize float NaNs
        for k, v in r.items():
            if v is not None and isinstance(v, float) and pd.isna(v):
                r[k] = None
    return {
        'total': len(records),
        'ipos': records
    }

@app.get('/api/v1/research/ablation')
def get_research_ablation_table():
    path = 'ml/artifacts/ablation_study_table.csv'
    if os.path.exists(path):
        df_abl = pd.read_csv(path)
        return {
            'ablation_table': df_abl.to_dict(orient='records')
        }
    return {'error': 'Ablation table not found'}
