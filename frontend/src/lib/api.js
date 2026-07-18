const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || '/api/v1').replace(/\/$/, '');

async function request(path, options = {}) {
  const { timeoutMs = 45000, ...fetchOptions } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...fetchOptions, signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...(fetchOptions.headers || {}) },
    });
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The data source took too long. Your saved data is unchanged; please retry.');
    throw error;
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      detail = payload.detail || payload.message || detail;
    } catch (_) {}
    throw new Error(detail);
  }

  return response.json();
}

export const api = {
  health: () => request('/health'),
  macro: () => request('/macro/live', { cache: 'no-store' }),
  calendar: (refresh = false) => request(`/live-ipo-calendar${refresh ? '?refresh=true' : ''}`),
  ipos: (limit = 800) => request(`/ipos?limit=${limit}`),
  benchmark: (limit = 200) => request(`/module1/benchmark?limit=${limit}`),
  performanceEvidence: () => request('/performance/evidence'),
  modelComparison: () => request('/performance/model-comparison'),
  performanceDashboard: () => request('/performance/dashboard'),
  twoTrackEvidence: () => request('/module1/two-track-evidence'),
  sequentialEvaluation: () => request('/module2/sequential-evaluation'),
  nextSessionEvaluation: () => request('/module2/next-session-evaluation'),
  nextSessionImprovement: () => request('/module2/next-session-improvement'),
  holdingPolicyEvaluation: () => request('/module2/holding-policy-evaluation'),
  multihorizonEvaluation: () => request('/module2/multihorizon-evaluation'),
  module2ShadowScorecard: () => request('/module2/shadow-scorecard'),
  predictionResults: () => request('/module1/prediction-results'),
  prospectiveResults: () => request('/module1/prospective'),
  priorities: () => request('/module1/priorities'),
  returnBenchmark: () => request('/module1/return-benchmark'),
  automaticDecisions: (company, limit = 24) => request(`/telemetry/automatic-decisions?company=${encodeURIComponent(company)}&limit=${limit}`),
  intradayForecastScorecard: () => request('/telemetry/intraday-forecast-scorecard'),
  prelistingReplayScorecard: () => request('/telemetry/prelisting-replay-scorecard'),
  prelistingOutcomeScorecard: () => request('/telemetry/prelisting-outcome-scorecard'),
  evaluate: (target, fetchNews = true, saveForecast = true) => request('/module1/scrape-and-evaluate', {
    method: 'POST',
    body: JSON.stringify({ url_or_name: target, fetch_news: fetchNews, save_forecast: saveForecast }),
  }),
  holding: (target) => request(`/module2/holdings?target=${encodeURIComponent(target)}`),
  module2Events: (ipoId) => request(`/telemetry/module2-events?ipo_id=${encodeURIComponent(ipoId)}`),
  module2CallOutcomes: (ipoId,horizon=1) => request(`/module2/call-outcomes?ipo_id=${encodeURIComponent(ipoId)}&horizon=${horizon}`),
  module2Intraday: (ipoId, refresh = false, company = '') => request(`/module2/intraday?${ipoId?`ipo_id=${encodeURIComponent(ipoId)}`:`company=${encodeURIComponent(company)}`}&refresh=${refresh}`),
  copilot: (companyName, question, contextData) => request('/ai/copilot', {
    method: 'POST',
    body: JSON.stringify({ company_name: companyName, question, context_data: contextData }),
  }),
};

export { API_BASE };
