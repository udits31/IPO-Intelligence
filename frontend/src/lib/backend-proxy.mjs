export function backendOrigin(env = process.env) {
  const bound = env.OPENBELL_APP_URL;
  const origin = bound || (!env.VERCEL ? env.OPENBELL_BACKEND_ORIGIN || 'http://127.0.0.1:8000' : null);
  if (!origin) throw new Error('Backend service binding is not configured');
  const url = new URL(origin);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('Backend service binding is invalid');
  }
  if (!url.pathname.endsWith('/')) url.pathname += '/';
  return url;
}

export function backendUrl(path, search = '', env = process.env) {
  if (!Array.isArray(path) || !path.length || path.some(p => !p || p === '.' || p === '..' || /[\\/]/.test(p))) {
    throw new Error('Invalid API path');
  }
  const url = new URL(`api/v1/${path.map(encodeURIComponent).join('/')}`, backendOrigin(env));
  url.search = search;
  return url;
}

export async function proxyBackend(request, path, {env = process.env, fetchImpl = fetch} = {}) {
  let target;
  try { target = backendUrl(path, new URL(request.url).search, env); }
  catch (_) { return Response.json({detail: 'The backend connection is not configured correctly.'}, {status: 503}); }
  const headers = new Headers();
  for (const name of ['accept', 'content-type']) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  try {
    const response = await fetchImpl(target, {
      method: request.method, headers, cache: 'no-store', redirect: 'manual',
      body: ['GET', 'HEAD'].includes(request.method) ? undefined : await request.arrayBuffer(),
      signal: AbortSignal.timeout(55000),
    });
    // Never forward cookies, internal redirects or binding hosts to the browser.
    const outgoing = new Headers({'Cache-Control': 'no-store'});
    const type = response.headers.get('content-type');
    if (type) outgoing.set('Content-Type', type);
    return new Response(response.body, {status: response.status, headers: outgoing});
  } catch (_) {
    return Response.json({detail: 'The backend could not be reached. Please retry.'}, {status: 502});
  }
}
