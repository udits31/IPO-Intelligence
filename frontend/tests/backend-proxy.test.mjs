import test from 'node:test';
import assert from 'node:assert/strict';
import {backendUrl, proxyBackend} from '../src/lib/backend-proxy.mjs';

test('runtime binding preserves a service URL prefix, API prefix and query', () => {
  assert.equal(backendUrl(['module2','holdings'],'?target=ESDS',{VERCEL:'1',OPENBELL_APP_URL:'https://internal.example/service/'}).href,
    'https://internal.example/service/api/v1/module2/holdings?target=ESDS');
});
test('production never falls back to a localhost or manually configured origin', () => {
  assert.throws(() => backendUrl(['health'],'',{VERCEL:'1',OPENBELL_BACKEND_ORIGIN:'https://wrong.example'}));
  assert.equal(backendUrl(['health'],'',{}).href,'http://127.0.0.1:8000/api/v1/health');
});
test('path traversal and absolute service destinations are rejected', () => {
  for (const path of [['..','health'],['https://evil.example'],['x/y']]) assert.throws(() => backendUrl(path));
});
test('POST proxy preserves JSON body, query and backend error status without forwarding cookies', async () => {
  let captured;
  const request = new Request('https://openbell.example/api/v1/module1/scrape-and-evaluate?x=1', {
    method:'POST',headers:{'Content-Type':'application/json','Cookie':'private=secret'},body:'{"url_or_name":"ESDS"}',
  });
  const result = await proxyBackend(request,['module1','scrape-and-evaluate'],{
    env:{VERCEL:'1',OPENBELL_APP_URL:'https://internal.example'},
    fetchImpl:async (url,options) => {captured={url,options};return Response.json({detail:'Missing evidence'},{status:422});},
  });
  assert.equal(captured.url.href,'https://internal.example/api/v1/module1/scrape-and-evaluate?x=1');
  assert.equal(new TextDecoder().decode(captured.options.body),'{"url_or_name":"ESDS"}');
  assert.equal(captured.options.headers.has('cookie'),false);
  assert.equal(result.status,422);
  assert.equal(result.headers.get('cache-control'),'no-store');
});
test('missing binding returns a service error without leaking private URLs', async () => {
  const result = await proxyBackend(new Request('https://public.example/api/v1/health'),['health'],{env:{VERCEL:'1'}});
  assert.equal(result.status,503);
});
