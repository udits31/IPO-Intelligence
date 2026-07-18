import {proxyBackend} from '../../../../lib/backend-proxy.mjs';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 60;

async function handle(request, {params}) {
  return proxyBackend(request, params.path);
}

export {handle as GET, handle as POST};
