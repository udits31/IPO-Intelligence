import test from 'node:test';
import assert from 'node:assert/strict';
import {marketStatus} from '../src/lib/marketStatus.mjs';

test('after the session shows closing data instead of a stale-looking clock',()=>{
  const s=marketStatus({timestamp:'2026-10-28T15:30:00+05:30'},'2026-10-28T19:30:00+05:30');
  assert.equal(s.title,'Market pulse');assert.equal(s.detail,'Market closed · Today’s close');assert.equal(s.warning,false);
});
test('checks freshness while the market is open',()=>{
  assert.equal(marketStatus({timestamp:'2026-10-28T12:00:00+05:30'},'2026-10-28T12:03:00+05:30').title,'Market pulse');
  assert.equal(marketStatus({timestamp:'2026-10-28T11:00:00+05:30'},'2026-10-28T12:03:00+05:30').title,'Market feed delayed');
});
test('does not call an unfinished or older reading today’s closing data',()=>{
  assert.equal(marketStatus({timestamp:'2026-10-28T14:00:00+05:30'},'2026-10-28T19:00:00+05:30').detail,'Market closed · Latest reading');
  assert.equal(marketStatus({timestamp:'2026-10-27T15:30:00+05:30'},'2026-10-28T19:00:00+05:30').detail,'Market closed · Latest close');
});
test('fallback remains visibly delayed even when it contains a closing time',()=>{
  assert.equal(marketStatus({timestamp:'2026-10-28T15:30:00+05:30',is_fallback:true},'2026-10-28T19:00:00+05:30').title,'Market feed delayed');
});
