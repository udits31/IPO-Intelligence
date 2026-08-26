import test from 'node:test';
import assert from 'node:assert/strict';
import {holdingValue} from '../src/lib/holding.mjs';

test('split/bonus P&L needs confirmation of user-entered share units', () => {
  // Actual Cordelia issue price and announced 10-for-1 split, not model data.
  const holding={price:808,quantity:1};
  const result={current_price_adjusted:107.97,price_basis_adjustment:{share_factor:10}};
  assert.equal(holdingValue(holding,result).pnl,null);
  assert.equal(holdingValue(holding,result).needsConfirmation,true);
  const confirmed={price:80.8,quantity:10,confirmedShareFactor:10};
  assert.ok(Math.abs(holdingValue(confirmed,result).pnl-271.7)<1e-8);
  assert.equal(holdingValue(confirmed,{...result,price_basis_adjustment:{share_factor:20}}).pnl,null);
});

test('confirmation cannot override an unresolved price basis', () => {
  const value=holdingValue({price:80.8,quantity:10,confirmedShareFactor:10},
    {status:'PRICE_BASIS_UNVERIFIED',current_price_adjusted:107.97,price_basis_adjustment:{share_factor:10}});
  assert.equal(value.pnl,null);
  assert.equal(value.unresolved,true);
  assert.equal(value.needsConfirmation,false);
});

test('missing current price does not become a zero-valued position', () => {
  assert.equal(holdingValue({price:808,quantity:1},{}).pnl,null);
});
