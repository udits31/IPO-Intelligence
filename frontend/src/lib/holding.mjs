// Entered holdings have no acquisition-date/share-basis evidence. Never guess
// whether a user's quantity and average already include a split or bonus.
export function holdingValue(holding, result) {
  const current = result?.current_price_adjusted;
  const basis = result?.price_basis_adjustment;
  const factor = Number(basis?.share_factor);
  const unresolved = result?.status === 'PRICE_BASIS_UNVERIFIED';
  const needsConfirmation = !unresolved && Number.isFinite(factor) && factor !== 1 &&
    Number(holding.confirmedShareFactor) !== factor;
  const valid = !unresolved && !needsConfirmation && current != null &&
    Number.isFinite(Number(current)) && Number(current) > 0 &&
    Number(holding.price) > 0 && Number(holding.quantity) > 0;
  return { current, unresolved, needsConfirmation, factor,
    pnl: valid ? (Number(current) - Number(holding.price)) * Number(holding.quantity) : null,
    returnPct: valid ? (Number(current) / Number(holding.price) - 1) * 100 : null };
}
