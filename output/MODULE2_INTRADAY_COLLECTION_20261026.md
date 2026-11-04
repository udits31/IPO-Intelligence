# Module 2 intraday collection

The one-minute collector is separate from the locked daily-close monitor and its 132-IPO evaluation. No model weights, daily policy thresholds, benchmark scores or existing forecast rows were changed.

## What is collected

- Completed one-minute OHLCV candles for exchange-bound recent IPO identities, plus recently evaluated holdings.
- Listing-day calendar candidates without a daily close can also qualify, but only when the explicit NSE ticker, INR currency, exchange and provider issuer name agree. The intraday panel remains available even when the daily assessment has no completed close yet.
- Raw source responses saved by SHA-256, append-only candle revisions, observation timestamps, and signal-change alerts in `data/telemetry/module2_intraday.db`.
- Background polling at approximately one-minute intervals during weekday 09:15–15:30 IST. A slow cycle finishes before the next starts. Old candles on a holiday cannot qualify for an active alert.
- Collection continues until 15:35 to save the final completed session candle; these after-close observations cannot trigger active intraday alerts.
- The Module 2 intraday panel refreshes every minute while open. A failed refresh keeps the previous prices visible and suppresses a fresh-alert claim.

## Signals

- VWAP lost / reclaimed: a crossing on consecutive completed minute candles.
- Review exit: two consecutive completed minutes below both the first 15-minute range low and bar-based VWAP. A partial opening range cannot qualify.
- Session range, approximate bar-based VWAP and volume relative to the preceding 20-minute median are supporting observations.

These are structural research alerts, not a newly trained return forecast. They do not replace the daily monitor verdict. Alerts record their actual capture time; historical candles downloaded now are not backdated as prospective calls. Identical persistent warnings are not repeatedly appended as new signal changes.

## Source limits and execution

The current source is Yahoo Finance's indicative one-minute chart feed. No broker market-data credentials were configured during implementation. Symbol and INR currency are checked; unfinished, future, malformed and duplicate minute bars are excluded or rejected. Completed candles older than three minutes are marked delayed and cannot trigger an active alert.

Three-minute freshness is an engineering expiry rule, not a latency guarantee or a tuned trading parameter. This source does not establish transaction-level VWAP, complete tick history, executable bid/ask liquidity or verified circuit bands. No orders are placed. An alert cannot guarantee an exit, especially when a stock is locked at its lower circuit without buyers. An authenticated broker/exchange feed is needed for dependable trading-time quote and depth coverage; its access has not been configured here.

## Evaluation boundary

The existing daily-close accuracy and net-return results do not validate these intraday signals. New observations and alerts are logged for a separate prospective evaluation with executable-price, cost and liquidity assumptions stated explicitly.
