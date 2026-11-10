# Automatic IPO data maintenance

Starting the backend starts the data workers. Collection and recovery do not
depend on an open browser, a chat prompt, or a recurring terminal command.

The data maintenance worker checks its durable queue every minute. It refreshes
the IPO calendar every 30 minutes, prioritises open and upcoming offers, and
processes at most six offer records per pass. Existing verified offer sources
are reused. Missing OFS, retail reservation, original lot size and use-of-proceeds
facts trigger a source fetch with issuer-name and offer-price checks. The raw
source, checksum and extracted facts are appended to SQLite and mirrored to
Supabase. Static facts are rechecked daily while an offer is active; completed
listed-offer jobs have a seven-day interval. Published price changes create a
new job version. Unpublished terms remain explicitly pending.

Parser upgrades first re-extract preserved source HTML and keep its original
fetch time. A versioned extraction record is stored alongside the unchanged
raw archive. More precise verified offer amounts are retained when a newer
summary rounds them; explicit capex amounts can fill the missing field using
that preserved fresh-issue denominator.

Source failures retry after 5, 10, 20, 40 and then 60 minutes. Partial extraction
rotates to another source on the next minute. Issuer-identity and conflicting
offer-total failures quarantine that source for 24 hours instead of repeatedly
requesting it. Alternate IPOji URLs are derived from the issuer name and original
offer slug; every alternate must pass the same name, cap-price and amount checks.
Per-source cooldowns persist across backend restarts. If all candidate sources
have been checked, the job enters `needs_document_review` with the remaining
fields and source list, and waits 24 hours. This does not certify an unknown
CAPEX figure: the page exposes the published use-of-proceeds breakdown and says
`Not separately itemised` where a separate accounting split is unavailable.
Verified use-of-proceeds sources retain their hashes and fetch times.
When the alternate source supplies the offer fields and published uses but no
separate CAPEX split, the job is `ready_with_limitations`: core offer recovery
has succeeded, the unquantified percentage remains null, and the limitation is
retained in the status/API. Listed offers with that limitation are rechecked
weekly rather than hourly; active offers are rechecked daily.
A five-minute persisted lease prevents overlapping work and
allows interrupted jobs to resume after a restart. One issuer's failure does
not stop the remaining queue. Unknown uses of proceeds are not labelled zero
capex. An entirely OFS offer has no fresh-issue capex allocation.

Existing dedicated workers retain responsibility for subscriptions, forecast
logging, intraday prices, daily price/news monitoring and verified listing
outcomes. Subscription capture runs every 15 minutes from 10:00 IST, then every
five minutes from 14:30 through 16:55. Application-time forecasts stop at the
final closing-day cutoff; later books remain separately tagged reconciliation
evidence. Saved forecasts and their inputs remain unchanged after listing.
The frozen forecast cycle is guarded against concurrent execution by its timer
and the subscription collector. Unexpected collector errors back off rather
than repeating an entire pass every 15 seconds.

Active IPO news is refreshed every ten minutes with source caching. Module 2's
minute collector runs during trading hours, and its daily collector retries
official closing records after the session. Published exchange listing archives
are reconciled automatically with preserved pre-close forecasts. An unpublished
archive remains pending; a current intraday candle is not a completed daily
close. A page's minute graph refreshes each minute, its Module 1 analysis each
minute, and its Module 2 daily assessment every five minutes and on window focus.
Background page refreshes do not create extra manual forecast records.

`GET /api/v1/telemetry/data-quality` exposes the persistent jobs, retry times,
source errors, capture coverage and database sync status. `/api/v1/health`
includes a compact worker status. The UI shows a delayed-source message while
retries or incomplete extractions remain. Successful HTTP responses containing
only retained cache data are not reported as a fresh calendar fetch.

The computer and backend must remain running to capture live observations.
Restarting catches up available public offer facts and exchange archives, but
cannot recreate an intraday observation or a prediction that was never saved.
