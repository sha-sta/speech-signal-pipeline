# Backtest harness

`pmlab.study` is a maker-side, strict-fill backtest simulator plus the candle store it reads
prices from. It is **strategy-agnostic**: it takes a frame of *your* model's predictions and
replays them against recorded market prices — it has no opinion on how those predictions were
produced. The parameter values shown below (and the CLI's defaults) are illustrative defaults
baked into the code for testability, not recommendations.

## Candle store (`pmlab.study.candles`)

Pulls and persists bid/ask OHLC bars for a set of markets via the venue's candlestick API,
storing them as a schema-validated parquet store keyed on `(ticker, ts, period_minutes)`.
Requests are self-throttled and cached forever — historical bars are immutable once settled, so
re-pulling the same window is free. Bar granularity (`period_minutes`) is a caller choice: coarser
bars are cheaper to pull across a large universe; finer bars capture more of a fast-moving
intraday price path at proportionally higher request volume. This is a real tradeoff worth naming
explicitly: coarse bars can *overstate* what a maker order would actually have filled, because a
bar's high/low can span an intraday move a resting order sitting at a fixed price might never have
touched.

## The simulator (`pmlab.study.backtest.run_backtest`)

For each market, at a caller-supplied decision timestamp `T`, the simulator does one thing at a
time, in order:

1. **Price.** Read the bar at or just before `T`. If the freshest available bar is older than a
   configured staleness limit, or its quote is missing/broken (crossed, non-finite), the market is
   priced `NaN` and never traded — a stale or broken quote is not a tradable price.
2. **Signal.** A market is only considered for a trade if its priced mid sits inside a configured
   price band and the supplied prediction disagrees with that price by more than a configured
   minimum edge (`theta`). Both the band and `theta` are constructor parameters
   (`BacktestParams`); the values wired as CLI defaults (`band_lo=0.15`, `band_hi=0.60`,
   `theta=0.05`) are illustrative, not tuned settings — set your own.
3. **Maker quote.** The simulated order rests one tick better than the relevant best quote, and
   never crosses the book — on a one-tick-wide (locked) book, no maker improvement is possible and
   no order is placed.
4. **Strict fill.** A resting order only fills if a *later* bar's trade prints actually cross its
   resting price — a buy fills when a subsequent bar's low reaches its price; a sell fills when a
   subsequent bar's high does. Only bars strictly after the decision bar are eligible, so a
   same-bar print that may have happened before the hypothetical order was even placed can never
   fill it. This is deliberately conservative: it will under-count fills relative to a live maker
   sitting continuously in the book, never over-count them.
5. **Settle.** A filled position is held to resolution; profit/loss is `payout − stake − fee`,
   with an optional per-fill maker fee (charged as a function of price and size) applied at fill
   time and settlement itself fee-free.

The result is one row per market: the priced market probability, the supplied prediction, whether
a trade was placed, whether it filled, and (if filled) the fill price, fee, and P&L. `occasion` is
carried through as a grouping key so an evaluation can cluster by occasion rather than treat every
market as an independent draw — markets from the same broadcast or speech are not independent
observations of whatever signal produced the predictions.

## Supplying predictions

The CLI (`pmlab study backtest`) takes a `--predictions` path to a parquet file with columns
`market_ticker, decision_ts, split, y, p_base, p_topic, p_cal` (`split` is your own train/test
label; `y` is the realized 0/1 outcome). Which column is actually traded against the market price
is a CLI flag (`--model-col`, default `p_cal`) — the column names themselves are just a fixed
frame shape the simulator expects, not a statement about what kind of model produced them. Build
that frame however you like: `pmlab.model.calibrate` ships generic isotonic/Platt calibration and
scoring primitives if you want to use them, but nothing in the simulator requires it.

## Occasion-clustered evaluation

Because markets sharing an `occasion` (the same speech, the same broadcast) are driven by
correlated events rather than independent draws, any aggregate metric computed over backtest rows
— hit rate, P&L, calibration error — should account for that clustering (e.g. a block bootstrap
resampling whole occasions rather than individual market rows) rather than treat every row as an
i.i.d. observation. The harness carries the `occasion` key through to the output specifically so a
downstream evaluation can do this; it does not compute or assert any particular metric itself.
