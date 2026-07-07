"""pmlab command-line interface (typer)."""

from __future__ import annotations

import itertools
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pmlab.config import Settings, get_settings
from pmlab.venues.kalshi import KalshiPublic, _f
from pmlab.venues.polymarket import ClobPublic, GammaClient, yes_token_id

app = typer.Typer(add_completion=False, help="pmlab — Kalshi mention-market research CLI")
venues_app = typer.Typer(help="Venue connectivity + data smoke tests")
mentions_app = typer.Typer(help="Mention-market universe discovery + resolution registry")
corpus_app = typer.Typer(help="Transcript + headline corpus scrapers")
study_app = typer.Typer(help="Candle history + maker-side backtest simulator")
app.add_typer(venues_app, name="venues")
app.add_typer(mentions_app, name="mentions")
app.add_typer(corpus_app, name="corpus")
app.add_typer(study_app, name="study")

from pmlab.probe.cli import probe_app  # noqa: E402 — sub-app import kept beside its registration

app.add_typer(probe_app, name="probe")

console = Console()

_PREDICTIONS_HELP = (
    "parquet of precomputed model predictions (market_ticker, decision_ts, split, y, p_base, "
    "p_topic, p_cal); defaults to data/study/predictions.parquet. Produce this frame with your "
    "own model — the simulator is strategy-agnostic."
)


class _Paths:
    """M1/M2 artifact locations under the configured data dir (§3, §4, §5)."""

    def __init__(self, s: Settings) -> None:
        self.catalog = s.data_dir / "catalog" / "mentions.parquet"
        self.raw = s.data_dir / "catalog" / "raw_events.jsonl"
        self.resolutions = s.data_dir / "universe" / "resolutions.jsonl"
        self.registry = s.data_dir / "universe" / "mention_registry.parquet"
        self.review = s.data_dir / "universe" / "review.csv"
        self.corpus = s.data_dir / "corpus" / "transcripts.parquet"
        self.headlines = s.data_dir / "corpus" / "headlines.parquet"
        self.predictions = s.data_dir / "study" / "predictions.parquet"
        self.candles = s.data_dir / "candles" / "candles.parquet"
        self.candles_1min = s.data_dir / "candles" / "candles_1min.parquet"
        self.backtest = s.data_dir / "study" / "backtest.parquet"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # quiet per-request request logs


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _find_candle(kalshi: KalshiPublic, *, sample: int = 3000, probe: int = 40) -> None:
    """Find and print one real candle row. The `status=open` slice reaches mostly MVE/long-tail
    markets, so we rank a bounded sample by lifetime volume and probe single-ticker 6-day windows
    (safely under the span cap) until a market with candles turns up."""
    now = int(datetime.now(tz=UTC).timestamp())
    window_start = now - 6 * 86_400
    pool = list(itertools.islice(kalshi.iter_markets(status="open"), sample))
    ranked = sorted(pool, key=lambda m: _f(m.get("volume_fp")), reverse=True)
    for market in ranked[:probe]:
        ticker = market["ticker"]
        candles = kalshi.candles_batch([ticker], window_start, now)
        if len(candles):
            row = candles.iloc[-1]
            mid = (row.yes_bid_close + row.yes_ask_close) / 2
            console.print(f"  candle       : {ticker}")
            console.print(
                f"                 {_iso(int(row.ts))}  "
                f"yes_bid OHLC=[{row.yes_bid_open:.2f} {row.yes_bid_high:.2f} "
                f"{row.yes_bid_low:.2f} {row.yes_bid_close:.2f}]  "
                f"yes_ask OHLC=[{row.yes_ask_open:.2f} {row.yes_ask_high:.2f} "
                f"{row.yes_ask_low:.2f} {row.yes_ask_close:.2f}]  mid={mid:.3f}"
            )
            return
    console.print(f"  candle       : [yellow]none found in top {probe} sampled markets[/yellow]")


@venues_app.command()
def smoke() -> None:
    """Hit both venues live and print a data sample — M0 acceptance (§4)."""
    _setup_logging()
    log = logging.getLogger("pmlab.smoke")

    console.rule("[bold]Kalshi[/bold]")
    kalshi = KalshiPublic()
    events = list(itertools.islice(kalshi.iter_events(status="open"), 200))
    console.print(f"  open events  : {len(events)} (first page)")
    for ev in events[:3]:
        console.print(f"                 · {ev.get('event_ticker')}  {ev.get('title', '')[:60]}")
    series_ticker = events[0]["series_ticker"]
    series = kalshi.get_series(series_ticker)
    console.print(
        f"  series       : {series_ticker}  fee_type={series.get('fee_type')}  "
        f"fee_multiplier={series.get('fee_multiplier')}  category={series.get('category')}"
    )
    _find_candle(kalshi)

    console.rule("[bold]Polymarket[/bold]")
    gamma = GammaClient()
    clob = ClobPublic()
    markets = list(itertools.islice(gamma.iter_markets(closed=False), 500))
    console.print(f"  open markets : {len(markets)} (first page)")
    for mk in markets[:3]:
        console.print(f"                 · {mk.get('question', '')[:70]}")
    market = next(m for m in markets if yes_token_id(m))
    token = yes_token_id(market)
    assert token is not None
    now = int(datetime.now(tz=UTC).timestamp())
    hist = clob.prices_history(token, now - 7 * 86_400, now)
    if len(hist):
        last = hist.iloc[-1]
        live_mid = clob.midpoint(token)
        console.print(f"  market       : {market.get('question', '')[:70]}")
        console.print(
            f"  price point  : {_iso(int(last.ts))}  p={last.p:.3f}   "
            f"(live midpoint={live_mid:.3f}, Δ={abs(last.p - live_mid):.3f})"
        )
    else:
        console.print("  price point  : [yellow]no history returned[/yellow]")

    console.rule()
    log.info("smoke OK — live data reached from both venues")


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@mentions_app.command()
def discover(
    statuses: str = typer.Option("settled,closed,open", help="comma-separated event statuses"),
    limit_series: int | None = typer.Option(None, help="cap series count (dev)"),
    coverage: bool = typer.Option(True, help="probe candle coverage (daily granularity)"),
) -> None:
    """Sweep the Kalshi mention universe → catalog.parquet + raw_events.jsonl (+candle coverage)."""
    from pmlab.mentions.discover import attach_candle_coverage, persist_catalog, sweep

    _setup_logging()
    s = get_settings()
    paths = _Paths(s)
    kalshi = KalshiPublic(settings=s)
    status_list = [x.strip() for x in statuses.split(",") if x.strip()]

    console.rule("[bold]Discover mention universe[/bold]")
    catalog, raw, stats = sweep(kalshi, status_list, limit_series=limit_series)
    _write_jsonl(raw, paths.raw)

    cutoff_ts = 0
    if coverage and len(catalog):
        console.print("  probing candle coverage …")
        catalog = attach_candle_coverage(kalshi, catalog)
        try:  # market_settled_ts is an ISO string, not a unix int
            raw_cutoff = str(kalshi.historical_cutoff().get("market_settled_ts", "") or "")
            if raw_cutoff:
                iso = datetime.fromisoformat(raw_cutoff.replace("Z", "+00:00"))
                cutoff_ts = int(iso.timestamp())
        except Exception as exc:  # noqa: BLE001 - coverage report is best-effort
            logging.getLogger("pmlab.mentions").warning("historical cutoff probe failed: %s", exc)
    persist_catalog(catalog, paths.catalog)

    console.print(
        f"  series={stats.n_series}  events={stats.n_events}  markets_seen={stats.n_markets_seen}  "
        f"parlay_excluded={stats.n_parlay_excluded}  [green]catalog={stats.n_catalog}[/green]"
    )
    console.print(f"  by format : {stats.by_format}")
    console.print(f"  by status : {stats.by_status}")
    if coverage and "has_candles" in catalog.columns and len(catalog):
        have = int(catalog["has_candles"].sum())
        pre = 0
        if cutoff_ts:
            pre = int(((~catalog["has_candles"]) & (catalog["close_ts"] < cutoff_ts)).sum())
        console.print(
            f"  candles    : {have}/{len(catalog)} markets have bars "
            f"({len(catalog) - have} without; {pre} settled pre-cutoff → /historical in M3)"
        )
    console.print(f"  wrote      : {paths.catalog}  +  {paths.raw}")


@mentions_app.command()
def parse() -> None:
    """Parse raw_events.jsonl → resolutions.jsonl (offline, deterministic baseline)."""
    from pmlab.mentions.resolution import parse_raw_records, write_resolutions

    _setup_logging()
    paths = _Paths(get_settings())
    if not paths.raw.exists():
        console.print(f"[red]missing {paths.raw}; run `pmlab mentions discover` first[/red]")
        raise typer.Exit(1)
    records = _read_jsonl(paths.raw)
    resolutions = parse_raw_records(records)
    n = write_resolutions(resolutions, paths.resolutions)

    from collections import Counter

    by_tmpl = Counter(r.template for r in resolutions)
    by_risk = Counter(r.resolution_risk for r in resolutions)
    elig = sum(r.eligible for r in resolutions)
    console.print(f"[green]parsed {n} resolutions[/green] → {paths.resolutions}")
    console.print(f"  template : {dict(by_tmpl)}")
    console.print(f"  risk     : {dict(by_risk)}")
    console.print(f"  eligible : {elig}/{n} (low/med risk, conf≥0.8, non-NQE)")
    other = [r.market_ticker for r in resolutions if r.template == "other"][:10]
    if other:
        console.print(f"  [yellow]unrecognized templates (sample): {other}[/yellow]")


@mentions_app.command()
def registry(
    review_ingest: bool = typer.Option(False, "--review", help="re-ingest edits from review.csv"),
) -> None:
    """Build mention_registry.parquet + review.csv from catalog + resolutions (offline)."""
    from pmlab.mentions.discover import load_catalog
    from pmlab.mentions.registry import (
        apply_review,
        build_registry,
        persist_registry,
        read_review_csv,
        write_review_csv,
    )
    from pmlab.mentions.resolution import read_resolutions

    _setup_logging()
    paths = _Paths(get_settings())
    catalog = load_catalog(paths.catalog)
    resolutions = read_resolutions(paths.resolutions)
    reg = build_registry(catalog, resolutions)

    if review_ingest and paths.review.exists():
        reg = apply_review(reg, read_review_csv(paths.review))
        console.print(f"  re-ingested owner edits from {paths.review}")

    persist_registry(reg, paths.registry)
    write_review_csv(reg, paths.review, only_flagged=True)
    n_flagged = int(((reg["resolution_risk"] != "low") | (~reg["eligible"])).sum())
    console.print(f"[green]registry: {len(reg)} markets[/green] → {paths.registry}")
    console.print(f"  review.csv ({n_flagged} flagged rows) → {paths.review}")
    console.print(
        f"  eligible={int(reg['eligible'].sum())}  "
        f"speakers={reg['speaker'].nunique()}  formats={sorted(reg['format'].unique())}"
    )
    _print_gate(reg)


@mentions_app.command()
def review() -> None:
    """Print the human-gate view (worst parses + flagged-high-risk) from the built registry."""
    from pmlab.mentions.registry import load_registry

    _setup_logging()
    paths = _Paths(get_settings())
    _print_gate(load_registry(paths.registry))


def _print_gate(reg: pd.DataFrame, n_low: int = 10, n_high: int = 5) -> None:
    """Owner spot-check surface: lowest-confidence parses + flagged-high-risk."""
    low = reg.sort_values("confidence").head(n_low)
    high = reg[reg["resolution_risk"] == "high"].sort_values("confidence").head(n_high)

    cols = {
        "market_ticker": 30, "speaker": 18, "format": 11, "phrase": 24,
        "variant_rule": 16, "risk": 6, "conf": 5, "src": 4,
    }

    def _table(title: str, df: pd.DataFrame) -> Table:
        t = Table(title=title, show_lines=False, expand=False, pad_edge=False)
        for col, width in cols.items():
            t.add_column(col, max_width=width, no_wrap=True, overflow="ellipsis")
        for _, r in df.iterrows():
            t.add_row(
                str(r["market_ticker"]), str(r["speaker"]), str(r["format"]),
                str(r["phrase"]), str(r["variant_rule"]), str(r["resolution_risk"]),
                f"{float(r['confidence']):.2f}", str(int(r["n_settlement_sources"])),
            )
        return t

    console.rule("[bold]Human gate — spot-check these[/bold]")
    console.print(_table(f"{n_low} lowest-confidence parses", low))
    if len(high):
        console.print(_table(f"{min(n_high, len(high))} flagged high-risk markets", high))
    else:
        console.print("[dim]no high-risk markets flagged[/dim]")


@corpus_app.command("fetch")
def corpus_fetch(
    limit_per_source: int | None = typer.Option(None, help="cap transcripts per source (dev)"),
    speech_years: str = typer.Option("2026,2025,2024", help="comma-separated Fed speech years"),
    statements: bool = typer.Option(True, help="include FOMC statements (the Chair)"),
    speeches: bool = typer.Option(True, help="include Fed speeches (named officials)"),
) -> None:
    """Scrape transcripts → data/corpus/transcripts.parquet (federalreserve.gov)."""
    from pmlab.corpus.sources import FederalReserveSource
    from pmlab.corpus.store import load_corpus, merge_corpus, persist_corpus, to_corpus_frame

    _setup_logging()
    s = get_settings()
    paths = _Paths(s)
    years = [int(x) for x in speech_years.split(",") if x.strip()]

    console.rule("[bold]Fetch transcript corpus[/bold]")
    fed = FederalReserveSource(settings=s)
    rows = [
        rec.to_row()
        for rec in fed.fetch(
            limit=limit_per_source, statements=statements, speeches=speeches, speech_years=years
        )
    ]
    fresh = to_corpus_frame(rows)
    existing = load_corpus(paths.corpus)  # returns empty frame if absent
    merged = merge_corpus(existing, fresh) if len(existing) else fresh
    persist_corpus(merged, paths.corpus)

    by_spk = merged["speaker"].value_counts()
    console.print(f"[green]corpus: {len(merged)} transcripts[/green] "
                  f"({len(fresh)} this run) → {paths.corpus}")
    console.print(f"  speakers   : {merged['speaker'].nunique()}  "
                  f"words={int(merged['n_words'].sum()):,}")
    for spk, cnt in by_spk.head(8).items():
        console.print(f"     · {cnt:>4}  {spk}")


@corpus_app.command("headlines")
def corpus_headlines(
    start_month: str = typer.Option("2024-01", help="first month YYYY-MM"),
    end_month: str = typer.Option("2026-07", help="last month YYYY-MM (inclusive)"),
    maxrecords: int = typer.Option(250, help="headlines per monthly GDELT query (≤250)"),
) -> None:
    """Fetch dated news headlines (GDELT) → data/corpus/headlines.parquet."""
    from pmlab.corpus.headlines import (
        GdeltHeadlineSource,
        fetch_headlines,
        load_headlines,
        persist_headlines,
    )

    _setup_logging()
    s = get_settings()
    paths = _Paths(s)
    console.rule("[bold]Fetch news headlines (GDELT)[/bold]")
    console.print(f"  throttled to ~1 req / 5s; {start_month} … {end_month}")
    src = GdeltHeadlineSource(settings=s)
    fresh = fetch_headlines(src, start_month, end_month, maxrecords=maxrecords)

    existing = load_headlines(paths.headlines)
    if len(existing):
        combined = pd.concat([existing, fresh], ignore_index=True)
        fresh = combined.drop_duplicates(subset="headline_id").reset_index(drop=True)
    persist_headlines(fresh, paths.headlines)

    span = "n/a"
    if len(fresh):
        lo = _iso(int(fresh["ts"].min()))
        hi = _iso(int(fresh["ts"].max()))
        span = f"{lo} … {hi}"
    n_src = fresh["source"].nunique() if len(fresh) else 0
    console.print(f"[green]headlines: {len(fresh):,}[/green] → {paths.headlines}")
    console.print(f"  span   : {span}  |  sources: {n_src}")


@study_app.command("candles")
def study_candles(
    period: str = typer.Option("daily", help="bar granularity: daily|hourly|minute"),
    limit: int | None = typer.Option(None, help="cap markets (dev/smoke)"),
    group_size: int = typer.Option(100, help="tickers per candles_batch request"),
) -> None:
    """Pull bid/ask OHLC for the backtestable markets → data/candles."""
    from pmlab.mentions.registry import load_registry
    from pmlab.study.candles import (
        PERIOD_ALIASES,
        coverage_stats,
        load_candles,
        merge_candles,
        persist_candles,
        pull_candles,
    )

    _setup_logging()
    s = get_settings()
    paths = _Paths(s)
    if not paths.registry.exists():
        console.print(f"[red]missing {paths.registry}; run `pmlab mentions registry` first[/red]")
        raise typer.Exit(1)
    if period not in PERIOD_ALIASES:
        console.print(f"[red]period must be one of {list(PERIOD_ALIASES)}[/red]")
        raise typer.Exit(1)

    registry = load_registry(paths.registry)
    kalshi = KalshiPublic(settings=s)
    console.rule("[bold]Pull candle history[/bold]")
    fresh = pull_candles(
        kalshi, registry, period_minutes=PERIOD_ALIASES[period], limit=limit, group_size=group_size
    )
    existing = load_candles(paths.candles)
    merged = merge_candles(existing, fresh) if len(existing) else fresh
    persist_candles(merged, paths.candles)

    st = coverage_stats(merged, registry)
    console.print(f"[green]candles: {st['n_bars']:,} bars[/green] "
                  f"({len(fresh):,} this run) → {paths.candles}")
    console.print(f"  covered    : {st['n_covered']:,}/{st['n_target']:,} backtestable markets "
                  f"({st['n_missing']:,} without bars)")


@study_app.command("backtest")
def study_backtest(
    predictions: Path | None = typer.Option(None, help=_PREDICTIONS_HELP),  # noqa: B008 - typer.Option is immutable; ruff's Path-typed heuristic false-positives here
    theta: float = typer.Option(0.05, help="min |model p − market price| to trade"),
    band_lo: float = typer.Option(0.15, help="lower price band"),
    band_hi: float = typer.Option(0.60, help="upper price band"),
    model_col: str = typer.Option("p_cal", help="prediction column to trade against market price"),
    trade_split: str = typer.Option("test", help="split to trade (test|train|all)"),
    max_staleness_days: float = typer.Option(7.0, help="max age of the decision-time quote"),
) -> None:
    """Run the maker-side, strict-fill backtest over precomputed predictions → data/study."""
    from pmlab.mentions.registry import load_registry
    from pmlab.study.backtest import BacktestParams, persist_backtest, run_backtest
    from pmlab.study.candles import load_candles

    _setup_logging()
    s = get_settings()
    paths = _Paths(s)
    pred_path = predictions or paths.predictions
    for label, path in [("predictions", pred_path), ("candles", paths.candles)]:
        if not path.exists():
            console.print(f"[red]missing {label} at {path}[/red]")
            raise typer.Exit(1)

    preds = pd.read_parquet(pred_path)
    candles = load_candles(paths.candles)
    registry = load_registry(paths.registry)
    fee_types = _fee_types(KalshiPublic(settings=s))
    params = BacktestParams(
        band_lo=band_lo, band_hi=band_hi, theta=theta, model_col=model_col,
        max_staleness_days=max_staleness_days,
        trade_split=None if trade_split == "all" else trade_split,
    )

    console.rule("[bold]Backtest[/bold]")
    bt = run_backtest(preds, candles, registry, params=params, fee_types=fee_types)
    persist_backtest(bt, paths.backtest)

    n_test_priced = int(((bt["split"] == "test") & bt["market_prob"].notna()).sum())
    console.print(f"[green]backtest: {len(bt):,} markets[/green] → {paths.backtest}")
    console.print(f"  priced test={n_test_priced:,}  traded={int(bt['traded'].sum()):,}  "
                  f"filled={int(bt['filled'].sum()):,}")


def _fee_types(kalshi: KalshiPublic) -> dict[str, str] | None:
    """``series_ticker → fee_type`` for maker fees; ``None`` (conservative) if unreachable."""
    try:
        series = kalshi.list_series(category="Mentions")
        return {s["ticker"]: str(s.get("fee_type", "")) for s in series if s.get("ticker")}
    except Exception as exc:  # noqa: BLE001 - fee map is best-effort; None charges the fee everywhere
        logging.getLogger("pmlab.study").warning("fee_type map unavailable (%s); charging maker "
                                                 "fee on every series (conservative)", exc)
        return None


def main() -> None:
    app()


if __name__ == "__main__":
    main()
