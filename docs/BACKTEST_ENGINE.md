# Native backtest engine (`cpp/`)

A C++20 reimplementation of the backtest hot path: the maker-side strict-fill simulator
(`pmlab.study.backtest.run_backtest`) and the tick-to-bar replay (`pmlab.probe.bars`). The
Python implementations remain the reference; the engine's definition of correct is **byte
identity with them**, enforced in CI.

## Why it exists, honestly

Ranked by actual value:

1. **A low-latency C++ artifact.** The engine is built and measured like a latency-sensitive
   system: flat cache-friendly column layouts, an allocation-free hot loop enforced by a
   counting `operator new` test, sanitizers in CI, exact-percentile latency reporting.
2. **A real speedup for the research loop.** A single backtest pass was never the problem
   (about 1.3 s in Python at full 15k-market scale). Parameter sweeps and occasion-block
   bootstraps are: Python re-pays the full cost per point, so a 200-point sweep costs minutes
   and a grid x bootstrap costs hours. The engine parses the corpus once and evaluates a point
   in under a millisecond, which turns that loop into seconds.
3. **A single-pass speedup that is real but irrelevant** (about 1500x on the hot loop; the
   end-to-end native path is only about 1.1x because pandas preparation dominates it, stated
   plainly in the numbers below).

## Architecture

```
src/pmlab/study/engine.py     pandas <-> flat arrays; reuses run_backtest's own helpers
        |                     (_meta, _bars_by_ticker, _has_maker_fee) so prep cannot drift
        v
cpp/binding/module.cpp        pmlab._engine: zero-copy numpy views, GIL released
        v
cpp/include/pmlab_engine/     pure C++20 static library, no Python anywhere:
  mechanics.hpp                 maker_fee / maker_quote / settle
  pyround.hpp                   CPython round(x, 2), bit for bit (see below)
  engine.hpp                    the per-market hot loop over SoA bar columns
  bars.hpp                      bucket_end / BarBuilder / assemble_bars (tick replay)
  tape.hpp                      PMTAPE01 mmap tick tape (24-byte packed records)
  corpus_io.hpp                 PMCORP01 binary corpus + results (engine_cli, benchmarks)
```

The extension is optional by design: `uv sync` never needs a compiler, `pmlab.study.backtest`
keeps working with zero C++ installed, and the engine-marked tests skip when the module is
absent. Strategy parameters stay out of the C++ code entirely; every knob arrives as an
argument, the same public/private line the rest of the repo draws.

## Parity: the acceptance test

`tests/test_engine_parity.py` and `tests/test_engine_bars_parity.py` run the Python reference
and the engine on identical inputs and require bit-level equality: float64 columns compare as
uint64 bit patterns (NaNs must pair positionally), everything else exactly, dtypes included.
Corpora are seeded and adversarial (`tests/parity_corpus.py`): NaN/crossed/locked quotes, stale
and boundary-exact decision times, NaN prints, duplicate registry rows, markets missing from the
registry, fee series known/unknown/fee-free. When a real `data/` corpus or recorded tapes exist
locally, the same gates run over those too.

The one genuinely hard piece is Python's `round(x, 2)`, which correctly rounds the exact binary
value with ties to even. `pyround2` does it with integer arithmetic (the double's mantissa times
100 over a power of two, exact in a u64), falling back to CPython's own snprintf method for
values above 2^53. A generated golden suite replays ~150k CPython-computed cases bit for bit,
covering the cent grid, exact dyadic ties like 0.125, and wide random fuzz.

## Build

```bash
# C++ library + tests (no Python needed)
cmake -S cpp -B cpp/build/release -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build/release --parallel && ctest --test-dir cpp/build/release

# the pybind11 module (drops pmlab._engine into src/pmlab/)
uv sync
cmake -S cpp -B cpp/build/python -DCMAKE_BUILD_TYPE=Release -DPMLAB_ENGINE_PYTHON=ON \
      -Dpybind11_DIR=$(uv run python -m pybind11 --cmakedir) \
      -DPython_EXECUTABLE=$(uv run python -c 'import sys; print(sys.executable)')
cmake --build cpp/build/python --parallel

uv run pytest -m engine      # the parity gates
```

Sanitizer preset: `cmake --preset sanitize` (on macOS 26 with Apple clang 15 use
`--preset sanitize-brew-llvm` and run ctest with `MallocNanoZone=0`; Apple's bundled ASan
runtime predates the OS and aborts at startup). CI runs {ubuntu, macos} x {Release, ASan/UBSan}
plus the parity gate on every push.

## Numbers

Measured 2026-07-10, Apple M3 Max, macOS 26.5.1, Apple clang 15 `-O3`-equivalent Release,
single thread. Corpus: seeded synthetic, 15,000 markets, 2.83M daily bars
(`scripts/bench_engine.py`, seed 42); reproduce with:

```bash
uv run python scripts/bench_engine.py --markets 15000 --bars 400 --corpus-out corpus.bin
cpp/build/release/engine_bench backtest corpus.bin 500
cpp/build/release/engine_bench ticks 10000000
```

| measurement | Python | C++ engine | speedup |
| --- | ---: | ---: | ---: |
| full backtest pass, end to end | 1271 ms | 1190 ms | 1.1x |
| backtest hot loop only | 1271 ms | 0.78 ms (19.3M markets/s) | ~1500x |
| 200-point parameter sweep | ~254 s (est: 200 x single pass) | 0.15 s | ~1700x |
| tick replay, 1M ticks (from Python) | 3175 ms (315k ticks/s) | 97 ms (10.4M ticks/s) | 33x |
| tick replay, 10M ticks (pure C++) | | 0.32 s (30.8M ticks/s) | |

Per-event latency, exact percentiles over all events (not a lossy histogram), single events
timed individually with `steady_clock`:

| path | mean | p50 | p90 | p99 | p99.9 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| backtest decision (search + gates + quote + fill scan + settle) | 92 ns | 83 ns | 167 ns | 209 ns | 291 ns | 3.1 us |
| BarBuilder::add per tick | 18 ns | <42 ns | 42 ns | 42 ns | 125 ns | 78 us |

Caveats, stated up front:

- `steady_clock` granularity on Apple Silicon is ~42 ns; per-tick samples are quantized to it
  (a p50 of "0" means faster than the clock can see, and the tick p99.9/max tail is dominated
  by minute-bucket creation and timer noise, not steady-state work).
- The end-to-end native path is only ~1.1x because pandas preparation (groupby + array
  extraction) dominates; that prep is identical for both paths by construction (it reuses the
  reference implementation's own helpers). The engine's win is amortizing it: prep once, then
  each additional evaluation costs ~0.8 ms instead of ~1.3 s.
- The Python sweep cost is estimated as 200 x the measured single pass; the current Python API
  has no way to amortize preparation across points, which is the point being measured.
- Synthetic corpus, single thread, one machine. Numbers are for shape, not for marketing.

## Profiling story (what actually got optimized)

`sample` on `engine_cli` showed a third of the hot loop inside `snprintf`/`__dtoa` and its
locale locks, all from the string-based `pyround2`. Replacing it with the exact integer
algorithm took the hot loop from 1617 us to 776 us. The remaining profile is the strict-fill
linear scan over later bars, which is the semantic core (first trade-through wins, so scan
length is data-dependent) and memory-bound; nothing else is worth touching until the parity
gates say otherwise. The allocation test (`cpp/tests/test_allocations.cpp`) pins the "no
allocations in the hot loop" claim so it cannot regress silently.
