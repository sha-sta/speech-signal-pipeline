"""THE acceptance gate for the C++ engine: byte-identical output vs the Python reference.

Every test runs run_backtest (pure Python) and run_backtest_native (C++ hot loop) on identical
inputs and requires bit-level equality: float64 columns compare as uint64 bit patterns (NaNs must
pair up positionally), everything else compares exactly, dtypes included. Skips cleanly when the
extension isn't built (`pytest -m engine` to select just these).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmlab.study.backtest import BacktestParams, run_backtest

pytest.importorskip("pmlab._engine", reason="native engine not built (see cpp/)")

from parity_corpus import FEE_TYPES, gen_corpus  # noqa: E402
from pmlab.study.engine import run_backtest_native  # noqa: E402

pytestmark = pytest.mark.engine

# Each entry: (params, fee_types, contracts, expect_trades). Covers the defaults, the no-gate
# wide-open case, alternate model column/tick, a degenerate band (mid must be exactly 0.40, so
# many seeds legitimately trade nothing), and fee/contracts variants.
CASES: list[tuple[BacktestParams, dict[str, str] | None, float, bool]] = [
    (BacktestParams(), FEE_TYPES, 1.0, True),
    (BacktestParams(trade_split=None), None, 1.0, True),
    (
        BacktestParams(
            theta=0.0, band_lo=0.0, band_hi=1.0, max_staleness_days=365.0, trade_split=None
        ),
        FEE_TYPES,
        100.0,
        True,
    ),
    (
        BacktestParams(model_col="p_base", theta=0.12, tick=0.02, trade_split=None),
        None,
        250.5,
        True,
    ),
    (BacktestParams(band_lo=0.4, band_hi=0.4, trade_split=None), FEE_TYPES, 1.0, False),
]


def assert_bit_identical(py_out: pd.DataFrame, native_out: pd.DataFrame) -> None:
    assert list(py_out.columns) == list(native_out.columns)
    assert len(py_out) == len(native_out)
    for col in py_out.columns:
        a, b = py_out[col], native_out[col]
        assert str(a.dtype) == str(b.dtype), f"{col}: dtype {a.dtype} vs {b.dtype}"
        if a.dtype == np.float64:
            va, vb = a.to_numpy(), b.to_numpy()
            same = (va.view(np.uint64) == vb.view(np.uint64)) | (np.isnan(va) & np.isnan(vb))
            bad = np.flatnonzero(~same)
            assert bad.size == 0, (
                f"{col}: {bad.size} rows differ, first at {bad[0]}: "
                f"{va[bad[0]]!r} vs {vb[bad[0]]!r}"
            )
        else:
            pd.testing.assert_series_equal(a, b, check_exact=True, obj=col)


@pytest.mark.parametrize("case_idx", range(len(CASES)))
@pytest.mark.parametrize("seed", range(8))
def test_parity_synthetic(seed: int, case_idx: int) -> None:
    params, fee_types, contracts, expect_trades = CASES[case_idx]
    preds, candles, registry = gen_corpus(seed, n_markets=400)
    py_out = run_backtest(preds, candles, registry, params=params, fee_types=fee_types,
                          contracts=contracts)
    native_out = run_backtest_native(preds, candles, registry, params=params,
                                     fee_types=fee_types, contracts=contracts)
    assert_bit_identical(py_out, native_out)
    if expect_trades:
        assert py_out["traded"].any(), "corpus produced no trades; the parity run proved nothing"


def test_parity_at_scale() -> None:
    preds, candles, registry = gen_corpus(1234, n_markets=5000, max_bars=60)
    params = BacktestParams(trade_split=None, max_staleness_days=30.0)
    py_out = run_backtest(preds, candles, registry, params=params, fee_types=FEE_TYPES)
    native_out = run_backtest_native(preds, candles, registry, params=params,
                                     fee_types=FEE_TYPES)
    assert_bit_identical(py_out, native_out)
    assert int(py_out["filled"].sum()) > 100


def test_parity_empty_predictions() -> None:
    preds, candles, registry = gen_corpus(7, n_markets=10)
    empty = preds.iloc[0:0]
    py_out = run_backtest(empty, candles, registry)
    native_out = run_backtest_native(empty, candles, registry)
    assert_bit_identical(py_out, native_out)


def test_parity_empty_candles() -> None:
    preds, candles, registry = gen_corpus(11, n_markets=50)
    py_out = run_backtest(preds, candles.iloc[0:0], registry, params=BacktestParams())
    native_out = run_backtest_native(preds, candles.iloc[0:0], registry,
                                     params=BacktestParams())
    assert_bit_identical(py_out, native_out)


def test_parity_real_corpus_if_present() -> None:
    """When a real data/ corpus exists locally (private research runs), gate on it too."""
    from pmlab.config import get_settings

    s = get_settings()
    pred_path = s.data_dir / "study" / "predictions.parquet"
    candles_path = s.data_dir / "candles" / "candles.parquet"
    registry_path = s.data_dir / "universe" / "mention_registry.parquet"
    if not (pred_path.exists() and candles_path.exists() and registry_path.exists()):
        pytest.skip("no real corpus under data/; synthetic parity still enforced")
    preds = pd.read_parquet(pred_path)
    candles = pd.read_parquet(candles_path)
    registry = pd.read_parquet(registry_path)
    for params in (BacktestParams(), BacktestParams(trade_split=None, theta=0.02)):
        py_out = run_backtest(preds, candles, registry, params=params)
        native_out = run_backtest_native(preds, candles, registry, params=params)
        assert_bit_identical(py_out, native_out)
