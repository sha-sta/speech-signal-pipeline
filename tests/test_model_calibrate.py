"""Calibration model (M3): scoring, time split, calibrators, topic layer, leakage.

The tests use synthetic data with known structure so the assertions are about *mechanism*
(monotonicity, out-of-sample improvement, no test-outcome leakage), not about the real market."""

from __future__ import annotations

import numpy as np
import pytest

from pmlab.model.calibrate import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier,
    log_loss,
    logit,
    make_calibrator,
    sigmoid,
    time_split,
)


def test_sigmoid_logit_inverse() -> None:
    p = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    assert np.allclose(sigmoid(logit(p)), p, atol=1e-6)


def test_brier_and_log_loss_known_values() -> None:
    y = np.array([1, 0, 1, 0])
    assert brier(np.array([1.0, 0.0, 1.0, 0.0]), y) == pytest.approx(0.0)
    assert brier(np.array([0.5, 0.5, 0.5, 0.5]), y) == pytest.approx(0.25)
    # perfect confident predictions → ~0 log loss; a coin flip → ln 2
    assert log_loss(np.array([0.5, 0.5, 0.5, 0.5]), y) == pytest.approx(np.log(2), abs=1e-9)


def test_brier_empty_is_nan() -> None:
    assert np.isnan(brier(np.array([]), np.array([])))


def test_time_split_fraction_and_order() -> None:
    ts = np.arange(100, dtype="float64")
    split = time_split(ts, train_frac=0.6)
    assert (split == "train").sum() == pytest.approx(60, abs=1)
    # chronological: the latest train time precedes the earliest test time (no ties here)
    assert ts[split == "train"].max() < ts[split == "test"].min()


def test_time_split_keeps_ties_together() -> None:
    ts = np.array([1, 1, 1, 1, 1, 2, 3, 4, 5, 6], dtype="float64")
    split = time_split(ts, train_frac=0.4)
    # all the ts==1 markets land on the same side of the split
    assert len(set(split[ts == 1])) == 1


def test_time_split_all_nan_is_train() -> None:
    split = time_split(np.array([np.nan, np.nan]), train_frac=0.6)
    assert list(split) == ["train", "train"]


def test_isotonic_is_monotone_and_improves_calibration() -> None:
    rng = np.random.default_rng(0)
    n = 6000
    true = rng.uniform(0.0, 1.0, n)
    raw = true**2  # monotone but badly miscalibrated (systematically too low)
    y = rng.binomial(1, true)
    split = time_split(np.arange(n), 0.6)
    tr, te = split == "train", split == "test"
    cal = IsotonicCalibrator.fit(raw[tr], y[tr])
    # monotone map
    grid = np.linspace(0, 1, 50)
    pred = cal.predict(grid)
    assert np.all(np.diff(pred) >= -1e-9)
    # out-of-sample Brier improves vs the raw score
    assert brier(cal.predict(raw[te]), y[te]) < brier(raw[te], y[te])


def test_platt_recovers_shrunk_logits() -> None:
    rng = np.random.default_rng(1)
    n = 6000
    true = rng.uniform(0.05, 0.95, n)
    raw = sigmoid(0.5 * logit(true))  # under-confident (logits shrunk toward 0)
    y = rng.binomial(1, true)
    split = time_split(np.arange(n), 0.6)
    tr, te = split == "train", split == "test"
    cal = PlattCalibrator.fit(raw[tr], y[tr])
    assert cal.a > 1.3  # recovers roughly the ~2x logit inflation
    assert brier(cal.predict(raw[te]), y[te]) < brier(raw[te], y[te])


def test_make_calibrator_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown calibration"):
        make_calibrator("bogus")


