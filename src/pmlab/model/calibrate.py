"""Probability model + calibration + scoring (§6, D8, D9).

The pipeline is **features → raw probability → time-split calibration → calibrated probability**,
scored by Brier / log-loss. Two raw models make the §0.3 control concrete:

* the **baseline** is the recency-decayed raw frequency itself (``base_rate_market`` — already a
  P(yes) estimate; ``base_rate_corpus`` where a transcript corpus exists), the published,
  already-priced signal;
* the **topic model** is a logistic combination of that baseline with the news-salience feature —
  the claimed uncrowded edge, which must beat the baseline *out-of-sample* or the study reports
  that it doesn't.

Calibration (isotonic or Platt) is fit on the **training** split only and applied to the held-out
split; nothing here reads a test-split outcome (trap #3, #7). The leakage boundary of the features
themselves was established in M2; this module adds the calibration-fit boundary on top.

No sklearn (it is only a transitive dependency, not declared): isotonic uses
``scipy.optimize.isotonic_regression`` and the logistic fits are a small L2-regularised L-BFGS on
the declared scipy stack — fully deterministic, so the calibration is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression, minimize

_EPS = 1e-6
CLIP = 1e-3  # keep probabilities off {0,1} so logit/log-loss stay finite

# Contract for prediction frames consumed by pmlab.study.backtest.run_backtest: one row per
# market with a chronological train/test split, the 0/1 outcome, and model probabilities.
PREDICTION_COLUMNS = ["market_ticker", "decision_ts", "split", "y", "p_base", "p_topic", "p_cal"]


# --- scalar helpers ------------------------------------------------------------------------------


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))), dtype="float64")


def logit(p: np.ndarray | float, eps: float = _EPS) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype="float64"), eps, 1.0 - eps)
    return np.asarray(np.log(q / (1.0 - q)), dtype="float64")


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of the probability against the 0/1 outcome (lower is better)."""
    p = np.clip(np.asarray(p, dtype="float64"), 0.0, 1.0)
    y = np.asarray(y, dtype="float64")
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1.0 - eps)
    y = np.asarray(y, dtype="float64")
    if p.size == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# --- time split ----------------------------------------------------------------------------------


def time_split(decision_ts: pd.Series | np.ndarray, train_frac: float = 0.6) -> np.ndarray:
    """Chronological ``train``/``test`` labels: the earliest ``train_frac`` of decision times train,
    the rest test (D9 held-out time split). Markets sharing a decision timestamp stay on the same
    side (a threshold split, not an index split) so an occasion doesn't straddle the boundary."""
    ts = pd.to_numeric(pd.Series(decision_ts), errors="coerce").to_numpy(dtype="float64")
    finite = ts[np.isfinite(ts)]
    n = len(ts)
    if finite.size == 0:
        return np.array(["train"] * n, dtype=object)
    cutoff = float(np.quantile(finite, train_frac))
    split = np.where(ts <= cutoff, "train", "test").astype(object)
    # Degenerate (all one side because of heavy ties) → fall back to a strict index split.
    if (split == "train").all() or (split == "test").all():
        order = np.argsort(np.where(np.isnan(ts), np.inf, ts), kind="mergesort")
        k = max(1, min(n - 1, int(round(n * train_frac))))
        split = np.array(["test"] * n, dtype=object)
        split[order[:k]] = "train"
    return split


# --- logistic (shared by the topic model and Platt scaling) --------------------------------------


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """MLE for ``P(y=1)=sigmoid(x @ beta)`` via L-BFGS; column 0 is the (unpenalised) intercept.

    A small L2 on the slopes keeps the fit finite under separation / constant columns (e.g. an
    all-NaN salience feature), so the topic model degrades gracefully to the base model rather than
    blowing up."""
    y = np.asarray(y, dtype="float64")

    def nll(b: np.ndarray) -> float:
        p = sigmoid(x @ b)
        ll = y * np.log(np.clip(p, 1e-12, 1.0)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1.0))
        return float(-np.sum(ll) + l2 * np.sum(b[1:] ** 2))

    def grad(b: np.ndarray) -> np.ndarray:
        g: np.ndarray = x.T @ (sigmoid(x @ b) - y)
        g[1:] += 2.0 * l2 * b[1:]
        return g

    res = minimize(nll, np.zeros(x.shape[1]), jac=grad, method="L-BFGS-B")
    return np.asarray(res.x, dtype="float64")


# --- calibrators ---------------------------------------------------------------------------------


@dataclass
class IsotonicCalibrator:
    """Monotone (PAVA) calibration map, fit on train, linearly interpolated at new scores."""

    x: np.ndarray
    yhat: np.ndarray

    @classmethod
    def fit(cls, p_raw: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        p = np.asarray(p_raw, dtype="float64")
        y = np.asarray(y, dtype="float64")
        if p.size == 0:
            return cls(np.empty(0), np.empty(0))
        order = np.argsort(p, kind="mergesort")
        xs, ys = p[order], y[order]
        fitted = np.asarray(isotonic_regression(ys).x, dtype="float64")
        # Collapse ties in x (np.interp wants strictly increasing xp): average fitted per unique x.
        ux, start = np.unique(xs, return_index=True)
        counts = np.diff(np.append(start, len(fitted)))
        uy = np.add.reduceat(fitted, start) / counts
        return cls(ux, uy)

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        p: np.ndarray = np.clip(np.asarray(p_raw, dtype="float64"), 0.0, 1.0)
        if self.x.size == 0:
            return p
        return np.asarray(np.interp(p, self.x, self.yhat), dtype="float64")


@dataclass
class PlattCalibrator:
    """Platt scaling: a 1-parameter logistic ``sigmoid(b + a·logit(p_raw))`` fit on train."""

    a: float
    b: float

    @classmethod
    def fit(cls, p_raw: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        y = np.asarray(y, dtype="float64")
        if y.size == 0:
            return cls(a=1.0, b=0.0)
        x = np.column_stack([np.ones(len(y)), logit(p_raw)])
        beta = _fit_logistic(x, y, l2=0.0)
        return cls(a=float(beta[1]), b=float(beta[0]))

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        return sigmoid(self.b + self.a * logit(p_raw))


Calibrator = IsotonicCalibrator | PlattCalibrator


def make_calibrator(method: str) -> type[IsotonicCalibrator] | type[PlattCalibrator]:
    if method == "isotonic":
        return IsotonicCalibrator
    if method == "platt":
        return PlattCalibrator
    raise ValueError(f"unknown calibration method: {method!r} (use 'isotonic' or 'platt')")


def calibrate_series(
    raw: np.ndarray, y: np.ndarray, split: np.ndarray, *, method: str = "isotonic",
    clip: float = CLIP,
) -> np.ndarray:
    """Fit a calibrator on the train rows and apply it to *all* rows (the out-of-sample map).

    Exposed so the report can calibrate the baseline model the same way as the primary, keeping the
    §0.3 raw-frequency-vs-topic comparison a like-for-like (both calibrated) contest."""
    raw = np.asarray(raw, dtype="float64")
    tr = np.asarray(split) == "train"
    if not tr.any():
        return np.asarray(np.clip(raw, clip, 1.0 - clip), dtype="float64")
    cal = make_calibrator(method).fit(raw[tr], np.asarray(y)[tr])
    return np.asarray(np.clip(cal.predict(raw), clip, 1.0 - clip), dtype="float64")


# --- orchestration -------------------------------------------------------------------------------
