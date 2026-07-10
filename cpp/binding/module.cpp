// pybind11 binding: pmlab._engine. A thin, allocation-boring layer — numpy columns in, numpy
// columns out, GIL released around the hot loop. All semantics live in the pure C++ library;
// all pandas-isms live in src/pmlab/study/engine.py.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <span>
#include <stdexcept>

#include "pmlab_engine/engine.hpp"
#include "pmlab_engine/version.hpp"

namespace py = pybind11;

namespace {

template <typename T>
using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
std::span<const T> in_span(const Array<T>& a) {
  return {a.data(), static_cast<std::size_t>(a.size())};
}

template <typename T>
std::span<T> out_span(py::array_t<T>& a) {
  return {a.mutable_data(), static_cast<std::size_t>(a.size())};
}

py::dict run_backtest(Array<double> bar_ts, Array<double> bar_bid_close,
                      Array<double> bar_ask_close, Array<double> bar_price_low,
                      Array<double> bar_price_high, Array<std::int64_t> offsets,
                      Array<std::int64_t> counts, Array<double> decision_ts,
                      Array<double> model_p, Array<std::int32_t> y,
                      Array<std::int64_t> bar_group, Array<std::uint8_t> trade_ok,
                      Array<std::uint8_t> has_maker_fee, Array<std::uint8_t> risk_technical,
                      double band_lo, double band_hi, double theta, double tick,
                      double max_staleness_s, double contracts) {
  const auto n_bars = bar_ts.size();
  if (bar_bid_close.size() != n_bars || bar_ask_close.size() != n_bars ||
      bar_price_low.size() != n_bars || bar_price_high.size() != n_bars) {
    throw std::invalid_argument("bar columns must all have the same length");
  }
  if (offsets.size() != counts.size()) {
    throw std::invalid_argument("offsets and counts must have the same length");
  }
  const auto n = decision_ts.size();
  if (model_p.size() != n || y.size() != n || bar_group.size() != n || trade_ok.size() != n ||
      has_maker_fee.size() != n || risk_technical.size() != n) {
    throw std::invalid_argument("prediction columns must all have the same length");
  }

  const pmlab::BarsView bars{in_span(bar_ts),        in_span(bar_bid_close),
                             in_span(bar_ask_close), in_span(bar_price_low),
                             in_span(bar_price_high), in_span(offsets), in_span(counts)};
  const pmlab::PredictionsView preds{in_span(decision_ts), in_span(model_p),
                                     in_span(y),           in_span(bar_group),
                                     in_span(trade_ok),    in_span(has_maker_fee),
                                     in_span(risk_technical)};
  const pmlab::BacktestParams params{band_lo, band_hi, theta, tick, max_staleness_s, contracts};

  py::array_t<double> market_prob(n), quote_price(n), fill_ts(n), fill_price(n), fee(n),
      contracts_out(n), stake(n), payout(n), pnl(n);
  py::array_t<std::int8_t> side(n);
  py::array_t<std::uint8_t> traded(n), filled(n), decided(n);
  const pmlab::BacktestOutputs out{
      out_span(market_prob), out_span(quote_price), out_span(fill_ts), out_span(fill_price),
      out_span(fee),         out_span(contracts_out), out_span(stake), out_span(payout),
      out_span(pnl),         out_span(side),        out_span(traded),  out_span(filled),
      out_span(decided)};

  {
    py::gil_scoped_release release;
    pmlab::run_backtest(bars, preds, params, out);
  }

  py::dict d;
  d["market_prob"] = market_prob;
  d["quote_price"] = quote_price;
  d["fill_ts"] = fill_ts;
  d["fill_price"] = fill_price;
  d["fee"] = fee;
  d["contracts"] = contracts_out;
  d["stake"] = stake;
  d["payout"] = payout;
  d["pnl"] = pnl;
  d["side"] = side;
  d["traded"] = traded;
  d["filled"] = filled;
  d["decided_by_technicality"] = decided;
  return d;
}

}  // namespace

PYBIND11_MODULE(_engine, m) {
  m.doc() = "Native C++ backtest engine (see cpp/); semantics identical to pmlab.study.backtest";
  m.def("version", &pmlab::engine_version);
  m.def("run_backtest", &run_backtest, py::arg("bar_ts"), py::arg("bar_bid_close"),
        py::arg("bar_ask_close"), py::arg("bar_price_low"), py::arg("bar_price_high"),
        py::arg("offsets"), py::arg("counts"), py::arg("decision_ts"), py::arg("model_p"),
        py::arg("y"), py::arg("bar_group"), py::arg("trade_ok"), py::arg("has_maker_fee"),
        py::arg("risk_technical"), py::arg("band_lo"), py::arg("band_hi"), py::arg("theta"),
        py::arg("tick"), py::arg("max_staleness_s"), py::arg("contracts"));
}
