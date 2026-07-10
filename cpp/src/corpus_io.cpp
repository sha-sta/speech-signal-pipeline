#include "pmlab_engine/corpus_io.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace pmlab {
namespace {

constexpr char kCorpusMagic[8] = {'P', 'M', 'C', 'O', 'R', 'P', '0', '1'};
constexpr char kResultsMagic[8] = {'P', 'M', 'R', 'E', 'S', '0', '0', '1'};

[[noreturn]] void fail(const std::filesystem::path& path, const std::string& what) {
  throw std::runtime_error(path.string() + ": " + what);
}

template <typename T>
void read_array(std::ifstream& in, const std::filesystem::path& path, std::vector<T>& out,
                std::size_t n) {
  out.resize(n);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * sizeof(T)));
  if (!in) fail(path, "truncated file");
}

template <typename T>
void write_array(std::ofstream& out, const std::vector<T>& v) {
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(T)));
}

std::uint64_t read_u64(std::ifstream& in, const std::filesystem::path& path) {
  std::uint64_t v = 0;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (!in) fail(path, "truncated file");
  return v;
}

void write_u64(std::ofstream& out, std::uint64_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

}  // namespace

BarsView Corpus::bars_view() const {
  return BarsView{bar_ts, bar_bid_close, bar_ask_close, bar_price_low, bar_price_high,
                  offsets, counts};
}

PredictionsView Corpus::predictions_view() const {
  return PredictionsView{decision_ts, model_p, y, bar_group, trade_ok, has_maker_fee,
                         risk_technical};
}

Results::Results(std::size_t n)
    : market_prob(n), quote_price(n), fill_ts(n), fill_price(n), fee(n), contracts(n), stake(n),
      payout(n), pnl(n), side(n), traded(n), filled(n), decided_by_technicality(n) {}

BacktestOutputs Results::outputs_view() {
  return BacktestOutputs{market_prob, quote_price, fill_ts, fill_price, fee, contracts,
                         stake, payout, pnl, side, traded, filled, decided_by_technicality};
}

Corpus read_corpus(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(path, "cannot open");
  char magic[8];
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, kCorpusMagic, sizeof(magic)) != 0) fail(path, "bad corpus magic");

  const auto n_bars = read_u64(in, path);
  const auto n_groups = read_u64(in, path);
  const auto n_preds = read_u64(in, path);

  Corpus c;
  double p[6];
  in.read(reinterpret_cast<char*>(p), sizeof(p));
  if (!in) fail(path, "truncated params");
  c.params = BacktestParams{p[0], p[1], p[2], p[3], p[4], p[5]};

  read_array(in, path, c.bar_ts, n_bars);
  read_array(in, path, c.bar_bid_close, n_bars);
  read_array(in, path, c.bar_ask_close, n_bars);
  read_array(in, path, c.bar_price_low, n_bars);
  read_array(in, path, c.bar_price_high, n_bars);
  read_array(in, path, c.offsets, n_groups);
  read_array(in, path, c.counts, n_groups);
  read_array(in, path, c.decision_ts, n_preds);
  read_array(in, path, c.model_p, n_preds);
  read_array(in, path, c.y, n_preds);
  read_array(in, path, c.bar_group, n_preds);
  read_array(in, path, c.trade_ok, n_preds);
  read_array(in, path, c.has_maker_fee, n_preds);
  read_array(in, path, c.risk_technical, n_preds);
  return c;
}

void write_corpus(const Corpus& c, const std::filesystem::path& path) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error(path.string() + ": cannot open for write");
  out.write(kCorpusMagic, sizeof(kCorpusMagic));
  write_u64(out, c.bar_ts.size());
  write_u64(out, c.offsets.size());
  write_u64(out, c.decision_ts.size());
  const double p[6] = {c.params.band_lo, c.params.band_hi,        c.params.theta,
                       c.params.tick,    c.params.max_staleness_s, c.params.contracts};
  out.write(reinterpret_cast<const char*>(p), sizeof(p));
  write_array(out, c.bar_ts);
  write_array(out, c.bar_bid_close);
  write_array(out, c.bar_ask_close);
  write_array(out, c.bar_price_low);
  write_array(out, c.bar_price_high);
  write_array(out, c.offsets);
  write_array(out, c.counts);
  write_array(out, c.decision_ts);
  write_array(out, c.model_p);
  write_array(out, c.y);
  write_array(out, c.bar_group);
  write_array(out, c.trade_ok);
  write_array(out, c.has_maker_fee);
  write_array(out, c.risk_technical);
  if (!out) throw std::runtime_error(path.string() + ": write failed");
}

void write_results(const Results& r, const std::filesystem::path& path) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error(path.string() + ": cannot open for write");
  out.write(kResultsMagic, sizeof(kResultsMagic));
  write_u64(out, r.market_prob.size());
  write_array(out, r.market_prob);
  write_array(out, r.quote_price);
  write_array(out, r.fill_ts);
  write_array(out, r.fill_price);
  write_array(out, r.fee);
  write_array(out, r.contracts);
  write_array(out, r.stake);
  write_array(out, r.payout);
  write_array(out, r.pnl);
  write_array(out, r.side);
  write_array(out, r.traded);
  write_array(out, r.filled);
  write_array(out, r.decided_by_technicality);
  if (!out) throw std::runtime_error(path.string() + ": write failed");
}

Results read_results(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(path, "cannot open");
  char magic[8];
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, kResultsMagic, sizeof(magic)) != 0) fail(path, "bad results magic");
  const auto n = read_u64(in, path);
  Results r(n);
  read_array(in, path, r.market_prob, n);
  read_array(in, path, r.quote_price, n);
  read_array(in, path, r.fill_ts, n);
  read_array(in, path, r.fill_price, n);
  read_array(in, path, r.fee, n);
  read_array(in, path, r.contracts, n);
  read_array(in, path, r.stake, n);
  read_array(in, path, r.payout, n);
  read_array(in, path, r.pnl, n);
  read_array(in, path, r.side, n);
  read_array(in, path, r.traded, n);
  read_array(in, path, r.filled, n);
  read_array(in, path, r.decided_by_technicality, n);
  return r;
}

}  // namespace pmlab
