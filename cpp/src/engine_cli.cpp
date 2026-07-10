// Standalone driver: run the engine over a binary corpus file with zero Python in the loop.
// Used by the benchmark scripts and for exercising the library directly.
//
//   engine_cli <corpus.bin> <results.bin> [repeats]
//
// repeats > 1 reruns the hot loop (results are identical each pass) so profilers have a
// process worth attaching to and per-run medians are measurable.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

#include "pmlab_engine/corpus_io.hpp"
#include "pmlab_engine/engine.hpp"

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::fprintf(stderr, "usage: %s <corpus.bin> <results.bin> [repeats]\n", argv[0]);
    return 2;
  }
  const long repeats = argc == 4 ? std::strtol(argv[3], nullptr, 10) : 1;
  try {
    const pmlab::Corpus corpus = pmlab::read_corpus(argv[1]);
    pmlab::Results results(corpus.n_predictions());
    const auto outputs = results.outputs_view();

    std::vector<double> us(static_cast<std::size_t>(std::max(repeats, 1L)));
    for (auto& sample : us) {
      const auto t0 = std::chrono::steady_clock::now();
      pmlab::run_backtest(corpus.bars_view(), corpus.predictions_view(), corpus.params, outputs);
      const auto t1 = std::chrono::steady_clock::now();
      sample = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    std::sort(us.begin(), us.end());

    pmlab::write_results(results, argv[2]);

    const double median = us[us.size() / 2];
    const auto n = corpus.n_predictions();
    std::printf("engine: %zu predictions, %zu bars -> median %.0f us over %ld runs "
                "(%.2fM predictions/s)\n",
                n, corpus.bar_ts.size(), median, repeats, n / median);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "engine_cli: %s\n", e.what());
    return 1;
  }
}
