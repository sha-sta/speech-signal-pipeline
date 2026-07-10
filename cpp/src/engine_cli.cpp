// Standalone driver: run the engine over a binary corpus file with zero Python in the loop.
// Used by the benchmark scripts and for exercising the library directly.
//
//   engine_cli <corpus.bin> <results.bin>
#include <chrono>
#include <cstdio>
#include <exception>

#include "pmlab_engine/corpus_io.hpp"
#include "pmlab_engine/engine.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <corpus.bin> <results.bin>\n", argv[0]);
    return 2;
  }
  try {
    const pmlab::Corpus corpus = pmlab::read_corpus(argv[1]);
    pmlab::Results results(corpus.n_predictions());
    const auto outputs = results.outputs_view();

    const auto t0 = std::chrono::steady_clock::now();
    pmlab::run_backtest(corpus.bars_view(), corpus.predictions_view(), corpus.params, outputs);
    const auto t1 = std::chrono::steady_clock::now();

    pmlab::write_results(results, argv[2]);

    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const auto n = corpus.n_predictions();
    std::printf("engine: %zu predictions, %zu bars -> %.0f us (%.2fM predictions/s)\n", n,
                corpus.bar_ts.size(), us, n / us);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "engine_cli: %s\n", e.what());
    return 1;
  }
}
