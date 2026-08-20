// Chebyshev-LOBPCG sweep helper (survey round 2026-08-19): runs the zero-setup
// polynomial baseline -- the SAME LOBPCG loop with the degree-k Chebyshev
// semi-iteration for L^{-1} in the preconditioner slot -- over a degree x
// interval-ratio grid and reports the fastest converged configuration. Shared
// by the family drivers so every table's polynomial column follows one
// protocol (Appendix A). BOCHNER_CHEB_SWEEP unset = no sweep; "1" = default
// grid; "d:r,d:r,..." = explicit list.
#pragma once

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "BenchTiming.h"
#include "solvers/GaugeEigen.h"

namespace chebsweep {

struct Best {
  int deg = 0;
  double ratio = 0.0;
  double ms = 0.0;
  int its = 0;
  bool any = false;
};

inline std::vector<std::pair<int, double>> configs() {
  std::vector<std::pair<int, double>> out;
  const char* e = std::getenv("BOCHNER_CHEB_SWEEP");
  if (!e || !*e) return out;
  const std::string s(e);
  if (s.find(':') != std::string::npos) {
    for (std::size_t p = 0; p < s.size();) {
      std::size_t q = s.find(',', p);
      if (q == std::string::npos) q = s.size();
      const std::string tok = s.substr(p, q - p);
      p = q + 1;
      const std::size_t c = tok.find(':');
      if (c != std::string::npos)
        out.emplace_back(std::atoi(tok.substr(0, c).c_str()), std::atof(tok.substr(c + 1).c_str()));
    }
  } else {
    for (int deg : {8, 16, 32, 64, 96})
      for (double ratio : {30.0, 100.0, 300.0, 1000.0, 3000.0}) out.emplace_back(deg, ratio);
  }
  return out;
}

// solve(opts) must run the eigensolve with opts.chebDegree/chebRatio set and
// return a GaugeEigenResult; base is the options to copy (tol etc.).
// BOCHNER_CHEB_SWEEP_REPS (default 1): repetitions per sweep point (counts are
// deterministic; the sweep only locates the optimum). The best configuration
// is then re-measured at the full BENCH_REPS median, which is the printed row.
inline int sweepReps() {
  const char* e = std::getenv("BOCHNER_CHEB_SWEEP_REPS");
  return (e && *e) ? std::max(1, std::atoi(e)) : 1;
}

template <class Solve>
Best run(const char* label, const bochner::GaugeEigenOptions& base, double covMs, Solve&& solve) {
  Best best;
  const auto cfgs = configs();
  if (cfgs.empty()) return best;
  std::printf("  cheb-LOBPCG sweep [%s] (same loop, polynomial slot), tol %.0e, %d rep(s) per point:\n",
              label, base.tol, sweepReps());
  for (const auto& cfg : cfgs) {
    bochner::GaugeEigenOptions ce = base;
    ce.chebDegree = cfg.first;
    ce.chebRatio = cfg.second;
    bochner::GaugeEigenResult C;
    const double ms = benchstat::medianMs([&] { C = solve(ce); }, sweepReps());
    std::printf("    deg %3d ratio %6.0f | %12.6f | %4d%s | %9.1f ms  res=%.1e\n", cfg.first, cfg.second,
                C.eigenvalue, C.iterations, C.converged ? "" : "!", ms, C.residual);
    if (C.converged && (!best.any || ms < best.ms)) {
      best = {cfg.first, cfg.second, ms, C.iterations, true};
    }
  }
  if (best.any) {
    bochner::GaugeEigenOptions ce = base;
    ce.chebDegree = best.deg;
    ce.chebRatio = best.ratio;
    bochner::GaugeEigenResult C;
    best.ms = benchstat::medianMs([&] { C = solve(ce); });
    best.its = C.iterations;
    std::printf("    best converged: deg %d ratio %.0f  %.1f ms (%d its, median of %d)  covMG %.1f ms  ratio %.2fx\n",
                best.deg, best.ratio, best.ms, best.its, benchstat::reps(), covMs, best.ms / covMs);
  } else
    std::printf("    best converged: NONE within the iteration cap\n");
  std::fflush(stdout);
  return best;
}

}  // namespace chebsweep
