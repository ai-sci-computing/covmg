// Measure-only diagnostic: per-level
// two-grid asymptotic convergence factors for the paper's seeded-ring
// connection (App. A "Seeded ring") vs the uniform-flux torus control.
//
// For each hierarchy level l (levels built by the library's own
// buildGaugeLevels, i.e. the level-l rediscretized operator with its own
// covariant transfer to level l+1), we measure the asymptotic per-cycle
// residual-reduction factor of the two-grid cycle at that level:
//   nu1 = nu2 = 2 red-black GS sweeps, omega = 1, alpha line search on the
//   coarse correction -- exactly the library cycle (paper App. A "Multigrid
//   constants") -- with the coarse problem solved either by
//     (a) 30 GS sweeps (the paper's coarsest-level constant), via the
//         library's own single-level vcycleSolve branch, or
//     (b) "exact-ish": unpreconditioned CG to relative residual 1e-12.
// Column (a) is measured two ways as a cross-check: through the library's
// vcycleSolve on the 2-entry level slice {levels[l], levels[l+1]} (fully
// library code), and through this file's driver (which must agree; it does
// to ~1e-3, differing only in fp summation order and the residual it logs).
// We also report rho of the full V-cycle started at level l (the slice
// levels[l..end]), again pure library code.
//
// rho = geometric mean of the last `tail` per-cycle relative-residual
// ratios on a fixed random RHS from a zero start (the convention of
// tools/convergence_factor_bench and scripts/frelaxation_twogrid.py).
//
// Ring: vortex ring R=0.7, Gamma=1, core 0.15, centered in [-0.8,0.8]^3,
// h=1.6/n, theta = u h / hbar at hbar = Gamma/2pi, homogeneous Neumann --
// construction copied verbatim from tools/ring_gap.cpp / eig_compare.cpp.
// Torus: periodic n^3, nPhi=4 quanta, Landau/seam gauge -- copied verbatim
// from tools/convergence_factor_bench.cpp.
//
// Usage: perlevel_twogrid ring N [N...] | perlevel_twogrid torus N [N...]

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "extraction/MacConnectionLaplacian.h"
#include "fluid/MacVortexRing.h"
#include "grid/MacGrid.h"
#include "solvers/GaugeMultigrid.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

// --- connections (copied from the repo's own generators; do not edit) ------

GaugeLattice ringLattice(int n) {  // tools/ring_gap.cpp, tools/eig_compare.cpp
  const MacGrid g(n, n, n, 1.6 / n, Vec3{-0.8, -0.8, -0.8});
  const double R = 0.7, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
  const auto u = vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, 0.15);
  const auto theta = connectionAngles(g, u, hbar);
  return gaugeLatticeFromFaces(g, theta);
}

GaugeLattice torusLattice(int n, int nPhi) {  // tools/convergence_factor_bench.cpp
  const double phi_p = 2.0 * M_PI * nPhi / (static_cast<double>(n) * n);
  const std::size_t N = static_cast<std::size_t>(n) * n * n;
  std::vector<double> lkx(N, 0.0), lky(N, 0.0), lkz(N, 0.0);
  const auto idx = [&](int i, int j, int k) { return (i * n + j) * n + k; };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        lkx[idx(i, j, k)] = -phi_p * j;
        if (j == n - 1) lky[idx(i, j, k)] = 2.0 * M_PI * nPhi * i / static_cast<double>(n);
      }
  return gaugeLatticePeriodic(n, n, n, static_cast<double>(n) * n, lkx, lky, lkz);
}

// --- helpers ---------------------------------------------------------------

double l2(const std::vector<cd>& v) {
  double s = 0.0;
  for (const cd& z : v) s += std::norm(z);
  return std::sqrt(s);
}

unsigned long long rhsSeed() {  // RF_SEED env override for seed-robustness checks
  const char* s = std::getenv("RF_SEED");
  return s ? std::strtoull(s, nullptr, 10) : 4321ull;
}

std::vector<cd> randomRhs(std::size_t n, unsigned long long seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<cd> b(n);
  for (auto& z : b) z = cd(g(rng), g(rng));
  return b;
}

// nu sweeps of the LIBRARY's red-black GS at level A: a single-entry level
// vector sends vcycleSolve down its coarsest-level branch, which runs exactly
// opts.coarseSweeps sweeps of smooth() -- so no smoother is reimplemented here.
void librarySweeps(const std::vector<GaugeLattice>& single, const std::vector<cd>& b,
                   std::vector<cd>& x, int nu) {
  MgOptions o;
  o.maxCycles = 1;
  o.tol = -1.0;  // fixed-cycle mode: run the sweeps, skip the residual report
  o.coarseSweeps = nu;
  vcycleSolve(single, b, x, o);
}

double tailRho(const std::vector<double>& res, int tail) {
  const int m = static_cast<int>(res.size());
  const int use = std::min(tail, m - 1);
  if (use < 1) return 0.0;
  double logsum = 0.0;
  for (int i = m - use; i < m; ++i) logsum += std::log(res[i] / res[i - 1]);
  return std::exp(logsum / use);
}

// (1) Pure-library factor on an arbitrary slice of the hierarchy: drive
// vcycleSolve one cycle at a time on a fixed random RHS (zero start),
// geometric-mean the last `tail` residual ratios. Same protocol as
// tools/convergence_factor_bench.cpp::asymptoticRho, but on a level slice and
// with tol > 0 so the library computes the relative residual each cycle.
double sliceRho(const std::vector<GaugeLattice>& slice, int maxCycles, int tail,
                double* firstCycle = nullptr) {
  std::vector<cd> b = randomRhs(static_cast<std::size_t>(slice.front().numNodes()), rhsSeed());
  std::vector<cd> x(b.size(), cd(0, 0));
  MgOptions one;
  one.maxCycles = 1;
  one.tol = 1e-300;  // > 0: residual is computed, but never stops a cycle early
  std::vector<double> res;
  res.reserve(maxCycles);
  for (int c = 0; c < maxCycles; ++c) {
    const MgResult r = vcycleSolve(slice, b, x, one);
    res.push_back(r.relResidual);
    if (r.relResidual < 1e-12) break;  // floor: stop before ratios saturate
  }
  if (firstCycle) *firstCycle = res.empty() ? 0.0 : res.front();
  return tailRho(res, tail);
}

// (2) This file's two-grid driver, library pieces only (librarySweeps /
// applyConnectionLaplacian / restrictGauge / prolongGauge / cgSolve), with a
// pluggable coarse solve. coarse solve: gsSweeps > 0 -> that many GS sweeps
// (30 = the paper's coarsest-level constant; must then match sliceRho on the
// 2-level slice); gsSweeps <= 0 -> unpreconditioned CG to rel. 1e-12.
double twogridRho(const GaugeLattice& fine, const GaugeLattice& coarse, int gsSweeps,
                  int maxCycles, int tail) {
  const std::vector<GaugeLattice> fineOnly{fine};
  const std::vector<GaugeLattice> coarseOnly{coarse};
  std::vector<cd> b = randomRhs(static_cast<std::size_t>(fine.numNodes()), rhsSeed());
  std::vector<cd> x(b.size(), cd(0, 0));
  const double bnorm = l2(b);
  std::vector<double> res;
  res.reserve(maxCycles);
  for (int c = 0; c < maxCycles; ++c) {
    librarySweeps(fineOnly, b, x, 2);  // nu1 = 2 (paper constant)
    std::vector<cd> Ax = applyConnectionLaplacian(fine, x);
    std::vector<cd> r(b.size());
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - Ax[i];
    const std::vector<cd> rc = restrictGauge(fine, r);
    std::vector<cd> ec(static_cast<std::size_t>(coarse.numNodes()), cd(0, 0));
    if (gsSweeps > 0)
      librarySweeps(coarseOnly, rc, ec, gsSweeps);
    else
      cgSolve(coarse, rc, ec, 1e-12, 200000);
    const std::vector<cd> pe = prolongGauge(fine, ec);
    // alpha line search, exactly as GaugeMultigrid.cpp::vcycle
    const std::vector<cd> Ape = applyConnectionLaplacian(fine, pe);
    cd num(0.0, 0.0);
    double den = 0.0;
    for (std::size_t i = 0; i < pe.size(); ++i) {
      num += std::conj(pe[i]) * r[i];
      den += std::real(std::conj(pe[i]) * Ape[i]);
    }
    const double alpha = den > 0.0 ? num.real() / den : 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += alpha * pe[i];
    librarySweeps(fineOnly, b, x, 2);  // nu2 = 2
    Ax = applyConnectionLaplacian(fine, x);
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - Ax[i];
    res.push_back(l2(r) / bnorm);
    if (res.back() < 1e-12) break;
  }
  return tailRho(res, tail);
}

void runFamily(const char* tag, const GaugeLattice& lat, int maxCycles, int tail) {
  const std::vector<GaugeLattice> levels = buildGaugeLevels(lat);
  const int L = static_cast<int>(levels.size());
  std::printf("\n%s: %d levels [", tag, L);
  for (int l = 0; l < L; ++l)
    std::printf("%s%dx%dx%d", l ? " " : "", levels[l].lx, levels[l].ly, levels[l].lz);
  std::printf("]\n");
  std::printf("  %-3s %-11s %-11s | %-12s %-12s %-12s | %-10s %-10s\n", "l", "fine", "coarse",
              "TG(30gs)lib", "TG(30gs)drv", "TG(exact)", "V-from-l", "V-cycle1");
  for (int l = 0; l + 1 < L; ++l) {
    const std::vector<GaugeLattice> two{levels[l], levels[l + 1]};
    const std::vector<GaugeLattice> down(levels.begin() + l, levels.end());
    const double tgLib = sliceRho(two, maxCycles, tail);
    const double tgDrv = twogridRho(levels[l], levels[l + 1], 30, maxCycles, tail);
    const double tgEx = twogridRho(levels[l], levels[l + 1], 0, maxCycles, tail);
    double v1 = 0.0;  // first-cycle reduction from zero start: the quality of ONE
                      // V-cycle application, i.e. what the LOBPCG preconditioner is
    const double vFrom = sliceRho(down, maxCycles, tail, &v1);
    char fdim[32], cdim[32];
    std::snprintf(fdim, sizeof fdim, "%dx%dx%d", levels[l].lx, levels[l].ly, levels[l].lz);
    std::snprintf(cdim, sizeof cdim, "%dx%dx%d", levels[l + 1].lx, levels[l + 1].ly,
                  levels[l + 1].lz);
    std::printf("  %-3d %-11s %-11s | %-12.4f %-12.4f %-12.4f | %-10.4f %-10.4f\n", l, fdim,
                cdim, tgLib, tgDrv, tgEx, vFrom, v1);
    std::fflush(stdout);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s {ring|torus} n [n...]\n", argv[0]);
    return 1;
  }
  const std::string family = argv[1];
  const int maxCycles = 200, tail = 10;
  std::printf("per-level two-grid asymptotic factors (nu1=nu2=2 RB-GS, omega=1, alpha line\n"
              "search; coarse solve = 30 GS sweeps [paper constant] or CG to 1e-12 [exact]);\n"
              "rho = geo-mean of last %d per-cycle relative-residual ratios, max %d cycles.\n",
              tail, maxCycles);
  for (int a = 2; a < argc; ++a) {
    const int n = std::atoi(argv[a]);
    char tag[64];
    if (family == "ring") {
      std::snprintf(tag, sizeof tag, "seeded ring n=%d", n);
      runFamily(tag, ringLattice(n), maxCycles, tail);
    } else if (family == "torus") {
      std::snprintf(tag, sizeof tag, "uniform-flux torus n=%d nPhi=4", n);
      runFamily(tag, torusLattice(n, 4), maxCycles, tail);
    } else {
      std::fprintf(stderr, "unknown family '%s'\n", family.c_str());
      return 1;
    }
  }
  return 0;
}
