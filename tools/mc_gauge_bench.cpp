// Monte-Carlo gauge-field benchmark for the paper.
//
// The existing SU(d) tables test two easy extremes: SMOOTH fields (flux -> 0
// under refinement) and i.i.d. HOT links (rough but uncorrelated). The missing
// hard regime is Monte-Carlo-sampled pure-gauge configurations at realistic
// coupling: ROUGH but CORRELATED. This tool samples 3D SU(3) (general d) Wilson
// configurations with the validated Cabibbo-Marinari / Kennedy-Pendleton
// heatbath (tests/test_mc_gauge.cpp) and runs the paper's full solver
// comparison on the massless covariant Laplacian (m^2 = 0, w = n^2, periodic
// n^3 -- the sun_gauge_bench conventions):
//
//   gauge-MG V-cycle to 1e-8   vs  unpreconditioned CG to 1e-8   (linear solve)
//   covMG-LOBPCG to 1e-7      vs  SLEPc Lanczos to 1e-7         (ground state)
//
// Three independent configurations (seeds) per point; per-config rows plus a
// median/range summary. Within each configuration every timed solve is the
// median of k identical runs (BenchTiming.h; the paper-wide statistic) -- the
// config median above is population variation, the per-solve median is timer
// noise. Thermalization: 300 heatbath sweeps (hot start for
// beta < 6, cold start for beta >= 6); <P> is the mean over the last 150
// sweeps, and the two quarters of that window are printed separately as the
// flatness diagnostic. Run single-threaded (OMP_NUM_THREADS=1) for the timings.
//
// Usage:  mc_gauge_bench [mode] [d]
//   mode = beta      : beta sweep at n=24, beta in {0,3,6,9,15,30}
//          refine    : refinement series at beta=6, n in {8,16,24,32}
//          refine48  : the n=48 point of the refinement series
//          traject   : matched continuum trajectory, (n, beta) jointly at
//                      n/beta = 8/3 (tree-level 3D matching a ~ 1/beta)
//          trajext   : trajectory extension, the (40,15) and (48,18) points
//          traj64    : single trajectory probe at (64,24)
//          all       : beta + refine   (default)
//   configs=N        : ensemble size per point (default 3)
//   d    = fiber dimension (default 3)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <array>
#include <random>
#include <string>
#include <vector>

#include "BenchTiming.h"
#include "solvers/SunGauge.h"
#include "solvers/WilsonMc.h"
#ifdef BOCHNER_WITH_PETSC
#include <slepc.h>

#include "grid/CooMatrix.h"
#include "solvers/EigenSolver.h"
#endif

using namespace bochner;
using cd = std::complex<double>;
using Clock = std::chrono::steady_clock;

namespace {

[[maybe_unused]] double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::vector<cd> randomRhs(long dof, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<cd> b(dof);
  for (auto& z : b) z = cd(g(rng), g(rng));
  return b;
}

#ifdef BOCHNER_WITH_PETSC
// Real-symmetric assembly of the SU(d) covariant Laplacian (complex->real 2x2
// embedding), for the production SLEPc Lanczos baseline -- identical to
// sun_gauge_bench's assembleReal.
CooMatrix assembleReal(const SunLattice& L) {
  const int d = L.d, dd = d * d;
  const long N = L.numNodes();
  const int M = static_cast<int>(2 * d * N);
  CooMatrix A(M, M);
  auto cplx = [&](long p, long q, cd v) {
    A.add(2 * p, 2 * q, v.real());
    A.add(2 * p, 2 * q + 1, -v.imag());
    A.add(2 * p + 1, 2 * q, v.imag());
    A.add(2 * p + 1, 2 * q + 1, v.real());
  };
  auto block = [&](long c, long n, const cd* U, bool adj) {
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        const cd t = adj ? std::conj(U[b * d + a]) : U[a * d + b];
        if (t != cd(0, 0)) cplx(c * d + a, n * d + b, -L.w * t);
      }
  };
  const int lx = L.lx, ly = L.ly, lz = L.lz;
  const double diag = L.w * 6.0 + L.mass2;
  for (int i = 0; i < lx; ++i)
    for (int j = 0; j < ly; ++j)
      for (int k = 0; k < lz; ++k) {
        const long c = L.index(i, j, k);
        for (int a = 0; a < d; ++a) cplx(c * d + a, c * d + a, cd(diag, 0));
        const int im = (i - 1 + lx) % lx, ip = (i + 1) % lx;
        const int jm = (j - 1 + ly) % ly, jp = (j + 1) % ly;
        const int km = (k - 1 + lz) % lz, kp = (k + 1) % lz;
        block(c, L.index(im, j, k), &L.ux[static_cast<size_t>(L.index(im, j, k)) * dd], false);
        block(c, L.index(ip, j, k), &L.ux[static_cast<size_t>(c) * dd], true);
        block(c, L.index(i, jm, k), &L.uy[static_cast<size_t>(L.index(i, jm, k)) * dd], false);
        block(c, L.index(i, jp, k), &L.uy[static_cast<size_t>(c) * dd], true);
        block(c, L.index(i, j, km), &L.uz[static_cast<size_t>(L.index(i, j, km)) * dd], false);
        block(c, L.index(i, j, kp), &L.uz[static_cast<size_t>(c) * dd], true);
      }
  return A;
}
#endif

struct PointResult {
  double plaq = 0.0, plaqQ3 = 0.0, plaqQ4 = 0.0;
  int mgCycles = 0;
  double mgMs = 0.0, mgRes = 0.0;
  int cgIters = 0;
  double cgMs = 0.0, cgRes = 0.0;
  int rqIters = 0;
  double rqMs = 0.0, rqLam = 0.0, rqRes = 0.0;
  bool rqConv = false;
  int lanIters = 0;
  double lanMs = 0.0, lanLam = 0.0;
  bool haveLan = false;
};

// One thermalized configuration + the full solver comparison.
bool g_relativeGsDrop = true;  // "absdrop" argv token selects the legacy absolute test

// "biasedstart" argv token: start covMG-LOBPCG from a constant-BIASED random
// vector (every real slot uniform on [0,1)) instead of the default psi = 1.
// That is the character of vector SLEPc's own default draw already has -- its
// measured constant-overlap is sqrt(3)/2 ~ 0.866 -- so it matches the baseline's
// start rather than handing either side domain knowledge. It also carries psi=1's
// warm-start information while breaking its symmetry, which on the coherent-flux
// operator was worth a factor of two (34 -> 17 outer iterations at SU(3) n=64).
// MC configurations are the roughest operators in the suite, so this is where
// the policy most plausibly moves the published verdict.
bool g_biasedStart = false;

std::vector<cd> constBiasedVec(long dof, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<cd> v(dof);
  for (auto& z : v) z = cd(u(rng), u(rng));
  return v;
}

PointResult runPoint(int d, int n, double beta, std::uint64_t seed, int thermSweeps) {
  PointResult r;
  std::vector<double> hist;
  const bool hotStart = beta < 6.0;  // either start thermalizes; pick the closer one
  const int sweeps = (beta == 0.0) ? 0 : thermSweeps;  // beta=0 = i.i.d. Haar (existing hot path)
  const SunLattice L = mcSunLattice(d, n, beta, sweeps, /*w=*/double(n) * n, /*mass2=*/0.0, seed,
                                    hotStart, &hist);
  if (sweeps > 0) {
    const int half = sweeps / 2, q = half / 2;
    double sQ3 = 0.0, sQ4 = 0.0;
    for (int s = half; s < half + q; ++s) sQ3 += hist[s];
    for (int s = half + q; s < sweeps; ++s) sQ4 += hist[s];
    r.plaqQ3 = sQ3 / q;
    r.plaqQ4 = sQ4 / (sweeps - half - q);
    double sP = 0.0;
    for (int s = half; s < sweeps; ++s) sP += hist[s];
    r.plaq = sP / (sweeps - half);
  } else {
    r.plaq = r.plaqQ3 = r.plaqQ4 = averagePlaquette(L);
  }

  const auto levels = buildSunLevels(L);
  const std::vector<cd> b = randomRhs(L.dof(), 12345 + seed);

  MgOptions mg;
  mg.tol = 1e-8;
  mg.maxCycles = 200;
  {
    MgResult mr;
    r.mgMs = benchstat::medianMs([&] {
      std::vector<cd> x(L.dof(), cd(0, 0));
      mr = vcycleSolveSun(levels, b, x, mg);
    });
    r.mgCycles = mr.cycles;
    r.mgRes = mr.relResidual;
  }
  {
    SolveStats cg;
    r.cgMs = benchstat::medianMs([&] {
      std::vector<cd> x(L.dof(), cd(0, 0));
      cg = cgSolveSun(L, b, x, 1e-8, 100000);
    });
    r.cgIters = cg.iterations;
    r.cgRes = cg.relResidual;
  }
  {
    GaugeEigenOptions eo;
    eo.tol = 1e-7;
    eo.maxIters = 300;
    eo.relativeGsDrop = g_relativeGsDrop;
    GaugeEigenResult er;
    const std::vector<cd> g0 = g_biasedStart ? constBiasedVec(L.dof(), 20260720) : std::vector<cd>();
    const std::vector<cd>* gp = g_biasedStart ? &g0 : nullptr;
    r.rqMs = benchstat::medianMs([&] { er = smallestEigenpairSunMG(L, gp, eo); });
    r.rqIters = er.iterations;
    r.rqLam = er.eigenvalue;
    r.rqRes = er.residual;
    r.rqConv = er.converged;
  }
#ifdef BOCHNER_WITH_PETSC
  {
    const CooMatrix A = assembleReal(L);
    EigenPair lp;
    r.lanMs = benchstat::medianMs([&] { lp = smallestEigenpairLanczos(A, 1e-7); });
    r.lanIters = lp.iterations;
    r.lanLam = lp.value;
    r.haveLan = true;
  }
#endif
  return r;
}

double median3(double a, double b, double c) { return std::max(std::min(a, b), std::min(std::max(a, b), c)); }
int median3i(int a, int b, int c) { return std::max(std::min(a, b), std::min(std::max(a, b), c)); }

void printHeader() {
  std::printf(
      "  %-5s %-8s %-8s  %-17s  %-20s  %-22s  %-18s\n", "seed", "<P>", "P(Q3|Q4)",
      "MG (cyc, ms)", "CG (its, ms)", "covMG-LOBPCG (out, ms, lam)", "SLEPc (its, ms)");
}

void printRow(std::uint64_t seed, const PointResult& r) {
  std::printf("  %-5llu %-8.5f %.4f|%.4f  %3d (%9.1f)  %6d (%10.1f)  %3d%s (%9.1f, %.4g)",
              static_cast<unsigned long long>(seed), r.plaq, r.plaqQ3, r.plaqQ4, r.mgCycles, r.mgMs,
              r.cgIters, r.cgMs, r.rqIters, r.rqConv ? "" : "!", r.rqMs, r.rqLam);
  if (r.haveLan)
    std::printf("  %5d (%9.1f)  |dlam|=%.1e", r.lanIters, r.lanMs, std::abs(r.rqLam - r.lanLam));
  std::printf("\n");
}

void printMedian(const PointResult* p) {
  std::printf("  med   %-8.5f %-17s %3d (%9.1f)  %6d (%10.1f)  %3d  (%9.1f)",
              median3(p[0].plaq, p[1].plaq, p[2].plaq), "",
              median3i(p[0].mgCycles, p[1].mgCycles, p[2].mgCycles),
              median3(p[0].mgMs, p[1].mgMs, p[2].mgMs),
              median3i(p[0].cgIters, p[1].cgIters, p[2].cgIters),
              median3(p[0].cgMs, p[1].cgMs, p[2].cgMs),
              median3i(p[0].rqIters, p[1].rqIters, p[2].rqIters),
              median3(p[0].rqMs, p[1].rqMs, p[2].rqMs));
  if (p[0].haveLan)
    std::printf("  %5d (%9.1f)", median3i(p[0].lanIters, p[1].lanIters, p[2].lanIters),
                median3(p[0].lanMs, p[1].lanMs, p[2].lanMs));
  std::printf("\n");
}

// Ensemble size. Three configurations is thin for a NEGATIVE result, and our
// own measurement says so from an unanticipated direction: the
// per-configuration scatter at fixed beta reaches a factor 2.5
// (189/134/76 outer iterations at beta=6), which is far larger than the ~20%
// a methodological change moves. At three configurations the median is unstable
// and the quoted multipliers cannot be given to two significant figures.
// g_configs sets it; "configs=N" on the command line.
int g_configs = 3;

void runSweepPoint(int d, int n, double beta, int thermSweeps) {
  std::printf("\n--- n=%d, beta=%g (DOF=%ld, therm=%d sweeps, %d configs) ---\n", n, beta,
              static_cast<long>(d) * n * n * n, (beta == 0.0) ? 0 : thermSweeps, g_configs);
  printHeader();
  std::vector<PointResult> res;
  std::vector<double> rqMs, lanMs;
  std::vector<int> rqIt;
  for (int s = 0; s < g_configs; ++s) {
    // Same first three seeds as every previously published MC table, so the
    // enlarged ensemble contains the old one rather than replacing it.
    const std::uint64_t seed = 2026ULL + 2026ULL * static_cast<std::uint64_t>(s);
    res.push_back(runPoint(d, n, beta, seed, thermSweeps));
    printRow(seed, res.back());
    std::fflush(stdout);
    rqMs.push_back(res.back().rqMs);
    lanMs.push_back(res.back().lanMs);
    rqIt.push_back(res.back().rqIters);
  }
  if (res.size() == 3) {
    PointResult r3[3] = {res[0], res[1], res[2]};
    printMedian(r3);
  }
  // Spread, which is the point of enlarging the ensemble: report the range and
  // the interquartile band alongside the median, not the median alone.
  auto stat = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size();
    return std::array<double, 4>{v.front(), v[m / 4], v[m / 2], v.back()};
  };
  const auto R = stat(rqMs), Ln = stat(lanMs);
  std::vector<double> itd(rqIt.begin(), rqIt.end());
  const auto I = stat(itd);
  std::printf("  spread over %d configs  ours ms min/q1/med/max %.0f/%.0f/%.0f/%.0f"
              "   Lanczos ms %.0f/%.0f/%.0f/%.0f   ours its %.0f/%.0f/%.0f/%.0f"
              "   med ratio Lanc/ours %.2fx\n",
              g_configs, R[0], R[1], R[2], R[3], Ln[0], Ln[1], Ln[2], Ln[3], I[0], I[1], I[2],
              I[3], Ln[2] / R[2]);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
#ifdef BOCHNER_WITH_PETSC
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
#endif
  const std::string mode = (argc > 1) ? argv[1] : "all";
  const int d = (argc > 2) ? std::atoi(argv[2]) : 3;
  const int thermSweeps = 300;
  for (int a = 1; a < argc; ++a)
    if (std::string(argv[a]) == "reldrop") g_relativeGsDrop = true;
    else if (std::string(argv[a]) == "absdrop") g_relativeGsDrop = false;
    else if (std::string(argv[a]) == "biasedstart") g_biasedStart = true;
    else if (std::string(argv[a]).rfind("configs=", 0) == 0)
      g_configs = std::atoi(argv[a] + 8);

  std::printf("=== SU(%d) Wilson-action MC configurations: massless covariant Laplacian ===\n", d);
  std::printf("    (w = n^2, m^2 = 0, periodic n^3; 3 configs/point; run OMP_NUM_THREADS=1)\n");
  std::printf("    thermalization: %d heatbath sweeps; <P> = mean over the last %d sweeps,\n",
              thermSweeps, thermSweeps / 2);
  std::printf("    P(Q3|Q4) = the two quarters of that window (flatness diagnostic)\n");
  if (g_relativeGsDrop)
    std::printf("    covMG-LOBPCG: relativeGsDrop ON (scale-relative MGS drop threshold)\n");

  if (mode == "beta" || mode == "all") {
    std::printf("\n[A] beta sweep at fixed n = 24\n");
    for (double beta : {0.0, 3.0, 6.0, 9.0, 15.0, 30.0}) runSweepPoint(d, 24, beta, thermSweeps);
  }
  if (mode == "refine" || mode == "all") {
    std::printf("\n[B] refinement series at beta = 6 (mesh-independence on correlated fields)\n");
    for (int n : {8, 16, 24, 32}) runSweepPoint(d, n, 6.0, thermSweeps);
  }
  if (mode == "refine48") {
    std::printf("\n[B+] refinement point n = 48, beta = 6\n");
    runSweepPoint(d, 48, 6.0, thermSweeps);
  }
  if (mode == "traject") {
    // Matched continuum trajectory: n and beta scaled JOINTLY. In 3D the coupling is dimensionful, beta ~ 1/(g^2 a)
    // at fixed g^2, so a ~ 1/beta and fixed physical volume L = n a means
    // n proportional to beta. This is tree-level matching -- adequate for a
    // solver benchmark's trajectory, not a physics-grade scaling study -- with
    // n/beta = 8/3 chosen so the line passes through existing measured points
    // ((16,6) near the refinement series, (24,9) a beta-sweep point).
    std::printf("\n[C] matched trajectory n/beta = 8/3 (continuum direction: n, beta jointly)\n");
    const int ns[] = {8, 16, 24, 32};
    const double betas[] = {3.0, 6.0, 9.0, 12.0};
    for (int i = 0; i < 4; ++i) runSweepPoint(d, ns[i], betas[i], thermSweeps);
  }
  if (mode == "trajext") {
    std::printf("\n[C+] matched-trajectory extension n/beta = 8/3: (40,15), (48,18)\n");
    const int ns[] = {40, 48};
    const double betas[] = {15.0, 18.0};
    for (int i = 0; i < 2; ++i) runSweepPoint(d, ns[i], betas[i], thermSweeps);
  }
  if (mode == "traj64") {
    std::printf("\n[C++] trajectory probe n/beta = 8/3: (64,24)\n");
    runSweepPoint(d, 64, 24.0, thermSweeps);
  }
  benchstat::printTimingSummary();
#ifdef BOCHNER_WITH_PETSC
  SlepcFinalize();
#endif
  return 0;
}
