/// \file
/// Anisotropy study (documentation, not remedy -- decision: scope stays
/// point-smoother honest): how do the V-cycle, CG, and MG-RQI degrade as
/// per-axis weight anisotropy grows, and does a smoothly varying
/// ("abstract metric") diagonal metric preserve mesh-independence?
///
///   [A] Anisotropy-ratio sweep on the uniform-flux torus (nPhi=4 through
///       the xy plane): constant per-axis weights with ratio r on one axis,
///       r in {1,2,4,8,16,32}, both orientations (stretch z = transverse to
///       the flux plane; stretch x = in-plane). V-cycles to 1e-8, CG its,
///       MG-RQI outer its (certified, tol 1e-7).
///   [B] Abstract-metric refinement: fixed smooth anisotropic diagonal
///       metric c_a(x) (per-axis different smooth conductances, pointwise
///       contrast up to 1+amp), refined n = 8..48 at fixed flux. Flat
///       counts = mesh-independence at fixed metric.
///
/// Times are median-of-5 (BenchTiming.h); iteration counts are
/// deterministic.
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "BenchTiming.h"
#include "ChebSweep.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"

using bochner::GaugeLattice;
using benchstat::medianMs;
using cd = std::complex<double>;

namespace {

// Uniform-flux torus in the Landau/seam gauge (same as torus_eig_compare).
GaugeLattice uniformFluxLattice(int n, int nPhi, double w) {
  const double phi_p = 2.0 * M_PI * nPhi / (double(n) * n);
  const std::size_t N = static_cast<std::size_t>(n) * n * n;
  std::vector<double> lkx(N, 0.0), lky(N, 0.0), lkz(N, 0.0);
  const auto idx = [&](int i, int j, int k) { return (i * n + j) * n + k; };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        lkx[idx(i, j, k)] = -phi_p * j;
        if (j == n - 1) lky[idx(i, j, k)] = 2.0 * M_PI * nPhi * i / double(n);
      }
  return bochner::gaugeLatticePeriodic(n, n, n, w, lkx, lky, lkz);
}

std::vector<cd> randomRhs(long dof, unsigned seed) {
  std::vector<cd> b(static_cast<std::size_t>(dof));
  unsigned s = seed;
  const auto next = [&s]() {
    s = 1664525u * s + 1013904223u;
    return (s >> 8) * (1.0 / 16777216.0) - 0.5;
  };
  for (auto& v : b) v = cd(next(), next());
  return b;
}

struct Row {
  int vcycles = 0;
  double mgMs = 0.0;
  int wbCycles = 0;
  double wbMs = 0.0;
  int nwCycles = 0;
  double nwRes = 0.0;
  int cgIts = 0;
  double cgMs = 0.0;
  int eigIts = 0;
  double eigMs = 0.0;
  bool eigConv = false;
};

Row measure(const GaugeLattice& L) {
  Row r;
  const std::vector<cd> b = randomRhs(L.numNodes(), 4242);
  const auto levels = bochner::buildGaugeLevels(L);
  bochner::MgOptions mo;  // paper defaults: 2+2 RB-GS, omega=1, 30 coarse sweeps
  bochner::MgResult mr;
  r.mgMs = medianMs([&] {
    std::vector<cd> x(L.numNodes(), cd(0, 0));
    mr = bochner::vcycleSolve(levels, b, x, mo);
  });
  r.vcycles = mr.cycles;
  // Weight-blind TRANSFER (transports kept, uniform averaging; coarse
  // operators stay the weighted rediscretization) -- is the weighted
  // averaging load-bearing, or would the PTMG-style unweighted transfer do?
  bochner::MgOptions wb = mo;
  wb.weightedTransfer = false;
  bochner::MgResult wr;
  r.wbMs = medianMs([&] {
    std::vector<cd> x(L.numNodes(), cd(0, 0));
    wr = bochner::vcycleSolve(levels, b, x, wb);
  });
  r.wbCycles = wr.cycles;
  // Fully weight-blind HIERARCHY (the PTMG-without-weights counterfactual):
  // strip the per-edge weights from every level below the finest (each falls
  // back to its uniform w = w0/4^l) and blind the transfer averaging too.
  // The finest operator keeps its weights -- it IS the problem being solved.
  {
    std::vector<GaugeLattice> blind = levels;
    for (std::size_t l = 1; l < blind.size(); ++l) {
      blind[l].wx.clear();
      blind[l].wy.clear();
      blind[l].wz.clear();
    }
    bochner::MgOptions nw = wb;
    bochner::MgResult nr;
    std::vector<cd> x(L.numNodes(), cd(0, 0));
    nr = bochner::vcycleSolve(blind, b, x, nw);
    r.nwCycles = nr.cycles;
    r.nwRes = nr.relResidual;
  }
  bochner::SolveStats cg;
  r.cgMs = medianMs([&] {
    std::vector<cd> x(L.numNodes(), cd(0, 0));
    cg = bochner::cgSolve(L, b, x, 1e-8, 200000);
  });
  r.cgIts = cg.iterations;
  bochner::GaugeEigenOptions eo;
  eo.tol = 1e-7;
  eo.maxIters = 500;
  bochner::GaugeEigenResult er;
  r.eigMs = medianMs([&] { er = bochner::smallestEigenpairGaugeMG(L, nullptr, eo); });
  r.eigIts = er.iterations;
  r.eigConv = er.converged;
  chebsweep::run("graded", eo, r.eigMs,
                 [&](const bochner::GaugeEigenOptions& ce) { return bochner::smallestEigenpairGaugeMG(L, nullptr, ce); });
  return r;
}

void printRow(const char* tag, const Row& r) {
  std::printf("  %-14s  V-cyc %3d (%7.1f ms)   wblindP %3d (%7.1f ms)   noW %3d (res %.0e)   "
              "CG %6d (%8.1f ms)   MG-RQI %3d%s (%8.1f ms)\n",
              tag, r.vcycles, r.mgMs, r.wbCycles, r.wbMs, r.nwCycles, r.nwRes, r.cgIts, r.cgMs,
              r.eigIts, r.eigConv ? "" : "*", r.eigMs);
}

}  // namespace

int main(int argc, char** argv) {
  const int nPhi = argc > 1 ? std::atoi(argv[1]) : 4;

  // [A] constant per-axis anisotropy, both orientations.
  for (int n : {16, 32, 48, 64}) {
    std::printf("\n[A] anisotropy sweep, flux torus n=%d nPhi=%d (w_base=n^2; '*' = not certified)\n",
                n, nPhi);
    const double w = double(n) * n;
    for (int r : {1, 2, 4, 8, 16, 32}) {
      for (int axis : {2, 0}) {  // 2 = stretch z (transverse), 0 = stretch x (in-plane)
        if (r == 1 && axis == 0) continue;  // r=1 identical for both orientations
        GaugeLattice L = uniformFluxLattice(n, nPhi, w);
        std::vector<double> wx(L.numLinksX(), w), wy(L.numLinksY(), w), wz(L.numLinksZ(), w);
        auto& stretched = (axis == 0) ? wx : wz;
        for (auto& v : stretched) v = r * w;
        L.setEdgeWeights(wx, wy, wz);
        char tag[32];
        std::snprintf(tag, sizeof(tag), "r=%d (%s)", r, axis == 0 ? "x, in-plane" : "z");
        printRow(tag, measure(L));
      }
    }
  }

  // [B] smooth anisotropic diagonal metric, refined at fixed metric + flux.
  for (double amp : {3.0, 7.0}) {
    std::printf("\n[B] abstract metric c_a(x), per-axis smooth conductances, contrast %gx, nPhi=%d\n",
                1.0 + amp, nPhi);
    for (int n : {8, 16, 24, 32, 48}) {
      const double w = double(n) * n, h = 1.0 / n;
      GaugeLattice L = uniformFluxLattice(n, nPhi, w);
      const auto s2 = [](double t) {
        const double s = std::sin(M_PI * t);
        return s * s;
      };
      const auto cfun = [&](int axis, double x, double y, double z) {
        if (axis == 0) return 1.0 + amp * s2(x) * s2(y);
        if (axis == 1) return 1.0 + amp * s2(y) * s2(z);
        return 1.0 + amp * s2(z) * s2(x);
      };
      const auto graded = [&](int axis) {
        std::vector<double> wv(static_cast<std::size_t>(axis == 0   ? L.numLinksX()
                                                        : axis == 1 ? L.numLinksY()
                                                                    : L.numLinksZ()));
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
              const double x = (i + (axis == 0 ? 0.5 : 0.0)) * h;
              const double y = (j + (axis == 1 ? 0.5 : 0.0)) * h;
              const double z = (k + (axis == 2 ? 0.5 : 0.0)) * h;
              wv[static_cast<std::size_t>((i * n + j) * n + k)] = w * cfun(axis, x, y, z);
            }
        return wv;
      };
      L.setEdgeWeights(graded(0), graded(1), graded(2));
      char tag[16];
      std::snprintf(tag, sizeof(tag), "n=%d", n);
      printRow(tag, measure(L));
    }
  }
  return 0;
}
