/// \file
/// Control experiment: separate "LOBPCG is the right outer loop" from "the
/// covariant V-cycle is the right preconditioner".
///
/// On the uniform-flux torus (the headline operator of tab:torus-eig), solve
/// the smallest eigenpair with the SAME preconditioner -- one covariant
/// V-cycle per apply, the exact configuration covMG-LOBPCG uses internally
/// (MgOptions defaults, tol=0, maxCycles=1, prebuilt hierarchy) -- inside
/// DIFFERENT outer loops, and with different preconditioners inside the same
/// SLEPc outer loops:
///
///   covMG-LOBPCG (ours)   -- native single-vector LOBPCG, complex arithmetic;
///   GD + covV             -- SLEPc Generalized Davidson, covariant V-cycle
///                            via PCSHELL on the real embedding;
///   LOBPCG(SLEPc) + covV  -- SLEPc block LOBPCG, same PCSHELL;
///   GD + AMG              -- SLEPc GD, PETSc GAMG (constant near-null);
///   LOBPCG(SLEPc) + AMG   -- SLEPc LOBPCG, GAMG;
///   LOBPCG(SLEPc) + Jacobi-- the cheap generic floor (diagonal scaling).
///
/// Protocol: single-threaded, eigen rtol 1e-7, median of 5, cold starts.
/// The shell rows charge the hierarchy build to setup (reported separately);
/// it is the same closed-form build the paper prices at <= one smoothing sweep.
///
/// Usage: covpc_outer_bench [nPhi] [n1 n2 ...]   (defaults: 4, sizes 16 24 32)
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <slepc.h>

#include "BenchTiming.h"
#include "grid/CooMatrix.h"
#include "solvers/EigenSolver.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"

using namespace bochner;
using benchstat::medianMs;
using cd = std::complex<double>;

namespace {

// Same operator as tools/torus_eig_compare (lattice-gauge-solvers
// Examples::uniformField): nPhi flux quanta through the x-y torus, Landau/seam
// gauge, forward links; kept textually in sync with that tool.
GaugeLattice uniformFluxLattice(int n, int nPhi, double h) {
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
  return gaugeLatticePeriodic(n, n, n, 1.0 / (h * h), lkx, lky, lkz);
}

CooMatrix assemblePeriodic(const GaugeLattice& L) {
  const int n = static_cast<int>(L.numNodes());
  CooMatrix A(2 * n, 2 * n);
  const auto blk = [&](int a, int b, double re, double im) {
    A.add(2 * a, 2 * b, re);
    A.add(2 * a, 2 * b + 1, -im);
    A.add(2 * a + 1, 2 * b, im);
    A.add(2 * a + 1, 2 * b + 1, re);
  };
  const auto lnk = [&](int a, int b, double th) {
    blk(a, a, L.w, 0.0);
    blk(b, b, L.w, 0.0);
    blk(a, b, -L.w * std::cos(th), L.w * std::sin(th));
    blk(b, a, -L.w * std::cos(th), -L.w * std::sin(th));
  };
  const auto id = [&](int i, int j, int k) { return (i * L.ly + j) * L.lz + k; };
  for (int i = 0; i < L.lx; ++i)
    for (int j = 0; j < L.ly; ++j)
      for (int k = 0; k < L.lz; ++k) {
        const int c = id(i, j, k);
        lnk(c, id((i + 1) % L.lx, j, k), L.lkx[c]);
        lnk(c, id(i, (j + 1) % L.ly, k), L.lky[c]);
        lnk(c, id(i, j, (k + 1) % L.lz), L.lkz[c]);
      }
  return A;
}

}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  {
    const double tol = 1e-7;
    const int nPhi = (argc > 1) ? std::atoi(argv[1]) : 4;
    std::vector<int> sizes;
    for (int a = 2; a < argc; ++a) sizes.push_back(std::atoi(argv[a]));
    if (sizes.empty()) sizes = {16, 24, 32};

    std::printf("same-preconditioner / different-outer-loop control, "
                "uniform-flux torus nPhi=%d, tol=%.0e, median of 5, cold starts\n\n",
                nPhi, tol);
    std::printf("%-22s %6s %12s %10s %10s %12s\n", "method", "n", "eig", "iters",
                "setup ms", "solve ms");

    for (const int n : sizes) {
      const double h = 1.0 / n;
      const GaugeLattice lat = uniformFluxLattice(n, nPhi, h);
      const CooMatrix E = assemblePeriodic(lat);
      const std::size_t N = static_cast<std::size_t>(lat.numNodes());

      // The shared preconditioner: one covariant V-cycle on the prebuilt
      // hierarchy -- the exact internal configuration of covMG-LOBPCG.
      double buildMs = 0.0;
      std::vector<GaugeLattice> levels;
      buildMs = medianMs([&] { levels = buildGaugeLevels(lat); });
      MgOptions pmg;
      pmg.tol = 0.0;
      pmg.maxCycles = 1;
      std::vector<cd> pb(N), px(N);
      std::vector<double> pin(2 * N), pout;
      const ShellPcApply covApply = [&](const double* in, double* out) {
        std::copy(in, in + 2 * N, pin.begin());
        pb = toComplex(pin);
        std::fill(px.begin(), px.end(), cd(0.0));
        vcycleSolve(levels, pb, px, pmg);
        pout = toInterleaved(px);
        std::copy(pout.begin(), pout.end(), out);
      };

      struct Row {
        const char* name;
        double eig = 0, tms = 0, setup = 0;
        long its = 0;
      };
      std::vector<Row> rows;

      {  // ours, native
        Row r{"covMG-LOBPCG (ours)"};
        GaugeEigenResult gr;
        r.tms = medianMs([&] {
          GaugeEigenOptions eo;
          eo.tol = tol;
          gr = smallestEigenpairGaugeMG(lat, nullptr, eo);
        });
        r.eig = gr.eigenvalue;
        r.its = gr.iterations;
        r.setup = buildMs;
        rows.push_back(r);
      }
      const auto slepcRow = [&](const char* name, auto&& solve) {
        Row r{name};
        EigenPair p;
        r.tms = medianMs([&] { p = solve(); });
        r.eig = p.value;
        r.its = p.iterations;
        r.setup = p.setupMs;
        rows.push_back(r);
      };
      slepcRow("GD + covV", [&] {
        auto p = smallestEigenpairDavidsonShell(E, tol, nullptr, covApply);
        p.setupMs += buildMs;
        return p;
      });
      slepcRow("LOBPCG(SLEPc) + covV", [&] {
        auto p = smallestEigenpairLOBPCGShell(E, tol, nullptr, covApply);
        p.setupMs += buildMs;
        return p;
      });
      slepcRow("GD + AMG", [&] {
        return smallestEigenpairDavidson(E, tol, nullptr, InnerPC::AMG);
      });
      slepcRow("LOBPCG(SLEPc) + AMG", [&] {
        return smallestEigenpairLOBPCG(E, tol, nullptr, InnerPC::AMG);
      });
      slepcRow("LOBPCG(SLEPc) + Jacobi", [&] {
        return smallestEigenpairLOBPCG(E, tol, nullptr, InnerPC::Jacobi);
      });

      for (const auto& r : rows)
        std::printf("%-22s %6d %12.6f %10ld %10.1f %12.1f\n", r.name, n, r.eig,
                    r.its, r.setup, r.tms);
      std::printf("\n");
    }
  }
  SlepcFinalize();
  return 0;
}
