/// \file
/// Served-application benchmark: the magnetic Schroedinger / linear
/// Ginzburg--Landau lowest eigenpair (superconducting nucleation, H_c2).
///
/// The operator is (-i\nabla - A)^2 on an OPEN n^3 box with Neumann boundaries
/// (periodic = false), uniform applied field B along z in the Landau gauge
/// A = (-By, 0, 0), so the Peierls phase sits on the x-links (theta = -phi_p * j)
/// and the y/z links are trivial. This is a bounded-domain operator -- NOT
/// diagonalized by an FFT, unlike the periodic torus or a Fourier-spectral BEC
/// solver -- so the comparison is fair to the method. The lowest eigenvector's
/// zero set seeds the Abrikosov vortex lattice; computing that eigenpair fast is
/// the deliverable.
///
/// We refine at fixed physical field (fixed nPhi = flux quanta through the box
/// face, h = 1/n, w = 1/h^2) and compare our covMG-LOBPCG against SLEPc
/// Krylov--Schur on the SAME operator (asserted equal), plus a linear V-cycle
/// ablation (covariant transfer vs plain averaging) on this physical operator.
///
/// Usage: magnetic_eig_compare [n] [nPhi]   (defaults 16 2)
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <slepc.h>

#include "BenchTiming.h"
#include "ChebSweep.h"
#include "grid/CooMatrix.h"
#include "solvers/EigenSolver.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"

using bochner::GaugeLattice;
using benchstat::medianMs;
using cd = std::complex<double>;

namespace {

// Uniform field on an OPEN box, Landau gauge A = (-By, 0, 0): the x-link from
// (i,j,k) to (i+1,j,k) carries theta = -phi_p * j; y/z links are trivial. No
// seam/wrap term is needed (the box does not close), so the connection is the
// smooth Landau gauge. phi_p = 2*pi*nPhi/n^2 is the flux per plaquette; nPhi
// fixed under refinement is a fixed physical field.
GaugeLattice openMagneticLattice(int n, int nPhi) {
  GaugeLattice L;
  L.lx = L.ly = L.lz = n;
  L.periodic = false;
  L.w = static_cast<double>(n) * n;  // 1/h^2, h = 1/n
  L.lkx.assign(L.numLinksX(), 0.0);
  L.lky.assign(L.numLinksY(), 0.0);
  L.lkz.assign(L.numLinksZ(), 0.0);
  const double phi_p = 2.0 * M_PI * nPhi / (static_cast<double>(n) * n);
  // x-link index (open): (i*ly + j)*lz + k, i in 0..n-2.
  for (int i = 0; i < n - 1; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) L.lkx[(i * n + j) * n + k] = -phi_p * j;
  L.buildTransports();
  return L;
}

// Assemble the open operator as a real 2N CooMatrix (the Lanczos input),
// matching applyConnectionLaplacian for periodic=false: interior links only, so
// a boundary node's diagonal is w * (its actual degree < 6) -- Neumann.
bochner::CooMatrix assembleOpen(const GaugeLattice& L) {
  const int n = static_cast<int>(L.numNodes());
  bochner::CooMatrix A(2 * n, 2 * n);
  const auto blk = [&](int a, int b, double re, double im) {
    A.add(2 * a, 2 * b, re);
    A.add(2 * a, 2 * b + 1, -im);
    A.add(2 * a + 1, 2 * b, im);
    A.add(2 * a + 1, 2 * b + 1, re);
  };
  const auto lnk = [&](int a, int b, double th) {
    blk(a, a, L.w, 0.0);
    blk(b, b, L.w, 0.0);
    blk(a, b, -L.w * std::cos(th), L.w * std::sin(th));   // -w e^{-i th}
    blk(b, a, -L.w * std::cos(th), -L.w * std::sin(th));  // conj
  };
  const auto id = [&](int i, int j, int k) { return (i * L.ly + j) * L.lz + k; };
  const auto xi = [&](int i, int j, int k) { return (i * L.ly + j) * L.lz + k; };
  const auto yi = [&](int i, int j, int k) { return (i * (L.ly - 1) + j) * L.lz + k; };
  const auto zi = [&](int i, int j, int k) { return (i * L.ly + j) * (L.lz - 1) + k; };
  for (int i = 0; i < L.lx; ++i)
    for (int j = 0; j < L.ly; ++j)
      for (int k = 0; k < L.lz; ++k) {
        const int c = id(i, j, k);
        if (i < L.lx - 1) lnk(c, id(i + 1, j, k), L.lkx[xi(i, j, k)]);
        if (j < L.ly - 1) lnk(c, id(i, j + 1, k), L.lky[yi(i, j, k)]);
        if (k < L.lz - 1) lnk(c, id(i, j, k + 1), L.lkz[zi(i, j, k)]);
      }
  return A;
}

}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  const int n = argc > 1 ? std::atoi(argv[1]) : 16;
  const int nPhi = argc > 2 ? std::atoi(argv[2]) : 2;
  const double tol = 1e-7;

  const GaugeLattice lat = openMagneticLattice(n, nPhi);
  const bochner::CooMatrix A = assembleOpen(lat);

  // Operator-equivalence guard: A (real 2N) == the matrix-free open lattice.
  std::vector<cd> probe(lat.numNodes());
  for (std::size_t c = 0; c < probe.size(); ++c) probe[c] = cd(std::cos(0.7 * c), std::sin(0.3 * c));
  const std::vector<cd> yLat = bochner::applyConnectionLaplacian(lat, probe);
  const std::vector<double> yA = [&] {
    std::vector<double> xr = bochner::toInterleaved(probe), yr(xr.size(), 0.0);
    for (const auto& t : A.compressed()) yr[t.row] += t.value * xr[t.col];
    return yr;
  }();
  const std::vector<cd> yAc = bochner::toComplex(yA);
  double mismatch = 0.0;
  for (std::size_t c = 0; c < yLat.size(); ++c) mismatch = std::max(mismatch, std::abs(yLat[c] - yAc[c]));

  std::printf("\n#### magnetic Schrodinger (open box, Neumann), n=%d (%d real DOFs), nPhi=%d ####\n", n,
              2 * n * n * n, nPhi);
  std::printf("   operator mismatch (assembled vs matrix-free) = %.2e\n", mismatch);

  // SLEPc Krylov--Schur baseline.
  bochner::EigenPair L;
  const double msL = medianMs([&] { L = bochner::smallestEigenpairLanczos(A, tol); });

  // Our covMG-LOBPCG (no SLEPc), cold constant start.
  bochner::GaugeEigenOptions eo;
  eo.tol = tol;
  bochner::GaugeEigenResult R;
  const double msR = medianMs([&] { R = bochner::smallestEigenpairGaugeMG(lat, nullptr, eo); });
  chebsweep::run("magnetic", eo, msR,
                 [&](const bochner::GaugeEigenOptions& ce) { return bochner::smallestEigenpairGaugeMG(lat, nullptr, ce); });

  const double lineDist = [&] {
    std::vector<cd> vL = bochner::toComplex(L.vector), vR = R.vector;
    const auto unit = [](std::vector<cd>& v) {
      double s = 0.0;
      for (const cd& z : v) s += std::norm(z);
      s = std::sqrt(s);
      if (s > 0) for (cd& z : v) z /= s;
    };
    unit(vL);
    unit(vR);
    cd ov(0, 0);
    for (std::size_t c = 0; c < vL.size(); ++c) ov += std::conj(vL[c]) * vR[c];
    const double o = std::min(1.0, std::abs(ov));
    return std::sqrt(std::max(0.0, 1.0 - o * o));
  }();

  // Linear V-cycle transfer ablation on this PHYSICAL operator: covariant vs
  // plain averaging, same hierarchy/smoother/line search.
  bochner::MgOptions mo;
  mo.tol = tol;
  std::vector<cd> b = bochner::applyConnectionLaplacian(lat, probe), x(lat.numNodes(), cd(0, 0));
  mo.covariantTransfer = true;
  std::fill(x.begin(), x.end(), cd(0, 0));
  const bochner::MgResult full = bochner::vcycleSolve(lat, b, x, mo);
  mo.covariantTransfer = false;
  std::fill(x.begin(), x.end(), cd(0, 0));
  const bochner::MgResult plain = bochner::vcycleSolve(lat, b, x, mo);

  std::printf("   lambda_min = %.6f   |Lanczos - ours| = %.2e   line dist = %.1e\n", R.eigenvalue,
              std::abs(L.value - R.eigenvalue), lineDist);
  std::printf("   Lanczos: %6.1f ms      covMG-LOBPCG: %3d its %7.1f ms (res %.1e)   speedup %.1fx\n", msL,
              R.iterations, msR, R.residual, msR > 0 ? msL / msR : 0.0);
  std::printf("   linear V-cycles: covariant %d   plain-P %d\n", full.cycles, plain.cycles);
  std::printf("ROW n=%d nPhi=%d lam=%.4f its=%d ms_ours=%.1f ms_lanc=%.1f speed=%.2f cyc=%d plain=%d\n", n,
              nPhi, R.eigenvalue, R.iterations, msR, msL, msR > 0 ? msL / msR : 0.0, full.cycles,
              plain.cycles);

  benchstat::printTimingSummary();
  SlepcFinalize();
  return 0;
}
