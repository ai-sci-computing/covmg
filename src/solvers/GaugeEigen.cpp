#include "solvers/GaugeEigen.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace bochner {
namespace {

using cd = std::complex<double>;

cd cdot(const std::vector<cd>& a, const std::vector<cd>& b) {
  cd s(0.0, 0.0);
  for (std::size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
  return s;
}

double norm2(const std::vector<cd>& v) { return std::sqrt(cdot(v, v).real()); }

// Operator diagonal scale for the certificate floor: 6w uniform, or an upper
// bound from the per-axis weight maxima when weighted. Tightness is
// irrelevant -- the floor sits orders below any gapped operator's lambda_min.
double diagScale(const GaugeLattice& L) {
  if (!L.weighted()) return 6.0 * L.w;
  auto mx = [](const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, x);
    return m;
  };
  return 2.0 * (mx(L.wx) + mx(L.wy) + mx(L.wz));
}

void scale(std::vector<cd>& v, cd s) {
  for (cd& z : v) z *= s;
}

// Degree-k Chebyshev approximation to L^{-1} r on [a, b] (the classical
// semi-iteration from z = 0): the zero-setup polynomial eigen-preconditioner
// of the survey baseline. Costs k-1 operator applies.
void chebPrecondition(const GaugeLattice& lat, const std::vector<cd>& r,
                      std::vector<cd>& z, int k, double a, double b) {
  const std::size_t n = r.size();
  const double theta = 0.5 * (b + a), delta = 0.5 * (b - a);
  const double sigma1 = theta / delta;
  double rhoPrev = 1.0 / sigma1;
  std::vector<cd> d(n);
  for (std::size_t i = 0; i < n; ++i) d[i] = r[i] / theta;
  z = d;
  for (int step = 2; step <= k; ++step) {
    const std::vector<cd> Lz = applyConnectionLaplacian(lat, z);
    const double rhoCur = 1.0 / (2.0 * sigma1 - rhoPrev);
    const double c1 = rhoCur * rhoPrev, c2 = 2.0 * rhoCur / delta;
    for (std::size_t i = 0; i < n; ++i) {
      d[i] = c1 * d[i] + c2 * (r[i] - Lz[i]);
      z[i] += d[i];
    }
    rhoPrev = rhoCur;
  }
}

// Smallest eigenpair of a small (m <= 3) dense complex-Hermitian matrix H, by
// cyclic Hermitian Jacobi rotations. Returns the smallest eigenvalue and its
// eigenvector c (length m). Each rotation makes the off-diagonal real with a
// diagonal phase, then applies a real Jacobi rotation.
void smallestHermitianEig(int m, cd H[3][3], double& lam, cd c[3]) {
  cd D[3][3], V[3][3];
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) {
      D[i][j] = H[i][j];
      V[i][j] = (i == j) ? cd(1.0, 0.0) : cd(0.0, 0.0);
    }
  for (int sweep = 0; sweep < 40; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < m; ++p)
      for (int q = p + 1; q < m; ++q) off += std::norm(D[p][q]);
    if (off < 1e-30) break;
    for (int p = 0; p < m; ++p)
      for (int q = p + 1; q < m; ++q) {
        const cd dpq = D[p][q];
        if (std::abs(dpq) < 1e-18) continue;
        const double a = D[p][p].real(), b = D[q][q].real(), mag = std::abs(dpq);
        const cd g = std::exp(cd(0.0, -std::arg(dpq)));   // makes the off-diag real
        const double theta = 0.5 * std::atan2(2.0 * mag, a - b);
        const double cc = std::cos(theta), ss = std::sin(theta);
        // 2x2 unitary on (p,q): U = diag(1,g) * realRotation(theta).
        const cd U00(cc, 0.0), U01(-ss, 0.0), U10 = g * ss, U11 = g * cc;
        const cd c00 = std::conj(U00), c01 = std::conj(U01), c10 = std::conj(U10),
                 c11 = std::conj(U11);
        for (int i = 0; i < m; ++i) {  // D <- D U (columns p,q)
          const cd dip = D[i][p], diq = D[i][q];
          D[i][p] = dip * U00 + diq * U10;
          D[i][q] = dip * U01 + diq * U11;
        }
        for (int j = 0; j < m; ++j) {  // D <- U^H D (rows p,q)
          const cd dpj = D[p][j], dqj = D[q][j];
          D[p][j] = c00 * dpj + c10 * dqj;
          D[q][j] = c01 * dpj + c11 * dqj;
        }
        for (int i = 0; i < m; ++i) {  // V <- V U
          const cd vip = V[i][p], viq = V[i][q];
          V[i][p] = vip * U00 + viq * U10;
          V[i][q] = vip * U01 + viq * U11;
        }
      }
  }
  int jmin = 0;
  for (int j = 1; j < m; ++j)
    if (D[j][j].real() < D[jmin][jmin].real()) jmin = j;
  lam = D[jmin][jmin].real();
  for (int i = 0; i < m; ++i) c[i] = V[i][jmin];
}

}  // namespace

GaugeEigenResult smallestEigenpairGaugeMG(const GaugeLattice& lat,
                                          const std::vector<cd>* initialGuess,
                                          const GaugeEigenOptions& opts) {
  const std::size_t n = static_cast<std::size_t>(lat.numNodes());
  std::vector<cd> x(n, cd(1.0, 0.0));
  if (initialGuess && initialGuess->size() == n) x = *initialGuess;
  {
    // Normalize the guess. A zero, denormal (1/norm overflows), or non-finite
    // guess (e.g. a warm-start vector decayed to ~0) would otherwise leave x at
    // 0/inf/NaN and the loop would report a false-converged zero eigenpair --
    // res.residual = 0 < tol on the very first step. Mirror the SLEPc guard
    // (EigenSolver.cpp: nrm0==0 -> VecSet(x,1)) and reset to all-ones.
    double nx = norm2(x);
    if (!std::isfinite(nx) || !(nx > 0.0) || !std::isfinite(1.0 / nx)) {
      x.assign(n, cd(1.0, 0.0));
      nx = norm2(x);
    }
    scale(x, cd(1.0 / nx, 0.0));
  }

  // Preconditioner = a fixed (small) number of gauge-MG V-cycles from zero.
  MgOptions pmg = opts.mg;
  pmg.tol = 0.0;
  pmg.maxCycles = opts.precCycles;
  // Build the coarse hierarchy ONCE and reuse it every outer step. Rebuilding it
  // per vcycleSolve would re-coarsen and re-evaluate every coarse level's cos/sin
  // transports maxIters times -- pure redundant work on the interactivity path.
  //
  // Staleness probe (opts.frozenPrecLevels; default nullptr = bit-identical
  // behavior, the project's hard convention): when set, skip the build and
  // precondition with the caller's FROZEN pyramid -- stale transfers + stale
  // coarse operators -- while the V-cycle's finest-level smoother/residual/
  // alpha and everything in THIS function (residuals, Rayleigh quotients,
  // certificate) use the current `lat`.
  const bool frozenHier = opts.frozenPrecLevels != nullptr;
  std::vector<GaugeLattice> pmgLevelsOwn;
  if (!frozenHier && opts.chebDegree <= 0) {
    pmgLevelsOwn = buildGaugeLevels(lat);
    if (opts.maxLevels > 0 && static_cast<int>(pmgLevelsOwn.size()) > opts.maxLevels)
      pmgLevelsOwn.resize(static_cast<std::size_t>(opts.maxLevels));
  }
  const std::vector<GaugeLattice>& pmgLevels = frozenHier ? *opts.frozenPrecLevels : pmgLevelsOwn;
  const GaugeLattice* fineOverride = frozenHier ? &lat : nullptr;
  const double certFloor = opts.certFloorRel * diagScale(lat);
  // Chebyshev-baseline interval (chebDegree > 0): Gershgorin upper bound.
  const double chebB = 2.0 * diagScale(lat);
  const double chebA = chebB / std::max(1.0, opts.chebRatio);

  GaugeEigenResult res;
  res.vector = x;
  std::vector<cd> xprev;  // previous iterate (the LOBPCG conjugate direction)

  for (int it = 1; it <= opts.maxIters; ++it) {
    const std::vector<cd> Ex = applyConnectionLaplacian(lat, x);
    const double rho = cdot(x, Ex).real();  // x is unit-norm
    res.eigenvalue = rho;
    res.iterations = it;

    std::vector<cd> r(n);
    for (std::size_t i = 0; i < n; ++i) r[i] = Ex[i] - rho * x[i];
    res.residual = norm2(r) / std::max({std::abs(rho), certFloor, 1e-300});
    if (res.residual < opts.tol) break;

    // Preconditioned residual.
    std::vector<cd> w(n, cd(0.0, 0.0));
    if (opts.chebDegree > 0)
      chebPrecondition(lat, r, w, opts.chebDegree, chebA, chebB);
    else
      vcycleSolve(pmgLevels, r, w, pmg, fineOverride);

    // Build the locally-optimal trial subspace {x, w, x_prev} by modified
    // Gram-Schmidt (drop near-dependent vectors). q[0] = x (already unit).
    std::vector<std::vector<cd>> q;
    q.push_back(x);
    auto orthonormalAdd = [&](std::vector<cd> v) {
      const double nv0 = norm2(v);
      for (const auto& qi : q) {
        const cd proj = cdot(qi, v);
        for (std::size_t t = 0; t < n; ++t) v[t] -= proj * qi[t];
      }
      const double nv = norm2(v);
      // Drop near-dependent directions: relative to the direction's own norm
      // (certifies tight tolerances on dense-edge spectra) or the legacy
      // absolute 1e-7 (doubles as the warm-start early-exit; see the option).
      if (nv > (opts.relativeGsDrop ? 1e-7 * nv0 : 1e-7)) {
        scale(v, cd(1.0 / nv, 0.0));
        q.push_back(std::move(v));
      }
    };
    orthonormalAdd(w);
    if (!xprev.empty()) orthonormalAdd(xprev);
    const int m = static_cast<int>(q.size());
    // No search direction survived orthogonalization: the preconditioned
    // residual is parallel to the current iterate, so the 2x2/3x3 step can do
    // nothing. Stop iterating -- but this is NOT asserted to be convergence:
    // res.converged is set below from the RECOMPUTED residual of the returned
    // vector, so a stagnation short of tolerance honestly returns false.
    if (m == 1) break;

    // Rayleigh-Ritz: H = Q^H A Q (m x m Hermitian). A q[0] = Ex is already known.
    std::vector<std::vector<cd>> Aq(m);
    Aq[0] = Ex;
    for (int i = 1; i < m; ++i) Aq[i] = applyConnectionLaplacian(lat, q[i]);
    cd H[3][3];
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j) H[i][j] = cdot(q[i], Aq[j]);

    double lam = 0.0;
    cd cc[3];
    smallestHermitianEig(m, H, lam, cc);

    // Next iterate = the optimal combination; remember the old one for next step.
    std::vector<cd> xn(n, cd(0.0, 0.0));
    for (int i = 0; i < m; ++i)
      for (std::size_t t = 0; t < n; ++t) xn[t] += cc[i] * q[i][t];
    const double nn = norm2(xn);
    if (nn < 1e-300) break;
    scale(xn, cd(1.0 / nn, 0.0));
    xprev = std::move(x);
    x = std::move(xn);
    res.vector = x;
  }

  // Report the eigenpair OF THE RETURNED VECTOR. On any early exit (maxIters
  // reached, m==1 stagnation, degenerate combination) x is one step ahead of the
  // rho/residual computed at the top of the loop, so recompute them here; then
  // `converged` reflects the returned vector honestly rather than the control-
  // flow path that stopped the iteration.
  const std::vector<cd> Ex = applyConnectionLaplacian(lat, x);
  const double rho = cdot(x, Ex).real();  // x is unit-norm
  std::vector<cd> r(n);
  for (std::size_t i = 0; i < n; ++i) r[i] = Ex[i] - rho * x[i];
  res.eigenvalue = rho;
  res.residual = norm2(r) / std::max({std::abs(rho), certFloor, 1e-300});
  res.vector = x;
  res.converged = res.residual < opts.tol;
  return res;
}

BlockEigResult lowestEigenpairsGaugeMG(const GaugeLattice& lat, int m,
                                       const std::vector<std::vector<cd>>* guess,
                                       const GaugeEigenOptions& opts) {
  MgOptions pmg = opts.mg;
  pmg.tol = 0.0;
  pmg.maxCycles = opts.precCycles;
  std::vector<GaugeLattice> levels = buildGaugeLevels(lat);
  if (opts.maxLevels > 0 && static_cast<int>(levels.size()) > opts.maxLevels)
    levels.resize(static_cast<std::size_t>(opts.maxLevels));
  BlockEigOptions bo;
  bo.maxIters = opts.maxIters;
  bo.tol = opts.tol;
  bo.relativeDrop = opts.relativeGsDrop;
  bo.wanted = m;
  bo.certFloor = opts.certFloorRel * diagScale(lat);
  bo.lockConverged = opts.blockLockConverged;
  bo.softLockConverged = opts.blockSoftLockConverged;
  const int mb = m + std::max(0, opts.blockGuard);
  const auto apply = [&](const std::vector<cd>& x) { return applyConnectionLaplacian(lat, x); };
  const auto prec = [&](const std::vector<cd>& r) {
    std::vector<cd> w(r.size(), cd(0.0, 0.0));
    vcycleSolve(levels, r, w, pmg);
    return w;
  };
  BlockEigResult r = blockLobpcg(static_cast<std::size_t>(lat.numNodes()), mb, apply, prec, guess, bo);
  // blockLobpcg signals "could not build a rank-mb block" (n < mb, or guesses
  // that collapse the block) by returning EMPTY eigenvalue/residual/vector
  // arrays with maxResidual = 1e300. The resize() below only ever shrinks on
  // the success path -- but on that failure path it would GROW the empty
  // arrays with zeros, the max over m zeros is 0 < tol, and the 1e300 sentinel
  // is overwritten: a hard failure returned as converged=true with m
  // zero-length eigenvectors, which any caller indexing r.vectors[j][i] reads
  // out of bounds. Propagate the failure before resizing.
  if (static_cast<int>(r.vectors.size()) < m || static_cast<int>(r.eigenvalues.size()) < m ||
      static_cast<int>(r.residuals.size()) < m) {
    r.converged = false;
    return r;
  }
  // Return only the wanted pairs; guards are internal.
  r.eigenvalues.resize(m);
  r.residuals.resize(m);
  r.vectors.resize(m);
  r.maxResidual = 0.0;
  for (double d : r.residuals) r.maxResidual = std::max(r.maxResidual, d);
  r.converged = r.maxResidual < opts.tol;
  return r;
}

}  // namespace bochner
