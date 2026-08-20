// The reference eigenvalues the paper's appendix offers for independent
// verification -- pinned to the precision at which they are published.
//
// The appendix states these as the mechanism by which a reader checks an
// independent implementation against ours. That makes them load-bearing: if a
// refactor shifts the sixth decimal, every one of those quoted numbers is
// wrong and nothing in the suite says so. Before this file exactly one of them
// was pinned anywhere, at a relative tolerance of 1e-5 -- which on 24.825546
// is +-0.00025, leaving the last three published decimals unchecked.
//
// Tolerance is 1e-7 relative: tight enough to pin every digit the paper
// prints (the observed run-to-run agreement is 4e-9 to 1e-8, and any real
// regression moves these by orders of magnitude more), loose enough not to be
// a floating-point equality check on a value published to six decimals.
#include <cmath>
#include <complex>
#include <vector>

#include "doctest.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"
#include "solvers/SunGauge.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

// The uniform-flux 3-torus of the paper's Section 5, Landau/seam gauge.
GaugeLattice fluxTorus(int n, int nPhi) {
  const double phi = 2.0 * M_PI * nPhi / (double(n) * n);
  const std::size_t N = static_cast<std::size_t>(n) * n * n;
  std::vector<double> lkx(N, 0.0), lky(N, 0.0), lkz(N, 0.0);
  const auto idx = [&](int i, int j, int k) { return (i * n + j) * n + k; };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        lkx[idx(i, j, k)] = -phi * j;
        if (j == n - 1) lky[idx(i, j, k)] = 2.0 * M_PI * nPhi * i / double(n);
      }
  return gaugeLatticePeriodic(n, n, n, double(n) * n, lkx, lky, lkz);
}

// The smooth SU(d) field of sun_gauge_bench / tab:nonabelian.
//
// AMPLITUDE MATTERS AND THE PAPER DOES NOT STATE IT. sun_gauge_bench.cpp:332
// uses amp = 4.0 and produces tab:nonabelian (lambda_min = 29.176453 at
// SU(2), n=16); ablation_bench.cpp:291 uses amp = 1.0 and produces
// tab:ablation (lambda_min = 3.114 on the same lattice). Two different
// operators, both called "the smooth SU(d') field" in the text. This test
// pins the one the reproducibility appendix cites.
void embedSu2(cd* M, int d, int p, int q, double v0, double v1, double v2) {
  const double a = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
  const double c = std::cos(a), s = (a > 1e-12) ? std::sin(a) / a : 1.0;
  M[p * d + p] = cd(c, s * v2);
  M[p * d + q] = cd(s * v1, s * v0);
  M[q * d + p] = cd(-s * v1, s * v0);
  M[q * d + q] = cd(c, -s * v2);
}
void setIdentity(cd* M, int d) {
  for (int i = 0; i < d * d; ++i) M[i] = cd(0, 0);
  for (int i = 0; i < d; ++i) M[i * d + i] = cd(1, 0);
}
void matmulInto(const cd* A, const cd* B, int d, cd* out) {
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      cd s(0, 0);
      for (int k = 0; k < d; ++k) s += A[i * d + k] * B[k * d + j];
      out[i * d + j] = s;
    }
}
void smoothLink(int d, int axis, int i, int j, int k, int n, double amp, cd* M) {
  const double h = 1.0 / n;
  const auto S = [n](int idx) { return std::sin(2.0 * M_PI * idx / n); };
  double v[3][3];
  v[0][0] = S(j); v[0][1] = S(k); v[0][2] = 0.0;
  v[1][0] = S(k); v[1][1] = 0.0;  v[1][2] = S(i);
  v[2][0] = 0.0;  v[2][1] = S(i); v[2][2] = S(j);
  const double rot = (axis == 0) ? 0.0 : (axis == 1) ? 2.09439510239 : 4.18879020479;
  for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
      v[b][c] = amp * h * (v[b][c] + 0.5 * std::sin(2.0 * M_PI * (i + j + k) / n + rot));
  if (d == 2) {
    setIdentity(M, 2);
    embedSu2(M, 2, 0, 1, v[0][0], v[0][1], v[0][2]);
    return;
  }
  cd R01[9], R02[9], R12[9], T[9];
  setIdentity(R01, 3); embedSu2(R01, 3, 0, 1, v[0][0], v[0][1], v[0][2]);
  setIdentity(R02, 3); embedSu2(R02, 3, 0, 2, v[1][0], v[1][1], v[1][2]);
  setIdentity(R12, 3); embedSu2(R12, 3, 1, 2, v[2][0], v[2][1], v[2][2]);
  matmulInto(R01, R02, 3, T);
  matmulInto(T, R12, 3, M);
}
SunLattice smoothSun(int d, int n, double amp) {
  SunLattice L;
  L.d = d;
  L.lx = L.ly = L.lz = n;
  L.periodic = true;
  L.w = static_cast<double>(n) * n;
  L.mass2 = 0.0;
  const int dd = d * d;
  L.ux.resize(static_cast<std::size_t>(L.numLinksX()) * dd);
  L.uy.resize(static_cast<std::size_t>(L.numLinksY()) * dd);
  L.uz.resize(static_cast<std::size_t>(L.numLinksZ()) * dd);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        const std::size_t e = static_cast<std::size_t>((i * n + j) * n + k) * dd;
        smoothLink(d, 0, i, j, k, n, amp, &L.ux[e]);
        smoothLink(d, 1, i, j, k, n, amp, &L.uy[e]);
        smoothLink(d, 2, i, j, k, n, amp, &L.uz[e]);
      }
  return L;
}

}  // namespace

TEST_CASE("published reference eigenvalue: uniform-flux torus, n=16, nPhi=4") {
  const GaugeLattice L = fluxTorus(16, 4);
  GaugeEigenOptions o;
  o.tol = 1e-11;
  o.maxIters = 500;
  const GaugeEigenResult r = smallestEigenpairGaugeMG(L, nullptr, o);
  REQUIRE(r.converged);
  // paper, reproducibility appendix: 24.825546
  CHECK(r.eigenvalue == doctest::Approx(24.825546).epsilon(1e-7));
}

TEST_CASE("published reference eigenvalue: uniform-flux torus, n=32, nPhi=4") {
  const GaugeLattice L = fluxTorus(32, 4);
  GaugeEigenOptions o;
  o.tol = 1e-11;
  o.maxIters = 500;
  const GaugeEigenResult r = smallestEigenpairGaugeMG(L, nullptr, o);
  REQUIRE(r.converged);
  // paper, reproducibility appendix: 25.055713
  CHECK(r.eigenvalue == doctest::Approx(25.055713).epsilon(1e-7));
}

TEST_CASE("published reference eigenvalue: smooth SU(2), n=16") {
  const SunLattice L = smoothSun(2, 16, 4.0);  // tab:nonabelian's amplitude
  GaugeEigenOptions o;
  o.tol = 1e-11;
  o.maxIters = 500;
  const GaugeEigenResult r = smallestEigenpairSunMG(L, nullptr, o);
  REQUIRE(r.converged);
  // paper, tab:nonabelian and the appendix: 29.176453
  CHECK(r.eigenvalue == doctest::Approx(29.176453).epsilon(1e-7));
}

TEST_CASE("the SU(d) flux embedding reproduces the abelian eigenvalue exactly") {
  // Not a published constant but the identity that makes two of the paper's
  // tables cross-checkable: the flux torus embedded in a Cartan direction with
  // traceless charges decouples into colour components, each seeing a U(1)
  // flux q_a * nPhi, so lambda_min must equal the abelian value at the
  // smallest |q_a| * nPhi. With charges (1,-1) at d=2 that is nPhi itself.
  const int n = 16, nPhi = 4;
  SunLattice S;
  S.d = 2;
  S.lx = S.ly = S.lz = n;
  S.periodic = true;
  S.w = static_cast<double>(n) * n;
  S.mass2 = 0.0;
  const int dd = 4;
  S.ux.resize(static_cast<std::size_t>(S.numLinksX()) * dd);
  S.uy.resize(static_cast<std::size_t>(S.numLinksY()) * dd);
  S.uz.resize(static_cast<std::size_t>(S.numLinksZ()) * dd);
  const double phi = 2.0 * M_PI * nPhi / (double(n) * n);
  const double q[2] = {1.0, -1.0};
  const auto setDiag = [&](cd* M, double th) {
    for (int a = 0; a < dd; ++a) M[a] = cd(0, 0);
    for (int a = 0; a < 2; ++a) M[a * 2 + a] = std::polar(1.0, q[a] * th);
  };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        const std::size_t e = static_cast<std::size_t>((i * n + j) * n + k) * dd;
        setDiag(&S.ux[e], -phi * j);
        setDiag(&S.uy[e], (j == n - 1) ? 2.0 * M_PI * nPhi * i / double(n) : 0.0);
        setDiag(&S.uz[e], 0.0);
      }

  GaugeEigenOptions o;
  o.tol = 1e-11;
  o.maxIters = 500;
  const GaugeEigenResult rs = smallestEigenpairSunMG(S, nullptr, o);
  REQUIRE(rs.converged);
  CHECK(rs.eigenvalue == doctest::Approx(24.825546).epsilon(1e-7));
}
