// Diagnostic for the SU(3) n=64 iteration jump on the coherent-flux torus.
//
// Observed (sun_gauge_bench section [4], nPhi=4): covMG-LOBPCG outer iterations
// are flat at 12-14 for SU(3) flux through n=48, then jump to 34 at n=64, while
// SU(2) stays flat at 10 across the same series. The eigenvalue is correct
// either way (agrees with SLEPc and with the abelian prediction), so this is a
// convergence-RATE effect.
//
// Two hypotheses, and this tool separates them:
//
//   (H1) HIERARCHY. The coarse levels renormalize the flux as phi -> 4 phi per
//        level, so a deep enough hierarchy can land a coarse level on (or near)
//        the flux-aliasing point phi = pi/2 of ex:alias, where the
//        rediscretized coarse operator is singular and spectral equivalence
//        fails. This would degrade the V-CYCLE itself.
//
//   (H2) OUTER LOOP. The ground state is highly degenerate (n_Phi Landau copies
//        x colour multiplicity), and single-vector LOBPCG can slow down on a
//        degenerate/near-degenerate ground state. This would leave the V-cycle
//        untouched and show up only in the eigensolver.
//
// The discriminator is running BOTH on the same operator: a flat V-cycle count
// with a jumping eigen count indicts the outer loop (H2); both jumping indicts
// the hierarchy (H1). No SLEPc baseline is needed, so the sweep is cheap.
//
// The nPhi sweep is the second discriminator. Aliasing is sharply
// flux-dependent: with phi = 2 pi nPhi / n^2 and phi_L = 4^L phi, the level-L
// flux hits pi/2 exactly when 4^L nPhi / n^2 = 1/4. At n=64 that is L=4 for
// nPhi=4 (coarse size 4^3) but no integer level for nPhi=3 or 5. A rate problem
// tied to degeneracy has no such arithmetic signature.
//
// Usage:  flux_alias_diag [d]        (d = 2 or 3; default runs both)

#include <cmath>
#include <complex>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"
#include "solvers/SunGauge.h"
#include "solvers/WilsonMc.h"
#include "extraction/MacConnectionLaplacian.h"
#include "fluid/MacVortexRing.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

// The coherently frustrated SU(d) field: uniform-flux torus in a diagonal
// generator with traceless charges. Mirrors fluxSun()/fluxLattice() in
// ablation_bench.cpp and sun_gauge_bench.cpp.
SunLattice fluxSun(int d, int n, int nPhi) {
  SunLattice L;
  L.d = d;
  L.lx = L.ly = L.lz = n;
  L.periodic = true;
  L.w = static_cast<double>(n) * n;
  L.mass2 = 0.0;
  const int dd = d * d;
  L.ux.resize(static_cast<size_t>(L.numLinksX()) * dd);
  L.uy.resize(static_cast<size_t>(L.numLinksY()) * dd);
  L.uz.resize(static_cast<size_t>(L.numLinksZ()) * dd);
  double q[3] = {1.0, -1.0, 0.0};
  if (d == 3) { q[0] = 1.0; q[1] = 1.0; q[2] = -2.0; }
  const double phi_p = 2.0 * M_PI * nPhi / (static_cast<double>(n) * n);
  auto setDiag = [&](cd* M, double th) {
    for (int a = 0; a < dd; ++a) M[a] = cd(0, 0);
    for (int a = 0; a < d; ++a) M[a * d + a] = std::polar(1.0, q[a] * th);
  };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        const size_t e = static_cast<size_t>((i * n + j) * n + k) * dd;
        setDiag(&L.ux[e], -phi_p * j);
        setDiag(&L.uy[e], (j == n - 1) ? 2.0 * M_PI * nPhi * i / static_cast<double>(n) : 0.0);
        setDiag(&L.uz[e], 0.0);
      }
  return L;
}

// ---- the headline operators, for the start-policy delta ----------------
// Mirrors of the constructors in sun_gauge_bench.cpp (smooth SU(d)) and
// ablation_bench.cpp (U(1) uniform flux). Duplicated per tool, as elsewhere.

void embedSu2(cd* M, int d, int p, int q, double v0, double v1, double v2) {
  const double a = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
  const double c = std::cos(a), s = (a > 1e-12) ? std::sin(a) / a : 1.0;
  const cd m00(c, s * v2), m01(s * v1, s * v0), m10(-s * v1, s * v0), m11(c, -s * v2);
  M[p * d + p] = m00;
  M[p * d + q] = m01;
  M[q * d + p] = m10;
  M[q * d + q] = m11;
}
void setIdentity(cd* M, int d) {
  for (int i = 0; i < d * d; ++i) M[i] = cd(0, 0);
  for (int i = 0; i < d; ++i) M[i * d + i] = cd(1, 0);
}
void matmulInto(const cd* A, const cd* B, int d, cd* out) {
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      cd sum(0, 0);
      for (int k = 0; k < d; ++k) sum += A[i * d + k] * B[k * d + j];
      out[i * d + j] = sum;
    }
}
void smoothLink(int d, int axis, int i, int j, int k, int n, double amp, cd* M) {
  const double h = 1.0 / n;
  auto S = [n](int idx) { return std::sin(2.0 * M_PI * idx / n); };
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
SunLattice smoothLattice(int d, int n, double amp) {
  SunLattice L;
  L.d = d;
  L.lx = L.ly = L.lz = n;
  L.periodic = true;
  L.w = static_cast<double>(n) * n;
  L.mass2 = 0.0;
  const int dd = d * d;
  L.ux.resize(static_cast<size_t>(L.numLinksX()) * dd);
  L.uy.resize(static_cast<size_t>(L.numLinksY()) * dd);
  L.uz.resize(static_cast<size_t>(L.numLinksZ()) * dd);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        const size_t e = static_cast<size_t>((i * n + j) * n + k) * dd;
        smoothLink(d, 0, i, j, k, n, amp, &L.ux[e]);
        smoothLink(d, 1, i, j, k, n, amp, &L.uy[e]);
        smoothLink(d, 2, i, j, k, n, amp, &L.uz[e]);
      }
  return L;
}

GaugeLattice uniformFluxLattice(int n, int nPhi) {
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
  return gaugeLatticePeriodic(n, n, n, double(n) * n, lkx, lky, lkz);
}

// A draw of the SAME CHARACTER as PETSc/SLEPc's default initial vector: every
// real slot uniform on [0,1), so the vector is constant-biased (measured
// constant-overlap 0.860 = sqrt(3)/2, structural for this distribution, not a
// seed accident -- see tools/start_symmetry_bench.cpp). In the interleaved real
// embedding that means BOTH parts of each complex component are uniform [0,1).
//
// The question this answers: psi=1 costs 34 outer iterations on SU(3) flux at
// n=64 where zero-mean random starts take 14-16. Constant-PLUS-noise carries
// the same warm-start information as psi=1 while breaking its symmetry, so it
// should land with the random starts, not with psi=1. If it does, adopting the
// baseline-matched start protocol removes the anomaly for free.
std::vector<cd> constBiasedVec(long dof, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<cd> v(dof);
  for (auto& z : v) z = cd(u(rng), u(rng));
  return v;
}

std::vector<cd> randomRhs(long dof, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<cd> b(dof);
  for (auto& z : b) z = cd(g(rng), g(rng));
  return b;
}

// Wrap a flux to (-pi, pi] -- the holonomy is what matters, not the winding.
double wrapPi(double x) {
  while (x > M_PI) x -= 2.0 * M_PI;
  while (x <= -M_PI) x += 2.0 * M_PI;
  return x;
}

// Per-level flux census: where does each charge sector's plaquette flux sit,
// and how close does it come to the aliasing point pi/2?
void printLevelCensus(int d, int n, int nPhi) {
  const int nLev = static_cast<int>(buildSunLevels(fluxSun(d, n, nPhi)).size());
  const double phi0 = 2.0 * M_PI * nPhi / (static_cast<double>(n) * n);
  const double q[3] = {1.0, -1.0, 0.0};
  const double q3[3] = {1.0, 1.0, -2.0};
  std::printf("    levels=%d  per-level plaquette flux (wrapped, /pi; aliasing at 0.5):\n", nLev);
  for (int L = 0; L < nLev; ++L) {
    const double phiL = std::pow(4.0, L) * phi0;
    std::printf("      L=%d size=%3d  ", L, n >> L);
    for (int a = 0; a < d; ++a) {
      const double qa = (d == 3) ? q3[a] : q[a];
      const double f = wrapPi(qa * phiL) / M_PI;
      const bool alias = std::abs(std::abs(f) - 0.5) < 1e-9;
      std::printf("q=%+d: %+.4f%s  ", static_cast<int>(qa), f, alias ? " <-ALIAS" : "");
    }
    std::printf("\n");
  }
}

void runSweep(int d) {
  std::printf("\n================ SU(%d) coherent flux ================\n", d);
  std::printf("  %-4s %-5s %-8s  %-12s  %-12s\n", "n", "nPhi", "DOF", "V-cycles", "eigen its");
  for (int n : {32, 48, 64}) {
    for (int nPhi : {2, 3, 4, 5}) {
      const SunLattice L = fluxSun(d, n, nPhi);
      const auto levels = buildSunLevels(L);
      const std::vector<cd> b = randomRhs(L.dof(), 12345);

      MgOptions mg;
      mg.tol = 1e-8;
      mg.maxCycles = 200;
      std::vector<cd> x(L.dof(), cd(0, 0));
      const MgResult mr = vcycleSolveSun(levels, b, x, mg);

      GaugeEigenOptions eo;
      eo.tol = 1e-7;
      eo.maxIters = 300;
      const GaugeEigenResult er = smallestEigenpairSunMG(L, nullptr, eo);

      std::printf("  %-4d %-5d %-8ld  %3d (%.0e)%s  %3d%s        lam=%.6f\n", n, nPhi, L.dof(),
                  mr.cycles, mr.relResidual, mr.cycles >= mg.maxCycles ? "!" : "  ", er.iterations,
                  er.converged ? " " : "!", er.eigenvalue);
    }
  }
  std::printf("\n  level census at the anomalous point (n=64, nPhi=4) and its neighbours:\n");
  for (int nPhi : {3, 4, 5}) {
    std::printf("  -- n=64 nPhi=%d\n", nPhi);
    printLevelCensus(d, 64, nPhi);
  }
}

// Start-dependence probe. A 2x spectral gap should converge fast, so 34 outer
// iterations at n=64 is anomalous on its face. The ground state is 2*nPhi-fold
// degenerate (nPhi Landau copies x the two |q|=1 colour sectors), and
// single-vector LOBPCG's rate on a degenerate ground state depends on how much
// of the initial vector lands in the eigenspace. If the count swings widely
// across random starts, the jump is start-dependence, not a structural loss of
// mesh-independence; if every start gives ~34, it is systematic and the series
// genuinely degrades at n=64.
void runSeedProbe(int d, int n, int nPhi, int nSeeds) {
  const SunLattice L = fluxSun(d, n, nPhi);
  std::printf("\n  seed probe: SU(%d) n=%d nPhi=%d, %d random starts\n", d, n, nPhi, nSeeds);
  int lo = 1 << 30, hi = 0;
  for (int s = 0; s < nSeeds; ++s) {
    const std::vector<cd> g = randomRhs(L.dof(), 9000 + 17 * s);
    GaugeEigenOptions eo;
    eo.tol = 1e-7;
    eo.maxIters = 300;
    const GaugeEigenResult er = smallestEigenpairSunMG(L, &g, eo);
    lo = std::min(lo, er.iterations);
    hi = std::max(hi, er.iterations);
    std::printf("    seed %2d: %3d its%s  lam=%.9f\n", s, er.iterations, er.converged ? " " : "!",
                er.eigenvalue);
  }
  std::printf("    -> range [%d, %d]\n", lo, hi);
}

// Why is the constant start bad at one configuration? smallestEigenpairSunMG
// defaults to psi = 1 (SunGauge.cpp:645; GaugeEigen.cpp:96 does the same for
// U(1)), which is the ground state of the TRIVIAL connection. On a flux
// operator its overlap with the true ground eigenspace is a lattice sum that
// can nearly vanish at particular commensurate (n, nPhi, charge) combinations
// -- and LOBPCG then spends its first iterations just building overlap.
//
// The Rayleigh quotient of the start is the cheap proxy: rho_0 / lambda_min
// near 1 means the start already sits in the low eigenspace, large means it
// does not. Matvecs only, no solves.
void printStartQuality(int d, int nPhi) {
  std::printf("\n  start quality, SU(%d) nPhi=%d: Rayleigh quotient of psi=1\n", d, nPhi);
  std::printf("    %-4s %-14s %-14s %-10s\n", "n", "rho_0", "lambda_min", "rho_0/lam");
  for (int n : {32, 48, 64}) {
    const SunLattice L = fluxSun(d, n, nPhi);
    const std::vector<cd> one(L.dof(), cd(1.0, 0.0));
    const std::vector<cd> Lone = applySunLaplacian(L, one);
    cd num(0, 0);
    double den = 0.0;
    for (std::size_t i = 0; i < one.size(); ++i) {
      num += std::conj(one[i]) * Lone[i];
      den += std::norm(one[i]);
    }
    const double rho0 = num.real() / den;
    GaugeEigenOptions eo;
    eo.tol = 1e-7;
    eo.maxIters = 300;
    const double lam = smallestEigenpairSunMG(L, nullptr, eo).eigenvalue;
    std::printf("    %-4d %-14.6f %-14.6f %-10.2f\n", n, rho0, lam, rho0 / lam);
  }
}

// WHY DOES plain-P FAIL ON THE FLUX TORUS BUT SURVIVE ON THE SMOOTH FIELD?
//
// The paper currently answers "the smooth field's holonomy vanishes under
// refinement". That cannot be the explanation: at fixed physical field BOTH
// operators have plaquette flux ~ h^2, so both shrink the same way. The phrase
// applies equally to the operator where plain-P works and the one where it
// dies, so it distinguishes nothing.
//
// The candidate this measures instead. Plain interpolation averages section
// values WITHOUT transporting them, so what it gets wrong is governed by how
// far each link is from the identity -- the LINK ANGLE, which is
// GAUGE-DEPENDENT -- and not by the plaquette flux, which is gauge-invariant.
// The two operators are represented in very different gauges: the flux torus
// sits in Landau/seam gauge (theta^x = -2 pi nPhi j / n^2, plus a seam row
// theta^y = 2 pi nPhi i / n that is O(1) at large i), while the smooth field
// has theta ~ amp*h uniformly small everywhere with no seam.
//
// If that is right, plain-P's failure is partly an artifact of the gauge
// REPRESENTATION rather than of the physical field -- which the covariant
// transfer is immune to by construction, since it is gauge-equivariant
// (Theorem thm:flat). That is a sharper and more useful statement than the
// one in the text, and it is exactly what a practitioner needs in order to
// predict when covariance will matter.
//
// So both quantities are reported per level, measured from the ACTUAL
// coarsened lattices (buildGaugeLevels / buildSunLevels), not from formulas:
//
//   linkDev  = ||U - I||_F / (2 sqrt(d))   gauge-DEPENDENT, what plain-P sees
//   plaqFrus = 1 - Re tr(U_p) / d          gauge-INVARIANT, the physical field
//
// Re tr is insensitive to plaquette orientation and cyclic ordering, so the
// invariant measure is robust to the convention used to walk the loop.
double linkDevU1(double theta) { return std::abs(std::sin(0.5 * theta)); }

double linkDevSun(const cd* U, int d) {
  double s = 0.0;
  for (int a = 0; a < d; ++a)
    for (int b = 0; b < d; ++b) {
      const cd delta = U[a * d + b] - ((a == b) ? cd(1, 0) : cd(0, 0));
      s += std::norm(delta);
    }
  return std::sqrt(s) / (2.0 * std::sqrt(static_cast<double>(d)));
}

void adjointInto(const cd* A, int d, cd* out) {
  for (int a = 0; a < d; ++a)
    for (int b = 0; b < d; ++b) out[a * d + b] = std::conj(A[b * d + a]);
}

void censusU1(const char* label, const GaugeLattice& lat) {
  const auto levels = buildGaugeLevels(lat);
  std::printf("  %-22s levels=%zu\n", label, levels.size());
  for (std::size_t l = 0; l < levels.size(); ++l) {
    const GaugeLattice& L = levels[l];
    const int nx = L.lx, ny = L.ly, nz = L.lz;
    const auto id = [&](int i, int j, int k) { return (i * ny + j) * nz + k; };
    double devSum = 0.0, devMax = 0.0, frSum = 0.0, frMax = 0.0;
    long cnt = 0;
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k) {
          const double dx = linkDevU1(L.lkx[id(i, j, k)]);
          const double dy = linkDevU1(L.lky[id(i, j, k)]);
          devSum += dx + dy;
          devMax = std::max(devMax, std::max(dx, dy));
          const double phi = L.lkx[id(i, j, k)] + L.lky[id((i + 1) % nx, j, k)] -
                             L.lkx[id(i, (j + 1) % ny, k)] - L.lky[id(i, j, k)];
          frSum += 1.0 - std::cos(phi);
          frMax = std::max(frMax, 1.0 - std::cos(phi));
          cnt++;
        }
    std::printf("    L=%zu size=%3d  linkDev mean %.4f max %.4f   plaqFrus mean %.4f max %.4f\n",
                l, nx, devSum / (2.0 * cnt), devMax, frSum / cnt, frMax);
  }
}

void censusSun(const char* label, const SunLattice& lat) {
  const auto levels = buildSunLevels(lat);
  const int d = lat.d, dd = d * d;
  std::printf("  %-22s levels=%zu\n", label, levels.size());
  std::vector<cd> T(dd), T2(dd), A1(dd), A2(dd);
  for (std::size_t l = 0; l < levels.size(); ++l) {
    const SunLattice& L = levels[l];
    const int nx = L.lx, ny = L.ly, nz = L.lz;
    double devSum = 0.0, devMax = 0.0, frSum = 0.0;
    long cnt = 0;
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k) {
          const std::size_t e = static_cast<std::size_t>(L.index(i, j, k)) * dd;
          const double dx = linkDevSun(&L.ux[e], d), dy = linkDevSun(&L.uy[e], d);
          devSum += dx + dy;
          devMax = std::max(devMax, std::max(dx, dy));
          // U_p = Ux(i,j) Uy(i+1,j) Ux(i,j+1)^H Uy(i,j)^H
          const std::size_t ex1 = static_cast<std::size_t>(L.index((i + 1) % nx, j, k)) * dd;
          const std::size_t ey1 = static_cast<std::size_t>(L.index(i, (j + 1) % ny, k)) * dd;
          adjointInto(&L.ux[ey1], d, A1.data());
          adjointInto(&L.uy[e], d, A2.data());
          matmulInto(&L.ux[e], &L.uy[ex1], d, T.data());
          matmulInto(T.data(), A1.data(), d, T2.data());
          matmulInto(T2.data(), A2.data(), d, T.data());
          cd tr(0, 0);
          for (int a = 0; a < d; ++a) tr += T[a * d + a];
          frSum += 1.0 - tr.real() / d;
          cnt++;
        }
    std::printf("    L=%zu size=%3d  linkDev mean %.4f max %.4f   plaqFrus mean %.4f\n", l, nx,
                devSum / (2.0 * cnt), devMax, frSum / cnt);
  }
}

// THE DECISIVE TEST for what plain-P actually reacts to.
//
// The census shows the gauge-INVARIANT frustration failing to separate the two
// operator families (at coarse levels the smooth fields are MORE frustrated
// than the flux torus, which is gauge-trivial at its coarsest level), while the
// gauge-DEPENDENT link deviation separates them cleanly: the flux torus carries
// links with U = -1 at every level, the smooth fields do not.
//
// So: apply a random gauge transformation to the SMOOTH field. The physics is
// untouched -- same bundle, same curvature, same spectrum, lambda_min invariant
// to round-off -- but every link becomes O(1) away from the identity. If
// plain-P's cycle count blows up while the covariant cycle is unchanged, then
// what plain interpolation reacts to is the GAUGE REPRESENTATION and not the
// physical field, and the paper's "holonomy vanishes under refinement" is not
// the explanation.
//
// The covariant column doubles as a check on gauge equivariance (thm:flat):
// it should be invariant, not merely similar. lambda_min invariance validates
// that we are using the library's own transformation convention correctly.
void runGaugeTest(double amp) {
  std::printf("\n=== GAUGE-TRANSFORM TEST: what does plain-P react to? ===\n");
  std::printf("  Same physical field, random gauge. Covariant transfer should be\n");
  std::printf("  INVARIANT (gauge equivariance); plain-P is predicted to break.\n\n");
  std::printf("  %-16s %-4s  %-13s %-13s   %-13s %-13s  %s\n", "operator", "n", "full (orig)",
              "full (gauged)", "plain-P (orig)", "plain-P (gauged)", "lambda_min drift");
  for (int d : {2, 3})
    for (int n : {16, 32}) {
      SunLattice A = smoothLattice(d, n, amp);
      SunLattice B = A;
      gaugeTransformSun(B, /*seed=*/4242);

      auto cycles = [&](const SunLattice& L, bool covariant) {
        MgOptions mg;
        mg.tol = 1e-8;
        mg.maxCycles = 200;
        mg.covariantTransfer = covariant;
        // tab:ablation's SU(d) rhs, so these cycles read against that table.
        const size_t dof = static_cast<size_t>(L.dof());
        std::vector<cd> b(dof);
        for (size_t i = 0; i < dof; ++i) b[i] = cd(std::sin(0.7 * i), std::cos(0.3 * i));
        std::vector<cd> x(dof, cd(0, 0));
        return vcycleSolveSun(buildSunLevels(L), b, x, mg).cycles;
      };
      auto lam = [&](const SunLattice& L) {
        GaugeEigenOptions eo;
        eo.tol = 1e-9;
        return smallestEigenpairSunMG(L, nullptr, eo).eigenvalue;
      };

      const int fA = cycles(A, true), fB = cycles(B, true);
      const int pA = cycles(A, false), pB = cycles(B, false);
      const double lA = lam(A), lB = lam(B);
      std::printf("  SU(%d) smooth     %-4d  %-13d %-13d   %-13d %-13d  %.2e\n", d, n, fA, fB, pA,
                  pB, std::abs(lA - lB) / lA);
      std::fflush(stdout);
    }
}

// THE NON-CONTRIVED VERSION of the gauge test.
//
// Re-gauging the smooth field proves plain-P reacts to the frame rather than
// the physics, but the fair objection is that nobody runs in a random
// gauge. Monte-Carlo configurations answer that: they are physical, they are
// what production lattice QCD actually hands a solver, and they carry NO gauge
// fixing -- the sampler leaves them in whatever frame the heatbath produced.
//
// If plain-P degrades on MC configurations the way it does on the re-gauged
// smooth field, the argument needs no artificial transformation at all. If it
// does not, the re-gauging result stays a mechanism demonstration rather than a
// practitioner-facing claim, and should be presented as such.
//
// beta is swept because the frame's roughness tracks it: large beta means links
// closer to a smooth configuration, small beta means a disordered frame.
void runMcAblation() {
  std::printf("\n=== ABLATION ON MC CONFIGURATIONS (physical, no gauge fixing) ===\n");
  std::printf("  Does plain-P degrade on operators production LQCD actually produces,\n");
  std::printf("  without any artificial re-gauging?\n\n");
  std::printf("  %-4s %-6s %-5s  %-10s %-12s  %-8s %s\n", "n", "beta", "seed", "full (cyc)",
              "plain-P (cyc)", "ratio", "<P>");
  for (int n : {16, 24})
    for (double beta : {2.0, 6.0, 15.0}) {
      for (std::uint64_t seed : {2026ULL, 4052ULL}) {
        const SunLattice L =
            mcSunLattice(3, n, beta, 300, static_cast<double>(n) * n, 0.0, seed, beta < 6.0);
        const auto levels = buildSunLevels(L);
        const std::vector<cd> b = randomRhs(L.dof(), 12345);
        auto cycles = [&](bool covariant) {
          MgOptions mg;
          mg.tol = 1e-8;
          mg.maxCycles = 200;
          mg.covariantTransfer = covariant;
          std::vector<cd> x(L.dof(), cd(0, 0));
          return vcycleSolveSun(levels, b, x, mg).cycles;
        };
        const int f = cycles(true), p = cycles(false);
        std::printf("  %-4d %-6.1f %-5llu  %-10d %-12d%s  %-8.1f %.5f\n", n, beta,
                    static_cast<unsigned long long>(seed), f, p, p >= 200 ? "!" : " ",
                    static_cast<double>(p) / f, averagePlaquette(L));
        std::fflush(stdout);
      }
    }
}

// ===========================================================================
// MISALIGNMENT: IS IT THE MAGNITUDE OR THE VARIATION?
//
// The census above reports linkDev = ||U-I||_F, the MAGNITUDE of the frame
// misalignment, and it separates the smooth field from the flux torus. But the
// 't Hooft twist refutes magnitude as the mechanism: its links are as far from
// the identity as the seam's, yet plain-P matches the covariant cycle there
// (9/7/7 against 9/7/7). The twist's links are CONSTANT over the lattice.
//
// So the proposed mechanism is not how far the frame is rotated but how much
// that rotation VARIES from site to site. Plain interpolation is covariant
// interpolation with every transport replaced by I, so what it commits is a
// position-dependent misalignment; a constant misalignment distorts the coarse
// space uniformly (and the line search absorbs a uniform rescaling), while a
// varying one scatters a covariantly smooth mode into modes the coarse space
// cannot carry.
//
// Two quantities per axis and level, both from the ACTUAL coarsened lattices:
//
//   mag = mean_e ||U_e - I||_F                    (what the census reports)
//   var = min_M mean_e ||U_e - M||_F              distance to the nearest
//                                                 CONSTANT field
//
// The minimizer of the second is the entrywise mean Mbar = mean_e U_e (not
// unitary, and it need not be -- it is a least-squares reference, not a gauge),
// so var is computed directly as the mean distance to Mbar. Constant links give
// var = 0 identically, which is the twist; a seam gives var = O(1).
//
// Both are normalized by 2 sqrt(d') like linkDev, so the columns are
// comparable, and both are deliberately GAUGE-DEPENDENT: that is the point.
//
// The prediction under test: plain-P's damage tracks var, not mag.
// Both a MEAN and a MAX aggregation, because they answer different questions
// and the mean turned out to be the wrong one. The flux torus carries its O(1)
// misalignment on the SEAM -- a codimension-one set, 1/n of the edges -- so a
// mean over the lattice washes it out (measured: flux var-mean 0.130 sits BELOW
// the smooth field's 0.137, while plain-P needs 59 cycles on the first and 11
// on the second). The census above already reports linkDev as a max for the
// same reason. A codimension-one defect is invisible to a mean and fatal to
// the coarse space.
struct Misalign {
  double mag = 0.0, magMax = 0.0;
  double var = 0.0, varMax = 0.0;
};

// Mean over the x and y links of a level. The axis means are taken SEPARATELY
// (a field can be constant along x and varying along y), then pooled.
Misalign misalignSun(const SunLattice& L) {
  const int d = L.d, dd = d * d;
  const double nrm = 2.0 * std::sqrt(static_cast<double>(d));
  Misalign out;
  for (int axis = 0; axis < 2; ++axis) {
    const std::vector<cd>& U = (axis == 0) ? L.ux : L.uy;
    // numLinks < lx*ly*lz on an OPEN lattice; iterate the array, not the cells.
    const long cells = static_cast<long>(U.size()) / dd;
    std::vector<cd> Mbar(dd, cd(0, 0));
    for (long c = 0; c < cells; ++c)
      for (int a = 0; a < dd; ++a) Mbar[a] += U[static_cast<size_t>(c) * dd + a];
    for (int a = 0; a < dd; ++a) Mbar[a] /= static_cast<double>(cells);
    double magSum = 0.0, varSum = 0.0;
    for (long c = 0; c < cells; ++c) {
      const cd* Ue = &U[static_cast<size_t>(c) * dd];
      double m = 0.0, v = 0.0;
      for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b) {
          const cd id = (a == b) ? cd(1, 0) : cd(0, 0);
          m += std::norm(Ue[a * d + b] - id);
          v += std::norm(Ue[a * d + b] - Mbar[a * d + b]);
        }
      magSum += std::sqrt(m);
      varSum += std::sqrt(v);
      out.magMax = std::max(out.magMax, std::sqrt(m) / nrm);
      out.varMax = std::max(out.varMax, std::sqrt(v) / nrm);
    }
    out.mag += magSum / (nrm * cells);
    out.var += varSum / (nrm * cells);
  }
  out.mag *= 0.5;
  out.var *= 0.5;
  return out;
}

Misalign misalignU1(const GaugeLattice& L) {
  Misalign out;
  for (int axis = 0; axis < 2; ++axis) {
    const std::vector<double>& th = (axis == 0) ? L.lkx : L.lky;
    const long cells = static_cast<long>(th.size());
    // U = exp(i theta) as a 1x1 matrix; same least-squares reference.
    cd Mbar(0, 0);
    for (long c = 0; c < cells; ++c) Mbar += std::polar(1.0, th[c]);
    Mbar /= static_cast<double>(cells);
    double magSum = 0.0, varSum = 0.0;
    for (long c = 0; c < cells; ++c) {
      const cd U = std::polar(1.0, th[c]);
      magSum += std::abs(U - cd(1, 0));
      varSum += std::abs(U - Mbar);
      out.magMax = std::max(out.magMax, std::abs(U - cd(1, 0)) / 2.0);
      out.varMax = std::max(out.varMax, std::abs(U - Mbar) / 2.0);
    }
    out.mag += magSum / (2.0 * cells);
    out.var += varSum / (2.0 * cells);
  }
  out.mag *= 0.5;
  out.var *= 0.5;
  return out;
}

// ---------------------------------------------------------------------------
// COARSE-SPACE DEFECT: the quantity the mechanism is supposed to control.
//
// Iteration counts confound the transfer with the gap: hot fields converge in
// a handful of cycles with ANY transfer because their gap is large. The
// transfer-only quantity is how well the coarse space range(P) can represent
// the ground state in the energy norm,
//
//   defect(P)^2 = min_z ||psi - P z||_L^2 / ||psi||_L^2
//               = 1 - (g^H z) / (psi^H L psi),   L P z = ... , g := P^H L psi,
//
// with z the solution of (P^H L P) z = g. This is the weak approximation
// property measured directly, and it is exactly what plain-P is accused of
// destroying. P^H L P is SPD (P injective), so CG on the matrix-free triple
// product suffices; the coarse problem is 1/8 the size.
//
// Note this is a two-grid, transfer-only diagnostic: no smoother, no cycle, no
// line search. That is deliberate -- alpha can rescale a uniformly distorted
// coarse correction but cannot undo scattering, so isolating the coarse space
// is what distinguishes the two failure modes.
// Trace form, so the answer does not depend on WHICH basis of a degenerate
// ground level the eigensolver happens to return:
//
//   defect^2 = tr(Psi^H L Psi - Psi^H L P (P^H L P)^-1 P^H L Psi)
//              -------------------------------------------------
//                            tr(Psi^H L Psi)
//
// Both traces are invariant under Psi -> Psi Q for unitary Q, so mixing within
// a degenerate level cannot move the number. This matters here: the twist
// ground level is four-fold and the flux torus n_Phi-fold, so the defect of a
// single eigenvector is not a well-defined property of the operator at all.
// (First version measured exactly that, and two random starts disagreed -- the
// disagreement was the diagnostic working, not the solver failing.)
template <typename ApplyL, typename ApplyP, typename ApplyR>
double coarseDefectBlock(const std::vector<std::vector<cd>>& psi, long coarseDof, ApplyL applyL,
                         ApplyP applyP, ApplyR applyR) {
  const auto dot = [](const std::vector<cd>& a, const std::vector<cd>& b) {
    cd s(0, 0);
    for (size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
    return s;
  };
  const auto Ac = [&](const std::vector<cd>& v) { return applyR(applyL(applyP(v))); };

  double numer = 0.0, denom = 0.0;
  for (const std::vector<cd>& v : psi) {
    const std::vector<cd> Lv = applyL(v);
    const double energy = dot(v, Lv).real();
    const std::vector<cd> g = applyR(Lv);

    std::vector<cd> z(static_cast<size_t>(coarseDof), cd(0, 0)), r = g, p = g;
    double rr = dot(r, r).real();
    const double rr0 = rr;
    for (int it = 0; it < 20000 && rr > 1e-24 * rr0; ++it) {
      const std::vector<cd> Ap = Ac(p);
      const double alpha = rr / dot(p, Ap).real();
      for (size_t i = 0; i < z.size(); ++i) z[i] += alpha * p[i];
      for (size_t i = 0; i < r.size(); ++i) r[i] -= alpha * Ap[i];
      const double rrNew = dot(r, r).real();
      const double beta = rrNew / rr;
      for (size_t i = 0; i < p.size(); ++i) p[i] = r[i] + beta * p[i];
      rr = rrNew;
    }
    const double captured = dot(g, z).real();
    // Each column's captured energy cannot exceed its total: a violation means
    // the coarse solve broke down (P^H L P near-singular). Reporting that as a
    // defect of zero -- a PERFECT coarse space -- is the most dangerous way
    // this diagnostic could fail, so it is surfaced instead of clamped.
    if (captured > energy * (1.0 + 1e-8)) return -1.0;
    numer += energy - captured;
    denom += energy;
  }
  return std::sqrt(std::max(0.0, numer / denom));
}

// Ground state for the diagnostic, with the stall guard this table NEEDS.
//
// The default psi=1 start is a stationary point of the Rayleigh quotient on
// any CONSTANT-link field -- exactly the twist, the discriminating row. The RQ
// iteration then exits with zero residual on an EXCITED eigenvector and every
// number computed from it is silently wrong (observed first time round: SU(2)
// twist returned lambda = 4w instead of the symbol's (4-2sqrt2)w). The
// pathology is measure zero, so a generic start avoids it; two independent
// starts must agree or the row is not trustworthy.
GaugeEigenResult groundStateSun(const SunLattice& L, bool* ok) {
  GaugeEigenOptions eo;
  eo.tol = 1e-10;
  const std::vector<cd> g1 = randomRhs(L.dof(), 777), g2 = randomRhs(L.dof(), 31337);
  const GaugeEigenResult a = smallestEigenpairSunMG(L, &g1, eo);
  const GaugeEigenResult b = smallestEigenpairSunMG(L, &g2, eo);
  *ok = std::abs(a.eigenvalue - b.eigenvalue) <= 1e-6 * std::abs(a.eigenvalue) && a.converged &&
        b.converged;
  return a.eigenvalue <= b.eigenvalue ? a : b;
}

GaugeEigenResult groundStateU1(const GaugeLattice& L, bool* ok) {
  GaugeEigenOptions eo;
  eo.tol = 1e-10;
  const long dof = static_cast<long>(L.lx) * L.ly * L.lz;
  const std::vector<cd> g1 = randomRhs(dof, 777), g2 = randomRhs(dof, 31337);
  const GaugeEigenResult a = smallestEigenpairGaugeMG(L, &g1, eo);
  const GaugeEigenResult b = smallestEigenpairGaugeMG(L, &g2, eo);
  *ok = std::abs(a.eigenvalue - b.eigenvalue) <= 1e-6 * std::abs(a.eigenvalue) && a.converged &&
        b.converged;
  return a.eigenvalue <= b.eigenvalue ? a : b;
}

// diag(L) for a periodic uniform-weight lattice, MEASURED at two sites rather
// than assumed from the stencil, and cross-checked for constancy.
// The full diagonal of L, probed correctly and cheaply.
//
// A strided unit-vector probe is WRONG: with a stride equal to the fiber
// dimension, neighbouring SITES are set too and the result picks up
// off-diagonal contributions. For the 7-point axis stencil the right probe is a
// red-black parity colouring -- within one parity class no two sites are
// adjacent -- so 2 * d' applies of L recover the exact diagonal.
// Constancy is reported, not assumed: on a bounded MAC grid the diagonal varies
// with the neighbour count, which is precisely the demonstration operator.
template <typename ApplyL>
std::vector<double> measuredDiag(ApplyL applyL, int lx, int ly, int lz, int fiber, bool* uniform) {
  const long dof = static_cast<long>(lx) * ly * lz * fiber;
  std::vector<double> d(static_cast<size_t>(dof), 0.0);
  for (int par = 0; par < 2; ++par)
    for (int c = 0; c < fiber; ++c) {
      std::vector<cd> e(static_cast<size_t>(dof), cd(0, 0));
      for (int i = 0; i < lx; ++i)
        for (int j = 0; j < ly; ++j)
          for (int k = 0; k < lz; ++k) {
            if (((i + j + k) & 1) != par) continue;
            e[static_cast<size_t>(((i * ly + j) * lz + k)) * fiber + c] = cd(1, 0);
          }
      const std::vector<cd> Le = applyL(e);
      for (int i = 0; i < lx; ++i)
        for (int j = 0; j < ly; ++j)
          for (int k = 0; k < lz; ++k) {
            if (((i + j + k) & 1) != par) continue;
            const size_t idx = static_cast<size_t>(((i * ly + j) * lz + k)) * fiber + c;
            d[idx] = Le[idx].real();
          }
    }
  *uniform = true;
  for (double x : d)
    if (std::abs(x - d[0]) > 1e-10 * std::abs(d[0])) { *uniform = false; break; }
  return d;
}

// THE LITERATURE'S CRITERION, so ours can be checked against a standard one.
//
// Vassilevski's two-grid characterization (lecture notes / Multilevel Block
// Factorization Preconditioners) is NECESSARY AND SUFFICIENT:
//
//     K_TG = max_v [ min_vc ||v - P vc||^2_Mtilde / ||v||^2_A ]
//
// with the "weak approximation property" ||v - P vc||_Mtilde <= sqrt(K_TG)
// ||v||_A, and Mtilde replaceable by any spectrally equivalent SPD D (the
// familiar form takes D = ||A|| I). Note it minimizes over the coarse space
// exactly as coarseDefectBlock does -- the ONLY difference from our quantity is
// the numerator norm: smoother-induced D rather than the energy norm A. With
// the D-norm the constant IS the sharp two-grid factor; with the A-norm it is
// a weaker cousin with no such interpretation.
//
// Our lattices are periodic with uniform weights, so diag(L) = 2*d*w is a
// CONSTANT multiple of the identity -- verified at run time rather than
// assumed. D therefore contributes an overall scale, which cancels in the
// plain/covariant RATIO, and that ratio is what we compare against.
//
// Falgout's AMG introduction describes the decomposition this feeds
// (falgout2004generalizing, already cited): K_star measures the quality of the
// coarse GRID (the best P possible) and eta the quality of P's COEFFICIENTS.
// Our two transfers share a coarse grid and differ only in coefficients, so
// K_star is common to both and the entire effect sits in eta.
template <typename ApplyP, typename ApplyPadj>
double wapDefectBlock(const std::vector<std::vector<cd>>& psi, long coarseDof,
                      const std::vector<double>& diagD, const std::vector<double>& energies,
                      ApplyP applyP, ApplyPadj applyPadj) {
  const auto dot = [](const std::vector<cd>& a, const std::vector<cd>& b) {
    cd s(0, 0);
    for (size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
    return s;
  };
  const auto Dmul = [&](std::vector<cd> v) {
    for (size_t i = 0; i < v.size(); ++i) v[i] *= diagD[i];
    return v;
  };
  // min ||v - P z||_D^2 over the coarse space: normal equations (P^H D P) z = P^H D v.
  // D is a genuine diagonal here, not a scalar -- on a bounded MAC grid diag(L)
  // varies with the neighbour count, so the scalar shortcut would be wrong.
  const auto PtP = [&](const std::vector<cd>& v) { return applyPadj(Dmul(applyP(v))); };
  double numer = 0.0, denom = 0.0;
  for (size_t c = 0; c < psi.size(); ++c) {
    const std::vector<cd>& v = psi[c];
    const std::vector<cd> g = applyPadj(Dmul(v));
    std::vector<cd> z(static_cast<size_t>(coarseDof), cd(0, 0)), r = g, q = g;
    double rr = dot(r, r).real();
    const double rr0 = rr;
    for (int it = 0; it < 20000 && rr > 1e-24 * rr0; ++it) {
      const std::vector<cd> Aq = PtP(q);
      const double a = rr / dot(q, Aq).real();
      for (size_t i = 0; i < z.size(); ++i) z[i] += a * q[i];
      for (size_t i = 0; i < r.size(); ++i) r[i] -= a * Aq[i];
      const double rrNew = dot(r, r).real();
      const double b = rrNew / rr;
      for (size_t i = 0; i < q.size(); ++i) q[i] = r[i] + b * q[i];
      rr = rrNew;
    }
    const std::vector<cd> Pz = applyP(z);
    double e2 = 0.0;
    for (size_t i = 0; i < v.size(); ++i) e2 += diagD[i] * std::norm(v[i] - Pz[i]);
    numer += e2;
    denom += energies[c];
  }
  return std::sqrt(numer / denom);
}

// ALIASED-FLAT COARSE LEVEL: the kernel, by parallel transport.
//
// ex:alias's singular case is a coarse connection whose links have aliased to
// a FLAT field with trivial holonomy. The SU(2) twist realizes it exactly:
// S^2 = C^2 = -I, so one coarsening makes every link central and every loop
// trivial, and the rediscretized coarse operator is gauge-equivalent to the
// plain massless Laplacian -- singular, kernel = the d covariantly constant
// sections. Those sections can be CONSTRUCTED: transport a color frame from
// the origin along a lex spanning tree, then let the operator itself verify
// global consistency -- ||L v|| ~ 0 fails under non-flatness OR nontrivial
// holonomy OR a wrong transport convention (both orientations are tried), so
// a false positive would require the operator to be singular anyway.
// Returns the normalized sections (mutually orthogonal: unitary transport of
// an orthonormal frame), or empty if the level is not flat-trivial.
std::vector<std::vector<cd>> flatKernelBasis(const SunLattice& Lc) {
  if (!Lc.periodic) return {};
  const int d = Lc.d, dd = d * d;
  const double wscale = 6.0 * Lc.w + Lc.mass2;
  for (int conv = 0; conv < 2; ++conv) {
    std::vector<std::vector<cd>> basis;
    for (int c = 0; c < d; ++c) {
      std::vector<cd> v(static_cast<size_t>(Lc.dof()), cd(0, 0));
      v[static_cast<size_t>(Lc.index(0, 0, 0)) * d + c] = cd(1, 0);
      const auto step = [&](const std::vector<cd>& links, int fi, int fj, int fk, int ti, int tj,
                            int tk) {
        const cd* U = &links[static_cast<size_t>(Lc.index(fi, fj, fk)) * dd];
        const cd* src = &v[static_cast<size_t>(Lc.index(fi, fj, fk)) * d];
        cd* dst = &v[static_cast<size_t>(Lc.index(ti, tj, tk)) * d];
        for (int a = 0; a < d; ++a) {
          cd s(0, 0);
          for (int b = 0; b < d; ++b) s += (conv ? std::conj(U[b * d + a]) : U[a * d + b]) * src[b];
          dst[a] = s;
        }
      };
      for (int i = 1; i < Lc.lx; ++i) step(Lc.ux, i - 1, 0, 0, i, 0, 0);
      for (int i = 0; i < Lc.lx; ++i) {
        for (int j = 1; j < Lc.ly; ++j) step(Lc.uy, i, j - 1, 0, i, j, 0);
        for (int j = 0; j < Lc.ly; ++j)
          for (int k = 1; k < Lc.lz; ++k) step(Lc.uz, i, j, k - 1, i, j, k);
      }
      basis.push_back(std::move(v));
    }
    bool ok = true;
    for (const std::vector<cd>& v : basis) {
      const std::vector<cd> Lv = applySunLaplacian(Lc, v);
      double n2 = 0, r2 = 0;
      for (const cd& z : v) n2 += std::norm(z);
      for (const cd& z : Lv) r2 += std::norm(z);
      if (!(std::sqrt(r2) <= 1e-10 * wscale * std::sqrt(n2))) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    for (std::vector<cd>& v : basis) {
      double n2 = 0;
      for (const cd& z : v) n2 += std::norm(z);
      const double inv = 1.0 / std::sqrt(n2);
      for (cd& z : v) z *= inv;
    }
    return basis;
  }
  return {};
}

// THE SAME QUESTION ASKED OF THE CYCLE WE ACTUALLY RUN.
//
// coarseDefectBlock optimizes over the WHOLE coarse space, which is the
// A-orthogonal projection -- i.e. the correction a GALERKIN coarse operator
// P^H L P would produce with an exact coarse solve. Our coarse operator is
// REdiscretized, so that is an idealization of a cycle we do not run, and the
// gap between the two is exactly the (P, L-tilde) mismatch of sec:harmonic.
//
// So measure the actual thing as well: form the correction with OUR coarse
// operator and OUR line search,
//
//     p = P L-tilde^{-1} P^H L psi,     alpha = <p, L psi> / <p, L p>,
//
// and report the residual energy of psi - alpha p. This optimizes over a
// single DIRECTION rather than the whole space, so it is necessarily >= the
// projection defect; the two agreeing means the rediscretized operator is
// extracting what the space has to offer.
template <typename ApplyL, typename ApplyLc, typename ApplyP, typename ApplyR>
double appliedDefectBlock(const std::vector<std::vector<cd>>& psi, long coarseDof, ApplyL applyL,
                          ApplyLc applyLc, ApplyP applyP, ApplyR applyR,
                          const std::vector<std::vector<cd>>* kernel = nullptr) {
  const auto dot = [](const std::vector<cd>& a, const std::vector<cd>& b) {
    cd s(0, 0);
    for (size_t i = 0; i < a.size(); ++i) s += std::conj(a[i]) * b[i];
    return s;
  };
  // With a singular L-tilde and a KNOWN kernel (flatKernelBasis), the solve
  // that makes sense is the pseudoinverse: project the kernel out of the RHS
  // and keep the residual out of it against roundoff drift -- CG then runs on
  // the orthogonal complement, where L-tilde is positive definite.
  const auto deflate = [&](std::vector<cd>& r) {
    if (!kernel) return;
    for (const std::vector<cd>& kv : *kernel) {
      cd s(0, 0);
      for (size_t i = 0; i < r.size(); ++i) s += std::conj(kv[i]) * r[i];
      for (size_t i = 0; i < r.size(); ++i) r[i] -= s * kv[i];
    }
  };
  double numer = 0.0, denom = 0.0;
  for (const std::vector<cd>& v : psi) {
    const std::vector<cd> Lv = applyL(v);
    const double energy = dot(v, Lv).real();
    std::vector<cd> g = applyR(Lv);
    deflate(g);

    // Coarse solve with the REDISCRETIZED operator (CG; it is SPD -- on the
    // deflated complement when a kernel is supplied).
    std::vector<cd> z(static_cast<size_t>(coarseDof), cd(0, 0)), r = g, q = g;
    double rr = dot(r, r).real();
    const double rr0 = rr;
    int used = 0;
    for (int it = 0; it < 20000 && rr > 1e-24 * rr0; ++it) {
      used = it + 1;
      const std::vector<cd> Aq = applyLc(q);
      const double a = rr / dot(q, Aq).real();
      for (size_t i = 0; i < z.size(); ++i) z[i] += a * q[i];
      for (size_t i = 0; i < r.size(); ++i) r[i] -= a * Aq[i];
      deflate(r);
      const double rrNew = dot(r, r).real();
      const double b = rrNew / rr;
      for (size_t i = 0; i < q.size(); ++i) q[i] = r[i] + b * q[i];
      rr = rrNew;
    }
    // The rediscretized coarse operator can be SINGULAR (ex:alias), in which
    // case this solve is meaningless and so is the number built from it --
    // unless the kernel was detected and deflated above, which is how the
    // aliased-flat rows avoid tripping this.
    if (used >= 20000 || !(rr <= 1e-16 * rr0)) return -2.0;
    const std::vector<cd> pvec = applyP(z);
    const std::vector<cd> Lp = applyL(pvec);
    const double pLp = dot(pvec, Lp).real();
    const double pLpsi = dot(pvec, Lv).real();
    // Energy after the alpha-optimal step: ||psi||^2_L - <p,Lpsi>^2/<p,Lp>.
    const double left = (pLp > 0.0) ? energy - pLpsi * pLpsi / pLp : energy;
    numer += std::max(0.0, left);
    denom += energy;
  }
  return std::sqrt(numer / denom);
}

// The twist: coherent, genuinely non-commuting, and CONSTANT over the lattice.
// Copied verbatim from tools/ablation_bench.cpp (whose validateTwistSun checks
// ||[S,C]|| > 0 and a central plaquette); mirrored here the same way this file
// already mirrors fluxSun/smoothLattice.
SunLattice twistSun(int d, int n) {
  SunLattice L;
  L.d = d;
  L.lx = L.ly = L.lz = n;
  L.periodic = true;
  L.w = static_cast<double>(n) * n;
  L.mass2 = 0.0;
  const int dd = d * d;
  L.ux.resize(static_cast<size_t>(L.numLinksX()) * dd);
  L.uy.resize(static_cast<size_t>(L.numLinksY()) * dd);
  L.uz.resize(static_cast<size_t>(L.numLinksZ()) * dd);
  std::vector<cd> S(dd, cd(0, 0)), C(dd, cd(0, 0)), I(dd, cd(0, 0));
  const cd ph = (d == 2) ? cd(0, 1) : cd(1, 0);  // det fix at d=2
  for (int j = 0; j < d; ++j) S[((j + 1) % d) * d + j] = ph;
  for (int a = 0; a < d; ++a) C[a * d + a] = ph * std::polar(1.0, 2.0 * M_PI * a / d);
  for (int a = 0; a < d; ++a) I[a * d + a] = cd(1, 0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        const size_t e = static_cast<size_t>((i * n + j) * n + k) * dd;
        for (int a = 0; a < dd; ++a) {
          L.ux[e + a] = S[a];
          L.uy[e + a] = C[a];
          L.uz[e + a] = I[a];
        }
      }
  return L;
}

// One row of the mechanism table. mBlock = size of the low subspace whose
// coarse-space representation is measured (see coarseDefectBlock).
constexpr int kBlock = 4;      // minimum block
constexpr int kBlockMax = 12;  // cap on multiplet closing

// Largest m in [kBlock, size] whose level is CLOSED, i.e. followed by a real
// gap. A block that splits a degenerate level spans a basis-dependent subspace,
// so the trace defect stops being a property of the operator -- measured
// directly: SU(2) twist gave applied defects 0.8319 and 0.8915 for the same
// physical field in two gauges until this was fixed. Multiplicities here are
// large and structural (Landau n_Phi-fold, SU(2) Kramers doubling).
int closedBlock(const std::vector<double>& lam) {
  const int cap = std::min<int>(kBlockMax, static_cast<int>(lam.size()) - 1);
  for (int m = kBlock; m <= cap; ++m)
    if (lam[m] - lam[m - 1] > 1e-6 * std::abs(lam[0])) return m;
  return -1;  // no closed block found within the cap
}


// Mechanism-row block-solve cap; BOCHNER_MECH_MAXITERS overrides (certified
// n=64 reruns need more than the default on the twist/MC rows).
int mechMaxIters() {
  const char* e = std::getenv("BOCHNER_MECH_MAXITERS");
  return e ? std::atoi(e) : 600;
}

void mechanismRowSun(const char* label, const SunLattice& L) {
  const auto levels = buildSunLevels(L);
  const Misalign m0 = misalignSun(levels[0]);
  const Misalign m1 = misalignSun(levels[1]);
  // ex:alias realized: if the coarse links have aliased to a flat trivial-
  // holonomy field (the SU(2) twist: S^2 = C^2 = -I), the rediscretized coarse
  // operator is EXACTLY singular. Then (a) the applied-defect solve is the
  // pseudoinverse -- deflate the constructed kernel; (b) the reference block's
  // V-cycle preconditioner is truncated to its single-level branch: every
  // coarser level is equally aliased, and the rows this happens on are hot
  // (twist kappa ~ 10), exactly where the smoother-only branch suffices.
  const std::vector<std::vector<cd>> ker = flatKernelBasis(levels[1]);

  GaugeEigenOptions eo;
  eo.tol = 1e-8;
  eo.maxIters = mechMaxIters();
  if (!ker.empty()) eo.maxLevels = 1;
  BlockEigResult er = lowestEigenpairsSunMG(L, kBlockMax + 1, nullptr, eo);
  const int mUse = closedBlock(er.eigenvalues);
  if (mUse > 0) er.vectors.resize(static_cast<size_t>(mUse));

  const long cdof = static_cast<long>(levels[1].dof());
  const auto applyL = [&](const std::vector<cd>& v) { return applySunLaplacian(L, v); };
  const auto defect = [&](bool cov) {
    return coarseDefectBlock(
        er.vectors, cdof, applyL,
        [&](const std::vector<cd>& v) { return prolongSun(L, v, cov); },
        [&](const std::vector<cd>& v) { return restrictSun(L, v, cov); });
  };
  const double dCov = defect(true), dPlain = defect(false);
  const auto applyLc = [&](const std::vector<cd>& v) { return applySunLaplacian(levels[1], v); };
  const auto applied = [&](bool cov) {
    return appliedDefectBlock(
        er.vectors, cdof, applyL, applyLc,
        [&](const std::vector<cd>& v) { return prolongSun(L, v, cov); },
        [&](const std::vector<cd>& v) { return restrictSun(L, v, cov); },
        ker.empty() ? nullptr : &ker);
  };
  const double aCov = applied(true), aPlain = applied(false);

  bool diagUniform = false;
  const std::vector<double> dD = measuredDiag(applyL, L.lx, L.ly, L.lz, L.d, &diagUniform);
  std::vector<double> energies;
  for (const std::vector<cd>& v : er.vectors) {
    const std::vector<cd> Lv = applyL(v);
    cd s(0, 0);
    for (size_t i = 0; i < v.size(); ++i) s += std::conj(v[i]) * Lv[i];
    energies.push_back(s.real());
  }
  const auto wap = [&](bool cov) {
    return wapDefectBlock(
        er.vectors, cdof, dD, energies,
        [&](const std::vector<cd>& v) { return prolongSun(L, v, cov); },
        [&](const std::vector<cd>& v) { return restrictSun(L, v, cov); });
  };
  const double wCov = wap(true), wPlain = wap(false);

  auto cycles = [&](bool cov) {
    MgOptions mg;
    mg.tol = 1e-8;
    mg.maxCycles = 200;
    mg.covariantTransfer = cov;
    // tab:ablation's deterministic structured RHS, so the cycle columns of the
    // two tables are directly comparable.
    const size_t dof = static_cast<size_t>(L.dof());
    std::vector<cd> b(dof);
    for (size_t i = 0; i < dof; ++i) b[i] = cd(std::sin(0.7 * i), std::cos(0.3 * i));
    std::vector<cd> x(dof, cd(0, 0));
    return vcycleSolveSun(levels, b, x, mg).cycles;
  };
  const int cCov = cycles(true), cPlain = cycles(false);

  std::printf("  %-20s %5.3f   %6.4f %6.4f %5.1f   %7.2f %7.2f %5.1f   %4d %4d%s  %.3f\n", label,
              m0.varMax, dCov, dPlain, dCov > 0 ? dPlain / dCov : -1.0, wCov, wPlain,
              wCov > 0 ? wPlain / wCov : -1.0, cCov, cPlain, cPlain >= 200 ? "!" : " ",
              er.eigenvalues.front());
  if (!diagUniform)
    std::printf("      ^^ diag(L) NOT constant -- D = diagD*I invalid, wap columns not "
                "trustworthy\n");
  std::printf("      [block m=%d closed=%s]\n", mUse, mUse > 0 ? "yes" : "NO");
  if (!ker.empty())
    std::printf("      [coarse rediscretization aliased FLAT (ex:alias): L-tilde exactly "
                "singular, kernel dim %d deflated; reference block at maxLevels=1]\n",
                static_cast<int>(ker.size()));
  if (mUse <= 0)
    std::printf("      ^^ NO CLOSED MULTIPLET within m<=%d -- defect is basis-dependent, "
                "row not trustworthy\n", kBlockMax);
  if (aCov <= -2.0 || aPlain <= -2.0)
    std::printf("      ^^ COARSE SOLVE with the rediscretized operator DID NOT CONVERGE "
                "(singular L-tilde?) -- applied columns not trustworthy\n");
  if (!er.converged)
    std::printf("      ^^ BLOCK EIGENSOLVER DID NOT CONVERGE (res %.1e) -- row not trustworthy\n",
                er.maxResidual);
  if (dCov < 0 || dPlain < 0)
    std::printf("      ^^ COARSE SOLVE BROKE DOWN (defect -1) -- row not trustworthy\n");
  std::fflush(stdout);
}

void mechanismRowU1(const char* label, const GaugeLattice& L) {
  const auto levels = buildGaugeLevels(L);
  const Misalign m0 = misalignU1(levels[0]);
  const Misalign m1 = misalignU1(levels[1]);

  GaugeEigenOptions eo;
  eo.tol = 1e-8;
  eo.maxIters = mechMaxIters();
  BlockEigResult er = lowestEigenpairsGaugeMG(L, kBlockMax + 1, nullptr, eo);
  const int mUse = closedBlock(er.eigenvalues);
  if (mUse > 0) er.vectors.resize(static_cast<size_t>(mUse));

  const long cdof = static_cast<long>(levels[1].lx) * levels[1].ly * levels[1].lz;
  const auto applyL = [&](const std::vector<cd>& v) { return applyConnectionLaplacian(L, v); };
  const auto defect = [&](bool cov) {
    return coarseDefectBlock(
        er.vectors, cdof, applyL,
        [&](const std::vector<cd>& v) { return prolongGauge(L, v, cov); },
        [&](const std::vector<cd>& v) { return restrictGauge(L, v, cov); });
  };
  const double dCov = defect(true), dPlain = defect(false);
  const auto applyLc = [&](const std::vector<cd>& v) {
    return applyConnectionLaplacian(levels[1], v);
  };
  const auto applied = [&](bool cov) {
    return appliedDefectBlock(
        er.vectors, cdof, applyL, applyLc,
        [&](const std::vector<cd>& v) { return prolongGauge(L, v, cov); },
        [&](const std::vector<cd>& v) { return restrictGauge(L, v, cov); });
  };
  const double aCov = applied(true), aPlain = applied(false);

  bool diagUniform = false;
  const std::vector<double> dD = measuredDiag(applyL, L.lx, L.ly, L.lz, 1, &diagUniform);
  std::vector<double> energies;
  for (const std::vector<cd>& v : er.vectors) {
    const std::vector<cd> Lv = applyL(v);
    cd s(0, 0);
    for (size_t i = 0; i < v.size(); ++i) s += std::conj(v[i]) * Lv[i];
    energies.push_back(s.real());
  }
  const auto wap = [&](bool cov) {
    return wapDefectBlock(
        er.vectors, cdof, dD, energies,
        [&](const std::vector<cd>& v) { return prolongGauge(L, v, cov); },
        [&](const std::vector<cd>& v) { return restrictGauge(L, v, cov); });
  };
  const double wCov = wap(true), wPlain = wap(false);

  auto cycles = [&](bool cov) {
    MgOptions mg;
    mg.tol = 1e-8;
    mg.maxCycles = 200;
    mg.covariantTransfer = cov;
    // tab:ablation uses a GAUSSIAN rhs for its U(1) rows and the structured
    // sin/cos one for its SU(d) rows -- two conventions in the one tool. Match
    // each per fiber so both cycle columns reproduce the published table.
    const std::vector<cd> b = randomRhs(static_cast<long>(L.lx) * L.ly * L.lz, 12345);
    std::vector<cd> x(b.size(), cd(0, 0));
    return vcycleSolve(levels, b, x, mg).cycles;
  };
  const int cCov = cycles(true), cPlain = cycles(false);

  std::printf("  %-20s %5.3f   %6.4f %6.4f %5.1f   %7.2f %7.2f %5.1f   %4d %4d%s  %.3f\n", label,
              m0.varMax, dCov, dPlain, dCov > 0 ? dPlain / dCov : -1.0, wCov, wPlain,
              wCov > 0 ? wPlain / wCov : -1.0, cCov, cPlain, cPlain >= 200 ? "!" : " ",
              er.eigenvalues.front());
  if (!diagUniform)
    std::printf("      ^^ diag(L) NOT constant -- D = diagD*I invalid, wap columns not "
                "trustworthy\n");
  std::printf("      [block m=%d closed=%s]\n", mUse, mUse > 0 ? "yes" : "NO");
  if (mUse <= 0)
    std::printf("      ^^ NO CLOSED MULTIPLET within m<=%d -- defect is basis-dependent, "
                "row not trustworthy\n", kBlockMax);
  if (aCov <= -2.0 || aPlain <= -2.0)
    std::printf("      ^^ COARSE SOLVE with the rediscretized operator DID NOT CONVERGE "
                "(singular L-tilde?) -- applied columns not trustworthy\n");
  if (!er.converged)
    std::printf("      ^^ BLOCK EIGENSOLVER DID NOT CONVERGE (res %.1e) -- row not trustworthy\n",
                er.maxResidual);
  if (dCov < 0 || dPlain < 0)
    std::printf("      ^^ COARSE SOLVE BROKE DOWN (defect -1) -- row not trustworthy\n");
  std::fflush(stdout);
}

// Multiplet census: the block defect is only a property of the OPERATOR if the
// block closes a degenerate level. If it cuts a multiplet in half, the trace is
// taken over a basis-dependent subspace and gauge invariance breaks (observed:
// SU(2) twist applied-defect 0.8319 against 0.8915 for the SAME physical field
// re-gauged). SU(2) is pseudo-real, so Kramers doubling multiplies every level
// by two on top of the geometric degeneracy -- exactly the multiplicity-
// integrity trap this project has hit before.
// AMPLITUDE CHECK: tab:ablation's smooth field is built with amp=1.0
// (ablation_bench.cpp), the gauge-transform test with amp=4.0. Both are called
// "the smooth field" in the paper, and they are not the same operator. This
// prints plain-P cycles for both so the two can be compared directly.
void runSmoothAmp() {
  // tab:ablation's harness exactly: amp=1.0 AND the deterministic structured
  // RHS b[i] = (sin 0.7i, cos 0.3i). Reproducing its published numbers is the
  // check that we are talking about the same operator at all.
  std::printf("\n  -- tab:ablation harness (amp=1.0, structured RHS) vs re-gauged --\n");
  std::printf("  %-8s %-4s %-9s %-9s %-9s %-9s\n", "fiber", "n", "full", "plainP",
              "full(gau)", "plainP(gau)");
  for (int d : {2, 3})
    for (int n : {8, 16, 32}) {
      SunLattice A = smoothLattice(d, n, 1.0);
      SunLattice G = A;
      gaugeTransformSun(G, 4242);
      auto cyc = [&](const SunLattice& L, bool cov) {
        const size_t dof = static_cast<size_t>(L.dof());
        std::vector<cd> b(dof);
        for (size_t i = 0; i < dof; ++i) b[i] = cd(std::sin(0.7 * i), std::cos(0.3 * i));
        MgOptions o;
        o.tol = 1e-8;
        o.maxCycles = 200;
        o.covariantTransfer = cov;
        std::vector<cd> x(dof, cd(0, 0));
        const MgResult r = vcycleSolveSun(L, b, x, o);
        return r.relResidual < o.tol ? r.cycles : -1;
      };
      std::printf("  SU(%d)    %-4d %-9d %-9d %-9d %-9d\n", d, n, cyc(A, true), cyc(A, false),
                  cyc(G, true), cyc(G, false));
      std::fflush(stdout);
    }
  std::printf("\n  -- amplitude sweep, random RHS (the mechanism table's harness) --\n");
  std::printf("\n=== SMOOTH-FIELD AMPLITUDE: which field is \"the smooth field\"? ===\n");
  std::printf("  %-8s %-4s %-6s %-8s %-8s\n", "fiber", "n", "amp", "full", "plain-P");
  for (int d : {2, 3})
    for (int n : {8, 16, 32})
      for (double amp : {1.0, 4.0}) {
        const SunLattice L = smoothLattice(d, n, amp);
        const auto levels = buildSunLevels(L);
        auto cyc = [&](bool cov) {
          MgOptions mg;
          mg.tol = 1e-8;
          mg.maxCycles = 200;
          mg.covariantTransfer = cov;
          const std::vector<cd> b = randomRhs(L.dof(), 12345);
          std::vector<cd> x(L.dof(), cd(0, 0));
          return vcycleSolveSun(levels, b, x, mg).cycles;
        };
        std::printf("  SU(%d)    %-4d %-6.1f %-8d %-8d\n", d, n, amp, cyc(true), cyc(false));
        std::fflush(stdout);
      }
}

void runSpectrum() {
  std::printf("\n=== MULTIPLET CENSUS: is a block of %d closing the lowest level? ===\n", kBlock);
  GaugeEigenOptions eo;
  eo.tol = 1e-9;
  const int n = 16, m = 10;
  auto show = [&](const char* label, const SunLattice& L) {
    const BlockEigResult r = lowestEigenpairsSunMG(L, m, nullptr, eo);
    std::printf("  %-20s conv=%d  ", label, static_cast<int>(r.converged));
    for (int i = 0; i < m && i < static_cast<int>(r.eigenvalues.size()); ++i)
      std::printf("%s%.4f ", i == kBlock ? "| " : "", r.eigenvalues[i]);
    std::printf("\n");
    std::fflush(stdout);
  };
  show("SU(2) twist", twistSun(2, n));
  show("SU(3) twist", twistSun(3, n));
  show("SU(2) smooth", smoothLattice(2, n, 4.0));
  show("SU(3) flux nPhi=4", fluxSun(3, n, 4));
  std::printf("  (bar marks where the block of %d cuts; a level split across it\n", kBlock);
  std::printf("   makes the defect basis-dependent and not a property of L.)\n");
}

// THE DEMONSTRATION OPERATOR, ablated on the transfer.
//
// The point: the ring/fluid operator is the paper's flagship application,
// and it must be ablated on the flagship ingredient. If plain
// interpolation also works on the ring, the demonstration does not exercise the
// covariant transfer; if it fails, that is a headline result. Either way it has
// to be measured rather than assumed.
//
// Same construction as tools/eig_compare.cpp (the seeded vortex ring whose
// eigenpair the demo solves per frame), so the operator is the tabulated one.
void mechanismRowRing(int n) {
  const MacGrid g(n, n, n, 1.6 / n, Vec3{-0.8, -0.8, -0.8});
  const double R = 0.7, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
  const auto u = vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, 0.15);
  const auto theta = connectionAngles(g, u, hbar);
  const GaugeLattice L = gaugeLatticeFromFaces(g, theta);

  const auto levels = buildGaugeLevels(L);
  const Misalign m0 = misalignU1(levels[0]);

  GaugeEigenOptions eo;
  eo.tol = 1e-8;
  eo.maxIters = mechMaxIters();
  BlockEigResult er = lowestEigenpairsGaugeMG(L, kBlockMax + 1, nullptr, eo);
  const int mUse = closedBlock(er.eigenvalues);
  if (mUse > 0) er.vectors.resize(static_cast<size_t>(mUse));

  const long cdof = static_cast<long>(levels[1].lx) * levels[1].ly * levels[1].lz;
  const auto applyL = [&](const std::vector<cd>& v) { return applyConnectionLaplacian(L, v); };
  bool diagUniform = false;
  const long dof = static_cast<long>(L.lx) * L.ly * L.lz;
  const std::vector<double> dD = measuredDiag(applyL, L.lx, L.ly, L.lz, 1, &diagUniform);
  std::vector<double> energies;
  for (const std::vector<cd>& v : er.vectors) {
    const std::vector<cd> Lv = applyL(v);
    cd sdot(0, 0);
    for (size_t i = 0; i < v.size(); ++i) sdot += std::conj(v[i]) * Lv[i];
    energies.push_back(sdot.real());
  }
  const auto wap = [&](bool cov) {
    return wapDefectBlock(
        er.vectors, cdof, dD, energies,
        [&](const std::vector<cd>& v) { return prolongGauge(L, v, cov); },
        [&](const std::vector<cd>& v) { return restrictGauge(L, v, cov); });
  };
  const double wCov = wap(true), wPlain = wap(false);

  auto cycles = [&](bool cov) {
    MgOptions mg;
    mg.tol = 1e-8;
    mg.maxCycles = 200;
    mg.covariantTransfer = cov;
    const std::vector<cd> b = randomRhs(dof, 12345);
    std::vector<cd> x(b.size(), cd(0, 0));
    return vcycleSolve(levels, b, x, mg).cycles;
  };
  const int cCov = cycles(true), cPlain = cycles(false);

  std::printf("  ring n=%-3d           %5.3f   %6s %6s %5s   %6.2f %6.2f %5.1f   %4d %4d%s  %.4f\n",
              n, m0.varMax, "-", "-", "-", wCov, wPlain, wCov > 0 ? wPlain / wCov : -1.0, cCov,
              cPlain, cPlain >= 200 ? "!" : " ", er.eigenvalues.front());
  std::printf("      [block m=%d closed=%s, diag uniform=%s]\n", mUse, mUse > 0 ? "yes" : "NO",
              diagUniform ? "yes" : "NO");
  std::fflush(stdout);
}

// THE DEMONSTRATION ABLATION, DONE ON THE RIGHT PROBLEM.
//
// A first version of this ablated the V-cycle as a LINEAR solver on the ring
// with a random right-hand side, and found plain-P nearly matching (20 vs 22
// cycles). That measures the wrong thing. The demo does not solve a linear
// system with a random rhs: it solves the smallest-EIGENVECTOR problem per
// frame, with the V-cycle as the preconditioner inside covMG-LOBPCG, and this
// project's own history is that the hardness there is the eigen-GAP rather
// than conditioning -- which is why linear-solve preconditioners did not help.
//
// The two differ in exactly the way that matters here. A random rhs is
// dominated by high-frequency error the smoother removes whatever the transfer
// does, so the coarse space is barely exercised. The eigenproblem needs the
// coarse space to carry a SMOOTH LOW MODE, which is where covariance should
// bite. So: same operator, same solver, transfer switched.
void runEigAblation() {
  std::printf("\n=== DEMONSTRATION OPERATOR: eigensolver ablated on the transfer ===\n");
  std::printf("  The demo's actual problem (smallest eigenpair, V-cycle as\n");
  std::printf("  preconditioner), not a linear solve with a random rhs.\n\n");
  std::printf("  %-22s %6s  %-14s %-14s %s\n", "operator", "n", "covariant", "plain-P", "ratio");
  for (int n : {16, 24, 32, 46}) {
    const MacGrid g(n, n, n, 1.6 / n, Vec3{-0.8, -0.8, -0.8});
    const double R = 0.7, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
    const auto u = vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, 0.15);
    const auto theta = connectionAngles(g, u, hbar);
    const GaugeLattice L = gaugeLatticeFromFaces(g, theta);
    auto run = [&](bool cov) {
      GaugeEigenOptions eo;
      eo.tol = 1e-8;
      eo.maxIters = 400;
      eo.mg.covariantTransfer = cov;
      return smallestEigenpairGaugeMG(L, nullptr, eo);
    };
    const GaugeEigenResult a = run(true), b = run(false);
    std::printf("  %-22s %6d  %4d it%-8s %4d it%-8s %5.1fx   lam %.6f / %.6f\n", "seeded ring", n,
                a.iterations, a.converged ? "" : " (NO)", b.iterations,
                b.converged ? "" : " (NO)",
                a.iterations > 0 ? double(b.iterations) / a.iterations : -1.0, a.eigenvalue,
                b.eigenvalue);
    std::fflush(stdout);
  }
  // Does the LIVE POLICY mask the transfer difference? The obstacle profiler
  // runs tol=1e-6 with the absolute-drop early exit, and there the two chains
  // came out bit-identical -- which is not a null result but a signature. Same
  // operator, same switch, the two stopping policies side by side.
  std::printf("\n  stopping-policy check (ring): certified vs the live demo policy\n");
  for (int n : {16, 32}) {
    const MacGrid g(n, n, n, 1.6 / n, Vec3{-0.8, -0.8, -0.8});
    const double R = 0.7, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
    const auto u = vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, 0.15);
    const auto theta = connectionAngles(g, u, hbar);
    const GaugeLattice L = gaugeLatticeFromFaces(g, theta);
    for (int live = 0; live < 2; ++live) {
      auto run = [&](bool cov) {
        GaugeEigenOptions eo;
        eo.maxIters = 400;
        eo.mg.covariantTransfer = cov;
        if (live) { eo.tol = 1e-6; eo.relativeGsDrop = false; }
        else      { eo.tol = 1e-8; }
        return smallestEigenpairGaugeMG(L, nullptr, eo);
      };
      const GaugeEigenResult a = run(true), b = run(false);
      std::printf("    n=%-3d %-10s cov %3d it (lam %.10f)  plain %3d it (lam %.10f)  dlam %.1e\n",
                  n, live ? "LIVE" : "certified", a.iterations, a.eigenvalue, b.iterations,
                  b.eigenvalue,
                  std::abs(a.eigenvalue - b.eigenvalue) / std::abs(a.eigenvalue));
      std::fflush(stdout);
    }
  }

  // WHY: the frame the ring operator is presented in. theta = u.h/hbar with a
  // smooth velocity field, so the per-link deviation should be small -- but the
  // circulation around the filament is one quantum (2 pi) by construction, so
  // the CURVATURE cannot vanish. Which of the two the transfer sees is exactly
  // the question tab:frame answers, so measure both here.
  std::printf("\n  frame census of the ring operator (same statistics as the paper's):\n");
  for (int n : {16, 32}) {
    const MacGrid g(n, n, n, 1.6 / n, Vec3{-0.8, -0.8, -0.8});
    const double R = 0.7, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
    const auto u = vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, 0.15);
    const auto theta = connectionAngles(g, u, hbar);
    const GaugeLattice L = gaugeLatticeFromFaces(g, theta);
    char lbl[64];
    std::snprintf(lbl, sizeof lbl, "seeded ring n=%d", n);
    censusU1(lbl, L);
    const Misalign m = misalignU1(buildGaugeLevels(L)[0]);
    std::printf("      misalignment: mag mean %.4f max %.4f   var mean %.4f max %.4f\n", m.mag,
                m.magMax, m.var, m.varMax);
  }

  // Control: an operator where tab:frame says covariance IS load-bearing, run
  // through the same eigensolver path, so the ring's answer can be read against
  // a positive result from the identical harness.
  for (int n : {16, 32}) {
    const GaugeLattice L = uniformFluxLattice(n, 4);
    auto run = [&](bool cov) {
      GaugeEigenOptions eo;
      eo.tol = 1e-8;
      eo.maxIters = 400;
      eo.mg.covariantTransfer = cov;
      return smallestEigenpairGaugeMG(L, nullptr, eo);
    };
    const GaugeEigenResult a = run(true), b = run(false);
    std::printf("  %-22s %6d  %4d it%-8s %4d it%-8s %5.1fx   lam %.6f / %.6f\n",
                "flux torus (control)", n, a.iterations, a.converged ? "" : " (NO)",
                b.iterations, b.converged ? "" : " (NO)",
                a.iterations > 0 ? double(b.iterations) / a.iterations : -1.0, a.eigenvalue,
                b.eigenvalue);
    std::fflush(stdout);
  }
}

// S4: the two coarsening ingredients the paper advertises but never ablated.
//
// tab:ablation carries two switches (line search, covariant transfer). The
// abstract names THREE transfer ingredients, and the other two -- the
// path-product coarse link and the series-conductance coarse weight -- were
// argued for but never run against their counterfactuals. Both counterfactuals
// are named in the literature: averaged-then-reunitarized coarse links (the
// 1990s alternative to an ordered product) and the arithmetic-mean coarse
// weight. Both now exist as CoarsenOptions, so the hierarchy is built by the
// library in either mode rather than reimplemented here.
GaugeLattice hotLatticeLocal(int n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> U(-M_PI, M_PI);
  const std::size_t N = static_cast<std::size_t>(n) * n * n;
  std::vector<double> lkx(N), lky(N), lkz(N);
  for (std::size_t i = 0; i < N; ++i) { lkx[i] = U(rng); lky[i] = U(rng); lkz[i] = U(rng); }
  return gaugeLatticePeriodic(n, n, n, 1.0, lkx, lky, lkz);
}

void runCoarsenAblation() {
  std::printf("\n=== COARSENING ABLATION (S4): the two rules never tested ===\n");
  std::printf("  V-cycles to 1e-8 (cap 200), same rhs and smoother as tab:ablation.\n\n");

  auto cyc = [&](const GaugeLattice& L, const CoarsenOptions& co) {
    const auto levels = buildGaugeLevels(L, co);
    MgOptions mg;
    mg.tol = 1e-8;
    mg.maxCycles = 200;
    const std::vector<cd> b = randomRhs(static_cast<long>(L.lx) * L.ly * L.lz, 12345);
    std::vector<cd> x(b.size(), cd(0, 0));
    const MgResult r = vcycleSolve(levels, b, x, mg);
    return r.relResidual < mg.tol ? r.cycles : -1;
  };

  std::printf("  -- coarse link: ordered PATH PRODUCT vs AVERAGED + reunitarized --\n");
  std::printf("  %-24s %10s %10s\n", "operator", "product", "averaged");
  CoarsenOptions prod, avg;
  avg.link = CoarsenOptions::Link::AveragedReunitarized;
  for (int n : {8, 16, 32}) {
    const GaugeLattice L = uniformFluxLattice(n, 4);
    char lbl[64];
    std::snprintf(lbl, sizeof lbl, "flux torus n=%d", n);
    const int a = cyc(L, prod), b2 = cyc(L, avg);
    std::printf("  %-24s %10d %10s\n", lbl, a, b2 < 0 ? "DIVERGED" : std::to_string(b2).c_str());
    std::fflush(stdout);
  }
  for (int n : {8, 16, 32}) {
    const GaugeLattice L = hotLatticeLocal(n, 2026);
    char lbl[64];
    std::snprintf(lbl, sizeof lbl, "hot links n=%d", n);
    const int a = cyc(L, prod), b2 = cyc(L, avg);
    std::printf("  %-24s %10d %10s\n", lbl, a, b2 < 0 ? "DIVERGED" : std::to_string(b2).c_str());
    std::fflush(stdout);
  }

  CoarsenOptions ser, ari;
  ari.weight = CoarsenOptions::Weight::Arithmetic;

  // Why the weight rule might not show: the line search rescales the coarse
  // correction optimally, so a coarse-operator SCALE error is exactly what it
  // absorbs. Re-run the weight comparison with the step fixed at 1 to see
  // whether the insensitivity is the line search's doing. Hot links, because
  // the no-alpha cycle diverges outright on the flux torus and would
  // discriminate nothing.
  auto cycAlpha = [&](const GaugeLattice& L, const CoarsenOptions& co, bool alpha) {
    const auto levels = buildGaugeLevels(L, co);
    MgOptions mg;
    mg.tol = 1e-8;
    mg.maxCycles = 200;
    mg.alphaStep = alpha;
    const std::vector<cd> b = randomRhs(static_cast<long>(L.lx) * L.ly * L.lz, 12345);
    std::vector<cd> x(b.size(), cd(0, 0));
    const MgResult r = vcycleSolve(levels, b, x, mg);
    return r.relResidual < mg.tol ? r.cycles : -1;
  };
  std::printf("\n  -- is the line search absorbing the weight rule? (hot links, graded) --\n");
  std::printf("  %-24s %9s %9s %9s %9s\n", "operator", "ser+a", "ari+a", "ser-a", "ari-a");
  for (int n : {16, 32}) {
    GaugeLattice L = hotLatticeLocal(n, 2026);
    const double h = 1.0 / n;
    const auto s2 = [](double t) { return std::sin(M_PI * t) * std::sin(M_PI * t); };
    const auto graded = [&](int axis) {
      std::vector<double> wv(static_cast<std::size_t>(
          axis == 0 ? L.numLinksX() : axis == 1 ? L.numLinksY() : L.numLinksZ()));
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          for (int k = 0; k < n; ++k) {
            const double x = (i + 0.5) * h, y = (j + 0.5) * h, z = (k + 0.5) * h;
            const double c = axis == 0   ? 1.0 + 24.0 * s2(x) * s2(y)
                             : axis == 1 ? 1.0 + 24.0 * s2(y) * s2(z)
                                         : 1.0 + 24.0 * s2(z) * s2(x);
            wv[static_cast<std::size_t>((i * n + j) * n + k)] = c;
          }
      return wv;
    };
    L.setEdgeWeights(graded(0), graded(1), graded(2));
    char lbl[64];
    std::snprintf(lbl, sizeof lbl, "hot graded n=%d 25x", n);
    auto fmt = [](int v) { return v < 0 ? std::string("DIV") : std::to_string(v); };
    std::printf("  %-24s %9s %9s %9s %9s\n", lbl, fmt(cycAlpha(L, ser, true)).c_str(),
                fmt(cycAlpha(L, ari, true)).c_str(), fmt(cycAlpha(L, ser, false)).c_str(),
                fmt(cycAlpha(L, ari, false)).c_str());
    std::fflush(stdout);
  }

  // A weight field whose ADJACENT values differ by design. The smooth grading
  // used above varies on the scale of the domain, so neighbouring edges are
  // nearly equal and the two rules differ there only by their constant factor
  // 2 -- which the line search absorbs, so that comparison measured nothing.
  // Alternating weights make the rules differ in SHAPE, not just scale.
  CoarsenOptions am;
  am.weight = CoarsenOptions::Weight::ArithmeticMatched;

  // Alternating weights are still a UNIFORM rescale: every coarse edge combines
  // the same pair (1,C), so matched/series is one constant and the line search
  // absorbs it exactly as before. The ratio has to VARY across edges, which
  // needs weights that are not periodic in the coarsening stride -- random ones.
  std::printf("\n  -- coarse weight, RANDOM per-edge weights in [1,C] --\n");
  std::printf("  %-26s %8s %8s %8s\n", "operator", "series", "arith", "matched");
  for (int n : {16, 32})
    for (double C : {4.0, 25.0}) {
      GaugeLattice L = uniformFluxLattice(n, 4);
      std::mt19937_64 rng(4242);
      std::uniform_real_distribution<double> U(1.0, C);
      const auto rnd = [&](long cnt) {
        std::vector<double> wv(static_cast<std::size_t>(cnt));
        for (auto& v : wv) v = U(rng);
        return wv;
      };
      L.setEdgeWeights(rnd(L.numLinksX()), rnd(L.numLinksY()), rnd(L.numLinksZ()));
      const auto l1 = buildGaugeLevels(L, ser), l3 = buildGaugeLevels(L, am);
      double rmin = 1e30, rmax = 0.0;
      for (std::size_t i = 0; i < l1[1].wx.size(); ++i) {
        const double r = l3[1].wx[i] / l1[1].wx[i];
        rmin = std::min(rmin, r);
        rmax = std::max(rmax, r);
      }
      char lbl[80];
      std::snprintf(lbl, sizeof lbl, "flux n=%d, random 1..%.0f", n, C);
      auto fmt = [](int v) { return v < 0 ? std::string("DIV") : std::to_string(v); };
      std::printf("  %-26s %8s %8s %8s   (matched/series in [%.2f,%.2f])\n", lbl,
                  fmt(cyc(L, ser)).c_str(), fmt(cyc(L, ari)).c_str(), fmt(cyc(L, am)).c_str(),
                  rmin, rmax);
      std::fflush(stdout);
    }

  std::printf("\n  -- coarse weight, ALTERNATING adjacent weights (1:C along each axis) --\n");
  std::printf("  matched = (wa+wb)/8, which equals the series rule when wa=wb, so this\n");
  std::printf("  isolates the rule's shape from the factor-2 scale the line search absorbs.\n");
  std::printf("  %-26s %8s %8s %8s\n", "operator", "series", "arith", "matched");
  for (int n : {16, 32})
    for (double C : {4.0, 25.0}) {
      GaugeLattice L = uniformFluxLattice(n, 4);
      const auto alt = [&](int axis) {
        std::vector<double> wv(static_cast<std::size_t>(
            axis == 0 ? L.numLinksX() : axis == 1 ? L.numLinksY() : L.numLinksZ()));
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
              const int idx = axis == 0 ? i : axis == 1 ? j : k;
              wv[static_cast<std::size_t>((i * n + j) * n + k)] = (idx & 1) ? C : 1.0;
            }
        return wv;
      };
      L.setEdgeWeights(alt(0), alt(1), alt(2));
      // Report how much the rules actually differ per edge, beyond the scale.
      const auto l1 = buildGaugeLevels(L, ser), l3 = buildGaugeLevels(L, am);
      double rmin = 1e30, rmax = 0.0;
      for (std::size_t i = 0; i < l1[1].wx.size(); ++i) {
        const double r = l3[1].wx[i] / l1[1].wx[i];
        rmin = std::min(rmin, r);
        rmax = std::max(rmax, r);
      }
      char lbl[80];
      std::snprintf(lbl, sizeof lbl, "flux n=%d, alt 1:%.0f", n, C);
      auto fmt = [](int v) { return v < 0 ? std::string("DIV") : std::to_string(v); };
      std::printf("  %-26s %8s %8s %8s   (matched/series in [%.2f,%.2f])\n", lbl,
                  fmt(cyc(L, ser)).c_str(), fmt(cyc(L, ari)).c_str(), fmt(cyc(L, am)).c_str(),
                  rmin, rmax);
      std::fflush(stdout);
    }

  std::printf("\n  -- coarse weight: SERIES conductance vs ARITHMETIC mean (smooth grading) --\n");
  std::printf("  (graded weights; uniform lattices give w/4 under both rules)\n");
  std::printf("  %-24s %10s %10s\n", "operator", "series", "arithmetic");
  for (int n : {16, 32})
    for (double amp : {3.0, 24.0}) {
      GaugeLattice L = uniformFluxLattice(n, 4);
      const double h = 1.0 / n, w = static_cast<double>(n) * n;
      const auto s2 = [](double t) { return std::sin(M_PI * t) * std::sin(M_PI * t); };
      const auto graded = [&](int axis) {
        std::vector<double> wv(static_cast<std::size_t>(
            axis == 0 ? L.numLinksX() : axis == 1 ? L.numLinksY() : L.numLinksZ()));
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
              const double x = (i + (axis == 0 ? 0.5 : 0.0)) * h;
              const double y = (j + (axis == 1 ? 0.5 : 0.0)) * h;
              const double z = (k + (axis == 2 ? 0.5 : 0.0)) * h;
              const double c = axis == 0   ? 1.0 + amp * s2(x) * s2(y)
                               : axis == 1 ? 1.0 + amp * s2(y) * s2(z)
                                           : 1.0 + amp * s2(z) * s2(x);
              wv[static_cast<std::size_t>((i * n + j) * n + k)] = w * c;
            }
        return wv;
      };
      L.setEdgeWeights(graded(0), graded(1), graded(2));
      char lbl[64];
      std::snprintf(lbl, sizeof lbl, "graded n=%d contrast %.0fx", n, 1.0 + amp);
      const int a = cyc(L, ser), b2 = cyc(L, ari);
      // Identical iteration counts are only meaningful if the rule actually
      // changed the coarse weights. Measure the difference rather than assume
      // the switch took effect.
      const auto lv1 = buildGaugeLevels(L, ser), lv2 = buildGaugeLevels(L, ari);
      double wdiff = 0.0, wref = 0.0;
      for (std::size_t i = 0; i < lv1[1].wx.size(); ++i) {
        wdiff = std::max(wdiff, std::abs(lv1[1].wx[i] - lv2[1].wx[i]));
        wref = std::max(wref, std::abs(lv1[1].wx[i]));
      }
      std::printf("  %-24s %10d %10s   (coarse w differ by %.2f%% max)\n", lbl, a,
                  b2 < 0 ? "DIVERGED" : std::to_string(b2).c_str(),
                  wref > 0 ? 100.0 * wdiff / wref : -1.0);
      std::fflush(stdout);
    }
}

// The setup side of the rediscretized-vs-Galerkin choice.
//
// Our coarse operator is three arrays of link angles, each entry the sum of the
// two fine angles the coarse edge spans -- one closed-form O(N) pass, and the
// solver never forms a matrix at all. The Galerkin alternative must assemble
// P and L and compute a sparse triple product per level, and in this paper's
// regime that is paid on EVERY rebuild because the connection changes every
// frame. This times our side; tools/galerkin_setup times MatPtAP on the same
// operators for the other.
void runCoarsenTiming() {
  std::printf("\n=== COARSENING SETUP COST: the closed-form link pass ===\n");
  std::printf("  %-8s %-10s %-8s %12s %14s\n", "n", "nodes", "levels", "build ms", "coarse MB");
  for (int n : {16, 32, 64, 96}) {
    const GaugeLattice L = uniformFluxLattice(n, 4);
    std::vector<double> ms;
    std::size_t lv = 0, bytes = 0;
    for (int r = 0; r < 5; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto levels = buildGaugeLevels(L);
      const auto t1 = std::chrono::steady_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      lv = levels.size();
      bytes = 0;
      for (std::size_t i = 1; i < levels.size(); ++i)
        bytes += (levels[i].lkx.size() + levels[i].lky.size() + levels[i].lkz.size()) *
                 sizeof(double);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("  %-8d %-10ld %-8zu %12.2f %14.2f\n", n, L.numNodes(), lv, ms[ms.size() / 2],
                bytes / 1e6);
    std::fflush(stdout);
  }
  std::printf("  (coarse MB = link-angle storage below the fine level; the fine\n");
  std::printf("   level is the operator itself and is common to both schemes)\n");
}

// The discriminating experiment. Families ordered so that the two candidate
// explanations disagree:
//
//   smooth          mag low,  var low   -> plain-P fine under both stories
//   twist           mag HIGH, var ZERO  -> DISCRIMINATOR: magnitude says fail,
//                                          variation says fine (observed: fine)
//   flux torus      mag HIGH, var HIGH  -> both stories say fail
//   gauged smooth   mag HIGH, var HIGH  -> both stories say fail
//   MC (unfixed)    mag HIGH, var HIGH  -> the physical version of the same
//
// n = 16 throughout: the defect solve is a full CG per family and the
// mechanism is a per-level property, not an asymptotic one.
void runMechanism(int n, const char* filter = nullptr) {
  // Optional label-substring filter ("mechanism 64 'SU(2) twist'"): rerun
  // selected rows without paying for the full table.
  const auto want = [&](const char* lbl) { return !filter || std::strstr(lbl, filter) != nullptr; };
  std::printf("\n=== MECHANISM: does plain-P react to the MAGNITUDE or the VARIATION\n");
  std::printf("    of the frame misalignment? ===\n");
  std::printf("  mag = mean ||U-I||    var = mean ||U - Ubar|| (distance to nearest\n");
  std::printf("  CONSTANT field).  defect = energy-norm coarse-space approximation\n");
  std::printf("  error of the true ground state, min_z ||psi - Pz||_L / ||psi||_L.\n");
  std::printf("  The twist is the discriminator: mag HIGH, var ZERO.\n\n");
  std::printf("                             --- ours: A-norm projection ---"
              "  --- literature: weak approx (D-norm) ---\n");
  std::printf("  %-20s %5s   %6s %6s %5s   %7s %7s %5s   %4s %4s  %s\n", "family", "varMx", "cov",
              "plain", "ratio", "cov", "plain", "ratio", "cyc", "plnP", "lam_min");
  std::printf("  lattice n=%d\n", n);
  if (want("SU(2) smooth")) mechanismRowSun("SU(2) smooth", smoothLattice(2, n, 1.0));
  if (want("SU(3) smooth")) mechanismRowSun("SU(3) smooth", smoothLattice(3, n, 1.0));
  if (want("SU(2) twist")) mechanismRowSun("SU(2) twist", twistSun(2, n));
  if (want("SU(3) twist")) mechanismRowSun("SU(3) twist", twistSun(3, n));
  // THE CONTROL that excludes the gap. The twist is far better conditioned
  // than the flux torus (lambda_min ~ 300 against 24.8), and a large gap alone
  // makes plain-P look harmless -- the paper already argues exactly that for
  // hot fields. Re-gauging the twist holds the bundle, the curvature and the
  // ENTIRE SPECTRUM fixed (lambda_min invariant) while making the frame vary
  // from site to site. Magnitude is ~unchanged too (||U-I|| ~ 0.7 either way).
  // So this pair isolates variation with everything else held fixed: if
  // plain-P collapses here, neither the gap nor the magnitude can be the
  // explanation.
  if (want("SU(2) twist gauged")) {
    SunLattice T = twistSun(2, n);
    gaugeTransformSun(T, 4242);
    mechanismRowSun("SU(2) twist gauged", T);
  }
  if (want("SU(3) twist gauged")) {
    SunLattice T = twistSun(3, n);
    gaugeTransformSun(T, 4242);
    mechanismRowSun("SU(3) twist gauged", T);
  }
  if (want("U(1) flux nPhi=4")) mechanismRowU1("U(1) flux nPhi=4", uniformFluxLattice(n, 4));
  if (want("SU(3) flux nPhi=4")) mechanismRowSun("SU(3) flux nPhi=4", fluxSun(3, n, 4));
  if (want("SU(2) smooth gauged")) {
    SunLattice G = smoothLattice(2, n, 1.0);
    gaugeTransformSun(G, 4242);
    mechanismRowSun("SU(2) smooth gauged", G);
  }
  if (want("SU(3) smooth gauged")) {
    SunLattice G = smoothLattice(3, n, 1.0);
    gaugeTransformSun(G, 4242);
    mechanismRowSun("SU(3) smooth gauged", G);
  }
  if (want("ring")) mechanismRowRing(n);
  for (double beta : {2.0, 6.0, 15.0}) {
    char lbl[64];
    std::snprintf(lbl, sizeof lbl, "SU(3) MC beta=%.0f", beta);
    if (want(lbl))
      mechanismRowSun(lbl, mcSunLattice(3, n, beta, 300, static_cast<double>(n) * n, 0.0, 2026ULL,
                                        beta < 6.0));
  }
}

// ---- 2026-08-04 revision experiments -------------------------------------
// Mechanism row on a FROZEN WAKE OPERATOR (revision-plan §5.2: "Run
// flux_alias_diag mechanism on a frozen wake operator and add it as a Table 13
// row"). The operator is dumped by tools/obstacle_profile under
// BOCHNER_DUMP_LAT (the final frame's GaugeLattice: open/Neumann box, uniform
// w = 1/h^2, theta = u*h/hbar from the live velocity field -- the obstacle
// enters through u, not through a mask, so the lattice type is exactly what
// mechanismRowU1 already takes). This mode loads the dump and runs the
// UNMODIFIED mechanismRowU1 on it; the existing guards (non-constant diag(L)
// -> wap warning, non-closed multiplet, non-converged solves) say which
// columns are trustworthy. Optionally (BOCHNER_MECH_EIGABL=<tol>) it also runs
// the certified COLD-start eigensolver ablation (covariant vs plain-P vs
// 1-level) on the same frozen operator.
GaugeLattice loadGaugeLattice(const char* path) {
  std::FILE* fp = std::fopen(path, "rb");
  if (!fp) throw std::runtime_error(std::string("cannot open ") + path);
  const auto rd = [&](void* p, size_t sz, size_t cnt) {
    if (std::fread(p, sz, cnt, fp) != cnt) {
      std::fclose(fp);
      throw std::runtime_error(std::string("short read in ") + path);
    }
  };
  char magic[4];
  int version = 0, per = 0;
  GaugeLattice L;
  rd(magic, 1, 4);
  if (std::memcmp(magic, "GLAT", 4) != 0)
    throw std::runtime_error(std::string("bad magic in ") + path);
  rd(&version, sizeof(int), 1);
  rd(&L.lx, sizeof(int), 1);
  rd(&L.ly, sizeof(int), 1);
  rd(&L.lz, sizeof(int), 1);
  rd(&per, sizeof(int), 1);
  rd(&L.w, sizeof(double), 1);
  L.periodic = per != 0;
  const auto arr = [&](std::vector<double>& a) {
    std::uint64_t sz = 0;
    rd(&sz, sizeof(std::uint64_t), 1);
    a.resize(static_cast<size_t>(sz));
    if (sz) rd(a.data(), sizeof(double), static_cast<size_t>(sz));
  };
  arr(L.lkx);
  arr(L.lky);
  arr(L.lkz);
  std::fclose(fp);
  L.buildTransports();
  return L;
}

// Certified eigensolve rows on a frozen operator file, covariant vs plain
// transfers. Cold constant start, certified stopping (relativeGsDrop),
// tol 1e-7, median of 5.
// 2026-08-19: BOCHNER_EIGFILE_REGAUGE=<seed>[:scale] applies randomGaugeTransform
// (solvers/GaugeMultigrid.h) to the loaded lattice AND to the start vector
// (psi0' = g . psi0), so every row is the exact conjugate of its original-gauge
// counterpart and counts must agree up to rounding.
void runEigFile(const char* path, const char* label) {
  GaugeLattice L = loadGaugeLattice(path);
  std::vector<cd> start(static_cast<std::size_t>(L.numNodes()), cd(1.0, 0.0));
  // BOCHNER_EIGFILE_RANDSTART=<seed>: generic complex-Gaussian start instead of
  // psi=1 (the constant start has near-zero overlap with a localized ground
  // state and its counts are perturbation-sensitive).
  if (const char* e = std::getenv("BOCHNER_EIGFILE_RANDSTART"); e && *e) {
    std::mt19937_64 rng(std::strtoull(e, nullptr, 10));
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& z : start) z = cd(nd(rng), nd(rng));
    std::printf("\n  [random start, seed %s]\n", e);
  }
  if (const char* e = std::getenv("BOCHNER_EIGFILE_REGAUGE"); e && *e) {
    std::uint64_t seed = 0; double scale = M_PI;
    std::sscanf(e, "%llu:%lf", &seed, &scale);
    const std::vector<cd> g = randomGaugeTransform(L, seed, scale);
    for (std::size_t c = 0; c < start.size(); ++c) start[c] *= g[c];
    std::printf("\n  [random re-gauge, seed %llu, |phi| <= %g; start vector transformed with the operator]\n",
                (unsigned long long)seed, scale);
  }
  std::printf("\n=== EIGENSOLVE ROWS on %s (%s) ===\n", path, label);
  std::printf("  lattice %dx%dx%d (periodic=%d), levels=%zu, tol 1e-7 certified, cold start,"
              " median of 5\n",
              L.lx, L.ly, L.lz, L.periodic ? 1 : 0, buildGaugeLevels(L).size());
  // Optional third row (2026-08-19, gauge-invariance check of the polynomial
  // slot): BOCHNER_EIGFILE_CHEB="deg:ratio" runs Chebyshev-LOBPCG (no
  // hierarchy) under the same loop and stopping rule.
  int chebDeg = 0;
  double chebRatio = 0.0;
  if (const char* e = std::getenv("BOCHNER_EIGFILE_CHEB")) std::sscanf(e, "%d:%lf", &chebDeg, &chebRatio);
  for (int row = 0; row < (chebDeg > 0 ? 3 : 2); ++row) {
    const bool cov = row == 0;
    GaugeEigenOptions eo;
    eo.tol = 1e-7;
    eo.relativeGsDrop = true;
    eo.mg.covariantTransfer = cov;
    if (row == 2) {
      eo.chebDegree = chebDeg;
      eo.chebRatio = chebRatio;
    }
    GaugeEigenResult r;
    std::vector<double> ms;
    for (int rep = 0; rep < 5; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      r = smallestEigenpairGaugeMG(L, &start, eo);
      const auto t1 = std::chrono::steady_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    char chebLabel[32];
    std::snprintf(chebLabel, sizeof chebLabel, "cheb %d:%g", chebDeg, chebRatio);
    std::printf("  %-18s | %12.6f | %4d%s | %9.1f ms  res=%.1e\n",
                row == 2 ? chebLabel : (cov ? "covariant transfer" : "plain transfer"), r.eigenvalue, r.iterations,
                r.converged ? "" : "!", ms[ms.size() / 2], r.residual);
  }
}

// Dump the uniform-flux torus as a GLAT file, so the frozen-operator modes
// (eigfile, fluxcensusfile) can run on a synthetic operator as well as on
// the wake dumps.
void dumpTorusGlat(int n, int nPhi, const char* path) {
  const GaugeLattice L = uniformFluxLattice(n, nPhi);
  std::FILE* fp = std::fopen(path, "wb");
  if (!fp) throw std::runtime_error(std::string("cannot open ") + path);
  const char magic[4] = {'G', 'L', 'A', 'T'};
  const int version = 1, per = L.periodic ? 1 : 0;
  std::fwrite(magic, 1, 4, fp);
  std::fwrite(&version, sizeof(int), 1, fp);
  std::fwrite(&L.lx, sizeof(int), 1, fp);
  std::fwrite(&L.ly, sizeof(int), 1, fp);
  std::fwrite(&L.lz, sizeof(int), 1, fp);
  std::fwrite(&per, sizeof(int), 1, fp);
  std::fwrite(&L.w, sizeof(double), 1, fp);
  const auto arr = [&](const std::vector<double>& a) {
    const std::uint64_t sz = a.size();
    std::fwrite(&sz, sizeof(std::uint64_t), 1, fp);
    std::fwrite(a.data(), sizeof(double), a.size(), fp);
  };
  arr(L.lkx);
  arr(L.lky);
  arr(L.lkz);
  std::fclose(fp);
  std::printf("dumped %dx%dx%d torus (nPhi=%d) -> %s\n", L.lx, L.ly, L.lz, nPhi, path);
}

// Per-level plaquette-flux census on a frozen operator. For every
// hierarchy level, histogram |phi| (principal value) over all three plaquette
// orientations, report the fraction past the aliasing point |phi| > pi/2 and
// the frustration moments. Open boundaries: plaquette loops stop one short of
// each non-periodic extent.
void runFluxCensusFile(const char* path) {
  const GaugeLattice L0 = loadGaugeLattice(path);
  const auto levels = buildGaugeLevels(L0);
  std::printf("\n=== PER-LEVEL PLAQUETTE-FLUX CENSUS on %s ===\n", path);
  std::printf("  lattice %dx%dx%d (periodic=%d), levels=%zu; bins are |phi|/pi octiles\n",
              L0.lx, L0.ly, L0.lz, L0.periodic ? 1 : 0, levels.size());
  std::printf("  %3s %12s  %-42s %9s %9s %9s\n", "lvl", "plaquettes",
              "histogram |phi|/pi in [0,1], 8 bins (%)", ">pi/2 (%)", "mean|phi|", "max|phi|");
  for (std::size_t l = 0; l < levels.size(); ++l) {
    const GaugeLattice& L = levels[l];
    const int nx = L.lx, ny = L.ly, nz = L.lz;
    const auto id = [&](int i, int j, int k) { return (i * ny + j) * nz + k; };
    long hist[8] = {0};
    long total = 0, past = 0;
    double sum = 0.0, mx = 0.0;
    const auto tally = [&](double phi) {
      phi = std::remainder(phi, 2.0 * M_PI);
      const double a = std::abs(phi);
      int b = static_cast<int>(a / M_PI * 8.0);
      if (b > 7) b = 7;
      hist[b]++;
      total++;
      if (a > 0.5 * M_PI) past++;
      sum += a;
      mx = std::max(mx, a);
    };
    const int ix = L.periodic ? nx : nx - 1, iy = L.periodic ? ny : ny - 1,
              iz = L.periodic ? nz : nz - 1;
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k) {
          if (i < ix && j < iy)  // xy-plaquette at (i,j,k)
            tally(L.lkx[id(i, j, k)] + L.lky[id((i + 1) % nx, j, k)] -
                  L.lkx[id(i, (j + 1) % ny, k)] - L.lky[id(i, j, k)]);
          if (j < iy && k < iz)  // yz
            tally(L.lky[id(i, j, k)] + L.lkz[id(i, (j + 1) % ny, k)] -
                  L.lky[id(i, j, (k + 1) % nz)] - L.lkz[id(i, j, k)]);
          if (k < iz && i < ix)  // zx
            tally(L.lkz[id(i, j, k)] + L.lkx[id(i, j, (k + 1) % nz)] -
                  L.lkz[id((i + 1) % nx, j, k)] - L.lkx[id(i, j, k)]);
        }
    std::printf("  %3zu %12ld  ", l, total);
    for (int b = 0; b < 8; ++b) std::printf("%4.1f ", 100.0 * hist[b] / std::max(1L, total));
    std::printf("  %8.2f %9.4f %9.4f\n", 100.0 * past / std::max(1L, total),
                sum / std::max(1L, total), mx);
  }
}

void runMechanismFile(const char* path, const char* label) {
  const GaugeLattice L = loadGaugeLattice(path);
  double thmax = 0.0;
  for (double t : L.lkx) thmax = std::max(thmax, std::abs(t));
  for (double t : L.lky) thmax = std::max(thmax, std::abs(t));
  for (double t : L.lkz) thmax = std::max(thmax, std::abs(t));
  const auto levels = buildGaugeLevels(L);
  std::printf("\n=== MECHANISM on a frozen operator from %s ===\n", path);
  std::printf("  lattice %dx%dx%d (periodic=%d, w=%.6g), max|theta|=%.4f, hierarchy levels=%zu\n",
              L.lx, L.ly, L.lz, L.periodic ? 1 : 0, L.w, thmax, levels.size());
  std::printf("\n                             --- ours: A-norm projection ---"
              "  --- literature: weak approx (D-norm) ---\n");
  std::printf("  %-20s %5s   %6s %6s %5s   %7s %7s %5s   %4s %4s  %s\n", "family", "varMx", "cov",
              "plain", "ratio", "cov", "plain", "ratio", "cyc", "plnP", "lam_min");
  mechanismRowU1(label, L);
  if (const char* e = std::getenv("BOCHNER_MECH_EIGABL")) {
    const double tol = std::atof(e);
    const auto run = [&](const char* name, bool cov, int lvl) {
      GaugeEigenOptions o;
      o.tol = tol;  // certified: relativeGsDrop defaults true
      o.maxIters = mechMaxIters();
      o.mg.covariantTransfer = cov;
      o.maxLevels = lvl;
      const auto t0 = std::chrono::steady_clock::now();
      const GaugeEigenResult r = smallestEigenpairGaugeMG(L, nullptr, o);
      const double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("    %-18s %4d its%s %9.1f ms  lambda %.6f  res %.1e\n", name, r.iterations,
                  r.converged ? " " : "!", ms, r.eigenvalue, r.residual);
      std::fflush(stdout);
    };
    std::printf("\n  certified COLD-start eigensolver ablation (tol %.0e, psi=1 start, "
                "single wall-clock run each):\n", tol);
    run("covariant MG", true, 0);
    run("plain-P MG", false, 0);
    run("1-level (GS only)", true, 1);
  }
}

void runCensus(double amp) {
  std::printf("\n=== PER-LEVEL CENSUS: what plain-P sees vs the physical field ===\n");
  std::printf("  linkDev = ||U-I||_F/(2 sqrt d)  GAUGE-DEPENDENT (plain-P's error driver)\n");
  std::printf("  plaqFrus = 1 - Re tr(U_p)/d     GAUGE-INVARIANT (the physical field)\n\n");
  for (int n : {32, 64}) {
    std::printf("  -- n=%d\n", n);
    censusU1("U(1) flux nPhi=4", uniformFluxLattice(n, 4));
    censusSun("SU(2) smooth", smoothLattice(2, n, amp));
    censusSun("SU(3) smooth", smoothLattice(3, n, amp));
    censusSun("SU(3) flux nPhi=4", fluxSun(3, n, 4));
  }
}

// START-POLICY DELTA on the operators the paper headlines. Iteration counts
// only -- no baselines, no timings -- so this is cheap. The question it answers
// is whether switching the default start from psi=1 to a deterministic
// pseudo-random vector would COST or BUY headline numbers, which is the one
// fact that should decide whether a full regeneration is worth it.
//
// Reports the constant start (today's default, hence today's published counts)
// against the min/median/max over several random starts.
void deltaRow(const char* label, int n, long dof,
              const std::function<GaugeEigenResult(const std::vector<cd>*)>& solve) {
  const GaugeEigenResult c = solve(nullptr);
  std::vector<int> r;
  for (int s = 0; s < 5; ++s) {
    const std::vector<cd> g = randomRhs(dof, 4000 + 31 * s);
    r.push_back(solve(&g).iterations);
  }
  std::sort(r.begin(), r.end());
  const int med = r[r.size() / 2];
  std::printf("  %-14s %-4d %-9ld  %4d%s      %2d / %2d / %-2d    %+d\n", label, n, dof,
              c.iterations, c.converged ? " " : "!", r.front(), med, r.back(), med - c.iterations);
}

void runDelta() {
  GaugeEigenOptions eo;
  eo.tol = 1e-7;
  eo.maxIters = 300;
  std::printf("\n=== START-POLICY DELTA: constant psi=1 vs random starts ===\n");
  std::printf("  (delta = median(random) - constant; POSITIVE means random is WORSE)\n\n");
  std::printf("  %-14s %-4s %-9s  %-9s  %-15s %s\n", "operator", "n", "DOF", "constant",
              "random min/med/max", "delta");

  std::printf("  -- U(1) flux torus (tab:torus-eig), nPhi=4\n");
  for (int n : {16, 24, 32, 48, 64}) {
    const GaugeLattice L = uniformFluxLattice(n, 4);
    const long dof = static_cast<long>(n) * n * n;  // GaugeLattice has no dof()
    deltaRow("U(1) flux", n, dof, [&](const std::vector<cd>* g) {
      return smallestEigenpairGaugeMG(L, g, eo);
    });
  }
  for (int d : {2, 3}) {
    std::printf("  -- smooth SU(%d) (tab:nonabelian-eig), the current headline\n", d);
    for (int n : {8, 16, 24, 32, 48}) {
      const SunLattice L = smoothLattice(d, n, 4.0);
      char lab[32];
      std::snprintf(lab, sizeof(lab), "SU(%d) smooth", d);
      deltaRow(lab, n, L.dof(), [&](const std::vector<cd>* g) {
        return smallestEigenpairSunMG(L, g, eo);
      });
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "mcablate") {
    runMcAblation();
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "gauge") {
    runGaugeTest(argc > 2 ? std::atof(argv[2]) : 1.0);
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "census") {
    runCensus(argc > 2 ? std::atof(argv[2]) : 1.0);
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "smoothamp") {
    runSmoothAmp();
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "spectrum") {
    runSpectrum();
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "coarsentime") {
    runCoarsenTiming();
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "coarsen") {
    runCoarsenAblation();
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "eigablate") {
    runEigAblation();
    return 0;
  }
  if (argc > 2 && std::string(argv[1]) == "mechanismfile") {
    runMechanismFile(argv[2], argc > 3 ? argv[3] : "wake (file)");
    return 0;
  }
  if (argc > 2 && std::string(argv[1]) == "fluxcensusfile") {
    runFluxCensusFile(argv[2]);
    return 0;
  }
  if (argc > 2 && std::string(argv[1]) == "eigfile") {
    runEigFile(argv[2], argc > 3 ? argv[3] : "frozen operator");
    return 0;
  }
  if (argc > 4 && std::string(argv[1]) == "dumptorus") {
    dumpTorusGlat(std::atoi(argv[2]), std::atoi(argv[3]), argv[4]);
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "mechanism") {
    runMechanism(argc > 2 ? std::atoi(argv[2]) : 16, argc > 3 ? argv[3] : nullptr);
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "biased") {
    GaugeEigenOptions eo;
    eo.tol = 1e-7;
    eo.maxIters = 300;
    std::printf("\n=== constant-biased (PETSc-character) start on the COHERENT FLUX operator ===\n");
    std::printf("  psi=1 gives 34 its at SU(3) n=64; zero-mean random gives 14-16.\n");
    std::printf("  %-6s %-4s %-10s %-14s\n", "fiber", "n", "psi=1", "const-biased x4");
    for (int d : {3, 2})
      for (int n : {48, 64}) {
        const SunLattice L = fluxSun(d, n, 4);
        const int c = smallestEigenpairSunMG(L, nullptr, eo).iterations;
        std::printf("  SU(%d)  %-4d %-10d ", d, n, c);
        for (int s = 0; s < 4; ++s) {
          const std::vector<cd> g = constBiasedVec(L.dof(), 7000 + 13 * s);
          std::printf("%d ", smallestEigenpairSunMG(L, &g, eo).iterations);
        }
        std::printf("\n");
        std::fflush(stdout);
      }
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "delta") {
    runDelta();
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "start") {
    printStartQuality(2, 4);
    printStartQuality(3, 4);
    return 0;
  }
  const bool seedMode = argc > 1 && std::string(argv[1]) == "seeds";
  if (seedMode) {
    // The anomalous point, plus the largest clean size as a control.
    runSeedProbe(3, 64, 4, 6);
    runSeedProbe(3, 48, 4, 6);
    runSeedProbe(2, 64, 4, 6);
    return 0;
  }
  if (argc > 1) {
    runSweep(std::atoi(argv[1]));
  } else {
    runSweep(2);
    runSweep(3);
  }
  return 0;
}
