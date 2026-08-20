// Ablation study for the paper: which ingredient of the gauge-aware V-cycle is
// load-bearing?
//
// Grid of variants on the SAME operator, hierarchy, and smoother:
//   full        alpha-step ON,  covariant transfer ON   (the paper's method)
//   no-alpha    alpha-step OFF (raw injection)          (historical-PTMG mode)
//   plain-P     alpha ON,  covariant transfer OFF        (isolates the transport)
//   neither     alpha OFF, covariant OFF                 (classical GMG on E)
//
// Operators: (a) uniform-flux torus (nPhi flux quanta, Landau/seam gauge) --
// the paper's Section 5 problem; (b) a hot (uniformly random links) periodic
// U(1) field -- the disordered regime where PTMG historically failed; and the
// non-abelian counterparts, (c) SMOOTH SU(2)/SU(3) and (d) hot SU(2)/SU(3).
//
// The smooth non-abelian rows are the ones that test the COVARIANT TRANSFER:
// on hot fields the random transports average out and plain interpolation
// nearly matches (7 vs 6 cycles, U(1) and SU(N) alike), so covariance only
// shows its value where the connection is coherent -- the flux torus for U(1),
// the smooth field for SU(N).
//
// Reported: V-cycles to relResidual < 1e-8 (cap 200), final residual, and a
// DIVERGED marker when the residual exceeds 1e3 (raw injection can diverge --
// that is the point of the experiment).
//
// Usage: ablation_bench [nPhi]   (default 4)

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "solvers/GaugeMultigrid.h"
#include "solvers/SunGauge.h"
#include "solvers/GaugeEigen.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

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

GaugeLattice hotLattice(int n, std::uint64_t seed) {
  const std::size_t N = static_cast<std::size_t>(n) * n * n;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-M_PI, M_PI);
  std::vector<double> lkx(N), lky(N), lkz(N);
  for (std::size_t e = 0; e < N; ++e) {
    lkx[e] = u(rng);
    lky[e] = u(rng);
    lkz[e] = u(rng);
  }
  return gaugeLatticePeriodic(n, n, n, 1.0, lkx, lky, lkz);
}

struct Variant {
  const char* name;
  bool alpha, covariant;
};

void runOperator(const char* label, const GaugeLattice& lat) {
  const auto levels = buildGaugeLevels(lat);
  std::vector<cd> b(static_cast<std::size_t>(lat.numNodes()));
  std::mt19937_64 rng(12345);
  std::normal_distribution<double> g(0.0, 1.0);
  for (auto& z : b) z = cd(g(rng), g(rng));

  const Variant variants[] = {{"full (alpha+cov)", true, true},
                              {"no-alpha        ", false, true},
                              {"plain-P (alpha) ", true, false},
                              {"neither         ", false, false}};
  std::printf("  %-18s", label);
  for (const auto& v : variants) {
    MgOptions opts;
    opts.tol = 1e-8;
    opts.maxCycles = 200;
    opts.alphaStep = v.alpha;
    opts.covariantTransfer = v.covariant;
    std::vector<cd> x(b.size(), cd(0, 0));
    const MgResult r = vcycleSolve(levels, b, x, opts);
    if (!std::isfinite(r.relResidual) || r.relResidual > 1e3)
      std::printf("  %10s", "DIVERGED");
    else if (r.relResidual > opts.tol)
      std::printf("  stall %.0e", r.relResidual);
    else
      std::printf("  %6d cyc", r.cycles);
  }
  std::printf("\n");
}

}  // namespace


// ---- SU(d) operators ------------------------------------------------------
// smoothLink/smoothSun mirror tools/sun_gauge_bench.cpp so the two tools
// exercise the same field. Faithfulness is checked by reproducing that tool's
// published smooth-field V-cycle counts (7/7/6 at n=8/16/24 for SU(3)) in the
// "full" column here.
void embedSu2Local(cd* M, int d, int p, int q, double v0, double v1, double v2) {
  const double a = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
  const double c = std::cos(a), sn = (a > 1e-12) ? std::sin(a) / a : 1.0;
  M[p * d + p] = cd(c, sn * v2);
  M[p * d + q] = cd(sn * v1, sn * v0);
  M[q * d + p] = cd(-sn * v1, sn * v0);
  M[q * d + q] = cd(c, -sn * v2);
}
void setIdentityLocal(cd* M, int d) {
  for (int i = 0; i < d * d; ++i) M[i] = cd(0, 0);
  for (int i = 0; i < d; ++i) M[i * d + i] = cd(1, 0);
}
void matmulLocal(const cd* A, const cd* B, int d, cd* out) {
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      cd sum(0, 0);
      for (int k = 0; k < d; ++k) sum += A[i * d + k] * B[k * d + j];
      out[i * d + j] = sum;
    }
}
void smoothLinkLocal(int d, int axis, int i, int j, int k, int n, double amp, cd* M) {
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
    setIdentityLocal(M, 2);
    embedSu2Local(M, 2, 0, 1, v[0][0], v[0][1], v[0][2]);
    return;
  }
  cd R01[9], R02[9], R12[9], T[9];
  setIdentityLocal(R01, 3); embedSu2Local(R01, 3, 0, 1, v[0][0], v[0][1], v[0][2]);
  setIdentityLocal(R02, 3); embedSu2Local(R02, 3, 0, 2, v[1][0], v[1][1], v[1][2]);
  setIdentityLocal(R12, 3); embedSu2Local(R12, 3, 1, 2, v[2][0], v[2][1], v[2][2]);
  matmulLocal(R01, R02, 3, T);
  matmulLocal(T, R12, 3, M);
}
// A COHERENTLY FRUSTRATED SU(d) field: the uniform-flux torus embedded in a
// diagonal (Cartan) generator with traceless charges. Unlike the smooth field
// (holonomy ~ h, vanishing under refinement) and the hot field (frustrated but
// incoherent), this carries O(1) STRUCTURED holonomy per plaquette at every n
// -- the non-abelian analogue of the U(1) flux torus, and the configuration
// needed to test whether the covariant transfer is load-bearing on the SU(d)
// code path.
//
// The links commute, so this does not exercise non-commutativity. That is the
// point rather than a limitation: the reason covariance is expected to carry
// over to SU(d) is that the transfer coefficients are real SCALARS, so nothing
// has to commute -- the mechanism under test involves no commutators. What it
// does exercise is the full matrix-transport path (matvec, adjoint, coarse
// link products) under coherent frustration.
//
// Validation: with charges q, the colour components decouple and each sees a
// U(1) flux q_a * nPhi, so lambda_min must equal the U(1) flux-torus value at
// the SMALLEST |q_a| * nPhi -- see validateFluxSun() below for why it is the
// least-frustrated component that sets the ground state. Checked at run time.
SunLattice fluxSun(int d, int n, int nPhi) {
  SunLattice L;
  L.d = d; L.lx = L.ly = L.lz = n; L.periodic = true;
  L.w = static_cast<double>(n) * n; L.mass2 = 0.0;
  const int dd = d * d;
  L.ux.resize(static_cast<size_t>(L.numLinksX()) * dd);
  L.uy.resize(static_cast<size_t>(L.numLinksY()) * dd);
  L.uz.resize(static_cast<size_t>(L.numLinksZ()) * dd);
  double q[3] = {1.0, -1.0, 0.0};                 // SU(2): traceless
  if (d == 3) { q[0] = 1.0; q[1] = 1.0; q[2] = -2.0; }  // SU(3): traceless
  const double phi_p = 2.0 * M_PI * nPhi / (static_cast<double>(n) * n);
  auto setDiag = [&](cd* M, double th) {
    for (int a = 0; a < dd; ++a) M[a] = cd(0, 0);
    for (int a = 0; a < d; ++a) M[a * d + a] = std::polar(1.0, q[a] * th);
  };
  const auto idx = [&](int i, int j, int k) { return (i * n + j) * n + k; };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        const size_t e = static_cast<size_t>(idx(i, j, k)) * dd;
        setDiag(&L.ux[e], -phi_p * j);
        setDiag(&L.uy[e], (j == n - 1) ? 2.0 * M_PI * nPhi * i / static_cast<double>(n) : 0.0);
        setDiag(&L.uz[e], 0.0);
      }
  return L;
}

// 'T HOOFT TWIST: the coherent, genuinely NON-COMMUTING configuration.
//
// The flux field above is coherent but its links commute by construction (a
// U(1) flux in a Cartan direction), and the smooth field is non-commuting but
// not coherent. Neither exercises the remaining corner: coherent frustration
// carried by links that genuinely fail to commute. That is what 't Hooft
// twisted boundary conditions supply, and without it the non-abelian claim
// would rest on commuting-link evidence alone.
//
// Construction: the clock and shift matrices, constant over the lattice.
//   S[i][j] = 1 iff i == (j+1) mod d      (shift)
//   C       = diag(1, w, ..., w^{d-1}),  w = exp(2 pi i / d)   (clock)
// They satisfy S C = w C S, so [S,C] != 0, while the xy plaquette
//   U_p = S C S^H C^H = w^{-1} I
// is CONSTANT and central -- coherent frustration at every plaquette and every
// n, with no position dependence at all. The xz and yz plaquettes are trivial
// (U_z = I and the links are constant), so this is a pure twist in one plane.
//
// Determinants: the 3-cycle shift and the d=3 clock already have det 1; at
// d=2 both S and C are Pauli matrices with det -1, so each is multiplied by i.
SunLattice twistSun(int d, int n) {
  SunLattice L;
  L.d = d; L.lx = L.ly = L.lz = n; L.periodic = true;
  L.w = static_cast<double>(n) * n; L.mass2 = 0.0;
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

// Confirm the twist is what it claims: links that do NOT commute, and a
// constant central plaquette. Printed so the table certifies itself.
void validateTwist(int d) {
  const int dd = d * d;
  const SunLattice L = twistSun(d, 8);
  const cd *S = &L.ux[0], *C = &L.uy[0];
  std::vector<cd> SC(dd), CS(dd), T(dd), P(dd), Sa(dd), Ca(dd);
  matmulLocal(S, C, d, SC.data());
  matmulLocal(C, S, d, CS.data());
  double comm = 0.0;
  for (int a = 0; a < dd; ++a) comm += std::norm(SC[a] - CS[a]);
  for (int a = 0; a < d; ++a)
    for (int b = 0; b < d; ++b) {
      Sa[a * d + b] = std::conj(S[b * d + a]);
      Ca[a * d + b] = std::conj(C[b * d + a]);
    }
  matmulLocal(SC.data(), Sa.data(), d, T.data());
  matmulLocal(T.data(), Ca.data(), d, P.data());
  cd tr(0, 0);
  double offdiag = 0.0;
  for (int a = 0; a < d; ++a) tr += P[a * d + a];
  for (int a = 0; a < d; ++a)
    for (int b = 0; b < d; ++b)
      if (a != b) offdiag += std::norm(P[a * d + b]);
  std::printf("  [validate] SU(%d) twist: ||[S,C]||=%.3f (must be > 0), plaquette "
              "tr/d=%.4f%+.4fi, off-diagonal=%.1e (central)\n",
              d, std::sqrt(comm), tr.real() / d, tr.imag() / d, std::sqrt(offdiag));
}

SunLattice smoothSun(int d, int n) {
  SunLattice L;
  L.d = d; L.lx = L.ly = L.lz = n; L.periodic = true;
  L.w = static_cast<double>(n) * n; L.mass2 = 0.0;
  const int dd = d * d;
  L.ux.resize(static_cast<size_t>(L.numLinksX()) * dd);
  L.uy.resize(static_cast<size_t>(L.numLinksY()) * dd);
  L.uz.resize(static_cast<size_t>(L.numLinksZ()) * dd);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        const size_t e = static_cast<size_t>((i * n + j) * n + k) * dd;
        smoothLinkLocal(d, 0, i, j, k, n, 1.0, &L.ux[e]);
        smoothLinkLocal(d, 1, i, j, k, n, 1.0, &L.uy[e]);
        smoothLinkLocal(d, 2, i, j, k, n, 1.0, &L.uz[e]);
      }
  return L;
}

// Same four-way sweep as runOperator, on the SU(d) path.
void runOperatorSun(const char* label, const SunLattice& L) {
  const size_t dof = static_cast<size_t>(L.dof());
  std::vector<cd> b(dof);
  for (size_t i = 0; i < dof; ++i) b[i] = cd(std::sin(0.7 * i), std::cos(0.3 * i));
  std::printf("  %-18s", label);
  for (int mode = 0; mode < 4; ++mode) {
    MgOptions o;
    o.tol = 1e-8;
    o.maxCycles = 200;
    o.alphaStep = !(mode & 1);
    o.covariantTransfer = !(mode & 2);
    std::vector<cd> x(dof, cd(0, 0));
    const MgResult r = vcycleSolveSun(L, b, x, o);
    if (r.relResidual < o.tol) std::printf("  %6d cyc", r.cycles);
    else std::printf("  %10s", "DIVERGED");
  }
  std::printf("\n");
}

// Confirm the Cartan embedding is what it claims: each colour component sees a
// U(1) flux q_a * nPhi, so the operator block-diagonalizes into d independent
// U(1) problems and lambda_min(SU(d) flux) must equal the U(1) flux-torus
// lambda_min at the SMALLEST |q_a| * nPhi -- more flux means more frustration
// means a HIGHER ground state, so the least-frustrated component sets it.
// (SU(2)'s charges are (1,-1), where largest and smallest coincide; SU(3)'s
// (1,1,-2) distinguishes the two, and it was SU(3) that exposed this predicate
// being written backwards the first time.)
void validateFluxSun(int d, int n, int nPhi) {
  const SunLattice S = fluxSun(d, n, nPhi);
  GaugeEigenOptions eo;
  eo.tol = 1e-9;
  const double lamSun = smallestEigenpairSunMG(S, nullptr, eo).eigenvalue;
  const int qmin = 1;  // min |q_a| is 1 for both (1,-1) and (1,1,-2)
  const double lamU1 = smallestEigenpairGaugeMG(uniformFluxLattice(n, qmin * nPhi), nullptr, eo)
                           .eigenvalue;
  const double rel = std::abs(lamSun - lamU1) / lamU1;
  std::printf("  [validate] SU(%d) flux n=%d nPhi=%d: lambda_min %.6f vs U(1) at nPhi=%d %.6f"
              "  rel=%.2e %s\n",
              d, n, nPhi, lamSun, qmin * nPhi, lamU1, rel, rel < 1e-6 ? "OK" : "*** MISMATCH ***");
}

// EIGENSOLVER ablation: the energy line search against a FIXED step alpha = 1/2.
//
// The line search is what makes the preconditioner nonlinear and non-Hermitian,
// which is precisely why the Knyazev-Neymeyr chain does not cover the solver we
// actually ship: the theory (Corollary cor:notay) describes the step-fixed
// LINEAR cycle. Running covMG-LOBPCG both ways settles which of two things is
// true. If fixing the step costs little, a variant the theory covers can be
// shipped; if it costs a lot, we have quantified what the nonlinearity buys.
// Either way it is worth a row, and the answer is not predictable from the
// linear-solve ablation, where raw injection (alpha = 1) simply diverges.
void runEigenAlpha(const char* label, const GaugeLattice& L) {
  GaugeEigenOptions ls;  // default: energy line search
  ls.tol = 1e-7;
  ls.maxIters = 300;
  GaugeEigenOptions half = ls;  // the linear, theory-covered cycle
  half.mg.alphaStep = false;
  half.mg.fixedAlpha = 0.5;

  const GaugeEigenResult a = smallestEigenpairGaugeMG(L, nullptr, ls);
  std::printf("  %-20s  %4d%s  ", label, a.iterations, a.converged ? " " : "!");
  // Sweep the constant. 1/2 is the 1D-exact step (Lemma lem:1d gives
  // P^H L P = 2 Lc there), but Theorem thm:flat only bounds P^H L P <= 2^d Lc,
  // so in 3D a fixed step that still descends is nearer 2^-d than 1/2 -- and a
  // 1/2 that overshoots would be a property of the CONSTANT, not evidence that
  // the line search is indispensable.
  for (double fa : {0.5, 0.25, 0.125, 0.0625}) {
    GaugeEigenOptions h = half;
    h.mg.fixedAlpha = fa;
    const GaugeEigenResult b = smallestEigenpairGaugeMG(L, nullptr, h);
    std::printf("%4d%s ", b.iterations, b.converged ? " " : "!");
  }
  std::printf("\n");
}

void runEigenAlphaSun(const char* label, const SunLattice& L) {
  GaugeEigenOptions ls;
  ls.tol = 1e-7;
  ls.maxIters = 300;
  GaugeEigenOptions half = ls;
  half.mg.alphaStep = false;
  half.mg.fixedAlpha = 0.5;

  const GaugeEigenResult a = smallestEigenpairSunMG(L, nullptr, ls);
  std::printf("  %-20s  %4d%s  ", label, a.iterations, a.converged ? " " : "!");
  for (double fa : {0.5, 0.25, 0.125, 0.0625}) {
    GaugeEigenOptions h = half;
    h.mg.fixedAlpha = fa;
    const GaugeEigenResult b = smallestEigenpairSunMG(L, nullptr, h);
    std::printf("%4d%s ", b.iterations, b.converged ? " " : "!");
  }
  std::printf("\n");
}

int main(int argc, char** argv) {
  const int nPhi = (argc > 1) ? std::atoi(argv[1]) : 4;
  std::printf("=== V-cycle ablations: alpha-step x covariant transfer (tol 1e-8, cap 200) ===\n");
  std::printf("  %-18s  %-10s  %-10s  %-10s  %-10s\n", "operator", "full", "no-alpha", "plain-P",
              "neither");
  for (int n : {8, 16, 32}) {
    char label[64];
    std::snprintf(label, sizeof label, "flux n=%d nPhi=%d", n, nPhi);
    runOperator(label, uniformFluxLattice(n, nPhi));
  }
  for (int n : {8, 16, 32}) {
    char label[64];
    std::snprintf(label, sizeof label, "hot  n=%d", n);
    runOperator(label, hotLattice(n, 777));
  }
  for (int d : {2, 3}) validateFluxSun(d, 16, nPhi);
  for (int d : {2, 3}) validateTwist(d);
  for (int d : {2, 3})
    for (int n : {8, 16, 32}) {
      char label[64];
      std::snprintf(label, sizeof label, "SU(%d) twist  n=%d", d, n);
      runOperatorSun(label, twistSun(d, n));
    }
  for (int d : {2, 3})
    for (int n : {8, 16, 32}) {
      char label[64];
      std::snprintf(label, sizeof label, "SU(%d) flux   n=%d", d, n);
      runOperatorSun(label, fluxSun(d, n, nPhi));
    }
  for (int d : {2, 3})
    for (int n : {8, 16, 32}) {
      char label[64];
      std::snprintf(label, sizeof label, "SU(%d) smooth n=%d", d, n);
      runOperatorSun(label, smoothSun(d, n));
    }
  for (int d : {2, 3})
    for (int n : {8, 16, 32}) {
      char label[64];
      std::snprintf(label, sizeof label, "SU(%d) hot    n=%d", d, n);
      runOperatorSun(label, randomSunLattice(d, n, n, n, 1.0, 0.0, 777u + d, true));
    }

  std::printf("\n=== Eigensolver: energy line search vs FIXED alpha=1/2 (tol 1e-7) ===\n");
  std::printf("  the fixed-step cycle is the one the theory covers (cor:notay)\n");
  std::printf("  %-20s  %-6s  %-5s%-5s%-5s%-5s\n", "operator", "search", "1/2", "1/4", "1/8",
              "1/16");
  for (int n : {16, 32, 48, 64}) {
    char label[64];
    std::snprintf(label, sizeof label, "flux n=%d nPhi=%d", n, nPhi);
    runEigenAlpha(label, uniformFluxLattice(n, nPhi));
  }
  for (int n : {16, 32, 48}) {
    char label[64];
    std::snprintf(label, sizeof label, "SU(3) smooth n=%d", n);
    runEigenAlphaSun(label, smoothSun(3, n));
  }
  return 0;
}
