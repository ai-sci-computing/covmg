// alphasweep: does the FIXED coarse-correction step alpha=2^{-d} (Table 4,
// tab:alpha) survive outside the two published families, or does the energy
// line search become load-bearing somewhere?
//
// Protocol matches tools/ablation_bench's eigen section exactly:
//   covMG-LOBPCG (smallestEigenpair{Gauge,Sun}MG), tol 1e-7, maxIters 300,
//   psi=1 start (nullptr guess); line search = default GaugeEigenOptions;
//   fixed step = mg.alphaStep=false, mg.fixedAlpha in {1/2,1/4,1/8,1/16}.
//
// Operator constructors are copied VERBATIM from the tools that own them
// (ablation_bench: flux torus, twist, smooth; sun_gauge_bench: smoothLattice
// amp=4 + graded 4x weights; aniso_bench: constant per-axis stretch;
// flux_alias_diag: GLAT loader). No repo files modified.
//
// Usage: alphasweep <family>   family in
//   control | twist | aniso | graded | wake | mc | fluxsun
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"
#include "solvers/SunGauge.h"
#include "solvers/WilsonMc.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

// ---- flux torus (ablation_bench) ------------------------------------------
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

// ---- SU(d) helpers (ablation_bench) ---------------------------------------
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

// smoothLattice with amp parameter: ablation_bench's smoothSun is amp=1
// (Table 4's smooth SU(3) rows); sun_gauge_bench's is amp=4 (tab:graded-eig).
SunLattice smoothSunAmp(int d, int n, double amp) {
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
        smoothLinkLocal(d, 0, i, j, k, n, amp, &L.ux[e]);
        smoothLinkLocal(d, 1, i, j, k, n, amp, &L.uy[e]);
        smoothLinkLocal(d, 2, i, j, k, n, amp, &L.uz[e]);
      }
  return L;
}

// 't Hooft twist (ablation_bench, verbatim).
SunLattice twistSun(int d, int n) {
  SunLattice L;
  L.d = d; L.lx = L.ly = L.lz = n; L.periodic = true;
  L.w = static_cast<double>(n) * n; L.mass2 = 0.0;
  const int dd = d * d;
  L.ux.resize(static_cast<size_t>(L.numLinksX()) * dd);
  L.uy.resize(static_cast<size_t>(L.numLinksY()) * dd);
  L.uz.resize(static_cast<size_t>(L.numLinksZ()) * dd);
  std::vector<cd> S(dd, cd(0, 0)), C(dd, cd(0, 0)), I(dd, cd(0, 0));
  const cd ph = (d == 2) ? cd(0, 1) : cd(1, 0);
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

// Cartan-embedded flux (ablation_bench, verbatim) -- the commuting coherent
// SU(d) family, for context alongside the twist.
SunLattice fluxSun(int d, int n, int nPhi) {
  SunLattice L;
  L.d = d; L.lx = L.ly = L.lz = n; L.periodic = true;
  L.w = static_cast<double>(n) * n; L.mass2 = 0.0;
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

// Graded 4x conductance weights (sun_gauge_bench [2b]/[3b], verbatim).
void setGradedWeights(SunLattice& L, int n) {
  const double h = 1.0 / n;
  const auto cfun = [&](double x, double y) {
    const double sx = std::sin(M_PI * x), sy = std::sin(M_PI * y);
    return 1.0 + 3.0 * sx * sx * sy * sy;
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
          wv[static_cast<std::size_t>((i * n + j) * n + k)] =
              static_cast<double>(n) * n * cfun(x + z, y + z);
        }
    return wv;
  };
  L.setEdgeWeights(graded(0), graded(1), graded(2));
}

// GLAT loader (flux_alias_diag, verbatim).
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

// ---- the sweep -------------------------------------------------------------
// Table-4 protocol. Prints: search its, then its at alpha 1/2,1/4,1/8,1/16.
// '!' = hit the 300-iteration cap; DIV = non-finite result; also prints
// each run's eigenvalue drift vs the line-search value (catches convergence
// to a wrong answer, which a bare count would hide).
void header() {
  std::printf("  %-24s %-7s %-11s %-11s %-11s %-11s  %s\n", "operator", "search", "a=1/2",
              "a=1/4", "a=1/8", "a=1/16", "lambda(search)");
}

int capIters() {
  const char* e = std::getenv("ALPHASWEEP_MAXITERS");
  return e ? std::atoi(e) : 300;
}

template <typename Solve>
void sweepRow(const char* label, const Solve& solve) {
  GaugeEigenOptions ls;
  ls.tol = 1e-7;
  ls.maxIters = capIters();
  const auto t0 = std::chrono::steady_clock::now();
  const GaugeEigenResult a = solve(ls);
  const double ms0 =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  %-24s %4d%s  ", label, a.iterations, a.converged ? " " : "!");
  std::fflush(stdout);
  for (double fa : {0.5, 0.25, 0.125, 0.0625}) {
    GaugeEigenOptions h = ls;
    h.mg.alphaStep = false;
    h.mg.fixedAlpha = fa;
    const GaugeEigenResult b = solve(h);
    const double drift = std::abs(b.eigenvalue - a.eigenvalue) /
                         std::max(std::abs(a.eigenvalue), 1e-30);
    const bool bad = !std::isfinite(b.eigenvalue) || !std::isfinite(b.residual);
    if (bad)
      std::printf("%10s ", "DIV");
    else if (!b.converged)
      std::printf("%4d! r%.0e ", b.iterations, b.residual);
    else if (b.converged && drift > 1e-5)
      std::printf("%4d d%.0e ", b.iterations, drift);  // converged to a DIFFERENT value
    else
      std::printf("%4d       ", b.iterations);
    std::fflush(stdout);
  }
  std::printf(" %.6f%s [%.0f ms]\n", a.eigenvalue, a.converged ? "" : " (search CAP!)", ms0);
  std::fflush(stdout);
}

void sweepU1(const char* label, const GaugeLattice& L) {
  sweepRow(label, [&](const GaugeEigenOptions& o) {
    return smallestEigenpairGaugeMG(L, nullptr, o);
  });
}
void sweepSun(const char* label, const SunLattice& L) {
  sweepRow(label, [&](const GaugeEigenOptions& o) {
    return smallestEigenpairSunMG(L, nullptr, o);
  });
}
// Deterministic pseudo-random start (aniso_bench's randomRhs LCG, seed fixed):
// needed on the twist, where the constant psi=1 start is an EXACT eigenvector
// of the d'=2 operator (S+S^H = C+C^H = 0 for the det-fixed Pauli-type links),
// so the default start certifies at the wrong (non-minimal) eigenvalue.
std::vector<cd> randStart(long dof, unsigned seed) {
  std::vector<cd> b(static_cast<std::size_t>(dof));
  unsigned s = seed;
  const auto next = [&s]() {
    s = 1664525u * s + 1013904223u;
    return (s >> 8) * (1.0 / 16777216.0) - 0.5;
  };
  for (auto& v : b) v = cd(next(), next());
  return b;
}
unsigned startSeed() {
  const char* e = std::getenv("ALPHASWEEP_SEED");
  return e ? static_cast<unsigned>(std::atoi(e)) : 4242u;
}
void sweepSunRand(const char* label, const SunLattice& L) {
  const std::vector<cd> g = randStart(L.dof(), startSeed());
  sweepRow(label, [&](const GaugeEigenOptions& o) {
    return smallestEigenpairSunMG(L, &g, o);
  });
}

}  // namespace

int main(int argc, char** argv) {
  const std::string fam = (argc > 1) ? argv[1] : "control";
  std::printf("=== alphasweep [%s]: line search vs fixed alpha, covMG-LOBPCG tol 1e-7 cap 300 ===\n",
              fam.c_str());
  header();

  if (fam == "control") {
    // Must reproduce Table 4 rows: 10/cap/10/10/14 and 11/cap/14/11/20.
    for (int n : {16, 32}) {
      char lbl[64];
      std::snprintf(lbl, sizeof lbl, "flux torus n=%d nPhi=4", n);
      sweepU1(lbl, uniformFluxLattice(n, 4));
    }
    // Secondary control on the SU(3) smooth family (Table 4: 11/177/56/14/18).
    sweepSun("SU(3) smooth n=16 (amp1)", smoothSunAmp(3, 16, 1.0));
  } else if (fam == "twist") {
    for (int d : {2, 3})
      for (int n : {16, 32}) {
        char lbl[64];
        std::snprintf(lbl, sizeof lbl, "SU(%d) twist n=%d", d, n);
        sweepSun(lbl, twistSun(d, n));
      }
  } else if (fam == "twistr") {
    // Random-start twist rows (see randStart). Analytic check for d'=2:
    // plane-wave band w[6 - 2cos kz +/- 2 sqrt(sin^2 kx + sin^2 ky)] has
    // minimum w(4 - 2 sqrt 2) ~ 1.17157 w at kx=ky=pi/2 (4 | n), so
    // lambda_min = 299.92 (n=16), 1199.68 (n=32).
    for (int d : {2, 3})
      for (int n : {16, 32}) {
        char lbl[64];
        std::snprintf(lbl, sizeof lbl, "SU(%d) twist n=%d rand", d, n);
        sweepSunRand(lbl, twistSun(d, n));
      }
  } else if (fam == "twistlin") {
    // LINEAR V-cycle on the twist: ablation_bench protocol (tol 1e-8, cap
    // 200, random rhs seed 12345-style; here the same b as runOperatorSun).
    // Columns: search (alphaStep=true), then fixed 1/2, 1/4, 1/8, 1/16.
    std::printf("  (linear V-cycle solve, tol 1e-8 cap 200: cycles or DIV)\n");
    for (int d : {2, 3})
      for (int n : {16, 32}) {
        const SunLattice L = twistSun(d, n);
        const size_t dof = static_cast<size_t>(L.dof());
        std::vector<cd> b(dof);
        for (size_t i = 0; i < dof; ++i) b[i] = cd(std::sin(0.7 * i), std::cos(0.3 * i));
        std::printf("  SU(%d) twist n=%-3d lin   ", d, n);
        const auto levels = buildSunLevels(L);
        for (int m = 0; m < 5; ++m) {
          MgOptions o;
          o.tol = 1e-8;
          o.maxCycles = 200;
          o.alphaStep = (m == 0);
          if (m > 0) o.fixedAlpha = std::pow(0.5, m);
          std::vector<cd> x(dof, cd(0, 0));
          const MgResult r = vcycleSolveSun(levels, b, x, o);
          if (!std::isfinite(r.relResidual) || r.relResidual > 1e3)
            std::printf("%10s ", "DIV");
          else if (r.relResidual > o.tol)
            std::printf("%4d! r%.0e ", r.cycles, r.relResidual);
          else
            std::printf("%4d       ", r.cycles);
        }
        std::printf("\n");
      }
  } else if (fam == "fluxsun") {
    for (int d : {2, 3})
      for (int n : {16, 32}) {
        char lbl[64];
        std::snprintf(lbl, sizeof lbl, "SU(%d) flux n=%d nPhi=4", d, n);
        sweepSun(lbl, fluxSun(d, n, 4));
      }
  } else if (fam == "aniso") {
    for (int n : {16, 32})
      for (int r : {8, 32})
        for (int axis : {2, 0}) {
          const double w = double(n) * n;
          GaugeLattice L = uniformFluxLattice(n, 4);
          std::vector<double> wx(L.numLinksX(), w), wy(L.numLinksY(), w), wz(L.numLinksZ(), w);
          auto& stretched = (axis == 0) ? wx : wz;
          for (auto& v : stretched) v = r * w;
          L.setEdgeWeights(wx, wy, wz);
          char lbl[64];
          std::snprintf(lbl, sizeof lbl, "aniso n=%d r=%d (%s)", n, r,
                        axis == 0 ? "x in-plane" : "z transv");
          sweepU1(lbl, L);
        }
  } else if (fam == "graded") {
    for (int d : {2, 3})
      for (int n : {16, 32}) {
        SunLattice L = smoothSunAmp(d, n, 4.0);  // sun_gauge_bench amp=4
        setGradedWeights(L, n);
        char lbl[64];
        std::snprintf(lbl, sizeof lbl, "SU(%d) graded4x n=%d", d, n);
        sweepSun(lbl, L);
      }
  } else if (fam == "wake") {
    // BOCHNER_GLAT_DIR: directory holding the frozen wake dumps
    // (steady_final.glat / shedding_final.glat, written by obstacle_profile
    // under BOCHNER_DUMP_LAT).
    const char* base = std::getenv("BOCHNER_GLAT_DIR");
    if (!base || !*base) {
      std::fprintf(stderr, "wake family needs BOCHNER_GLAT_DIR (frozen .glat dumps)\n");
      return;
    }
    const std::string steady = std::string(base) + "/steady_final.glat";
    const std::string shedding = std::string(base) + "/shedding_final.glat";
    for (const auto& p : {std::make_pair("wake steady", steady),
                          std::make_pair("wake shedding", shedding)}) {
      const GaugeLattice L = loadGaugeLattice(p.second.c_str());
      std::printf("  [loaded %s: %dx%dx%d periodic=%d w=%g]\n", p.first, L.lx, L.ly, L.lz,
                  L.periodic ? 1 : 0, L.w);
      sweepU1(p.first, L);
    }
  } else if (fam == "mc") {
    // mc_gauge_bench protocol: seed 2026 (first published seed), beta=6,
    // 300 heatbath sweeps, cold start (beta >= 6), w=n^2, mass2=0.
    for (int n : {16}) {
      const SunLattice L = mcSunLattice(3, n, 6.0, 300, double(n) * n, 0.0, 2026ULL,
                                        /*hotStart=*/false);
      char lbl[64];
      std::snprintf(lbl, sizeof lbl, "MC SU(3) b=6 n=%d s2026", n);
      sweepSun(lbl, L);
    }
  } else {
    std::fprintf(stderr, "unknown family %s\n", fam.c_str());
    return 1;
  }
  return 0;
}
