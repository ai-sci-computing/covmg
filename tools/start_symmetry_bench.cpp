// Start-policy SYMMETRY control for the eigensolver comparison.
//
// The published tables run covMG-LOBPCG from psi = 1 (the hardcoded default of
// smallestEigenpairSunMG / smallestEigenpairGaugeMG) against a SLEPc
// Krylov-Schur baseline that gets no initial vector at all, so it starts from
// SLEPc's own random one. That is an asymmetry: psi = 1 is the exact ground
// state of the TRIVIAL connection, so on a smooth (near-trivial) field it is a
// warm start carrying real information about the answer -- information the
// baseline is not given. Measured separately, that start is worth about 25% of
// the iteration count on the SU(3) smooth series.
//
// Three configurations, so the effect is bounded from both sides:
//
//   [1] psi=1  /  SLEPc default      -- what the tables currently report
//   [2] psi=1  /  psi=1              -- does the BASELINE benefit from it too?
//                                       If yes, the published margin is partly
//                                       a warm-start artifact.
//   [3] random /  same random        -- fully matched: identical generic start
//                                       for both solvers, no domain knowledge
//                                       on either side. The conservative number.
//
// Configuration [3] is the honest headline if [1] and [3] differ materially:
// it removes the advantage rather than sharing it, and it is the one a hostile
// reader cannot object to. Fixed seed throughout, so [3] is reproducible.
//
// Usage:  start_symmetry_bench [d]     (d = 3 default; 2 also accepted)

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <slepc.h>

#include "BenchTiming.h"
#include "grid/CooMatrix.h"
#include "solvers/EigenSolver.h"
#include "solvers/SunGauge.h"

using namespace bochner;
using cd = std::complex<double>;
using benchstat::medianMs;

namespace {

// --- smooth SU(d) field: mirrors smoothLattice() in sun_gauge_bench.cpp ---
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

// Real embedding: complex dof p -> real indices (2p, 2p+1) = (Re, Im).
// Mirrors assembleReal() in sun_gauge_bench.cpp.
CooMatrix assembleReal(const SunLattice& L) {
  const int d = L.d;
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
  const int lx = L.lx, ly = L.ly, lz = L.lz, dd = d * d;
  const double diag = L.w * 6.0 + L.mass2;
  for (int i = 0; i < lx; ++i)
    for (int j = 0; j < ly; ++j)
      for (int k = 0; k < lz; ++k) {
        const long c = L.index(i, j, k);
        for (int a = 0; a < d; ++a) cplx(c * d + a, c * d + a, cd(diag, 0));
        const int im = (i - 1 + lx) % lx, ip = (i + 1) % lx;
        const int jm = (j - 1 + ly) % ly, jp = (j + 1) % ly;
        const int km = (k - 1 + lz) % lz, kp = (k + 1) % lz;
        block(c, L.index(ip, j, k), &L.ux[static_cast<size_t>(L.index(i, j, k)) * dd], false);
        block(c, L.index(im, j, k), &L.ux[static_cast<size_t>(L.index(im, j, k)) * dd], true);
        block(c, L.index(i, jp, k), &L.uy[static_cast<size_t>(L.index(i, j, k)) * dd], false);
        block(c, L.index(i, jm, k), &L.uy[static_cast<size_t>(L.index(i, jm, k)) * dd], true);
        block(c, L.index(i, j, kp), &L.uz[static_cast<size_t>(L.index(i, j, k)) * dd], false);
        block(c, L.index(i, j, km), &L.uz[static_cast<size_t>(L.index(i, j, km)) * dd], true);
      }
  return A;
}

// Draw a vector the way PETSc/SLEPc draws its own default initial vector, so
// the baseline runs in its natural configuration and OUR solver can be handed
// the identical vector. (A reconstruction of the same draw, not an interception
// of SLEPc's internal one -- same PetscRandom, same distribution.)
//
// NOTE the distribution: PETSc's default RNG is rand48, uniform on [0,1), so
// every component is POSITIVE. Such a vector is ~0.5*(1,1,...,1) plus noise --
// i.e. constant-biased, with large overlap on psi=1. If that is what SLEPc has
// been starting from, then the baseline was already receiving a quasi-warm
// start of the same character as ours, and the "asymmetry" is much smaller than
// it appears. constOverlap() below measures exactly that.
std::vector<double> petscDefaultRandomVec(int M) {
  Vec v;
  VecCreateSeq(PETSC_COMM_SELF, M, &v);
  PetscRandom r;
  PetscRandomCreate(PETSC_COMM_SELF, &r);
  PetscRandomSetFromOptions(r);
  VecSetRandom(v, r);
  const PetscScalar* a = nullptr;
  VecGetArrayRead(v, &a);
  std::vector<double> out(static_cast<std::size_t>(M));
  for (int i = 0; i < M; ++i) out[static_cast<std::size_t>(i)] = PetscRealPart(a[i]);
  VecRestoreArrayRead(v, &a);
  PetscRandomDestroy(&r);
  VecDestroy(&v);
  return out;
}

// Normalized overlap with the constant vector: |<1,v>| / (||1|| ||v||).
// ~1/sqrt(M) for a zero-mean draw; ~0.87 for uniform [0,1).
double constOverlap(const std::vector<double>& v) {
  double s = 0.0, n2 = 0.0;
  for (double x : v) { s += x; n2 += x * x; }
  return std::abs(s) / (std::sqrt(static_cast<double>(v.size())) * std::sqrt(n2));
}

std::vector<cd> fromReal(const std::vector<double>& r) {
  std::vector<cd> v(r.size() / 2);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = cd(r[2 * i], r[2 * i + 1]);
  return v;
}

std::vector<cd> randomVec(long dof, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<cd> v(dof);
  for (auto& z : v) z = cd(g(rng), g(rng));
  return v;
}

// Complex vector -> the interleaved real embedding SLEPc sees.
std::vector<double> toReal(const std::vector<cd>& v) {
  std::vector<double> r(2 * v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    r[2 * i] = v[i].real();
    r[2 * i + 1] = v[i].imag();
  }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  const int d = (argc > 1) ? std::atoi(argv[1]) : 3;

  std::printf("=== START-POLICY SYMMETRY, smooth SU(%d) (the headline operator) ===\n", d);
  std::printf("  [1] psi=1/default  [2] psi=1/psi=1  [3] ours-random/same"
              "  [5] BASELINE'S OWN vector, given to both\n\n");
  std::printf("  %-4s %-9s  %-22s  %-22s  %-22s  %-22s\n", "n", "DOF", "[1] ours|Lanc|speedup",
              "[2] ours|Lanc|speedup", "[3] ours|Lanc|speedup", "[5] ours|Lanc|speedup");

  for (int n : {8, 16, 24, 32, 48}) {
    const SunLattice L = smoothLattice(d, n, 4.0);
    const CooMatrix A = assembleReal(L);
    const long dof = L.dof();
    const std::vector<cd> one(dof, cd(1.0, 0.0));
    const std::vector<cd> rnd = randomVec(dof, 20260720);
    const std::vector<double> oneR = toReal(one), rndR = toReal(rnd);

    GaugeEigenOptions eo;
    eo.tol = 1e-7;
    eo.maxIters = 300;

    // ours: psi=1 (nullptr hits the same default) and random
    GaugeEigenResult e1, e3;
    const double ms1 = medianMs([&] { e1 = smallestEigenpairSunMG(L, nullptr, eo); });
    const double ms3 = medianMs([&] { e3 = smallestEigenpairSunMG(L, &rnd, eo); });

    // baseline: SLEPc default, psi=1, and the same random vector
    EigenPair lD, lOne, lRnd;
    const double msD = medianMs([&] { lD = smallestEigenpairLanczos(A, 1e-7); });
    const double msOne = medianMs([&] { lOne = smallestEigenpairLanczos(A, 1e-7, &oneR); });
    const double msRnd = medianMs([&] { lRnd = smallestEigenpairLanczos(A, 1e-7, &rndR); });

    // [5] the baseline's own kind of vector, handed to BOTH solvers
    const std::vector<double> pR = petscDefaultRandomVec(static_cast<int>(2 * dof));
    const std::vector<cd> pC = fromReal(pR);
    GaugeEigenResult e5;
    EigenPair l5;
    const double ms5 = medianMs([&] { e5 = smallestEigenpairSunMG(L, &pC, eo); });
    const double msP = medianMs([&] { l5 = smallestEigenpairLanczos(A, 1e-7, &pR); });

    std::printf("  %-4d %-9ld  %3d|%8.1f %3d|%8.1f %5.1fx  %3d|%8.1f %3d|%8.1f %5.1fx  "
                "%3d|%8.1f %3d|%8.1f %5.1fx  %3d|%8.1f %3d|%8.1f %5.1fx\n",
                n, dof, e1.iterations, ms1, lD.iterations, msD, msD / ms1, e1.iterations, ms1,
                lOne.iterations, msOne, msOne / ms1, e3.iterations, ms3, lRnd.iterations, msRnd,
                msRnd / ms3, e5.iterations, ms5, l5.iterations, msP, msP / ms5);
    if (n == 8)
      std::printf("      [diag] constant-overlap of the PETSc-default draw = %.3f "
                  "(zero-mean would be ~%.3f)\n",
                  constOverlap(pR), 1.0 / std::sqrt(2.0 * static_cast<double>(dof)));
    std::fflush(stdout);
  }

  benchstat::printTimingSummary();
  SlepcFinalize();
  return 0;
}
