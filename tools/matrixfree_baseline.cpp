// The data-structure control for the eigensolver comparison.
//
// The paper's baseline is SLEPc Krylov-Schur on an ASSEMBLED CSR matrix, while
// covMG-LOBPCG runs on a bespoke matrix-free stencil apply. A matrix-free
// 7-point apply typically beats CSR SpMV by 1.5-3x on memory-bound problems, so
// part of the reported speedup could be the data structure rather than the
// algorithm. The iteration counts carry the structural claim regardless -- flat
// against growing -- but the wall-time factors need this control before they
// are quotable.
//
// WHAT THE CONTROL HAS TO BE. A hand-rolled matrix-free Lanczos is NOT it:
// sun_gauge_bench already has one and marks it an upper bound, because full
// reorthogonalization without thick restart is not competitive with
// Krylov-Schur, and comparing against it would conflate implementation quality
// with data structure. Instead this wraps OUR matrix-free apply in a PETSc
// MatShell and hands it to the SAME EPSKRYLOVSCHUR, with the same problem type,
// tolerance and subspace size. The algorithm is then bit-for-bit the same code
// on the same operator; only the matvec implementation differs.
//
// That design self-validates: the shell and CSR runs must return the SAME
// ITERATION COUNT (they are the same Krylov method on the same operator, up to
// floating-point associativity in the matvec). If they do, the wall-time ratio
// between them is a clean measurement of the data structure alone, and the
// remaining ratio to covMG-LOBPCG is the preconditioner's contribution.
//
// Reported per size:
//   [csr]   Krylov-Schur, assembled CSR      -- the paper's baseline
//   [shell] Krylov-Schur, our matrix-free apply -- new: same algorithm, our data structure
//   [ours]  covMG-LOBPCG, our matrix-free apply
// with csr/shell isolating the data structure and shell/ours the preconditioner.
//
// Usage:  matrixfree_baseline [d]      (d = 3 default; 2 also accepted)

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
        // Adjoint convention copied verbatim from sun_gauge_bench.cpp: the
        // BACKWARD neighbour uses the link stored at that neighbour un-adjointed,
        // the FORWARD neighbour uses the link stored here adjointed. Writing it
        // the other way round assembles the transpose -- same spectrum, different
        // matvec, which operatorMismatch() below catches and an eigenvalue
        // comparison alone would not.
        block(c, L.index(im, j, k), &L.ux[static_cast<size_t>(L.index(im, j, k)) * dd], false);
        block(c, L.index(ip, j, k), &L.ux[static_cast<size_t>(c) * dd], true);
        block(c, L.index(i, jm, k), &L.uy[static_cast<size_t>(L.index(i, jm, k)) * dd], false);
        block(c, L.index(i, jp, k), &L.uy[static_cast<size_t>(c) * dd], true);
        block(c, L.index(i, j, km), &L.uz[static_cast<size_t>(L.index(i, j, km)) * dd], false);
        block(c, L.index(i, j, kp), &L.uz[static_cast<size_t>(c) * dd], true);
      }
  return A;
}

// ---- U(1) flux torus: the operator the paper now headlines ----------------
// Both of these are copied VERBATIM from their originals (uniformFluxLattice
// from ablation_bench.cpp, assemblePeriodic from torus_eig_compare.cpp) rather
// than reconstructed. Reconstructing the SU(d) assembler from its surrounding
// pattern is exactly how the adjoint convention got reversed above.
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
    blk(a, b, -L.w * std::cos(th), L.w * std::sin(th));   // -w e^{-i th}
    blk(b, a, -L.w * std::cos(th), -L.w * std::sin(th));  // conj
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

// The shell's payload: the lattice, plus scratch so MatMult allocates nothing.
struct ShellCtx {
  const SunLattice* lat;
  mutable std::vector<cd> in, out;
};

// y = L x, with x and y in the interleaved real embedding (2p, 2p+1) = (Re, Im)
// that assembleReal() uses -- so the shell and the CSR matrix are the same
// operator in the same basis, and the two runs are directly comparable.
PetscErrorCode shellMatMult(Mat A, Vec x, Vec y) {
  ShellCtx* ctx = nullptr;
  PetscCall(MatShellGetContext(A, &ctx));
  const PetscScalar* xa = nullptr;
  PetscScalar* ya = nullptr;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(y, &ya));
  const std::size_t n = ctx->in.size();
  for (std::size_t i = 0; i < n; ++i)
    ctx->in[i] = cd(PetscRealPart(xa[2 * i]), PetscRealPart(xa[2 * i + 1]));
  ctx->out = applySunLaplacian(*ctx->lat, ctx->in);
  for (std::size_t i = 0; i < n; ++i) {
    ya[2 * i] = ctx->out[i].real();
    ya[2 * i + 1] = ctx->out[i].imag();
  }
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscCall(VecRestoreArray(y, &ya));
  return PETSC_SUCCESS;
}

// Operator-agreement check, run before any timing is believed. The shell and
// the assembled matrix are supposed to BE the same operator in the same basis;
// if they are not, the csr/shell ratio measures two different problems and is
// meaningless. Iteration-count agreement is necessary but NOT sufficient for
// this -- two nearby operators can take the same number of Krylov steps.
double operatorMismatch(const SunLattice& L, const CooMatrix& A) {
  const long dof = L.dof();
  std::vector<cd> x(static_cast<std::size_t>(dof));
  for (long i = 0; i < dof; ++i)
    x[static_cast<std::size_t>(i)] = cd(std::sin(0.7 * i + 0.3), std::cos(0.4 * i + 1.1));

  // matrix-free, embedded to real interleaved
  const std::vector<cd> yMf = applySunLaplacian(L, x);
  std::vector<double> ref(static_cast<std::size_t>(2 * dof));
  for (long i = 0; i < dof; ++i) {
    ref[static_cast<std::size_t>(2 * i)] = yMf[static_cast<std::size_t>(i)].real();
    ref[static_cast<std::size_t>(2 * i + 1)] = yMf[static_cast<std::size_t>(i)].imag();
  }

  // assembled, same embedding
  std::vector<double> xr(static_cast<std::size_t>(2 * dof)), yr(static_cast<std::size_t>(2 * dof), 0.0);
  for (long i = 0; i < dof; ++i) {
    xr[static_cast<std::size_t>(2 * i)] = x[static_cast<std::size_t>(i)].real();
    xr[static_cast<std::size_t>(2 * i + 1)] = x[static_cast<std::size_t>(i)].imag();
  }
  for (const auto& e : A.compressed())
    yr[static_cast<std::size_t>(e.row)] += e.value * xr[static_cast<std::size_t>(e.col)];

  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < yr.size(); ++i) {
    const double d0 = yr[i] - ref[i];
    num += d0 * d0;
    den += ref[i] * ref[i];
  }
  return std::sqrt(num) / std::sqrt(den);
}

// U(1) analogue of ShellCtx/shellMatMult/krylovSchurShell.
struct ShellCtxU1 {
  const GaugeLattice* lat;
  mutable std::vector<cd> in, out;
};

PetscErrorCode shellMatMultU1(Mat A, Vec x, Vec y) {
  ShellCtxU1* ctx = nullptr;
  PetscCall(MatShellGetContext(A, &ctx));
  const PetscScalar* xa = nullptr;
  PetscScalar* ya = nullptr;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(y, &ya));
  const std::size_t n = ctx->in.size();
  for (std::size_t i = 0; i < n; ++i)
    ctx->in[i] = cd(PetscRealPart(xa[2 * i]), PetscRealPart(xa[2 * i + 1]));
  ctx->out = applyConnectionLaplacian(*ctx->lat, ctx->in);
  for (std::size_t i = 0; i < n; ++i) {
    ya[2 * i] = ctx->out[i].real();
    ya[2 * i + 1] = ctx->out[i].imag();
  }
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscCall(VecRestoreArray(y, &ya));
  return PETSC_SUCCESS;
}

double operatorMismatchU1(const GaugeLattice& L, const CooMatrix& A) {
  const long dof = L.numNodes();
  std::vector<cd> x(static_cast<std::size_t>(dof));
  for (long i = 0; i < dof; ++i)
    x[static_cast<std::size_t>(i)] = cd(std::sin(0.7 * i + 0.3), std::cos(0.4 * i + 1.1));
  const std::vector<cd> yMf = applyConnectionLaplacian(L, x);
  std::vector<double> ref(static_cast<std::size_t>(2 * dof)), xr(static_cast<std::size_t>(2 * dof)),
      yr(static_cast<std::size_t>(2 * dof), 0.0);
  for (long i = 0; i < dof; ++i) {
    ref[static_cast<std::size_t>(2 * i)] = yMf[static_cast<std::size_t>(i)].real();
    ref[static_cast<std::size_t>(2 * i + 1)] = yMf[static_cast<std::size_t>(i)].imag();
    xr[static_cast<std::size_t>(2 * i)] = x[static_cast<std::size_t>(i)].real();
    xr[static_cast<std::size_t>(2 * i + 1)] = x[static_cast<std::size_t>(i)].imag();
  }
  for (const auto& e : A.compressed())
    yr[static_cast<std::size_t>(e.row)] += e.value * xr[static_cast<std::size_t>(e.col)];
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < yr.size(); ++i) {
    const double d0 = yr[i] - ref[i];
    num += d0 * d0;
    den += ref[i] * ref[i];
  }
  return std::sqrt(num) / std::sqrt(den);
}

struct KsResult {
  double lambda = 0.0;
  int iterations = 0;
};

// Krylov-Schur on a MatShell around our matrix-free apply. Every EPS setting
// matches smallestEigenpairLanczos()'s assembled path, so the only difference
// between the two runs is how the matvec is performed.
KsResult krylovSchurShell(const SunLattice& L, double tol) {
  const long dof = L.dof();
  const PetscInt M = static_cast<PetscInt>(2 * dof);
  ShellCtx ctx;
  ctx.lat = &L;
  ctx.in.resize(static_cast<std::size_t>(dof));
  ctx.out.resize(static_cast<std::size_t>(dof));

  Mat A;
  MatCreateShell(PETSC_COMM_SELF, M, M, M, M, &ctx, &A);
  MatShellSetOperation(A, MATOP_MULT, reinterpret_cast<void (*)(void)>(shellMatMult));
  // The operator IS Hermitian (real-symmetric in this embedding); EPS_HEP below
  // asserts it, and MAT_SYMMETRIC lets SLEPc skip its own transpose path.
  MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);

  EPS eps;
  EPSCreate(PETSC_COMM_SELF, &eps);
  EPSSetOperators(eps, A, nullptr);
  EPSSetProblemType(eps, EPS_HEP);
  EPSSetType(eps, EPSKRYLOVSCHUR);
  EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
  EPSSetDimensions(eps, 1, PETSC_DEFAULT, PETSC_DEFAULT);
  EPSSetTolerances(eps, tol, PETSC_DEFAULT);
  EPSSetFromOptions(eps);
  EPSSolve(eps);

  KsResult r;
  PetscInt nconv = 0, its = 0;
  EPSGetConverged(eps, &nconv);
  EPSGetIterationNumber(eps, &its);
  r.iterations = static_cast<int>(its);
  if (nconv > 0) {
    PetscScalar kr = 0.0, ki = 0.0;
    EPSGetEigenvalue(eps, 0, &kr, &ki);
    r.lambda = PetscRealPart(kr);
  }
  EPSDestroy(&eps);
  MatDestroy(&A);
  return r;
}

KsResult krylovSchurShellU1(const GaugeLattice& L, double tol) {
  const long dof = L.numNodes();
  const PetscInt M = static_cast<PetscInt>(2 * dof);
  ShellCtxU1 ctx;
  ctx.lat = &L;
  ctx.in.resize(static_cast<std::size_t>(dof));
  ctx.out.resize(static_cast<std::size_t>(dof));
  Mat A;
  MatCreateShell(PETSC_COMM_SELF, M, M, M, M, &ctx, &A);
  MatShellSetOperation(A, MATOP_MULT, reinterpret_cast<void (*)(void)>(shellMatMultU1));
  MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);
  EPS eps;
  EPSCreate(PETSC_COMM_SELF, &eps);
  EPSSetOperators(eps, A, nullptr);
  EPSSetProblemType(eps, EPS_HEP);
  EPSSetType(eps, EPSKRYLOVSCHUR);
  EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
  EPSSetDimensions(eps, 1, PETSC_DEFAULT, PETSC_DEFAULT);
  EPSSetTolerances(eps, tol, PETSC_DEFAULT);
  EPSSetFromOptions(eps);
  EPSSolve(eps);
  KsResult r;
  PetscInt nconv = 0, its = 0;
  EPSGetConverged(eps, &nconv);
  EPSGetIterationNumber(eps, &its);
  r.iterations = static_cast<int>(its);
  if (nconv > 0) {
    PetscScalar kr = 0.0, ki = 0.0;
    EPSGetEigenvalue(eps, 0, &kr, &ki);
    r.lambda = PetscRealPart(kr);
  }
  EPSDestroy(&eps);
  MatDestroy(&A);
  return r;
}

// Same decomposition on the U(1) uniform-flux torus -- the operator the paper
// headlines, and therefore the one whose
// wall-time factors most need this control.
void runTorus() {
  std::printf("\n=== MATRIX-FREE CONTROL, U(1) flux torus nPhi=4 (the headline operator) ===\n");
  std::printf("  %-4s %-9s  %-16s %-16s %-16s  %-9s %-9s %s\n", "n", "DOF", "csr (its, ms)",
              "shell (its, ms)", "ours (its, ms)", "csr/shell", "shell/ours", "csr/ours");
  for (int n : {16, 24, 32, 48, 64}) {
    const GaugeLattice L = uniformFluxLattice(n, 4);
    const CooMatrix A = assemblePeriodic(L);
    const double mism = operatorMismatchU1(L, A);
    if (mism > 1e-12) {
      std::printf("  n=%-4d *** OPERATOR MISMATCH %.3e\n", n, mism);
      continue;
    }
    EigenPair csr;
    const double msCsr = medianMs([&] { csr = smallestEigenpairLanczos(A, 1e-7); });
    KsResult sh;
    const double msSh = medianMs([&] { sh = krylovSchurShellU1(L, 1e-7); });
    GaugeEigenOptions eo;
    eo.tol = 1e-7;
    eo.maxIters = 300;
    GaugeEigenResult er;
    const double msOurs = medianMs([&] { er = smallestEigenpairGaugeMG(L, nullptr, eo); });
    std::printf("  %-4d %-9ld  %4d %10.1f %4d %10.1f %4d %10.1f  %8.2fx %8.2fx %7.2fx  |dl|=%.1e\n",
                n, 2 * L.numNodes(), csr.iterations, msCsr, sh.iterations, msSh, er.iterations,
                msOurs, msCsr / msSh, msSh / msOurs, msCsr / msOurs,
                std::abs(csr.value - sh.lambda));
    std::fflush(stdout);
  }
}

}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  const int d = (argc > 1) ? std::atoi(argv[1]) : 3;

  std::printf("=== MATRIX-FREE CONTROL, smooth SU(%d): what is algorithm, what is layout? ===\n", d);
  std::printf("  [csr]   Krylov-Schur on the assembled matrix   (the paper's baseline)\n");
  std::printf("  [shell] Krylov-Schur on OUR matrix-free apply  (same algorithm, our layout)\n");
  std::printf("  [ours]  covMG-LOBPCG on the same apply\n");
  std::printf("  csr/shell isolates the data structure; shell/ours the preconditioner.\n");
  std::printf("  Iteration counts of csr and shell must agree -- that validates the shell.\n\n");
  std::printf("  %-4s %-9s  %-16s %-16s %-16s  %-9s %-9s %s\n", "n", "DOF", "csr (its, ms)",
              "shell (its, ms)", "ours (its, ms)", "csr/shell", "shell/ours", "csr/ours");

  for (int n : {8, 16, 24, 32, 48}) {
    const SunLattice L = smoothLattice(d, n, 4.0);
    const CooMatrix A = assembleReal(L);
    const double mism = operatorMismatch(L, A);
    if (mism > 1e-12) {
      std::printf("  n=%-4d *** OPERATOR MISMATCH %.3e -- shell and CSR are NOT the same\n", n,
                  mism);
      std::fflush(stdout);
      continue;
    }

    EigenPair csr;
    const double msCsr = medianMs([&] { csr = smallestEigenpairLanczos(A, 1e-7); });
    KsResult sh;
    const double msSh = medianMs([&] { sh = krylovSchurShell(L, 1e-7); });
    GaugeEigenOptions eo;
    eo.tol = 1e-7;
    eo.maxIters = 300;
    GaugeEigenResult er;
    const double msOurs = medianMs([&] { er = smallestEigenpairSunMG(L, nullptr, eo); });

    std::printf("  %-4d %-9ld  %4d %10.1f %4d %10.1f %4d %10.1f  %8.2fx %8.2fx %7.2fx  "
                "|dl|=%.1e\n",
                n, L.dof(), csr.iterations, msCsr, sh.iterations, msSh, er.iterations, msOurs,
                msCsr / msSh, msSh / msOurs, msCsr / msOurs, std::abs(csr.value - sh.lambda));
    std::fflush(stdout);
  }

  runTorus();

  benchstat::printTimingSummary();
  SlepcFinalize();
  return 0;
}
