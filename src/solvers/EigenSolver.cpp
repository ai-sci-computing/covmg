#include "solvers/EigenSolver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#include <petscksp.h>
#include <slepceps.h>

namespace bochner {

namespace {

void check(PetscErrorCode ierr, const char* what) {
  if (ierr != 0) {
    throw std::runtime_error(std::string("PETSc/SLEPc error (") + std::to_string(ierr) + ") in " +
                             what);
  }
}

template <class T, PetscErrorCode (*Destroy)(T*)>
struct Owned {
  T obj = nullptr;
  ~Owned() {
    if (obj) Destroy(&obj);
  }
};

// Assemble a sequential AIJ PETSc matrix from the COO operator; also report the
// largest |diagonal| (used to scale the inverse-iteration shift).
void assembleMat(const CooMatrix& A, Mat* out, double* maxDiag) {
  const PetscInt n = A.rows();
  const auto entries = A.compressed();
  std::vector<PetscInt> nnzPerRow(static_cast<std::size_t>(n), 0);
  *maxDiag = 0.0;
  for (const auto& e : entries) {
    ++nnzPerRow[static_cast<std::size_t>(e.row)];
    if (e.row == e.col) *maxDiag = std::max(*maxDiag, std::abs(e.value));
  }
  check(MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 0, nnzPerRow.data(), out), "MatCreateSeqAIJ");
  check(MatSetOption(*out, MAT_SYMMETRIC, PETSC_TRUE), "MatSetOption");
  for (const auto& e : entries)
    check(MatSetValue(*out, e.row, e.col, e.value, INSERT_VALUES), "MatSetValue");
  check(MatAssemblyBegin(*out, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
  check(MatAssemblyEnd(*out, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
}

// The two preconditioners that survive the eigensolver comparison: ICC (the
// fastest backsolve for this SPD operator, used by the CG-based methods) and
// AMG (hypre BoomerAMG; the block methods substitute GAMG in setupBlockPreconditioner).
void setInnerPC(PC pc, InnerPC kind) {
  switch (kind) {
    case InnerPC::ICC:
      check(PCSetType(pc, PCICC), "PCSetType icc");
      break;
    case InnerPC::AMG: {
      // Probe availability instead of branching on a raw PETSc return: a failed
      // PCSetType pushes onto the error stack (tracebacks on hypre-less builds,
      // death under -on_error_abort) even though the GAMG fallback would work.
      PetscBool hasHypre = PETSC_FALSE;
      check(PetscHasExternalPackage("hypre", &hasHypre), "PetscHasExternalPackage");
      if (hasHypre) {
        check(PCSetType(pc, PCHYPRE), "PCSetType hypre");
        check(PCHYPRESetType(pc, "boomeramg"), "PCHYPRESetType");
      } else {
        check(PCSetType(pc, PCGAMG), "PCSetType gamg fallback");
      }
      break;
    }
    case InnerPC::Jacobi:
      check(PCSetType(pc, PCJACOBI), "PCSetType jacobi");
      break;
  }
}

// PCSHELL plumbing for the caller-supplied preconditioner (the covariant
// V-cycle inside a foreign outer loop). PETSc's default PCMatApply loops the
// single apply over block columns, so the block methods (LOBPCG/GD) work
// without a dedicated shell matapply.
struct ShellPcCtx {
  const ShellPcApply* apply = nullptr;
};

PetscErrorCode shellPcApplyCb(PC pc, Vec x, Vec y) {
  void* raw = nullptr;
  PetscCall(PCShellGetContext(pc, &raw));
  auto* ctx = static_cast<ShellPcCtx*>(raw);
  const PetscScalar* xa = nullptr;
  PetscScalar* ya = nullptr;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(y, &ya));
  (*ctx->apply)(xa, ya);
  PetscCall(VecRestoreArray(y, &ya));
  PetscCall(VecRestoreArrayRead(x, &xa));
  return PETSC_SUCCESS;
}

void setupShellPreconditioner(EPS eps, ShellPcCtx* ctx) {
  ST st = nullptr;
  check(EPSGetST(eps, &st), "EPSGetST");
  check(STSetType(st, STPRECOND), "STSetType precond");
  KSP ksp = nullptr;
  check(STGetKSP(st, &ksp), "STGetKSP");
  check(KSPSetType(ksp, KSPPREONLY), "KSPSetType preonly");
  PC pc = nullptr;
  check(KSPGetPC(ksp, &pc), "KSPGetPC");
  check(PCSetType(pc, PCSHELL), "PCSetType shell");
  check(PCShellSetContext(pc, ctx), "PCShellSetContext");
  check(PCShellSetApply(pc, shellPcApplyCb), "PCShellSetApply");
}

// Block-method preconditioner plug point: ST = STPRECOND with KSPPREONLY, so the
// PC IS the inner solve. The block eigensolvers (LOBPCG, GD) precondition the
// whole trial block via PCMatApply, and hypre BoomerAMG has no working block
// apply in this build -- so for AMG we use PETSc's native GAMG (smoothed
// aggregation), which does. Shared so this GAMG-vs-hypre special case lives once.
void setupBlockPreconditioner(EPS eps, InnerPC pcKind) {
  ST st = nullptr;
  check(EPSGetST(eps, &st), "EPSGetST");
  check(STSetType(st, STPRECOND), "STSetType precond");
  KSP ksp = nullptr;
  check(STGetKSP(st, &ksp), "STGetKSP");
  check(KSPSetType(ksp, KSPPREONLY), "KSPSetType preonly");
  PC pc = nullptr;
  check(KSPGetPC(ksp, &pc), "KSPGetPC");
  if (pcKind == InnerPC::AMG)
    check(PCSetType(pc, PCGAMG), "PCSetType gamg");
  else
    setInnerPC(pc, pcKind);
  // The ST-owned PC is tunable from the options database under SLEPc's "st_"
  // prefix (-st_pc_gamg_threshold, -st_pc_gamg_agg_nsmooths, ...), consumed
  // by EPSSetFromOptions -- how the GAMG parameter sweep drives this harness.
}

// Attach `m` COMPUTED near-null vectors to `M` as its GAMG near-null space:
// relax `steps` weighted-Jacobi sweeps on deterministic random starts, then
// modified Gram-Schmidt + normalize. This is the adaptive smoothed-aggregation /
// DD-alpha-AMG idea (mirrors the sibling lattice-gauge-solvers attachAdaptiveNearNull);
// for the covariant Laplacian these vectors go oscillatory with the connection,
// so GAMG's coarse space becomes gauge-aware instead of assuming a constant
// near-kernel. Deterministic (fixed seed) so the baseline is reproducible.
void attachComputedNearNull(Mat M, int m, int steps) {
  Owned<Vec, VecDestroy> dinv, tmp;
  check(MatCreateVecs(M, &dinv.obj, nullptr), "MatCreateVecs dinv");
  check(MatGetDiagonal(M, dinv.obj), "MatGetDiagonal");
  check(VecReciprocal(dinv.obj), "VecReciprocal");
  check(MatCreateVecs(M, &tmp.obj, nullptr), "MatCreateVecs tmp");

  const PetscReal omega = 2.0 / 3.0;  // damps high modes (spec(D^-1 A) in [0,2])
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::vector<Vec> vecs(static_cast<std::size_t>(m), nullptr);
  for (int i = 0; i < m; ++i) {
    check(MatCreateVecs(M, &vecs[static_cast<std::size_t>(i)], nullptr), "MatCreateVecs null");
    Vec vi = vecs[static_cast<std::size_t>(i)];
    {
      PetscScalar* a = nullptr;
      PetscInt nloc = 0;
      check(VecGetLocalSize(vi, &nloc), "VecGetLocalSize");
      check(VecGetArray(vi, &a), "VecGetArray null");
      for (PetscInt k = 0; k < nloc; ++k) a[k] = uni(rng);
      check(VecRestoreArray(vi, &a), "VecRestoreArray null");
    }
    for (int s = 0; s < steps; ++s) {  // v <- v - omega D^-1 (A v)
      check(MatMult(M, vi, tmp.obj), "MatMult relax");
      check(VecPointwiseMult(tmp.obj, tmp.obj, dinv.obj), "VecPointwiseMult");
      check(VecAXPY(vi, -omega, tmp.obj), "VecAXPY relax");
    }
    for (int j = 0; j < i; ++j) {  // modified Gram-Schmidt
      PetscScalar dot = 0.0;
      check(VecDot(vi, vecs[static_cast<std::size_t>(j)], &dot), "VecDot");
      check(VecAXPY(vi, -dot, vecs[static_cast<std::size_t>(j)]), "VecAXPY gs");
    }
    PetscReal nrm = 0.0;
    check(VecNorm(vi, NORM_2, &nrm), "VecNorm null");
    if (nrm > 0.0) check(VecScale(vi, 1.0 / nrm), "VecScale null");
  }
  // Keep only unit-normalized survivors: MatNullSpaceCreate requires normalized
  // entries, and a vector the relax+MGS pass annihilated (or sent non-finite)
  // would silently corrupt the GAMG near-null baseline instead of erroring.
  std::vector<Vec> kept;
  for (Vec& v : vecs) {
    PetscReal nrm = 0.0;
    check(VecNorm(v, NORM_2, &nrm), "VecNorm keep");
    if (std::isfinite(static_cast<double>(nrm)) && std::abs(static_cast<double>(nrm) - 1.0) < 1e-6)
      kept.push_back(v);
    else
      VecDestroy(&v);
  }
  if (!kept.empty()) {
    MatNullSpace ns = nullptr;
    check(MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_FALSE, static_cast<PetscInt>(kept.size()),
                             kept.data(), &ns),
          "MatNullSpaceCreate");
    check(MatSetNearNullSpace(M, ns), "MatSetNearNullSpace");
    MatNullSpaceDestroy(&ns);                      // M holds a reference
  }
  for (Vec& v : kept) VecDestroy(&v);              // ns holds references
}

void readVec(Vec v, std::vector<double>& out, PetscInt n) {
  out.resize(static_cast<std::size_t>(n));
  const PetscScalar* a = nullptr;
  check(VecGetArrayRead(v, &a), "VecGetArrayRead");
  for (PetscInt i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = PetscRealPart(a[i]);
  check(VecRestoreArrayRead(v, &a), "VecRestoreArrayRead");
}

}  // namespace

EigenPair smallestEigenpair(const CooMatrix& A, double tol, const std::vector<double>* initialGuess,
                            InnerPC pcKind) {
  if (A.rows() != A.cols()) throw std::invalid_argument("smallestEigenpair: matrix must be square");
  const PetscInt n = A.rows();

  Owned<Mat, MatDestroy> M, Msh;
  Owned<Vec, VecDestroy> x, y, Ax, r;
  Owned<KSP, KSPDestroy> ksp;

  double maxDiag = 0.0;
  assembleMat(A, &M.obj, &maxDiag);
  const double shift = 1e-6 * std::max(maxDiag, 1.0);  // keep A + shift I SPD when lambda_min ~ 0
  check(MatDuplicate(M.obj, MAT_COPY_VALUES, &Msh.obj), "MatDuplicate");
  check(MatShift(Msh.obj, shift), "MatShift");
  check(MatSetOption(Msh.obj, MAT_SYMMETRIC, PETSC_TRUE), "MatSetOption sym");
  check(MatSetOption(Msh.obj, MAT_SPD, PETSC_TRUE), "MatSetOption spd");

  check(MatCreateVecs(M.obj, &x.obj, &Ax.obj), "MatCreateVecs");
  check(VecDuplicate(x.obj, &y.obj), "VecDuplicate");
  check(VecDuplicate(x.obj, &r.obj), "VecDuplicate r");

  if (initialGuess && initialGuess->size() == static_cast<std::size_t>(n)) {
    PetscScalar* xa = nullptr;
    check(VecGetArray(x.obj, &xa), "VecGetArray init");
    for (PetscInt i = 0; i < n; ++i) xa[i] = (*initialGuess)[static_cast<std::size_t>(i)];
    check(VecRestoreArray(x.obj, &xa), "VecRestoreArray init");
  } else {
    check(VecSet(x.obj, 1.0), "VecSet");
  }
  PetscReal nrm0 = 0.0;
  check(VecNorm(x.obj, NORM_2, &nrm0), "VecNorm");
  if (nrm0 == 0.0) check(VecSet(x.obj, 1.0), "VecSet fallback");
  check(VecNormalize(x.obj, nullptr), "VecNormalize");

  // Iterative inner solve: CG + the chosen preconditioner (ICC by default --
  // the fastest backsolve for this SPD operator).
  check(KSPCreate(PETSC_COMM_SELF, &ksp.obj), "KSPCreate");
  check(KSPSetOperators(ksp.obj, Msh.obj, Msh.obj), "KSPSetOperators");
  // Inner accuracy must beat the outer target, or the outer residual floors
  // above it and the outer loop just runs to its cap. NOTE: KSP judges rtol in
  // PETSc's default norm -- the PRECONDITIONED residual -- not the true one.
  const double innerRtol = std::max(1e-10, 0.05 * tol);
  check(KSPSetType(ksp.obj, KSPCG), "KSPSetType");
  check(KSPSetTolerances(ksp.obj, innerRtol, PETSC_DEFAULT, PETSC_DEFAULT, 2000),
        "KSPSetTolerances");
  check(KSPSetInitialGuessNonzero(ksp.obj, PETSC_TRUE), "KSPSetInitialGuessNonzero");
  {
    PC pc = nullptr;
    check(KSPGetPC(ksp.obj, &pc), "KSPGetPC");
    setInnerPC(pc, pcKind);
  }
  check(KSPSetFromOptions(ksp.obj), "KSPSetFromOptions");

  // Inverse iteration: x <- normalize((A+shift I)^{-1} x). Stop on the
  // *relative* eigen-residual ||A x - lambda x|| / |lambda| of the unshifted A.
  double lambda = 0.0;
  int totalInner = 0, outer = 0;
  bool converged = false;
  const int maxIter = 100;
  check(VecZeroEntries(y.obj), "VecZeroEntries");
  for (int it = 0; it < maxIter; ++it) {
    ++outer;
    check(KSPSolve(ksp.obj, x.obj, y.obj), "KSPSolve");
    // A diverged inner solve (CG breakdown, ICC pivot failure) returns success
    // from KSPSolve; consuming it as a solve silently corrupts the iterate.
    // Mirror PetscSolver::solveSpdCG and honor the header's throw contract.
    {
      KSPConvergedReason reason{};
      check(KSPGetConvergedReason(ksp.obj, &reason), "KSPGetConvergedReason");
      if (reason < 0)
        throw std::runtime_error("smallestEigenpair: inner KSP diverged (reason " +
                                 std::to_string(static_cast<int>(reason)) + ")");
    }
    PetscInt its = 0;
    check(KSPGetIterationNumber(ksp.obj, &its), "KSPGetIterationNumber");
    totalInner += static_cast<int>(its);
    check(VecNormalize(y.obj, nullptr), "VecNormalize y");
    check(VecCopy(y.obj, x.obj), "VecCopy");
    check(MatMult(M.obj, x.obj, Ax.obj), "MatMult");
    PetscScalar rq = 0.0;
    check(VecDot(x.obj, Ax.obj, &rq), "VecDot");
    lambda = PetscRealPart(rq);
    check(VecWAXPY(r.obj, -lambda, x.obj, Ax.obj), "VecWAXPY");
    PetscReal resid = 0.0;
    check(VecNorm(r.obj, NORM_2, &resid), "VecNorm r");
    if (resid <= tol * std::max(std::abs(lambda), shift)) {
      converged = true;
      break;
    }
  }

  EigenPair out;
  out.value = lambda;
  out.iterations = totalInner;
  out.outerIterations = outer;
  out.converged = converged;
  readVec(x.obj, out.vector, n);
  return out;
}

EigenPair smallestEigenpairLanczos(const CooMatrix& A, double tol,
                                   const std::vector<double>* initialGuess) {
  if (A.rows() != A.cols())
    throw std::invalid_argument("smallestEigenpairLanczos: matrix must be square");
  const PetscInt n = A.rows();

  Owned<Mat, MatDestroy> M;
  Owned<Vec, VecDestroy> xr, v0;
  Owned<EPS, EPSDestroy> eps;

  double maxDiag = 0.0;
  assembleMat(A, &M.obj, &maxDiag);
  check(MatCreateVecs(M.obj, nullptr, &xr.obj), "MatCreateVecs");

  check(EPSCreate(PETSC_COMM_SELF, &eps.obj), "EPSCreate");
  check(EPSSetOperators(eps.obj, M.obj, nullptr), "EPSSetOperators");
  check(EPSSetProblemType(eps.obj, EPS_HEP), "EPSSetProblemType");
  check(EPSSetWhichEigenpairs(eps.obj, EPS_SMALLEST_REAL), "EPSSetWhichEigenpairs");
  check(EPSSetDimensions(eps.obj, 1, PETSC_DEFAULT, PETSC_DEFAULT), "EPSSetDimensions");
  check(EPSSetTolerances(eps.obj, tol, PETSC_DEFAULT), "EPSSetTolerances");

  if (initialGuess && initialGuess->size() == static_cast<std::size_t>(n)) {
    check(MatCreateVecs(M.obj, nullptr, &v0.obj), "MatCreateVecs v0");
    PetscScalar* va = nullptr;
    check(VecGetArray(v0.obj, &va), "VecGetArray v0");
    for (PetscInt i = 0; i < n; ++i) va[i] = (*initialGuess)[static_cast<std::size_t>(i)];
    check(VecRestoreArray(v0.obj, &va), "VecRestoreArray v0");
    check(EPSSetInitialSpace(eps.obj, 1, &v0.obj), "EPSSetInitialSpace");
  }

  check(EPSSetFromOptions(eps.obj), "EPSSetFromOptions");
  check(EPSSolve(eps.obj), "EPSSolve");

  PetscInt nconv = 0;
  check(EPSGetConverged(eps.obj, &nconv), "EPSGetConverged");
  if (nconv < 1) throw std::runtime_error("smallestEigenpairLanczos: no eigenpair converged");

  PetscScalar kr = 0.0, ki = 0.0;
  check(EPSGetEigenpair(eps.obj, 0, &kr, &ki, xr.obj, nullptr), "EPSGetEigenpair");
  PetscInt its = 0;
  check(EPSGetIterationNumber(eps.obj, &its), "EPSGetIterationNumber");

  EigenPair out;
  out.value = PetscRealPart(kr);
  out.iterations = static_cast<int>(its);
  out.converged = true;  // non-convergence throws above
  readVec(xr.obj, out.vector, n);
  return out;
}

std::vector<EigenPair> smallestEigenpairsLanczos(const CooMatrix& A, int nev, double tol) {
  if (A.rows() != A.cols())
    throw std::invalid_argument("smallestEigenpairsLanczos: matrix must be square");
  const PetscInt n = A.rows();

  Owned<Mat, MatDestroy> M;
  Owned<Vec, VecDestroy> xr;
  Owned<EPS, EPSDestroy> eps;

  double maxDiag = 0.0;
  assembleMat(A, &M.obj, &maxDiag);
  check(MatCreateVecs(M.obj, nullptr, &xr.obj), "MatCreateVecs");

  check(EPSCreate(PETSC_COMM_SELF, &eps.obj), "EPSCreate");
  check(EPSSetOperators(eps.obj, M.obj, nullptr), "EPSSetOperators");
  check(EPSSetProblemType(eps.obj, EPS_HEP), "EPSSetProblemType");
  check(EPSSetWhichEigenpairs(eps.obj, EPS_SMALLEST_REAL), "EPSSetWhichEigenpairs");
  // Ask for nev with a generous subspace so a clustered low spectrum resolves.
  const PetscInt ncv = std::min<PetscInt>(n, std::max<PetscInt>(2 * nev + 8, 16));
  check(EPSSetDimensions(eps.obj, nev, ncv, PETSC_DEFAULT), "EPSSetDimensions");
  check(EPSSetTolerances(eps.obj, tol, PETSC_DEFAULT), "EPSSetTolerances");
  check(EPSSetFromOptions(eps.obj), "EPSSetFromOptions");
  check(EPSSolve(eps.obj), "EPSSolve");

  PetscInt nconv = 0;
  check(EPSGetConverged(eps.obj, &nconv), "EPSGetConverged");
  if (nconv < nev)
    throw std::runtime_error("smallestEigenpairsLanczos: fewer than nev eigenpairs converged");

  PetscInt its = 0;
  check(EPSGetIterationNumber(eps.obj, &its), "EPSGetIterationNumber");

  std::vector<EigenPair> out(nev);
  for (int e = 0; e < nev; ++e) {
    PetscScalar kr = 0.0, ki = 0.0;
    check(EPSGetEigenpair(eps.obj, e, &kr, &ki, xr.obj, nullptr), "EPSGetEigenpair");
    out[e].value = PetscRealPart(kr);
    out[e].iterations = static_cast<int>(its);  // EPS-level count, same for all pairs
    out[e].converged = true;  // fewer than nev converged throws above
    readVec(xr.obj, out[e].vector, n);
  }
  return out;
}

EigenPair smallestEigenpairShiftInvert(const CooMatrix& A, double tol,
                                       const std::vector<double>* initialGuess, InnerPC pcKind) {
  if (A.rows() != A.cols())
    throw std::invalid_argument("smallestEigenpairShiftInvert: matrix must be square");
  const PetscInt n = A.rows();

  Owned<Mat, MatDestroy> M;
  Owned<Vec, VecDestroy> xr, v0;
  Owned<EPS, EPSDestroy> eps;

  double maxDiag = 0.0;
  assembleMat(A, &M.obj, &maxDiag);
  check(MatSetOption(M.obj, MAT_SPD, PETSC_TRUE), "MatSetOption spd");
  check(MatCreateVecs(M.obj, nullptr, &xr.obj), "MatCreateVecs");

  check(EPSCreate(PETSC_COMM_SELF, &eps.obj), "EPSCreate");
  check(EPSSetOperators(eps.obj, M.obj, nullptr), "EPSSetOperators");
  check(EPSSetProblemType(eps.obj, EPS_HEP), "EPSSetProblemType");
  // Shift-invert about sigma=0: eigenvalues nearest 0 (the smallest, since A is
  // SPD) become the dominant modes 1/lambda of the inverted operator.
  check(EPSSetWhichEigenpairs(eps.obj, EPS_TARGET_REAL), "EPSSetWhichEigenpairs");
  check(EPSSetTarget(eps.obj, 0.0), "EPSSetTarget");
  check(EPSSetDimensions(eps.obj, 1, PETSC_DEFAULT, PETSC_DEFAULT), "EPSSetDimensions");
  check(EPSSetTolerances(eps.obj, tol, PETSC_DEFAULT), "EPSSetTolerances");

  // ST = shift-and-invert with sigma=0; the inner solve is CG + chosen PC (the
  // SPD solve is only moderately conditioned -> cheap).
  {
    ST st = nullptr;
    check(EPSGetST(eps.obj, &st), "EPSGetST");
    check(STSetType(st, STSINVERT), "STSetType sinvert");
    KSP ksp = nullptr;
    check(STGetKSP(st, &ksp), "STGetKSP");
    check(KSPSetType(ksp, KSPCG), "KSPSetType cg");
    // Inexact shift-invert discipline: the inner solve must clearly out-resolve
    // the eigentolerance, judged in the TRUE residual norm -- a 10x margin in
    // the preconditioned norm can under-solve by the preconditioner's
    // conditioning and silently corrupt the Krylov space.
    check(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED), "KSPSetNormType");
    check(KSPSetTolerances(ksp, std::max(1e-10, 1e-2 * tol), PETSC_DEFAULT, PETSC_DEFAULT, 2000),
          "KSPSetTolerances");
    PC pc = nullptr;
    check(KSPGetPC(ksp, &pc), "KSPGetPC");
    setInnerPC(pc, pcKind);
  }

  if (initialGuess && initialGuess->size() == static_cast<std::size_t>(n)) {
    check(MatCreateVecs(M.obj, nullptr, &v0.obj), "MatCreateVecs v0");
    PetscScalar* va = nullptr;
    check(VecGetArray(v0.obj, &va), "VecGetArray v0");
    for (PetscInt i = 0; i < n; ++i) va[i] = (*initialGuess)[static_cast<std::size_t>(i)];
    check(VecRestoreArray(v0.obj, &va), "VecRestoreArray v0");
    check(EPSSetInitialSpace(eps.obj, 1, &v0.obj), "EPSSetInitialSpace");
  }

  check(EPSSetFromOptions(eps.obj), "EPSSetFromOptions");
  check(EPSSolve(eps.obj), "EPSSolve");

  PetscInt nconv = 0;
  check(EPSGetConverged(eps.obj, &nconv), "EPSGetConverged");
  if (nconv < 1) throw std::runtime_error("smallestEigenpairShiftInvert: no eigenpair converged");

  PetscScalar kr = 0.0, ki = 0.0;
  check(EPSGetEigenpair(eps.obj, 0, &kr, &ki, xr.obj, nullptr), "EPSGetEigenpair");
  PetscInt its = 0;
  check(EPSGetIterationNumber(eps.obj, &its), "EPSGetIterationNumber");

  EigenPair out;
  out.value = PetscRealPart(kr);
  out.iterations = static_cast<int>(its);
  out.converged = true;  // non-convergence throws above
  readVec(xr.obj, out.vector, n);
  return out;
}

EigenPair smallestEigenpairLOBPCG(const CooMatrix& A, double tol,
                                  const std::vector<double>* initialGuess, InnerPC pcKind,
                                  int blockSize, int adaptiveNull) {
  if (A.rows() != A.cols())
    throw std::invalid_argument("smallestEigenpairLOBPCG: matrix must be square");
  if (blockSize < 1) throw std::invalid_argument("smallestEigenpairLOBPCG: blockSize must be >= 1");
  const PetscInt n = A.rows();
  using clk = std::chrono::steady_clock;
  const auto ms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  Owned<Mat, MatDestroy> M;
  Owned<Vec, VecDestroy> xr, v0;
  Owned<EPS, EPSDestroy> eps;

  double maxDiag = 0.0;
  assembleMat(A, &M.obj, &maxDiag);
  check(MatSetOption(M.obj, MAT_SPD, PETSC_TRUE), "MatSetOption spd");
  const auto tNull0 = clk::now();
  if (adaptiveNull > 0 && pcKind == InnerPC::AMG) attachComputedNearNull(M.obj, adaptiveNull, 10);
  const double nullMs = ms(tNull0, clk::now());
  check(MatCreateVecs(M.obj, nullptr, &xr.obj), "MatCreateVecs");

  check(EPSCreate(PETSC_COMM_SELF, &eps.obj), "EPSCreate");
  check(EPSSetOperators(eps.obj, M.obj, nullptr), "EPSSetOperators");
  check(EPSSetProblemType(eps.obj, EPS_HEP), "EPSSetProblemType");
  check(EPSSetWhichEigenpairs(eps.obj, EPS_SMALLEST_REAL), "EPSSetWhichEigenpairs");
  check(EPSSetType(eps.obj, EPSLOBPCG), "EPSSetType lobpcg");
  // One eigenpair wanted, but a block of >=2 spans the degenerate lowest pair
  // (the real 2x2 embedding doubles every eigenvalue).
  check(EPSSetDimensions(eps.obj, 1, PETSC_DEFAULT, PETSC_DEFAULT), "EPSSetDimensions");
  check(EPSLOBPCGSetBlockSize(eps.obj, static_cast<PetscInt>(blockSize)), "EPSLOBPCGSetBlockSize");
  check(EPSSetTolerances(eps.obj, tol, PETSC_DEFAULT), "EPSSetTolerances");

  // Preconditioned LOBPCG: STPRECOND applies the PC directly (KSPPREONLY), no
  // inner Krylov solve -- this is the structured-preconditioner plug point.
  setupBlockPreconditioner(eps.obj, pcKind);

  if (initialGuess && initialGuess->size() == static_cast<std::size_t>(n)) {
    check(MatCreateVecs(M.obj, nullptr, &v0.obj), "MatCreateVecs v0");
    PetscScalar* va = nullptr;
    check(VecGetArray(v0.obj, &va), "VecGetArray v0");
    for (PetscInt i = 0; i < n; ++i) va[i] = (*initialGuess)[static_cast<std::size_t>(i)];
    check(VecRestoreArray(v0.obj, &va), "VecRestoreArray v0");
    check(EPSSetInitialSpace(eps.obj, 1, &v0.obj), "EPSSetInitialSpace");
  }

  check(EPSSetFromOptions(eps.obj), "EPSSetFromOptions");
  // Split setup (PC/GAMG construction, includes the near-null build above) from
  // the iteration wall time, so the baseline reports a setup column like PyAMG.
  const auto tSetup0 = clk::now();
  check(EPSSetUp(eps.obj), "EPSSetUp");
  const double setupMs = nullMs + ms(tSetup0, clk::now());
  const auto tSolve0 = clk::now();
  check(EPSSolve(eps.obj), "EPSSolve");
  const double solveMs = ms(tSolve0, clk::now());

  PetscInt nconv = 0;
  check(EPSGetConverged(eps.obj, &nconv), "EPSGetConverged");
  if (nconv < 1) throw std::runtime_error("smallestEigenpairLOBPCG: no eigenpair converged");

  PetscScalar kr = 0.0, ki = 0.0;
  check(EPSGetEigenpair(eps.obj, 0, &kr, &ki, xr.obj, nullptr), "EPSGetEigenpair");
  PetscInt its = 0;
  check(EPSGetIterationNumber(eps.obj, &its), "EPSGetIterationNumber");

  EigenPair out;
  out.value = PetscRealPart(kr);
  out.iterations = static_cast<int>(its);
  out.outerIterations = static_cast<int>(its);
  out.setupMs = setupMs;
  out.solveMs = solveMs;
  out.converged = true;  // non-convergence throws above
  readVec(xr.obj, out.vector, n);
  return out;
}

EigenPair smallestEigenpairDavidson(const CooMatrix& A, double tol,
                                    const std::vector<double>* initialGuess, InnerPC pcKind) {
  if (A.rows() != A.cols())
    throw std::invalid_argument("smallestEigenpairDavidson: matrix must be square");
  const PetscInt n = A.rows();

  Owned<Mat, MatDestroy> M;
  Owned<Vec, VecDestroy> xr, v0;
  Owned<EPS, EPSDestroy> eps;

  double maxDiag = 0.0;
  assembleMat(A, &M.obj, &maxDiag);
  check(MatSetOption(M.obj, MAT_SPD, PETSC_TRUE), "MatSetOption spd");
  check(MatCreateVecs(M.obj, nullptr, &xr.obj), "MatCreateVecs");

  check(EPSCreate(PETSC_COMM_SELF, &eps.obj), "EPSCreate");
  check(EPSSetOperators(eps.obj, M.obj, nullptr), "EPSSetOperators");
  check(EPSSetProblemType(eps.obj, EPS_HEP), "EPSSetProblemType");
  check(EPSSetWhichEigenpairs(eps.obj, EPS_SMALLEST_REAL), "EPSSetWhichEigenpairs");
  check(EPSSetType(eps.obj, EPSGD), "EPSSetType gd");
  check(EPSSetDimensions(eps.obj, 1, PETSC_DEFAULT, PETSC_DEFAULT), "EPSSetDimensions");
  // Expand the search space by 2 vectors per step -> spans the degenerate lowest
  // pair, the analog of LOBPCG's block.
  check(EPSGDSetBlockSize(eps.obj, 2), "EPSGDSetBlockSize");
  check(EPSSetTolerances(eps.obj, tol, PETSC_DEFAULT), "EPSSetTolerances");

  // Generalized Davidson expands by the *preconditioned* residual via STPRECOND
  // (KSPPREONLY) -- no inner Krylov solve. GAMG for the same block-apply reason
  // as LOBPCG (hypre BoomerAMG has no working block PCMatApply in this build).
  setupBlockPreconditioner(eps.obj, pcKind);

  if (initialGuess && initialGuess->size() == static_cast<std::size_t>(n)) {
    check(MatCreateVecs(M.obj, nullptr, &v0.obj), "MatCreateVecs v0");
    PetscScalar* va = nullptr;
    check(VecGetArray(v0.obj, &va), "VecGetArray v0");
    for (PetscInt i = 0; i < n; ++i) va[i] = (*initialGuess)[static_cast<std::size_t>(i)];
    check(VecRestoreArray(v0.obj, &va), "VecRestoreArray v0");
    check(EPSSetInitialSpace(eps.obj, 1, &v0.obj), "EPSSetInitialSpace");
  }

  check(EPSSetFromOptions(eps.obj), "EPSSetFromOptions");
  check(EPSSolve(eps.obj), "EPSSolve");

  PetscInt nconv = 0;
  check(EPSGetConverged(eps.obj, &nconv), "EPSGetConverged");
  if (nconv < 1) throw std::runtime_error("smallestEigenpairDavidson: no eigenpair converged");

  PetscScalar kr = 0.0, ki = 0.0;
  check(EPSGetEigenpair(eps.obj, 0, &kr, &ki, xr.obj, nullptr), "EPSGetEigenpair");
  PetscInt its = 0;
  check(EPSGetIterationNumber(eps.obj, &its), "EPSGetIterationNumber");

  EigenPair out;
  out.value = PetscRealPart(kr);
  out.iterations = static_cast<int>(its);
  out.outerIterations = static_cast<int>(its);
  out.converged = true;  // non-convergence throws above
  readVec(xr.obj, out.vector, n);
  return out;
}

namespace {

// Shared EPS driver for the two Shell variants: identical to the InnerPC-based
// LOBPCG/GD paths except the preconditioner is the caller's PCSHELL.
EigenPair shellEigenSolve(const CooMatrix& A, double tol,
                          const std::vector<double>* initialGuess,
                          const ShellPcApply& apply, bool lobpcg, int blockSize) {
  if (A.rows() != A.cols())
    throw std::invalid_argument("shellEigenSolve: matrix must be square");
  const PetscInt n = A.rows();
  using clk = std::chrono::steady_clock;
  const auto ms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  Owned<Mat, MatDestroy> M;
  Owned<Vec, VecDestroy> xr, v0;
  Owned<EPS, EPSDestroy> eps;

  double maxDiag = 0.0;
  assembleMat(A, &M.obj, &maxDiag);
  check(MatSetOption(M.obj, MAT_SPD, PETSC_TRUE), "MatSetOption spd");
  check(MatCreateVecs(M.obj, nullptr, &xr.obj), "MatCreateVecs");

  check(EPSCreate(PETSC_COMM_SELF, &eps.obj), "EPSCreate");
  check(EPSSetOperators(eps.obj, M.obj, nullptr), "EPSSetOperators");
  check(EPSSetProblemType(eps.obj, EPS_HEP), "EPSSetProblemType");
  check(EPSSetWhichEigenpairs(eps.obj, EPS_SMALLEST_REAL), "EPSSetWhichEigenpairs");
  if (lobpcg) {
    check(EPSSetType(eps.obj, EPSLOBPCG), "EPSSetType lobpcg");
    check(EPSLOBPCGSetBlockSize(eps.obj, static_cast<PetscInt>(blockSize)),
          "EPSLOBPCGSetBlockSize");
  } else {
    check(EPSSetType(eps.obj, EPSGD), "EPSSetType gd");
    check(EPSGDSetBlockSize(eps.obj, 2), "EPSGDSetBlockSize");
  }
  check(EPSSetDimensions(eps.obj, 1, PETSC_DEFAULT, PETSC_DEFAULT), "EPSSetDimensions");
  check(EPSSetTolerances(eps.obj, tol, PETSC_DEFAULT), "EPSSetTolerances");

  ShellPcCtx ctx;
  ctx.apply = &apply;
  setupShellPreconditioner(eps.obj, &ctx);

  if (initialGuess && initialGuess->size() == static_cast<std::size_t>(n)) {
    check(MatCreateVecs(M.obj, nullptr, &v0.obj), "MatCreateVecs v0");
    PetscScalar* va = nullptr;
    check(VecGetArray(v0.obj, &va), "VecGetArray v0");
    for (PetscInt i = 0; i < n; ++i) va[i] = (*initialGuess)[static_cast<std::size_t>(i)];
    check(VecRestoreArray(v0.obj, &va), "VecRestoreArray v0");
    check(EPSSetInitialSpace(eps.obj, 1, &v0.obj), "EPSSetInitialSpace");
  }

  check(EPSSetFromOptions(eps.obj), "EPSSetFromOptions");
  const auto tSetup0 = clk::now();
  check(EPSSetUp(eps.obj), "EPSSetUp");
  const double setupMs = ms(tSetup0, clk::now());
  const auto tSolve0 = clk::now();
  check(EPSSolve(eps.obj), "EPSSolve");
  const double solveMs = ms(tSolve0, clk::now());

  PetscInt nconv = 0;
  check(EPSGetConverged(eps.obj, &nconv), "EPSGetConverged");
  if (nconv < 1) throw std::runtime_error("shellEigenSolve: no eigenpair converged");

  PetscScalar kr = 0.0, ki = 0.0;
  check(EPSGetEigenpair(eps.obj, 0, &kr, &ki, xr.obj, nullptr), "EPSGetEigenpair");
  PetscInt its = 0;
  check(EPSGetIterationNumber(eps.obj, &its), "EPSGetIterationNumber");

  EigenPair out;
  out.value = PetscRealPart(kr);
  out.iterations = static_cast<int>(its);
  out.outerIterations = static_cast<int>(its);
  out.setupMs = setupMs;
  out.solveMs = solveMs;
  out.converged = true;
  readVec(xr.obj, out.vector, n);
  return out;
}

}  // namespace

EigenPair smallestEigenpairLOBPCGShell(const CooMatrix& A, double tol,
                                       const std::vector<double>* initialGuess,
                                       const ShellPcApply& apply, int blockSize) {
  return shellEigenSolve(A, tol, initialGuess, apply, /*lobpcg=*/true, blockSize);
}

EigenPair smallestEigenpairDavidsonShell(const CooMatrix& A, double tol,
                                         const std::vector<double>* initialGuess,
                                         const ShellPcApply& apply) {
  return shellEigenSolve(A, tol, initialGuess, apply, /*lobpcg=*/false, 0);
}

EigenPair largestEigenvalue(const CooMatrix& A, double tol) {
  if (A.rows() != A.cols())
    throw std::invalid_argument("largestEigenvalue: matrix must be square");
  Owned<Mat, MatDestroy> M;
  Owned<Vec, VecDestroy> xr;
  Owned<EPS, EPSDestroy> eps;
  double maxDiag = 0.0;
  assembleMat(A, &M.obj, &maxDiag);
  check(MatCreateVecs(M.obj, nullptr, &xr.obj), "MatCreateVecs");
  check(EPSCreate(PETSC_COMM_SELF, &eps.obj), "EPSCreate");
  check(EPSSetOperators(eps.obj, M.obj, nullptr), "EPSSetOperators");
  check(EPSSetProblemType(eps.obj, EPS_HEP), "EPSSetProblemType");
  check(EPSSetWhichEigenpairs(eps.obj, EPS_LARGEST_REAL), "EPSSetWhichEigenpairs");
  check(EPSSetDimensions(eps.obj, 1, PETSC_DEFAULT, PETSC_DEFAULT), "EPSSetDimensions");
  check(EPSSetTolerances(eps.obj, tol, PETSC_DEFAULT), "EPSSetTolerances");
  check(EPSSetFromOptions(eps.obj), "EPSSetFromOptions");
  check(EPSSolve(eps.obj), "EPSSolve");
  PetscInt nconv = 0;
  check(EPSGetConverged(eps.obj, &nconv), "EPSGetConverged");
  if (nconv < 1) throw std::runtime_error("largestEigenvalue: no eigenpair converged");
  PetscScalar kr = 0.0, ki = 0.0;
  check(EPSGetEigenpair(eps.obj, 0, &kr, &ki, xr.obj, nullptr), "EPSGetEigenpair");
  PetscInt its = 0;
  check(EPSGetIterationNumber(eps.obj, &its), "EPSGetIterationNumber");
  EigenPair out;
  out.value = PetscRealPart(kr);
  out.iterations = static_cast<int>(its);
  out.converged = true;  // non-convergence throws above
  readVec(xr.obj, out.vector, A.rows());
  return out;
}

}  // namespace bochner
