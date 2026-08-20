/// \file
/// Does MPI help or regress the SLEPc Krylov--Schur baseline at the paper's
/// largest sizes?
///
/// The earlier 1->2->4-rank pass ran the *sequential* solver under mpirun --
/// each rank owning a full COMM_SELF copy -- so its "regression" conflated MPI
/// scaling with memory-bandwidth contention between duplicated solves. This
/// tool is a genuinely distributed baseline: the real 2N embedding of the
/// uniform-flux torus operator assembled as MPIAIJ on PETSC_COMM_WORLD
/// (row-range ownership, no duplication), solved by EPSKRYLOVSCHUR with the
/// paper's baseline configuration (EPS_HEP, smallest-real, default ncv,
/// rtol 1e-7). Median of 5 fresh EPSSolve runs, barrier-bracketed, rank-0
/// wall clock.
///
/// Usage: mpiexec -n R mpi_baseline_bench [n] [nPhi]   (defaults 64 4)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <slepc.h>

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  {
    const int n = (argc > 1) ? std::atoi(argv[1]) : 64;
    const int nPhi = (argc > 2) ? std::atoi(argv[2]) : 4;
    const double tol = 1e-7;
    const double h = 1.0 / n, w = 1.0 / (h * h);
    const PetscInt N = static_cast<PetscInt>(n) * n * n;  // complex nodes
    const PetscInt NR = 2 * N;                            // real rows

    PetscMPIInt rank = 0, size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    const double phi_p = 2.0 * M_PI * nPhi / (double(n) * n);
    const auto idx = [&](int i, int j, int k) {
      return (static_cast<PetscInt>(i) * n + j) * n + k;
    };
    // Forward link angles, same formula as torus_eig_compare/covpc_outer_bench.
    const auto lkx = [&](int, int j, int) { return -phi_p * j; };
    const auto lky = [&](int i, int j, int) {
      return (j == n - 1) ? 2.0 * M_PI * nPhi * i / double(n) : 0.0;
    };
    const auto lkz = [&](int, int, int) { return 0.0; };

    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, NR, NR);
    MatSetType(A, MATAIJ);
    MatSetFromOptions(A);
    MatMPIAIJSetPreallocation(A, 16, nullptr, 16, nullptr);
    MatSeqAIJSetPreallocation(A, 16, nullptr);
    MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);

    PetscInt rstart = 0, rend = 0;
    MatGetOwnershipRange(A, &rstart, &rend);
    // Each rank walks every link (the formula is free) and adds only entries
    // whose ROW it owns -- no duplication, no communication during insert.
    const auto blk = [&](PetscInt a, PetscInt b, double re, double im) {
      const PetscInt r0 = 2 * a, r1 = 2 * a + 1;
      if (r0 >= rstart && r0 < rend) {
        MatSetValue(A, r0, 2 * b, re, ADD_VALUES);
        MatSetValue(A, r0, 2 * b + 1, -im, ADD_VALUES);
      }
      if (r1 >= rstart && r1 < rend) {
        MatSetValue(A, r1, 2 * b, im, ADD_VALUES);
        MatSetValue(A, r1, 2 * b + 1, re, ADD_VALUES);
      }
    };
    const auto lnk = [&](PetscInt a, PetscInt b, double th) {
      blk(a, a, w, 0.0);
      blk(b, b, w, 0.0);
      blk(a, b, -w * std::cos(th), w * std::sin(th));
      blk(b, a, -w * std::cos(th), -w * std::sin(th));
    };
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k) {
          const PetscInt c = idx(i, j, k);
          lnk(c, idx((i + 1) % n, j, k), lkx(i, j, k));
          lnk(c, idx(i, (j + 1) % n, k), lky(i, j, k));
          lnk(c, idx(i, j, (k + 1) % n), lkz(i, j, k));
        }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    std::vector<double> times;
    PetscReal lam = 0;
    PetscInt its = 0;
    for (int run = 0; run < 5; ++run) {
      EPS eps;
      EPSCreate(PETSC_COMM_WORLD, &eps);
      EPSSetOperators(eps, A, nullptr);
      EPSSetProblemType(eps, EPS_HEP);
      EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
      EPSSetType(eps, EPSKRYLOVSCHUR);
      EPSSetDimensions(eps, 1, PETSC_DEFAULT, PETSC_DEFAULT);
      EPSSetTolerances(eps, tol, PETSC_DEFAULT);
      EPSSetFromOptions(eps);
      EPSSetUp(eps);
      MPI_Barrier(PETSC_COMM_WORLD);
      const double t0 = MPI_Wtime();
      EPSSolve(eps);
      MPI_Barrier(PETSC_COMM_WORLD);
      times.push_back((MPI_Wtime() - t0) * 1000.0);
      PetscInt nconv = 0;
      EPSGetConverged(eps, &nconv);
      if (nconv > 0) {
        PetscScalar kr, ki;
        EPSGetEigenpair(eps, 0, &kr, &ki, nullptr, nullptr);
        lam = PetscRealPart(kr);
      }
      EPSGetIterationNumber(eps, &its);
      EPSDestroy(&eps);
    }
    std::sort(times.begin(), times.end());
    if (rank == 0)
      std::printf("n=%d nPhi=%d ranks=%d  DOF(real)=%lld  lambda=%.9f  its=%d  "
                  "median=%.1f ms  [min %.1f, max %.1f]\n",
                  n, nPhi, size, static_cast<long long>(NR), (double)lam,
                  (int)its, times[2], times.front(), times.back());
    MatDestroy(&A);
  }
  SlepcFinalize();
  return 0;
}
