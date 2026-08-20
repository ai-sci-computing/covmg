/// \file
/// Baseline: does an
/// *adaptive gauge-aware* coarse space -- PETSc GAMG with a COMPUTED near-null
/// space (the SPD analogue of DD-alpha-AMG) -- close the eigensolver gap without
/// our closed-form covariant transfer?
///
/// Loads a complex Hermitian connection Laplacian (the MatrixMarket file written
/// by tools/export_operator), builds the real 2N embedding that bochner's SLEPc
/// baselines use, and solves for the smallest eigenpair with preconditioned
/// LOBPCG (EPSLOBPCG + STPRECOND + GAMG) in two configurations:
///
///   constant near-null  -- GAMG's default constant near-kernel (gauge-blind);
///   computed near-null  -- `adaptiveNull` vectors relaxed from the operator
///                          (adaptive smoothed-aggregation / DD-alpha-AMG idea;
///                          gauge-aware, since the vectors go oscillatory with
///                          the connection).
///
/// Reports iterations, setup ms (near-null build + GAMG construction), and
/// solve ms for each -- the head-to-head row that complements the PyAMG
/// adaptive-SA comparison (tools/pyamg_baseline.py) with the PETSc-side
/// gauge-aware-adaptive datapoint. Compare against covMG-LOBPCG on the SAME
/// operator (torus_eig_compare / sun_gauge_bench): our closed-form transfer,
/// zero setup.
///
/// Usage: gauge_aware_amg_eig <operator.mtx> [adaptiveNull=4]
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include <algorithm>
#include <vector>

#include <slepc.h>

#include "BenchTiming.h"
#include "grid/CooMatrix.h"
#include "solvers/EigenSolver.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

// Load a MatrixMarket "coordinate complex general" file (as written by
// export_operator) and return the real 2N embedding a+ib -> [[a,-b],[b,a]] with
// bochner's interleaving (complex node p -> real rows 2p, 2p+1), so the operator
// is bit-identical to torus_eig_compare / sun_gauge_bench's real CooMatrix.
CooMatrix loadRealEmbedding(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string line;
  // header + banner comments
  std::getline(in, line);
  if (line.find("coordinate complex") == std::string::npos)
    throw std::runtime_error("expected 'coordinate complex' MatrixMarket: " + path);
  while (std::getline(in, line))
    if (!line.empty() && line[0] != '%') break;
  long rows = 0, cols = 0, nnz = 0;
  {
    std::istringstream hs(line);
    hs >> rows >> cols >> nnz;
  }
  if (rows != cols) throw std::runtime_error("matrix not square");
  CooMatrix A(2 * rows, 2 * cols);
  const auto blk = [&](long a, long b, double re, double im) {
    A.add(2 * a, 2 * b, re);
    A.add(2 * a, 2 * b + 1, -im);
    A.add(2 * a + 1, 2 * b, im);
    A.add(2 * a + 1, 2 * b + 1, re);
  };
  long r = 0, c = 0;
  double re = 0, im = 0;
  long read = 0;
  while (in >> r >> c >> re >> im) {
    blk(r - 1, c - 1, re, im);  // MatrixMarket is 1-indexed
    ++read;
  }
  if (read != nnz)
    std::fprintf(stderr, "warning: read %ld entries, header said %ld\n", read, nnz);
  return A;
}

}  // namespace

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  {
    if (argc < 2) {
      std::fprintf(stderr, "usage: %s <operator.mtx> [adaptiveNull=4]\n", argv[0]);
      return 2;
    }
    const std::string path = argv[1];
    const int adaptiveNull = argc > 2 ? std::atoi(argv[2]) : 4;
    const double tol = 1e-7;

    const CooMatrix A = loadRealEmbedding(path);
    std::printf("operator %s : real-embedded dim %lld, tol %.0e, adaptiveNull %d\n", path.c_str(),
                static_cast<long long>(A.rows()), tol, adaptiveNull);
    std::printf("%-42s | %5s | %9s | %9s | %-12s\n", "LOBPCG + GAMG preconditioner", "iters",
                "setup ms", "solve ms", "eigenvalue");
    std::printf("--------------------------------------------------------------------------------\n");

    // Median over BENCH_REPS identical solves (each rebuilds its own Mat), the
    // paper-wide timing statistic; iterations and eigenvalue are deterministic.
    const int reps = benchstat::reps();
    const auto row = [&](const char* label, int nullv) {
      std::vector<double> su, so;
      EigenPair e;
      for (int r = 0; r < reps; ++r) {
        e = smallestEigenpairLOBPCG(A, tol, nullptr, InnerPC::AMG, /*blockSize=*/4, nullv);
        su.push_back(e.setupMs);
        so.push_back(e.solveMs);
      }
      std::sort(su.begin(), su.end());
      std::sort(so.begin(), so.end());
      std::printf("%-42s | %5d | %9.1f | %9.1f | %.6f\n", label, e.iterations, su[reps / 2],
                  so[reps / 2], e.value);
    };
    row("constant near-null (gauge-blind, default)", 0);
    char lbl[64];
    std::snprintf(lbl, sizeof(lbl), "computed near-null x%d (gauge-aware)", adaptiveNull);
    row(lbl, adaptiveNull);
  }
  SlepcFinalize();
  return 0;
}
