/// \file
/// How long does it take to FORM a Galerkin coarse operator?
///
/// The paper chooses a rediscretized coarse operator over the Galerkin one
/// P^H L P. The solve-side argument is weak: measured V-cycle counts are equal
/// at low flux and better for Galerkin at intermediate flux. The real cost is
/// on the SETUP side, and in this paper's regime setup is not amortized -- the
/// connection changes every frame, so the hierarchy is rebuilt every frame.
/// This bench measures that: the wall time of the sparse triple product, per
/// level, against which tools/flux_alias_diag `coarsentime` measures the
/// closed-form O(N) link pass the shipped solver uses instead.
///
/// Real embedding, deliberately. The operator is complex Hermitian, and PETSc's
/// fused MatPtAP computes P^T A P rather than P^H A P. Under the embedding
/// a+ib -> [a -b; b a] the embedding of a conjugate is the transpose of the
/// embedding, E(conj z) = E(z)^T, so E(P^H L P) = E(P)^T E(L) E(P) exactly --
/// MatPtAP on the embeddings IS the Hermitian triple product. This also uses
/// the configuration the project already measured to be the faster of the two
/// (tools/complex_baseline: real-embedded Krylov-Schur beats native complex by
/// 2.4-3.2x), so Galerkin is being timed in its best light rather than its
/// worst.
///
/// Inputs are the MatrixMarket files tools/export_operator writes:
///   export_operator u1      <n>   <nPhi> L.mtx
///   export_operator prolong <n>   <nPhi> P.mtx
///
/// Usage: galerkin_setup_bench <L.mtx> <P.mtx> [reps=5]
/// Prints: fine N, coarse N, nnz/row of L and of P^H L P, and the median
/// MatPtAP time.
#include <petscmat.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Coo {
  PetscInt rows = 0, cols = 0;
  std::vector<PetscInt> row, col;
  std::vector<double> re, im;
};

Coo readMtx(const char* path) {
  std::FILE* f = std::fopen(path, "r");
  if (!f) throw std::runtime_error(std::string("cannot open ") + path);
  char line[512];
  if (!std::fgets(line, sizeof line, f)) throw std::runtime_error("empty file");
  if (!std::strstr(line, "coordinate") || !std::strstr(line, "complex"))
    throw std::runtime_error("expected MatrixMarket coordinate complex header");
  do {
    if (!std::fgets(line, sizeof line, f)) throw std::runtime_error("truncated header");
  } while (line[0] == '%');
  long rows = 0, cols = 0, nnz = 0;
  if (std::sscanf(line, "%ld %ld %ld", &rows, &cols, &nnz) != 3)
    throw std::runtime_error("bad size line");
  Coo c;
  c.rows = static_cast<PetscInt>(rows);
  c.cols = static_cast<PetscInt>(cols);
  c.row.reserve(nnz);
  c.col.reserve(nnz);
  c.re.reserve(nnz);
  c.im.reserve(nnz);
  for (long k = 0; k < nnz; ++k) {
    long i = 0, j = 0;
    double a = 0.0, b = 0.0;
    if (std::fscanf(f, "%ld %ld %lf %lf", &i, &j, &a, &b) != 4)
      throw std::runtime_error("truncated entries");
    c.row.push_back(static_cast<PetscInt>(i - 1));
    c.col.push_back(static_cast<PetscInt>(j - 1));
    c.re.push_back(a);
    c.im.push_back(b);
  }
  std::fclose(f);
  return c;
}

void check(PetscErrorCode e, const char* what) {
  if (e) {
    std::fprintf(stderr, "PETSc error in %s\n", what);
    std::exit(2);
  }
}

/// Assemble the real 2R x 2C embedding of a complex COO matrix.
Mat assembleEmbedded(const Coo& c) {
  const PetscInt R = 2 * c.rows, C = 2 * c.cols;
  std::vector<PetscInt> nnzRow(static_cast<std::size_t>(R), 0);
  for (std::size_t k = 0; k < c.row.size(); ++k) {
    nnzRow[static_cast<std::size_t>(2 * c.row[k])] += 2;
    nnzRow[static_cast<std::size_t>(2 * c.row[k] + 1)] += 2;
  }
  Mat M = nullptr;
  check(MatCreateSeqAIJ(PETSC_COMM_SELF, R, C, 0, nnzRow.data(), &M), "MatCreateSeqAIJ");
  for (std::size_t k = 0; k < c.row.size(); ++k) {
    const PetscInt i = c.row[k], j = c.col[k];
    const double a = c.re[k], b = c.im[k];
    check(MatSetValue(M, 2 * i, 2 * j, a, INSERT_VALUES), "MatSetValue");
    check(MatSetValue(M, 2 * i, 2 * j + 1, -b, INSERT_VALUES), "MatSetValue");
    check(MatSetValue(M, 2 * i + 1, 2 * j, b, INSERT_VALUES), "MatSetValue");
    check(MatSetValue(M, 2 * i + 1, 2 * j + 1, a, INSERT_VALUES), "MatSetValue");
  }
  check(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
  check(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
  return M;
}

double nnzPerRow(Mat M) {
  MatInfo info;
  check(MatGetInfo(M, MAT_LOCAL, &info), "MatGetInfo");
  PetscInt r = 0, c = 0;
  check(MatGetSize(M, &r, &c), "MatGetSize");
  return r ? info.nz_used / static_cast<double>(r) : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
  check(PetscInitialize(&argc, &argv, nullptr, nullptr), "PetscInitialize");
  if (argc < 3) {
    std::fprintf(stderr, "usage: galerkin_setup_bench <L.mtx> <P.mtx> [reps=5]\n");
    return 1;
  }
  const int reps = (argc > 3) ? std::atoi(argv[3]) : 5;

  const Coo cl = readMtx(argv[1]);
  const Coo cp = readMtx(argv[2]);
  if (cl.rows != cp.rows) {
    std::fprintf(stderr, "shape mismatch: L is %ldx%ld, P is %ldx%ld\n", (long)cl.rows,
                 (long)cl.cols, (long)cp.rows, (long)cp.cols);
    return 1;
  }
  Mat L = assembleEmbedded(cl);
  Mat P = assembleEmbedded(cp);

  // Two numbers, because a per-frame rebuild does not pay the same cost twice.
  //
  //   INITIAL: symbolic + numeric. What the first build costs.
  //   REUSE:   numeric only, same sparsity pattern, new values. This is the
  //            HONEST best case for Galerkin in our regime -- the connection
  //            changes every frame but the grid does not, so the pattern of
  //            both P and L is fixed and a real implementation would cache the
  //            symbolic phase and redo only the numeric one.
  //
  // Quoting only INITIAL would overstate the Galerkin cost; quoting only REUSE
  // would ignore that the pattern must be built once per operator family.
  std::vector<double> msInit, msReuse;
  for (int r = 0; r < reps; ++r) {
    Mat G = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    check(MatPtAP(L, P, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &G), "MatPtAP initial");
    const auto t1 = std::chrono::steady_clock::now();
    msInit.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

    const auto t2 = std::chrono::steady_clock::now();
    check(MatPtAP(L, P, MAT_REUSE_MATRIX, PETSC_DEFAULT, &G), "MatPtAP reuse");
    const auto t3 = std::chrono::steady_clock::now();
    msReuse.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());

    if (r + 1 == reps) {
      std::printf("  fine %ld (embedded %ld), coarse %ld (embedded %ld)\n", (long)cl.rows,
                  (long)(2 * cl.rows), (long)cp.cols, (long)(2 * cp.cols));
      std::printf("  nnz/row: L %.1f   P %.1f   P^H L P %.1f\n", nnzPerRow(L), nnzPerRow(P),
                  nnzPerRow(G));
    }
    check(MatDestroy(&G), "MatDestroy");
  }
  std::sort(msInit.begin(), msInit.end());
  std::sort(msReuse.begin(), msReuse.end());
  std::printf("  MatPtAP median over %d reps:  initial %.2f ms  |  reuse (numeric only) %.2f ms\n",
              reps, msInit[msInit.size() / 2], msReuse[msReuse.size() / 2]);

  check(MatDestroy(&L), "MatDestroy");
  check(MatDestroy(&P), "MatDestroy");
  check(PetscFinalize(), "PetscFinalize");
  return 0;
}
