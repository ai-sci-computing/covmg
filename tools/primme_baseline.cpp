/// \file
/// Survey round 2026-08-15: PRIMME baseline for the smallest eigenpair of
/// the U(1) connection Laplacian -- the second production route on exactly this
/// operator class (lattice distillation, e.g. arXiv:2510.26459 uses PRIMME for
/// the 3D covariant Laplacian per time-slice). Unpreconditioned, zero setup,
/// as deployed there. Methods: PRIMME_DEFAULT_MIN_TIME (JDQMR) and
/// PRIMME_DEFAULT_MIN_MATVECS (GD+k), both reported (the package's own two
/// presets; the sweep convention). PRIMME's eps is relative to an estimate of
/// ||A||, not to lambda, so we sweep eps and report the LOOSEST setting whose
/// returned vector certifies the paper-wide criterion, rel eigen-residual
/// ||Lv - rho v|| / rho <= 1e-7, recomputed here in double (the fairness
/// convention: the baseline gets its best working configuration).
///
/// NOT in CMake (local-only bench dep). Build:
///   clang++ -O2 -std=c++17 -Isrc -I<primme>/include tools/primme_baseline.cpp \
///     build/src/solvers/libbochner_solvers.a build/src/grid/libbochner_grid.a \
///     <primme>/lib/libprimme.a <path-to-libomp>/libomp.dylib \
///     <petsc>/lib/libpetsc.dylib <slepc>/lib/libslepc.dylib \
///     -framework Accelerate -o build/tools/primme_baseline
/// Usage: primme_baseline torus <n> <nPhi> | file <path.glat>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "primme.h"

#include "solvers/GaugeMultigrid.h"

using bochner::GaugeLattice;
using cd = std::complex<double>;

namespace {

// == torus_eig_compare's uniformFluxLattice (nPhi quanta, Landau/seam gauge).
GaugeLattice uniformFluxLattice(int n, int nPhi, double h) {
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
  return bochner::gaugeLatticePeriodic(n, n, n, 1.0 / (h * h), lkx, lky, lkz);
}

// == flux_alias_diag's loadGaugeLattice (GLAT v1).
GaugeLattice loadGlat(const char* path) {
  std::FILE* fp = std::fopen(path, "rb");
  if (!fp) throw std::runtime_error(std::string("cannot open ") + path);
  const auto rd = [&](void* p, size_t sz, size_t cnt) {
    if (std::fread(p, sz, cnt, fp) != cnt) throw std::runtime_error("short read");
  };
  char magic[4];
  int version = 0, per = 0;
  GaugeLattice L;
  rd(magic, 1, 4);
  if (std::memcmp(magic, "GLAT", 4) != 0) throw std::runtime_error("bad magic");
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

void matvec(void* x, PRIMME_INT* ldx, void* y, PRIMME_INT* ldy, int* blockSize,
            primme_params* primme, int* ierr) {
  const GaugeLattice* L = static_cast<const GaugeLattice*>(primme->matrix);
  const std::size_t n = static_cast<std::size_t>(primme->n);
  for (int b = 0; b < *blockSize; ++b) {
    const cd* xin = static_cast<const cd*>(x) + static_cast<std::size_t>(*ldx) * b;
    cd* yout = static_cast<cd*>(y) + static_cast<std::size_t>(*ldy) * b;
    const std::vector<cd> xv(xin, xin + n);
    const std::vector<cd> yv = bochner::applyConnectionLaplacian(*L, xv);
    std::copy(yv.begin(), yv.end(), yout);
  }
  *ierr = 0;
}

double certResidual(const GaugeLattice& L, const std::vector<cd>& v, double* lambdaOut) {
  const std::vector<cd> Lv = bochner::applyConnectionLaplacian(L, v);
  cd num(0, 0);
  double den = 0.0, rr = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    num += std::conj(v[i]) * Lv[i];
    den += std::norm(v[i]);
  }
  const double rho = num.real() / den;
  for (std::size_t i = 0; i < v.size(); ++i) rr += std::norm(Lv[i] - rho * v[i]);
  *lambdaOut = rho;
  return std::sqrt(rr / den) / std::max(std::abs(rho), 1e-300);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: primme_baseline torus <n> <nPhi> | file <path.glat>\n");
    return 2;
  }
  GaugeLattice L;
  std::string label;
  if (std::string(argv[1]) == "torus") {
    const int n = std::atoi(argv[2]);
    const int nPhi = argc > 3 ? std::atoi(argv[3]) : 4;
    L = uniformFluxLattice(n, nPhi, 1.0 / n);
    label = "torus n=" + std::to_string(n) + " nPhi=" + std::to_string(nPhi);
  } else {
    L = loadGlat(argv[2]);
    label = argv[2];
  }
  const std::size_t N = static_cast<std::size_t>(L.numNodes());
  std::printf("PRIMME baseline on %s (%zu complex nodes), certified target rel res 1e-7\n",
              label.c_str(), N);
  std::printf("%-16s %8s | %12s | %6s/%7s | %9s | %s\n", "method", "eps", "lambda", "outer",
              "matvec", "ms(med5)", "certified res");

  const struct { primme_preset_method m; const char* name; } methods[] = {
      {PRIMME_DEFAULT_MIN_TIME, "JDQMR(min-time)"},
      {PRIMME_DEFAULT_MIN_MATVECS, "GD+k(min-mv)"}};
  for (const auto& meth : methods) {
    bool done = false;
    for (const double eps : {1e-8, 1e-9, 1e-10, 1e-11}) {  // loosest certifying wins
      primme_params primme;
      primme_initialize(&primme);
      primme.n = static_cast<PRIMME_INT>(N);
      primme.matrixMatvec = matvec;
      primme.matrix = &L;
      primme.numEvals = 1;
      primme.target = primme_smallest;
      primme.eps = eps;
      primme_set_method(meth.m, &primme);
      std::vector<cd> evecs(N, cd(1.0 / std::sqrt(double(N)), 0.0));  // constant start
      std::vector<double> evals(1, 0.0), rnorms(1, 0.0);
      primme.initSize = 1;
      std::vector<double> ms;
      int ret = 0;
      primme_params saved = primme;  // stats of the last rep are reported
      for (int rep = 0; rep < 5; ++rep) {
        primme = saved;
        std::fill(evecs.begin(), evecs.end(), cd(1.0 / std::sqrt(double(N)), 0.0));
        primme.initSize = 1;
        const auto t0 = std::chrono::steady_clock::now();
        ret = zprimme(evals.data(), reinterpret_cast<PRIMME_COMPLEX_DOUBLE*>(evecs.data()),
                      rnorms.data(), &primme);
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (ret != 0) break;
      }
      if (ret != 0) {
        std::printf("%-16s %8.0e | zprimme returned %d\n", meth.name, eps, ret);
        primme_free(&primme);
        continue;
      }
      double lambda = 0.0;
      const double res = certResidual(L, evecs, &lambda);
      std::sort(ms.begin(), ms.end());
      if (res <= 1e-7) {
        std::printf("%-16s %8.0e | %12.6f | %6lld/%7lld | %9.1f | %.1e  <- certified\n",
                    meth.name, eps, lambda,
                    static_cast<long long>(primme.stats.numOuterIterations),
                    static_cast<long long>(primme.stats.numMatvecs), ms[ms.size() / 2], res);
        primme_free(&primme);
        done = true;
        break;
      }
      std::printf("%-16s %8.0e | %12.6f | %6lld/%7lld | %9.1f | %.1e  (not certified)\n",
                  meth.name, eps, lambda, static_cast<long long>(primme.stats.numOuterIterations),
                  static_cast<long long>(primme.stats.numMatvecs), ms[ms.size() / 2], res);
      primme_free(&primme);
    }
    if (!done) std::printf("%-16s: no eps in the grid certified 1e-7\n", meth.name);
  }
  return 0;
}
