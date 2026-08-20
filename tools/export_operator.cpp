/// \file
/// Export the connection Laplacian as a COMPLEX MatrixMarket file, for external
/// eigensolver baselines (the PyAMG adaptive-SA / SA LOBPCG comparison). Two
/// operators, matching the paper's benchmarks exactly:
///
///   u1  <n> <nPhi> <out.mtx>  -- the uniform-flux U(1) torus (the
///                                lattice-gauge-solvers Examples::uniformField,
///                                same construction as torus_eig_compare);
///   sun <d> <n>    <out.mtx>  -- the smooth SU(d) field of sun_gauge_bench
///                                (amp = 4, massless, continuum weight w = n^2).
///
/// The matrix is written in coordinate format, field `complex`, symmetry
/// `general` with BOTH triangles present (readers stay trivial; the operator is
/// Hermitian, which we assert before writing). Entries are the N x N (u1) or
/// dN x dN (sun) complex operator -- NOT the real 2N embedding; scripts that
/// need the embedding build it from the complex data.
///
/// No PETSc/SLEPc required; links only bochner_core. Each export is verified
/// against the matrix-free operator (applyConnectionLaplacian /
/// applySunLaplacian) on a deterministic probe before the file is written.
#include <cmath>
#include <array>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "solvers/GaugeMultigrid.h"
#include "solvers/SunGauge.h"
#include "solvers/WilsonMc.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

/// Complex COO accumulator (duplicate (row, col) contributions sum), ordered by
/// (row, col) for a canonical .mtx output.
class ComplexCoo {
 public:
  explicit ComplexCoo(long dim) : dim_(dim) {}
  void add(long i, long j, cd v) { entries_[{i, j}] += v; }
  long dim() const { return dim_; }
  const std::map<std::pair<long, long>, cd>& entries() const { return entries_; }

  /// y = A x (for the matrix-free cross-check).
  std::vector<cd> apply(const std::vector<cd>& x) const {
    std::vector<cd> y(static_cast<std::size_t>(dim_), cd(0, 0));
    for (const auto& [rc, v] : entries_) y[static_cast<std::size_t>(rc.first)] += v * x[static_cast<std::size_t>(rc.second)];
    return y;
  }

  /// max_{i,j} |A_ij - conj(A_ji)| -- 0 for an exactly Hermitian assembly.
  double hermitianDefect() const {
    double defect = 0.0;
    for (const auto& [rc, v] : entries_) {
      const auto it = entries_.find({rc.second, rc.first});
      const cd t = (it != entries_.end()) ? it->second : cd(0, 0);
      defect = std::max(defect, std::abs(v - std::conj(t)));
    }
    return defect;
  }

 private:
  long dim_;
  std::map<std::pair<long, long>, cd> entries_;
};

// ---------------------------------------------------------------------------
// U(1): the lattice-gauge-solvers Examples::uniformField -- nPhi flux quanta
// through the x-y torus in the Landau/seam gauge (same as torus_eig_compare).
// ---------------------------------------------------------------------------

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
  return gaugeLatticePeriodic(n, n, n, 1.0 / (h * h), lkx, lky, lkz);
}

// Assemble the periodic U(1) connection Laplacian as the COMPLEX N x N matrix,
// matching applyConnectionLaplacian: diagonal 6w, off-diagonal -w e^{-i theta}
// from neighbour to centre (and its conjugate the other way).
ComplexCoo assembleU1(const GaugeLattice& L) {
  ComplexCoo A(L.numNodes());
  const auto lnk = [&](long a, long b, double th, double w) {
    A.add(a, a, cd(w, 0));
    A.add(b, b, cd(w, 0));
    A.add(a, b, -w * cd(std::cos(th), -std::sin(th)));  // -w e^{-i th}
    A.add(b, a, -w * cd(std::cos(th), std::sin(th)));   // conj
  };
  // Per-edge weight with uniform fallback (unweighted path unchanged).
  const auto ew = [&](const std::vector<double>& wv, long e) {
    return L.weighted() ? wv[static_cast<std::size_t>(e)] : L.w;
  };
  // Link indexing as in GaugeMultigrid.cpp (open lattices shorten the
  // coarsened extent of the y/z link arrays; no wrap link on open axes).
  const long ly1 = L.periodic ? L.ly : L.ly - 1, lz1 = L.periodic ? L.lz : L.lz - 1;
  for (int i = 0; i < L.lx; ++i)
    for (int j = 0; j < L.ly; ++j)
      for (int k = 0; k < L.lz; ++k) {
        const long c = L.index(i, j, k);
        if (L.periodic || i + 1 < L.lx) {
          const long e = (static_cast<long>(i) * L.ly + j) * L.lz + k;
          lnk(c, L.index((i + 1) % L.lx, j, k), L.lkx[static_cast<std::size_t>(e)], ew(L.wx, e));
        }
        if (L.periodic || j + 1 < L.ly) {
          const long e = (static_cast<long>(i) * ly1 + j) * L.lz + k;
          lnk(c, L.index(i, (j + 1) % L.ly, k), L.lky[static_cast<std::size_t>(e)], ew(L.wy, e));
        }
        if (L.periodic || k + 1 < L.lz) {
          const long e = (static_cast<long>(i) * L.ly + j) * lz1 + k;
          lnk(c, L.index(i, j, (k + 1) % L.lz), L.lkz[static_cast<std::size_t>(e)], ew(L.wz, e));
        }
      }
  return A;
}

// BOCHNER_EXPORT_REGAUGE=seed[:scale]: random U(1) re-gauge of a lattice before
// assembly (unitarily equivalent operator; 2026-08-19 gauge-dependence study).
void maybeRegauge(GaugeLattice& L, char* comment, size_t cap) {
  const char* e = std::getenv("BOCHNER_EXPORT_REGAUGE");
  if (!e || !*e) return;
  unsigned long long seed = 0;
  double scale = M_PI;
  std::sscanf(e, "%llu:%lf", &seed, &scale);
  randomGaugeTransform(L, seed, scale);
  const size_t n = std::strlen(comment);
  std::snprintf(comment + n, cap - n, " | random re-gauge seed=%llu |phi|<=%g", seed, scale);
}

// The smooth anisotropic diagonal metric of tools/aniso_bench section [B]
// (per-axis different smooth conductances, contrast 1+amp), applied to the
// uniform-flux lattice. Copied verbatim so the exported operator matches the
// one aniso_bench measures.
void applyAbstractMetric(GaugeLattice& L, int n, double amp) {
  const double w = double(n) * n, h = 1.0 / n;
  const auto s2 = [](double t) {
    const double s = std::sin(M_PI * t);
    return s * s;
  };
  const auto cfun = [&](int axis, double x, double y, double z) {
    if (axis == 0) return 1.0 + amp * s2(x) * s2(y);
    if (axis == 1) return 1.0 + amp * s2(y) * s2(z);
    return 1.0 + amp * s2(z) * s2(x);
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
          wv[static_cast<std::size_t>((i * n + j) * n + k)] = w * cfun(axis, x, y, z);
        }
    return wv;
  };
  L.setEdgeWeights(graded(0), graded(1), graded(2));
}

// ---------------------------------------------------------------------------
// SU(d): the smooth link field of sun_gauge_bench (copied helpers) -- exp of a
// smooth su(d) field, amplitude ~ amp*h, so links -> I under refinement.
// ---------------------------------------------------------------------------

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
  v[0][0] = S(j); v[0][1] = S(k); v[0][2] = 0.0;   // block (0,1)
  v[1][0] = S(k); v[1][1] = 0.0;  v[1][2] = S(i);   // block (0,2)
  v[2][0] = 0.0;  v[2][1] = S(i); v[2][2] = S(j);   // block (1,2)
  const double rot = (axis == 0) ? 0.0 : (axis == 1) ? 2.09439510239 : 4.18879020479;  // +/-120deg phase
  for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c) v[b][c] = amp * h * (v[b][c] + 0.5 * std::sin(2.0 * M_PI * (i + j + k) / n + rot));
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

SunLattice smoothLattice(int d, int n, double mass2, double amp) {
  SunLattice L;
  L.d = d;
  L.lx = L.ly = L.lz = n;
  L.periodic = true;
  L.w = static_cast<double>(n) * n;  // 1/h^2 with h = 1/n (continuum refinement)
  L.mass2 = mass2;
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

// Assemble the periodic SU(d) covariant Laplacian as the COMPLEX dN x dN matrix
// (the complex-valued twin of sun_gauge_bench's assembleReal, without the real
// 2x2 embedding): diagonal (6w + m^2) I_d, off-diagonal -w U_{n->c}.
ComplexCoo assembleSun(const SunLattice& L) {
  const int d = L.d, dd = d * d;
  ComplexCoo A(L.dof());
  auto block = [&](long c, long nb, const cd* U, bool adj) {  // -w * U_{nb->c}[a,b]
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        const cd t = adj ? std::conj(U[b * d + a]) : U[a * d + b];
        if (t != cd(0, 0)) A.add(c * d + a, nb * d + b, -L.w * t);
      }
  };
  const int lx = L.lx, ly = L.ly, lz = L.lz;
  const double diag = L.w * 6.0 + L.mass2;  // periodic degree = 6
  for (int i = 0; i < lx; ++i)
    for (int j = 0; j < ly; ++j)
      for (int k = 0; k < lz; ++k) {
        const long c = L.index(i, j, k);
        for (int a = 0; a < d; ++a) A.add(c * d + a, c * d + a, cd(diag, 0));
        const int im = (i - 1 + lx) % lx, ip = (i + 1) % lx;
        const int jm = (j - 1 + ly) % ly, jp = (j + 1) % ly;
        const int km = (k - 1 + lz) % lz, kp = (k + 1) % lz;
        block(c, L.index(im, j, k), &L.ux[static_cast<size_t>(L.index(im, j, k)) * dd], false);
        block(c, L.index(ip, j, k), &L.ux[static_cast<size_t>(c) * dd], true);
        block(c, L.index(i, jm, k), &L.uy[static_cast<size_t>(L.index(i, jm, k)) * dd], false);
        block(c, L.index(i, jp, k), &L.uy[static_cast<size_t>(c) * dd], true);
        block(c, L.index(i, j, km), &L.uz[static_cast<size_t>(L.index(i, j, km)) * dd], false);
        block(c, L.index(i, j, kp), &L.uz[static_cast<size_t>(c) * dd], true);
      }
  return A;
}

// ---------------------------------------------------------------------------
// Verification + MatrixMarket output.
// ---------------------------------------------------------------------------

std::vector<cd> probeVector(long n) {
  std::vector<cd> x(static_cast<std::size_t>(n));
  for (long c = 0; c < n; ++c) x[static_cast<std::size_t>(c)] = cd(std::cos(0.7 * c), std::sin(0.3 * c));
  return x;
}

double maxAbsDiff(const std::vector<cd>& a, const std::vector<cd>& b) {
  double m = 0.0;
  for (std::size_t c = 0; c < a.size(); ++c) m = std::max(m, std::abs(a[c] - b[c]));
  return m;
}

bool writeMatrixMarket(const ComplexCoo& A, const std::string& path, const std::string& comment) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "error: cannot open '%s' for writing\n", path.c_str());
    return false;
  }
  std::fprintf(f, "%%%%MatrixMarket matrix coordinate complex general\n");
  std::fprintf(f, "%% %s\n", comment.c_str());
  std::fprintf(f, "%% Hermitian operator; both triangles stored (symmetry 'general').\n");
  std::fprintf(f, "%ld %ld %zu\n", A.dim(), A.dim(), A.entries().size());
  for (const auto& [rc, v] : A.entries())
    std::fprintf(f, "%ld %ld %.16e %.16e\n", rc.first + 1, rc.second + 1, v.real(), v.imag());
  std::fclose(f);
  return true;
}

// Export the covariant PROLONGATION as a MatrixMarket matrix, by applying the
// library's own prolongGauge to each coarse unit basis vector. Reconstructing
// the multi-pass fill by hand is error-prone -- a first attempt at it silently
// implemented only pass 1, leaving face-centre and cell-centre rows empty --
// so anything that needs P as a matrix should take it from here.
bool exportProlongation(int n, int nPhi, const std::string& path) {
  const GaugeLattice fine = uniformFluxLattice(n, nPhi, 1.0 / n);
  const int nc = n / 2;
  const long Nc = static_cast<long>(nc) * nc * nc;
  const long N = static_cast<long>(n) * n * n;
  std::vector<std::array<long, 2>> ij;
  std::vector<std::complex<double>> val;
  for (long c = 0; c < Nc; ++c) {
    std::vector<std::complex<double>> e(static_cast<std::size_t>(Nc), std::complex<double>(0, 0));
    e[static_cast<std::size_t>(c)] = std::complex<double>(1, 0);
    const std::vector<std::complex<double>> col = prolongGauge(fine, e);
    for (long r = 0; r < N; ++r)
      if (std::abs(col[static_cast<std::size_t>(r)]) > 1e-15) {
        ij.push_back({r, c});
        val.push_back(col[static_cast<std::size_t>(r)]);
      }
  }
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return false;
  std::fprintf(f, "%%%%MatrixMarket matrix coordinate complex general\n");
  std::fprintf(f, "%% covariant prolongation from the library's prolongGauge\n");
  std::fprintf(f, "%ld %ld %zu\n", N, Nc, val.size());
  for (std::size_t t = 0; t < val.size(); ++t)
    std::fprintf(f, "%ld %ld %.16e %.16e\n", ij[t][0] + 1, ij[t][1] + 1, val[t].real(),
                 val[t].imag());
  std::fclose(f);
  return true;
}

int usage() {
  std::fprintf(stderr,
               "usage: export_operator u1 <n> <nPhi> <out.mtx>\n"
               "       export_operator glat <path.glat> <out.mtx>\n"
               "         (BOCHNER_EXPORT_REGAUGE=seed[:scale] re-gauges u1/glat before export)\n"
               "       export_operator u1g <n> <nPhi> <out.mtx>   (abstract 4x metric)\n"
               "       export_operator prolong <n> <nPhi> <out.mtx>\n"
               "       export_operator sun <d> <n> <out.mtx>\n"
               "       export_operator mc <d> <n> <beta> <seed> <out.mtx>\n"
               "         (Wilson-action MC config, 300 sweeps, w=n^2, m^2=0;\n"
               "          hot start for beta<6, cold otherwise -- the\n"
               "          mc_gauge_bench protocol)\n");
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (std::string(argv[1]) == "prolong") {
    if (argc != 5) return usage();
    return exportProlongation(std::atoi(argv[2]), std::atoi(argv[3]), argv[4]) ? 0 : 1;
  }
  if (argc != 4 && argc != 5 && argc != 7) return usage();
  const std::string kind = argv[1];
  if (argc == 4 && kind != "glat") return usage();

  if (kind == "glat") {
    // export_operator glat <path.glat> <out.mtx> (argc == 4 handled below)
    GaugeLattice lat = loadGaugeLatticeFile(argv[2]);
    char comment[200];
    std::snprintf(comment, sizeof comment, "U(1) connection Laplacian from %s (%dx%dx%d, periodic=%d)",
                  argv[2], lat.lx, lat.ly, lat.lz, lat.periodic ? 1 : 0);
    maybeRegauge(lat, comment, sizeof comment);
    const ComplexCoo A = assembleU1(lat);
    const std::vector<cd> x = probeVector(A.dim());
    const double mismatch = maxAbsDiff(A.apply(x), applyConnectionLaplacian(lat, x));
    const double herm = A.hermitianDefect();
    std::printf("glat: %s dim=%ld nnz=%zu  matrix-free mismatch=%.2e  hermitian defect=%.2e\n",
                argv[2], A.dim(), A.entries().size(), mismatch, herm);
    if (mismatch > 1e-10 || herm > 1e-12) {
      std::fprintf(stderr, "error: assembly verification failed\n");
      return 1;
    }
    return writeMatrixMarket(A, argv[3], comment) ? 0 : 1;
  }
  if (kind == "u1") {
    const int n = std::atoi(argv[2]), nPhi = std::atoi(argv[3]);
    if (n < 4 || n % 2 != 0) {
      std::fprintf(stderr, "error: n must be even and >= 4\n");
      return 1;
    }
    GaugeLattice lat = uniformFluxLattice(n, nPhi, 1.0 / n);
    char comment[200];
    std::snprintf(comment, sizeof comment,
                  "U(1) connection Laplacian, uniform-flux 3-torus: n=%d, nPhi=%d, w=n^2", n, nPhi);
    maybeRegauge(lat, comment, sizeof comment);
    const ComplexCoo A = assembleU1(lat);
    const std::vector<cd> x = probeVector(A.dim());
    const double mismatch = maxAbsDiff(A.apply(x), applyConnectionLaplacian(lat, x));
    const double herm = A.hermitianDefect();
    std::printf("u1: n=%d nPhi=%d  dim=%ld nnz=%zu  matrix-free mismatch=%.2e  hermitian defect=%.2e\n",
                n, nPhi, A.dim(), A.entries().size(), mismatch, herm);
    if (mismatch > 1e-10 || herm > 1e-12) {
      std::fprintf(stderr, "error: assembly verification failed\n");
      return 1;
    }
    return writeMatrixMarket(A, argv[4], comment) ? 0 : 1;
  }

  if (kind == "u1g") {
    const int n = std::atoi(argv[2]), nPhi = std::atoi(argv[3]);
    if (n < 4 || n % 2 != 0) {
      std::fprintf(stderr, "error: n must be even and >= 4\n");
      return 1;
    }
    GaugeLattice lat = uniformFluxLattice(n, nPhi, 1.0 / n);
    applyAbstractMetric(lat, n, 3.0);  // 4x contrast, aniso_bench [B]
    const ComplexCoo A = assembleU1(lat);
    const std::vector<cd> x = probeVector(A.dim());
    const double mismatch = maxAbsDiff(A.apply(x), applyConnectionLaplacian(lat, x));
    const double herm = A.hermitianDefect();
    std::printf("u1g: n=%d nPhi=%d  dim=%ld nnz=%zu  matrix-free mismatch=%.2e  hermitian defect=%.2e\n",
                n, nPhi, A.dim(), A.entries().size(), mismatch, herm);
    if (mismatch > 1e-10 || herm > 1e-12) {
      std::fprintf(stderr, "error: assembly verification failed\n");
      return 1;
    }
    char comment[200];
    std::snprintf(comment, sizeof comment,
                  "U(1) connection Laplacian, uniform-flux 3-torus + abstract metric (4x contrast, "
                  "aniso_bench [B]): n=%d, nPhi=%d, w=n^2 c_a(x)", n, nPhi);
    return writeMatrixMarket(A, argv[4], comment) ? 0 : 1;
  }

  if (kind == "sun") {
    const int d = std::atoi(argv[2]), n = std::atoi(argv[3]);
    if (d < 2 || d > 3 || n < 4 || n % 2 != 0) {
      std::fprintf(stderr, "error: need d in {2,3} and n even, >= 4\n");
      return 1;
    }
    const SunLattice lat = smoothLattice(d, n, /*mass2=*/0.0, /*amp=*/4.0);
    const ComplexCoo A = assembleSun(lat);
    const std::vector<cd> x = probeVector(A.dim());
    const double mismatch = maxAbsDiff(A.apply(x), applySunLaplacian(lat, x));
    const double herm = A.hermitianDefect();
    std::printf("sun: d=%d n=%d  dim=%ld nnz=%zu  matrix-free mismatch=%.2e  hermitian defect=%.2e\n",
                d, n, A.dim(), A.entries().size(), mismatch, herm);
    if (mismatch > 1e-10 || herm > 1e-12) {
      std::fprintf(stderr, "error: assembly verification failed\n");
      return 1;
    }
    char comment[160];
    std::snprintf(comment, sizeof comment,
                  "SU(%d) covariant Laplacian, smooth field (amp=4, massless): n=%d, w=n^2", d, n);
    return writeMatrixMarket(A, argv[4], comment) ? 0 : 1;
  }

  if (kind == "mc" && argc == 7) {
    const int d = std::atoi(argv[2]), n = std::atoi(argv[3]);
    const double beta = std::atof(argv[4]);
    const std::uint64_t seed = std::strtoull(argv[5], nullptr, 10);
    if (d < 2 || d > 3 || n < 4 || n % 2 != 0) {
      std::fprintf(stderr, "error: need d in {2,3} and n even, >= 4\n");
      return 1;
    }
    const SunLattice lat = mcSunLattice(d, n, beta, /*sweeps=*/300,
                                        /*w=*/double(n) * n, /*mass2=*/0.0,
                                        seed, /*hotStart=*/beta < 6.0);
    const ComplexCoo A = assembleSun(lat);
    const std::vector<cd> x = probeVector(A.dim());
    const double mismatch = maxAbsDiff(A.apply(x), applySunLaplacian(lat, x));
    const double herm = A.hermitianDefect();
    std::printf("mc: d=%d n=%d beta=%g seed=%llu  <P>=%.5f  dim=%ld nnz=%zu  "
                "matrix-free mismatch=%.2e  hermitian defect=%.2e\n",
                d, n, beta, static_cast<unsigned long long>(seed),
                averagePlaquette(lat), A.dim(), A.entries().size(), mismatch, herm);
    if (mismatch > 1e-10 || herm > 1e-12) {
      std::fprintf(stderr, "error: assembly verification failed\n");
      return 1;
    }
    char comment[200];
    std::snprintf(comment, sizeof comment,
                  "SU(%d) covariant Laplacian, Wilson MC config: n=%d, beta=%g, "
                  "seed=%llu, 300 sweeps, w=n^2, massless", d, n, beta,
                  static_cast<unsigned long long>(seed));
    return writeMatrixMarket(A, argv[6], comment) ? 0 : 1;
  }

  return usage();
}
