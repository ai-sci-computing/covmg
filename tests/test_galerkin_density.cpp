// The fill-in that decides the rediscretized-vs-Galerkin design choice.
//
// The paper chooses a REdiscretized coarse operator over the Galerkin one
// P^H L P, and stakes that choice on speed ("if Galerkin also solved faster,
// our choice would be indefensible in a paper about speed"). The reason it does
// not is structural rather than algorithmic: in three dimensions the Galerkin
// triple product fills a 7-point coarse stencil into a 27-point one, so every
// operator apply and -- far more importantly -- every smoother sweep on the
// coarse levels touches ~3.9x as many entries. The V-cycle spends most of its
// time smoothing, so that density ratio is what a wall-time argument rests on.
//
// This pins the density, not a time. Timing is machine- and
// implementation-dependent and would make a flaky test; the fill-in is exact,
// deterministic, and is the CAUSE of the timing effect. It is also the fact
// that differs between two and three dimensions: in 2D the same construction
// goes 5 -> 9 (a 1.8x penalty, which the Galerkin operator's better
// convergence factor can and does outweigh), in 3D 7 -> 27 (3.9x, which it
// does not). The paper's operator is three-dimensional, which is why the
// choice goes the way it does.
//
// P is taken from the library's own prolongGauge rather than rebuilt here: a
// hand-rolled version of the multi-pass covariant fill silently implemented
// only pass 1 once, leaving face- and cell-centre rows empty and producing a P
// with 7 nonzeros per coarse column instead of 27.
#include <complex>
#include <vector>

#include "doctest.h"
#include "solvers/GaugeMultigrid.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

// Uniform-flux 3-torus in Landau/seam gauge -- the paper's Section 5 operator.
GaugeLattice fluxTorus(int n, int nPhi) {
  const double phi = 2.0 * M_PI * nPhi / (double(n) * n);
  const std::size_t N = static_cast<std::size_t>(n) * n * n;
  std::vector<double> lkx(N, 0.0), lky(N, 0.0), lkz(N, 0.0);
  const auto idx = [&](int i, int j, int k) { return (i * n + j) * n + k; };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        lkx[idx(i, j, k)] = -phi * j;
        if (j == n - 1) lky[idx(i, j, k)] = 2.0 * M_PI * nPhi * i / double(n);
      }
  return gaugeLatticePeriodic(n, n, n, double(n) * n, lkx, lky, lkz);
}

// Dense column-major-ish helper: apply `f` to each unit vector to materialize a
// matrix. Only used at tiny sizes.
std::vector<std::vector<cd>> materialize(long rows, long cols,
                                         const std::function<std::vector<cd>(const std::vector<cd>&)>& f) {
  std::vector<std::vector<cd>> M(static_cast<std::size_t>(cols));
  for (long c = 0; c < cols; ++c) {
    std::vector<cd> e(static_cast<std::size_t>(cols), cd(0, 0));
    e[static_cast<std::size_t>(c)] = cd(1, 0);
    M[static_cast<std::size_t>(c)] = f(e);
    REQUIRE(static_cast<long>(M[static_cast<std::size_t>(c)].size()) == rows);
  }
  return M;
}

// Mean nonzeros per row of a dense matrix given column-wise.
double nnzPerRow(const std::vector<std::vector<cd>>& cols, long rows, double tol = 1e-12) {
  long nnz = 0;
  for (const auto& col : cols)
    for (const cd& v : col)
      if (std::abs(v) > tol) ++nnz;
  return static_cast<double>(nnz) / static_cast<double>(rows);
}

}  // namespace

TEST_CASE("3D Galerkin coarse operator fills 7-point into 27-point") {
  const int n = 8;
  const GaugeLattice L = fluxTorus(n, 2);
  const auto levels = buildGaugeLevels(L);
  REQUIRE(levels.size() >= 2);

  const long nf = L.numNodes();
  const long nc = levels[1].numNodes();
  REQUIRE(nc == nf / 8);

  // The REdiscretized coarse operator: the connection Laplacian of the
  // coarsened lattice. Periodic 3D => exactly 7 entries per row (self + 6
  // axis neighbours).
  const auto redis =
      materialize(nc, nc, [&](const std::vector<cd>& v) { return applyConnectionLaplacian(levels[1], v); });
  const double dRedis = nnzPerRow(redis, nc);
  CHECK(dRedis == doctest::Approx(7.0));

  // The Galerkin coarse operator P^H L P, with P the library's own covariant
  // prolongation. Built column by column: (P^H L P) e_c = P^H (L (P e_c)).
  const auto galerkin = materialize(nc, nc, [&](const std::vector<cd>& v) {
    return restrictGauge(L, applyConnectionLaplacian(L, prolongGauge(L, v)));
  });
  const double dGalerkin = nnzPerRow(galerkin, nc);

  // 27 = the full 3x3x3 stencil. This is the fill-in the design choice turns
  // on: every coarse smoother sweep touches this many entries per row.
  CHECK(dGalerkin == doctest::Approx(27.0));
  CHECK(dGalerkin / dRedis > 3.5);

  // Sanity: P really is the multi-pass covariant fill (27 nonzeros per coarse
  // column in 3D), not a truncated one. A pass-1-only fill would give 7 and
  // would also make the Galerkin operator look deceptively sparse.
  const auto P = materialize(nf, nc, [&](const std::vector<cd>& v) { return prolongGauge(L, v); });
  CHECK(nnzPerRow(P, nc) == doctest::Approx(27.0));
}

TEST_CASE("Galerkin coarse operator is Hermitian and positive definite") {
  // Not a rival to the density claim but a guard on the construction: if
  // P^H L P came out non-Hermitian the density above would be measuring
  // something other than a coarse operator.
  const int n = 8;
  const GaugeLattice L = fluxTorus(n, 2);
  const long nc = L.numNodes() / 8;
  const auto G = materialize(nc, nc, [&](const std::vector<cd>& v) {
    return restrictGauge(L, applyConnectionLaplacian(L, prolongGauge(L, v)));
  });
  double asym = 0.0;
  for (long i = 0; i < nc; ++i)
    for (long j = 0; j < nc; ++j)
      asym = std::max(asym, std::abs(G[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] -
                                     std::conj(G[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)])));
  CHECK(asym < 1e-10);

  // Positive on a random vector (a full eigen-decomposition is overkill here).
  std::vector<cd> x(static_cast<std::size_t>(nc));
  for (long i = 0; i < nc; ++i) x[static_cast<std::size_t>(i)] = cd(std::sin(0.7 * i), std::cos(0.3 * i));
  cd q(0, 0);
  for (long i = 0; i < nc; ++i)
    for (long j = 0; j < nc; ++j)
      q += std::conj(x[static_cast<std::size_t>(i)]) * G[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] *
           x[static_cast<std::size_t>(j)];
  CHECK(q.real() > 0.0);
  CHECK(std::abs(q.imag()) < 1e-8);
}
