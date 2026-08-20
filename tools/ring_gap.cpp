// Throwaway: two lowest eigenvalues of the seeded-ring connection Laplacian
// across n, to settle whether the relative gap lambda1/lambda2 narrows under
// refinement or converges to a limit. Construction identical to eig_compare.
#include <cmath>
#include <cstdio>
#include <vector>

#include "grid/MacGrid.h"
#include "fluid/MacVortexRing.h"
#include "solvers/GaugeEigen.h"
#include "extraction/MacConnectionLaplacian.h"

using namespace bochner;

int main(int argc, char** argv) {
  std::printf("%5s %12s %12s %14s %14s\n", "n", "lam1", "lam2", "lam1/lam2",
              "(l2-l1)/l1");
  for (int a = 1; a < argc; ++a) {
    const int n = std::atoi(argv[a]);
    const MacGrid g(n, n, n, 1.6 / n, Vec3{-0.8, -0.8, -0.8});
    const double R = 0.7, Gamma = 1.0, hbar = Gamma / (2.0 * M_PI);
    const auto u = vortexRingFaceField(g, {0, 0, 0}, {0, 0, 1}, R, Gamma, 0.15);
    const auto theta = connectionAngles(g, u, hbar);
    const GaugeLattice lat = gaugeLatticeFromFaces(g, theta);

    GaugeEigenOptions opts;
    opts.tol = 1e-7;
    opts.maxIters = 600;
    const BlockEigResult r = lowestEigenpairsGaugeMG(lat, 2, nullptr, opts);
    const double l1 = r.eigenvalues[0], l2 = r.eigenvalues[1];
    std::printf("%5d %12.6f %12.6f %14.8f %14.8f   %s\n", n, l1, l2, l1 / l2,
                (l2 - l1) / l1, r.maxResidual < 1e-7 ? "" : "NOT CERTIFIED");
    std::fflush(stdout);
  }
  return 0;
}
