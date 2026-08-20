#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "grid/GridOperators.h"
#include "grid/MacGrid.h"

namespace bochner {

/// \brief A U(1) connection on a regular 3D lattice -- the substrate-independent
/// core of the gauge-aware multigrid.
///
/// This deliberately knows nothing about the MAC grid, PETSc, or the fluid
/// solver: it is just a lattice of complex nodes with a forward parallel
/// transport \f$e^{i\theta}\f$ on each axis edge and a connection-Laplacian edge
/// weight \f$w\f$ (= \f$1/h^2\f$ at this level). Everything the multigrid needs
/// -- the operator, smoother, and intergrid transfers -- is expressed from these
/// fields alone, so the solver can be lifted into a stand-alone library.
///
/// `lkx[i,j,k]` is the angle transporting node `(i,j,k)` to `(i+1,j,k)` (low to
/// high), with `i` in `0..lx-2`; `lky`/`lkz` analogously. Node linear index is
/// `(i*ly+j)*lz+k` (\ref index), matching MacGrid::cellIndex.
struct GaugeLattice {
  int lx = 0, ly = 0, lz = 0;
  bool periodic = false;              ///< true = 3-torus (links wrap); false = open (Neumann)
  double w = 1.0;                     ///< uniform edge weight (1/h^2) -- used iff wx/wy/wz are empty
  std::vector<double> lkx, lky, lkz;  ///< forward link angle (low->high) per axis
  std::vector<std::complex<double>> tx, ty, tz;  ///< precomputed e^{i*lk} (built by buildTransports)

  /// \name Optional per-edge weights (variable edge lengths / graded resolution)
  /// Same indexing and length as lkx/lky/lkz. Empty (the default) means the
  /// uniform weight \ref w on every edge -- the original solver, bit-for-bit.
  /// Non-empty replaces \f$w\f$ per edge: the operator diagonal becomes the sum
  /// of incident edge weights, the prolongation averages with the connecting
  /// edges' weights (the A-harmonic fill), and coarsening combines the two fine
  /// weights of a coarse edge by series conductance,
  /// \f$W = w_a w_b / (2(w_a + w_b))\f$ (uniform \f$w \to w/4\f$, so the
  /// weighted rules contain the uniform ones). All three arrays must be set
  /// together (\ref setEdgeWeights validates this).
  /// \{
  std::vector<double> wx, wy, wz;
  bool weighted() const { return !wx.empty(); }
  /// Install per-edge weights (sizes must match numLinksX/Y/Z; all entries > 0).
  /// \throws std::invalid_argument on a size mismatch or a non-positive weight.
  void setEdgeWeights(std::vector<double> wx_, std::vector<double> wy_, std::vector<double> wz_);
  /// \}

  int index(int i, int j, int k) const { return (i * ly + j) * lz + k; }
  long numNodes() const { return static_cast<long>(lx) * ly * lz; }

  /// Number of forward links along each axis: `lx` per (j,k) row when periodic
  /// (the i=lx-1 link wraps lx-1 -> 0), else `lx-1` (open, no wrap). The link
  /// index formula `(i*ly+j)*lz+k` is shared by both -- only the range of `i`
  /// (and hence the array length) differs.
  long numLinksX() const { return static_cast<long>(periodic ? lx : (lx > 0 ? lx - 1 : 0)) * ly * lz; }
  long numLinksY() const { return static_cast<long>(lx) * (periodic ? ly : (ly > 0 ? ly - 1 : 0)) * lz; }
  long numLinksZ() const { return static_cast<long>(lx) * ly * (periodic ? lz : (lz > 0 ? lz - 1 : 0)); }

  /// Populate tx/ty/tz = e^{i*lk} from the link angles -- call after the lk
  /// arrays are set, so the matvec/smoother multiply by a stored transport
  /// instead of evaluating a transcendental per edge per sweep.
  void buildTransports();
};

/// Apply a random U(1) gauge transformation g_i = e^{i phi_i}, phi ~ U(-scale, scale),
/// to the link angles in place (theta'_{ij} = theta_{ij} + phi_j - phi_i, i.e.
/// U' = g_head U g_tail^H on every forward link, open or periodic) and rebuild
/// the transports. Returns g (one unit complex per node) so that a start or
/// reference vector can be transformed along: psi' = g . psi. The operator
/// G^H L G so obtained is unitarily equivalent to L (same spectrum).
std::vector<std::complex<double>> randomGaugeTransform(GaugeLattice& L, std::uint64_t seed,
                                                       double scale = 3.14159265358979323846);

/// Read a lattice from the GLAT dump format written by the frozen-operator
/// tools (magic "GLAT", version, lx, ly, lz, periodic, w, then lkx/lky/lkz).
/// Transports are built before returning.
GaugeLattice loadGaugeLatticeFile(const char* path);

/// Build the finest lattice from a MAC face connection \p theta (interior faces)
/// with weight \f$w = 1/h^2\f$ -- the bridge from the fluid side to the solver.
GaugeLattice gaugeLatticeFromFaces(const MacGrid& g, const FaceField& theta);

/// \brief Build a **periodic** (3-torus) lattice directly from raw forward-link
/// angle arrays -- the bridge for the lattice-gauge benchmark, whose operator is
/// periodic and whose connection is a bare gauge field (not a MAC velocity).
///
/// Each array is `lx*ly*lz` long (one forward link per node per axis; the
/// `i=lx-1` x-link wraps `lx-1 -> 0`, etc.), indexed by the low node
/// `(i*ly+j)*lz+k`. `lkx[i,j,k]` is the angle transporting node `(i,j,k)` to
/// `((i+1) mod lx, j, k)`. Every dimension must be even (the wrap link would put
/// two same-parity nodes on one red-black color otherwise). \throws
/// std::invalid_argument on an odd dimension or a link-array size mismatch.
GaugeLattice gaugeLatticePeriodic(int lx, int ly, int lz, double w,
                                  const std::vector<double>& lkx, const std::vector<double>& lky,
                                  const std::vector<double>& lkz);

/// \brief The gauge-aware subdivision section: seed \f$\psi = 1\f$ on the
/// coarsest of \p numLevels decimations of \p finest and prolong up by covariant
/// transport averaging. This is the SAME prolongation the linear V-cycle uses,
/// exposed so \ref subdivisionSection (the covariant-subdivision warm
/// start) shares one lattice + transfer implementation with the multigrid.
std::vector<std::complex<double>> subdivisionSectionFromLattice(const GaugeLattice& finest,
                                                                int numLevels);

/// \name Intergrid transfers (the V-cycle's prolongation P and its exact adjoint R)
/// Exposed so the adjoint identity \f$\langle f, Pc\rangle = \langle Rf, c\rangle\f$
/// is directly testable in double precision: the alpha-scaled coarse correction
/// guarantees descent for ANY transfer, so a scaling or conjugation error
/// confined to the transfers would pass every solve-to-tolerance test. The
/// coarse vector lives on the decimated lattice ((lx/2)*(ly/2)*(lz/2) nodes);
/// every dimension of \p fine must be even.
/// \{
/// \p covariant = false selects the plain (untransported) averaging of the
/// ablation switch \ref MgOptions::covariantTransfer, so a diagnostic can form
/// BOTH transfer operators from the library's own construction rather than
/// reimplementing the plain one.
std::vector<std::complex<double>> prolongGauge(const GaugeLattice& fine,
                                               const std::vector<std::complex<double>>& coarse,
                                               bool covariant = true);
std::vector<std::complex<double>> restrictGauge(const GaugeLattice& fine,
                                                const std::vector<std::complex<double>>& fineVec,
                                                bool covariant = true);
/// \}

/// \brief Matrix-free connection ("magnetic") Laplacian: \f$y = E x\f$.
///
/// \f$(Ex)_c = w\sum_{n}\big(x_c - P^\nabla_{n\to c}\,x_n\big)\f$ over the
/// existing axis neighbours \f$n\f$ of cell \f$c\f$ (homogeneous-Neumann
/// boundary = missing neighbours simply absent). This is exactly the operator
/// \ref connectionLaplacian assembles, evaluated without forming the matrix.
std::vector<std::complex<double>> applyConnectionLaplacian(
    const GaugeLattice& lat, const std::vector<std::complex<double>>& x);

/// Tuning + reporting for the multigrid linear solve.
struct MgOptions {
  int nu1 = 2;           ///< pre-smoothing sweeps per level
  int nu2 = 2;           ///< post-smoothing sweeps per level
  int coarseSweeps = 30;  ///< smoothing sweeps at the coarsest level
  int maxCycles = 100;   ///< V-cycle iteration cap
  double omega = 1.0;    ///< red-black GS relaxation (1 = Gauss-Seidel; <2 = SOR)
  double tol = 1e-8;     ///< target relative residual ||b - Ex|| / ||b||

  /// 2026-08-15 survey round (experimental): > 0 swaps every red-black GS
  /// smoothing call in the V-cycle for ONE degree-k Chebyshev semi-iteration
  /// on [lambda_max/chebSmootherRatio, lambda_max] (Gershgorin lambda_max =
  /// 2 x diagonal scale), the classical reduction-free multigrid smoother --
  /// the "absorb the competitor" hybrid: polynomial as smoother, hierarchy as
  /// accelerator. The coarsest level uses degree = coarseSweeps on the same
  /// interval (matched work). 0 = shipped red-black GS, bit-identical.
  /// Unweighted lattices only; weighted levels fall back to GS.
  int chebSmootherDegree = 0;
  double chebSmootherRatio = 6.0;  ///< smoother interval [lambda_max/ratio, lambda_max]

  /// \name Ablation switches (benchmarking only -- leave true in production)
  /// \{
  /// Apply the A-energy-optimal step alpha = <p,r>/<p,Ap> to each coarse
  /// correction. false = raw injection (alpha = 1), the historical PTMG
  /// behaviour -- the non-Galerkin cycle then has no descent guarantee and can
  /// diverge on frustrated configurations.
  bool alphaStep = true;
  /// The step used for the coarse correction when \ref alphaStep is false.
  /// Default 1 = raw injection, the historical PTMG mode of the ablation table
  /// (which diverges on frustrated configurations, as it did then).
  ///
  /// Setting it to 1/2 selects the LINEAR, step-fixed cycle -- the one the
  /// spectral-equivalence chain actually describes, since the line search makes
  /// the preconditioner nonlinear and non-Hermitian and so puts the practical
  /// solver outside the Knyazev-Neymeyr hypotheses. Comparing the two measures
  /// what the nonlinearity buys. Ignored when \ref alphaStep is true.
  double fixedAlpha = 1.0;
  /// Parallel-transport the transferred values through the links. false =
  /// plain (non-covariant) averaging with the same stencil -- isolates the
  /// covariant transfer as the load-bearing ingredient.
  bool covariantTransfer = true;
  /// Ablation switch mirroring \ref covariantTransfer for the WEIGHTS: when
  /// false, the transfers average with uniform weights (transports kept),
  /// while the coarse operators keep the weighted rediscretization -- the
  /// unweighted, PTMG-style transfer applied to a weighted operator. Default
  /// true (bit-identical; inert on unweighted lattices). U(1)-path only:
  /// SunLattice carries a single uniform weight by construction, so the SU(d)
  /// transfers have no per-edge weights to honor and ignore this flag.
  bool weightedTransfer = true;
  /// \}
};

struct MgResult {
  int cycles = 0;
  /// True relative residual after the last cycle; -1 when not computed
  /// (tol <= 0 = fixed-cycle preconditioner mode skips the residual matvec).
  double relResidual = 0.0;
};

/// Result of an iterative linear solve (for the CG baseline / comparisons).
struct SolveStats {
  int iterations = 0;
  double relResidual = 0.0;
};

/// \brief Baseline: solve \f$E x = b\f$ by **unpreconditioned conjugate
/// gradients** on the same matrix-free Hermitian-SPD operator -- the reference
/// the gauge multigrid is measured against.
/// \param lat      The connection lattice (the matrix-free Hermitian-SPD operator).
/// \param b        Right-hand side.
/// \param x in/out: initial guess in, solution out.
/// \param tol      Relative-residual stopping tolerance.
/// \param maxIters Iteration cap.
SolveStats cgSolve(const GaugeLattice& lat, const std::vector<std::complex<double>>& b,
                   std::vector<std::complex<double>>& x, double tol = 1e-8, int maxIters = 5000);

/// \name Matvec instrumentation (benchmarking) -- counts applyConnectionLaplacian calls.
/// \{
long gaugeMatvecCount();
/// Node-touches accumulated over all applies + smoother sweeps; divide by the
/// finest level's node count for the FINE-GRID-EQUIVALENT apply count (the fair
/// multilevel work metric -- a raw sweep count over-weights cheap coarse levels).
/// Convention: the unit is one U(1) node-touch. An SU(d) node charges d^2
/// (its apply multiplies d*d link blocks), so U(1) is the d=1 case and
/// cross-fiber comparisons of this counter are FLOP-proportional.
long long gaugeMatvecNodeWork();
void resetGaugeMatvecCount();
/// Credit work from another fiber's operator path (the SU(d) apply and smoother
/// in SunGauge.cpp, at d^2 node-work per node) to the same counters, so
/// cross-fiber work-count comparisons do not read SU(d) as zero.
void gaugeMatvecAdd(long matvecs, long long nodeWork);
/// \}

/// \brief Solve \f$E x = b\f$ for the connection Laplacian by gauge-aware
/// multigrid V-cycles (matrix-free).
///
/// The hierarchy coarsens by decimation (a coarse node is every other fine node;
/// a coarse link is the sum of the two fine links it spans -- the restricted
/// U(1) connection); the prolongation is the parallel-transport averaging of the
/// covariant subdivision, the restriction is its exact
/// adjoint, and each coarse operator is the connection Laplacian re-evaluated on
/// that level's links. Smoother is red-black Gauss-Seidel (parallel within a
/// colour); each coarse correction is scaled by the A-energy-optimal step.
///
/// \warning **The V-cycle is neither symmetric nor linear -- by design.** The
/// *operator* \f$E\f$ is Hermitian-SPD (as the surrounding docs say), but do not
/// transfer that to the preconditioner this cycle defines. Two independent
/// reasons: (1) the smoother always sweeps red-then-black, whose A-adjoint is
/// black-then-red, giving a measured relative asymmetry
/// \f$|\langle u,Mv\rangle - \langle Mu,v\rangle| / |\langle u,Mv\rangle|
/// \approx 2\cdot 10^{-1}\f$ at production settings; and (2), the binding one,
/// the A-energy-optimal step \ref MgOptions::alphaStep makes \f$M\f$ *nonlinear*
/// (homogeneous of degree 1 but not additive; measured additivity defect
/// \f$2.3\cdot 10^{-2}\f$ at scale 3.2, versus \f$4\cdot 10^{-15}\f$ with the
/// step off). So this must **not** be used as a CG preconditioner. It is used as
/// a LOBPCG preconditioner (GaugeEigen, SunGauge), which only needs search
/// directions, not an SPD linear \f$T\f$ -- note that this is weaker than the
/// hypothesis of Knyazev's convergence theory.
///
/// Reversing the colour order would fix (1) (measured: asymmetry to
/// \f$10^{-16}\f$) but buys nothing -- V-cycle counts are unchanged to slightly
/// worse and LOBPCG iteration counts are bit-identical, because the dual cell
/// graph is bipartite, so red-then-black and black-then-red are the same
/// two-block Gauss-Seidel and share a smoothing factor. Fixing (2) requires
/// `alphaStep = false`, which `tools/ablation_bench` shows **diverges** on the
/// flux operator. The non-symmetry is intrinsic to why the method converges.
///
/// \param lat  The connection lattice (the finest level).
/// \param b    Right-hand side.
/// \param x in/out: initial guess on entry (zero or a warm start), solution out.
/// \param opts V-cycle controls (max cycles, tolerance, smoothing).
/// \returns the V-cycle count and final relative residual.
MgResult vcycleSolve(const GaugeLattice& lat, const std::vector<std::complex<double>>& b,
                     std::vector<std::complex<double>>& x, const MgOptions& opts = {});

/// \brief Build the V-cycle level pyramid for \p lat once (coarsen while every
/// dimension stays coarsenable: open needs each dim even and >= 4; periodic needs
/// each divisible by 4 so the coarsened dim stays even). `levels.front()` is the
/// finest (== \p lat). Reuse the result across many \ref vcycleSolve calls -- the
/// eigensolver preconditions with one V-cycle per covMG-LOBPCG step -- to avoid
/// rebuilding the hierarchy (and recomputing every coarse level's transports) on
/// each solve.
/// \brief Ablation switches for the COARSENING rule (benchmarking only).
///
/// Two of the three transfer ingredients the paper advertises live here, and
/// both have a named 1990s counterfactual, so both need to be reachable to be
/// ablated rather than merely asserted.
struct CoarsenOptions {
  /// Coarse link along an axis. \c PathProduct is the ordered product of the
  /// two fine links the coarse edge spans (Balaban block spin / PTMG, and
  /// this paper). \c AveragedReunitarized instead averages the transporters
  /// over the parallel paths across the block and projects the mean back onto
  /// the group -- the averaged-then-reunitarized coarse link of the same era.
  enum class Link { PathProduct, AveragedReunitarized } link = Link::PathProduct;
  /// Coarse edge weight from the two fine weights. \c Series is the
  /// series-conductance rule w_a w_b / (2(w_a+w_b)) (which preserves the 1D
  /// identity exactly); \c Arithmetic is the (w_a+w_b)/4 mean.
  /// \c ArithmeticMatched is (w_a+w_b)/8, which AGREES with the series rule
  /// when the two fine weights are equal. That matters: the raw (w_a+w_b)/4
  /// mean differs from the series rule by a constant factor 2 on any
  /// near-uniform field, and a uniform rescaling of the coarse operator is
  /// exactly what the line search absorbs -- so comparing against it tests
  /// alpha's scale invariance, not the weight rule. Use the matched variant to
  /// isolate the rule's SHAPE, and weights whose ADJACENT values differ.
  enum class Weight { Series, Arithmetic, ArithmeticMatched } weight = Weight::Series;
};

std::vector<GaugeLattice> buildGaugeLevels(const GaugeLattice& lat,
                                           const CoarsenOptions& copts = {});

/// \brief V-cycle solve reusing a prebuilt \p levels pyramid (see \ref
/// buildGaugeLevels) -- identical to the single-lattice \ref vcycleSolve but
/// skips reconstructing the hierarchy per call.
///
/// \param levels prebuilt hierarchy (finest first, \ref buildGaugeLevels)
/// \param b      right-hand side on the finest level
/// \param x      iterate (in: initial guess, out: solution)
/// \param opts   cycle options (\ref MgOptions)
///
/// \param fineSmoothOverride EXPERIMENTAL (hierarchy-staleness probe; default
/// nullptr = bit-identical to the historical behavior, the project's hard
/// convention). When non-null, the FINEST level's smoother, its residual
/// computation, and its A-energy alpha step apply THIS operator instead of
/// `levels[0]`, while the level-0 -> level-1 transfer transports/weights and
/// every coarse level (operators, transfers) still come from \p levels. This
/// is the seam for measuring how a FROZEN hierarchy (captured on an earlier
/// frame's lattice) stales as the fine operator advances: stale P/R and stale
/// coarse operators, current fine operator. Must have the same dimensions as
/// `levels[0]`.
MgResult vcycleSolve(const std::vector<GaugeLattice>& levels,
                     const std::vector<std::complex<double>>& b,
                     std::vector<std::complex<double>>& x, const MgOptions& opts = {},
                     const GaugeLattice* fineSmoothOverride = nullptr);

/// \name Interleaved <-> complex conversion (bridge to traceZeroSet / eigensolver)
/// \{
std::vector<std::complex<double>> toComplex(const std::vector<double>& interleaved);
std::vector<double> toInterleaved(const std::vector<std::complex<double>>& z);
/// \}

}  // namespace bochner
