/// \file
/// Headless per-frame profiler for the FLOW-PAST-OBSTACLE demo (the paper's
/// Section "demonstration" figure): the same numerical loop as the obstacle
/// viewer (tools/viewer/obstacle_main.cpp), stripped of GL/imgui/smoke, so the
/// demo numbers are reproducible without a display.
///
/// Setup mirrors the viewer defaults: fixed tank 6 x 2 x 1.2 (long x = flow),
/// res cells across the height (res=32 -> 96 x 32 x 19 cells), SPHERE obstacle
/// (radius 0.35) at the upstream third with a half-cell y-offset (symmetry
/// breaker), inflow U = 1, explicit viscous substeps nu = 0.008 (the no-slip
/// vorticity source), CFL-adaptive dt (target 0.6, cap 0.1). Extraction every
/// frame: theta = u h / hbar (hbar = 0.1), warm-started covMG-LOBPCG (absolute-drop
/// early exit -- the live policy), zero-set trace + filament linking.
///
/// Per frame it times the stages
///   sim:      cfl scan | BFECC covector advect | viscous diffuse | MGPCG project
///   extract:  connection build | eigensolve | trace+link
/// and reports the median [min..max] over the last `window` frames (the flow
/// evolves, so this is a sequence statistic, not repeated identical solves).
/// At the last frame it also runs the OTHER eigensolvers on the same operator:
/// CPU double covMG-LOBPCG and SLEPc Lanczos (both warm-started where supported), and
/// reports wall time, eigenvalue agreement, and the chordal projective distance
/// between the eigenvector lines (the A/B check that the float GPU path traces
/// the same filaments).
///
/// Usage: obstacle_profile [res] [frames] [mode]
///   res    = cells across the tank height  (default 32)
///   frames = frames to run                 (default 200; stats over last 50)
///   mode   = gpu | cpu | ablate | gpuablate | gpucheb
///            (default gpu when Metal is built; ablate/gpuablate/gpucheb run
///            the base cpu/gpu loop plus extra warm ablation chains)
/// CPU thread count comes from OMP_NUM_THREADS.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <slepc.h>

#include "BenchTiming.h"
#include "solvers/EigenSolver.h"
#include "solvers/GaugeEigen.h"
#include "solvers/GaugeMultigrid.h"
#include "fluid/MacAdvection.h"
#include "fluid/MacExtrapolate.h"
#ifdef BOCHNER_WITH_METAL
#include "gpu/MetalContext.h"
#endif
#include "extraction/MacConnectionLaplacian.h"
#include "extraction/MacFilaments.h"
#include "grid/GridOperators.h"
#include "grid/MacGrid.h"
#include "grid/MacObstacle.h"
#include "fluid/MacProjection.h"

using namespace bochner;
using cd = std::complex<double>;

namespace {

double nowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct StageStat {
  std::vector<double> v;
  void add(double ms) { v.push_back(ms); }
  double median() const {
    std::vector<double> t = v;
    std::sort(t.begin(), t.end());
    return t.empty() ? 0.0 : (t.size() % 2 ? t[t.size() / 2]
                                           : 0.5 * (t[t.size() / 2 - 1] + t[t.size() / 2]));
  }
  double mn() const { return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); }
  double mx() const { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }
};

void printStat(const char* name, const StageStat& s) {
  std::printf("  %-10s %8.2f ms   [%7.2f .. %7.2f]\n", name, s.median(), s.mn(), s.mx());
}

double maxSpeed(const MacGrid& g, const FaceField& u) {
  double m = 0.0;
  for (int i = 0; i < g.nx(); ++i)
    for (int j = 0; j < g.ny(); ++j)
      for (int k = 0; k < g.nz(); ++k) {
        const double ux = 0.5 * (u.x[g.faceXIndex(i, j, k)] + u.x[g.faceXIndex(i + 1, j, k)]);
        const double uy = 0.5 * (u.y[g.faceYIndex(i, j, k)] + u.y[g.faceYIndex(i, j + 1, k)]);
        const double uz = 0.5 * (u.z[g.faceZIndex(i, j, k)] + u.z[g.faceZIndex(i, j, k + 1)]);
        m = std::max(m, std::sqrt(ux * ux + uy * uy + uz * uz));
      }
  return m;
}

// 2026-08-04 revision experiments: dump a frozen GaugeLattice (the final
// frame's wake operator) to a binary file, so flux_alias_diag's `mechanismfile`
// mode can run the Table-13 mechanism diagnostic on the REAL wake operator
// (revision-plan §5.2). Format: "GLAT" magic, version, lx/ly/lz/periodic,
// uniform w, then the three forward-link-angle arrays. The wake lattice is
// unweighted (gaugeLatticeFromFaces: uniform w = 1/h^2), asserted here.
bool dumpGaugeLattice(const GaugeLattice& L, const char* path) {
  if (L.weighted()) {
    std::fprintf(stderr, "dumpGaugeLattice: weighted lattice unsupported\n");
    return false;
  }
  std::FILE* fp = std::fopen(path, "wb");
  if (!fp) return false;
  const char magic[4] = {'G', 'L', 'A', 'T'};
  const int version = 1;
  const int per = L.periodic ? 1 : 0;
  std::fwrite(magic, 1, 4, fp);
  std::fwrite(&version, sizeof(int), 1, fp);
  std::fwrite(&L.lx, sizeof(int), 1, fp);
  std::fwrite(&L.ly, sizeof(int), 1, fp);
  std::fwrite(&L.lz, sizeof(int), 1, fp);
  std::fwrite(&per, sizeof(int), 1, fp);
  std::fwrite(&L.w, sizeof(double), 1, fp);
  const auto arr = [&](const std::vector<double>& a) {
    const std::uint64_t sz = a.size();
    std::fwrite(&sz, sizeof(std::uint64_t), 1, fp);
    std::fwrite(a.data(), sizeof(double), a.size(), fp);
  };
  arr(L.lkx);
  arr(L.lky);
  arr(L.lkz);
  std::fclose(fp);
  return true;
}

// Chordal projective distance between the complex lines spanned by two unit
// eigenvectors: sqrt(1 - |<a,b>|^2) (phase-invariant; 0 = same filaments).
double lineDistance(const std::vector<cd>& a, const std::vector<cd>& b) {
  cd ip(0, 0);
  double na = 0, nb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ip += std::conj(a[i]) * b[i];
    na += std::norm(a[i]);
    nb += std::norm(b[i]);
  }
  const double ov = std::min(1.0, std::abs(ip) / std::sqrt(std::max(1e-300, na * nb)));
  return std::sqrt(std::max(0.0, 1.0 - ov * ov));
}

}  // namespace

// Rayleigh quotient of a (unit) vector against the lattice operator -- used to
// check the two ablation chains converge to the SAME mode, not merely in the
// same number of steps.
double rayleighOf(const GaugeLattice& lat, const std::vector<cd>& v) {
  if (v.empty()) return 0.0;
  const std::vector<cd> Lv = applyConnectionLaplacian(lat, v);
  cd num(0, 0);
  double den = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    num += std::conj(v[i]) * Lv[i];
    den += std::norm(v[i]);
  }
  return den > 0 ? num.real() / den : 0.0;
}

int main(int argc, char** argv) {
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
  {
    const int res = (argc > 1) ? std::atoi(argv[1]) : 32;
    const int frames = (argc > 2) ? std::atoi(argv[2]) : 200;
    std::string mode = (argc > 3) ? argv[3] : "";
#ifdef BOCHNER_WITH_METAL
    const bool haveGpu = gpu::metalAvailable();
#else
    const bool haveGpu = false;
#endif
    if (mode.empty()) mode = haveGpu ? "gpu" : "cpu";
    // `ablate` = cpu run that additionally carries the plain-P warm-start chain.
    const bool ablate = (mode == "ablate");
    if (ablate) mode = "cpu";
    // `gpuablate` = gpu run that additionally carries a single-level (GS-only)
    // GPU warm-start chain: the "port plain Gauss-Seidel to the GPU" baseline.
    // With one uploaded level the device V-cycle degenerates to its
    // coarsest-level branch (encodeGaugeVcycle at L=1): coarseSweeps GS sweeps
    // at full resolution, the GPU analogue of the CPU ablation's 1-level chain.
    const bool gpuAblate = (mode == "gpuablate");
    if (gpuAblate) mode = "gpu";
    // `gpucheb` = gpu run that additionally carries Chebyshev-preconditioned
    // GPU chain(s) (lobpcgSolveGaugeCheb): the zero-setup polynomial baseline
    // ported to the device -- §6.2's one port-promotable candidate. Each chain
    // uploads the FINEST level only (no hierarchy is built or uploaded; the
    // apply is matvecs alone) and warm-starts from its OWN previous frame.
    // BOCHNER_GPU_CHEB = comma-separated deg:ratio list (default 64:1000, the
    // wake sweep winner of the survey campaign).
    const bool gpuCheb = (mode == "gpucheb");
    if (gpuCheb) mode = "gpu";
    const bool useGpu = (mode == "gpu");
    if (useGpu && !haveGpu) {
      std::fprintf(stderr, "error: mode=gpu but Metal is unavailable\n");
      return 2;
    }
    // 2026-08 hierarchy-staleness probe (reviewer question: "why not amortize
    // the hierarchy setup over frames?"). BOCHNER_FREEZE_HIER=<k0>: from frame
    // k0 (0-based) on, the CPU eigensolver's PRECONDITIONER reuses the MG
    // hierarchy captured from frame k0's lattice -- stale intergrid transfers
    // and stale coarse-level operators -- while the fine-level operator
    // (outer residuals, Rayleigh quotients, and the V-cycle's finest-level
    // smoother) stays current (see GaugeEigenOptions::frozenPrecLevels).
    // Default unset = bit-identical to the published behavior, the project's
    // hard convention. BOCHNER_ITER_LOG=1 prints one machine-readable line per
    // frame (frame, outer iterations, residual, converged, eigenvalue) for the
    // staleness curve; also default-off.
    int freezeFrame = -1;
    if (const char* e = std::getenv("BOCHNER_FREEZE_HIER")) freezeFrame = std::atoi(e);
    if (freezeFrame >= 0 && useGpu) {
      // The GPU solver consumes the uploaded hierarchy for the fine operator
      // AND the preconditioner; freezing there would stale the residuals too,
      // which is not the seam this probe measures. CPU only.
      std::fprintf(stderr, "error: BOCHNER_FREEZE_HIER requires mode=cpu\n");
      return 2;
    }
    const bool iterLog = std::getenv("BOCHNER_ITER_LOG") != nullptr;
    std::vector<GaugeLattice> frozenLevels;  // captured at frame freezeFrame
    const int window = std::min(50, std::max(1, frames / 4));

    // --- viewer-default setup (sphere; see file header) ---
    const float Lx = 6.0f, Ly = 2.0f, Lz = 1.2f;
    // 2026-08-04 revision experiments: BOCHNER_R / BOCHNER_HBAR override the
    // obstacle radius and the circulation quantum, so the profiler can measure
    // Figure 1's capture configuration (r=0.357, hbar=0.061) and not only the
    // profiler defaults. Unset = 0.35 / 0.1, bit-identical to the published
    // behavior. Both are echoed in the header line below, so logs are
    // self-describing.
    const char* rEnv = std::getenv("BOCHNER_R");
    const float radius = rEnv ? std::atof(rEnv) : 0.35f;
    // BOCHNER_U / BOCHNER_NU sweep the Reynolds number Re = U*(2*radius)/nu,
    // for checking the sphere-wake regime transitions (steady axisymmetric ->
    // steady planar-symmetric ~210 -> unsteady loop shedding ~270).
    const char* uEnv = std::getenv("BOCHNER_U");
    const char* nuEnv = std::getenv("BOCHNER_NU");
    const float U = uEnv ? std::atof(uEnv) : 1.0f;
    const float nu = nuEnv ? std::atof(nuEnv) : 0.008f;
    const char* hbarEnv = std::getenv("BOCHNER_HBAR");
    const double hbar = hbarEnv ? std::atof(hbarEnv) : 0.1;
    const double cflTarget = 0.6;
    const double h = double(Ly) / res;
    const int nx = std::max(12, int(std::lround(Lx / h)));
    const int ny = res;
    const int nz = std::max(4, int(std::lround(Lz / h)));
    const MacGrid grid(nx, ny, nz, h, Vec3{-Lx / 2, -Ly / 2, -Lz / 2});
    // Centred exactly. The half-cell y-offset this used to carry is no longer
    // needed to seed shedding (the cylinder sheds 20-24 filaments centred) and
    // was wrong for the sphere, whose sub-critical wake is axisymmetric.
    // BOCHNER_YOFFSET=1 restores it for comparison.
    const bool yoff = std::getenv("BOCHNER_YOFFSET") != nullptr;
    const Vec3 center{-Lx * 0.25, yoff ? 0.5 * h : 0.0, 0.0};
    // BOCHNER_OBSTACLE=cylinder selects the shedding geometry. The sphere's
    // shedding onset is Re ~ 270, so at this demo's Re ~ 87 a steady
    // axisymmetric wake is correct physics and the wake treatment cannot be
    // assessed on it; a cylinder sheds from Re ~ 47.
    const char* obsEnv = std::getenv("BOCHNER_OBSTACLE");
    const bool useCylinder = obsEnv && std::string(obsEnv) == "cylinder";
    const SolidMask solid = useCylinder
        ? cylinderMask(grid, center, Vec3{0, 0, 1}, radius)
        : sphereMask(grid, center, radius);
    int solidCells = 0;
    for (auto s : solid) solidCells += s;

    MacProjector proj(grid, PoissonMgOptions{}, BoundarySpec::channelFlow(U), solid);
    proj.setGpuProjection(useGpu);
    FaceField u = ops::zeroFaceField(grid);
    for (double& v : u.x) v = U;
    int cycles = 0;
    u = proj.project(u, &cycles);

    std::printf("obstacle profile: %dx%dx%d = %d cells (%d solid), %s r=%.3f, U=%g,"
                " nu=%g, hbar=%g\n",
                nx, ny, nz, grid.numCells(), solidCells, useCylinder ? "cylinder" : "sphere",
                radius, U, nu, hbar);
    std::printf("Re = U*D/nu = %.0f\n", U * 2.0 * radius / nu);
    std::printf("mode=%s  OMP_NUM_THREADS=%s  frames=%d (stats over last %d)\n\n", mode.c_str(),
                std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "(unset)",
                frames, window);

    StageStat sCfl, sAdv, sDif, sPrj, sSim, sConn, sEig, sTrc, sExt;
    StageStat sGpuBuild, sGpuUpload, sGpuSolve;  // GPU eigensolve row, split
    std::vector<cd> prevVec;
    std::size_t nFilaments = 0;
    GaugeLattice lastLat;  // last frame's operator, for the A/B block below
    std::vector<cd> lastWarmGuess;  // guess the loop actually used at the final frame
    std::vector<cd> lastVec;
    int lastIts = 0;
    // Transfer ablation on the LIVE demo operator (does the flagship
    // application exercise the flagship ingredient?). Second warm-start
    // chain, preconditioner transports stripped, same frames.
    std::vector<cd> prevVecPlain, prevVecCov, prevVecSmooth;
    long itsSmoothSum = 0, itsSmoothMax = 0;
    double msCovSum = 0, msPlainSum = 0, msSmoothSum = 0;
    // BOCHNER_ABL_LIVE=1 keeps the live absolute-drop policy; default runs the
    // certified policy, under which the preconditioned direction is not dropped.
    const bool liveAbl = std::getenv("BOCHNER_ABL_LIVE") != nullptr;
    const bool probe = std::getenv("BOCHNER_WAKE_PROBE") != nullptr;
    double simTime = 0.0, transKE = 0.0;
    long itsCovSum = 0, itsPlainSum = 0, itsCovMax = 0, itsPlainMax = 0, ablFrames = 0;
    int plainFailures = 0;
    // GPU single-level baseline chain (mode gpuablate).
    std::vector<cd> prevVecGpu1;
    long itsGpuMgSum = 0, itsGpuMgMax = 0, itsGpu1Sum = 0, itsGpu1Max = 0, gablFrames = 0;
    double msGpuMgSum = 0, msGpu1Sum = 0;
    int gpu1Failures = 0;
    // GPU Chebyshev chains (mode gpucheb), one warm chain per deg:ratio config.
    struct GpuChebChain {
      int deg = 64;
      double ratio = 1000.0;
      bool hybrid = false;  // E3: full hierarchy, Chebyshev SMOOTHER (deg:ratio
                            // are then the smoother's, not the preconditioner's)
      std::vector<cd> prev;
      long itsSum = 0, itsMax = 0;
      double msSum = 0;
      double resSum = 0, resMax = 0;  // double-recomputed exit eigen-residuals
    };
    std::vector<GpuChebChain> chebChains;
    if (gpuCheb) {
      const auto parseInto = [&](const char* env, const char* dflt, bool hybrid) {
        const char* e = std::getenv(env);
        std::string s = e ? e : (dflt ? dflt : "");
        for (std::size_t p = 0; p < s.size();) {
          std::size_t q = s.find(',', p);
          if (q == std::string::npos) q = s.size();
          const std::string tok = s.substr(p, q - p);
          p = q + 1;
          const std::size_t colon = tok.find(':');
          if (colon == std::string::npos) continue;
          GpuChebChain c;
          c.deg = std::atoi(tok.substr(0, colon).c_str());
          c.ratio = std::atof(tok.substr(colon + 1).c_str());
          c.hybrid = hybrid;
          if (c.deg > 0 && c.ratio >= 1.0) chebChains.push_back(std::move(c));
        }
      };
      parseInto("BOCHNER_GPU_CHEB", "64:1000", false);
      // E3 hybrid chains: BOCHNER_GPU_HYBRID = smootherDeg:smootherRatio list
      // (no default -- opt-in).
      parseInto("BOCHNER_GPU_HYBRID", nullptr, true);
    }
    long gchebFrames = 0;
    double resGpuMgSum = 0, resGpuMgMax = 0;  // main chain's exit residuals, same window

    for (int f = 0; f < frames; ++f) {
      const bool inWindow = f >= frames - window;
      // -- sim --
      double t0 = nowMs();
      const double Umax = std::max(1e-6, maxSpeed(grid, u));
      const double dtEff = std::min(0.1, cflTarget * h / Umax);
      // Give the solid band a plausible velocity before backtracing into it.
      // Without this, faces one layer outside the body sample the identically
      // zero interior and a sheet of dead fluid is injected every step.
      // BOCHNER_NO_EXTRAP=1 restores the old behaviour for A/B measurement.
      static const bool kNoExtrap = std::getenv("BOCHNER_NO_EXTRAP") != nullptr;
      if (!kNoExtrap) {
        const int band = static_cast<int>(std::ceil(cflTarget)) + 2;
        extrapolateIntoSolid(grid, u, solid, band);
      }
      double t1 = nowMs();
      FaceField w;
#ifdef BOCHNER_WITH_METAL
      if (useGpu)
        w = advectCovectorBFECCGpu(grid, u, u, dtEff);
      else
#endif
        w = advectCovectorBFECC(grid, u, u, dtEff);
      double t2 = nowMs();
      const double lim = 0.16 * h * h;
      const int nsub = std::max(1, int(std::ceil(nu * dtEff / lim)));
      for (int s = 0; s < nsub; ++s) w = ops::diffuseVelocity(grid, w, nu, dtEff / nsub, solid);
      double t3 = nowMs();
      u = proj.project(w, &cycles);
      double t4 = nowMs();

      simTime += dtEff;
      // -- extraction (every frame, as in the GPU viewer) --
      const FaceField theta = connectionAngles(grid, u, hbar);
      const GaugeLattice lat = gaugeLatticeFromFaces(grid, theta);
      // Capture the warm guess the loop is about to use at the final frame
      // (the previous frame's solution, i.e. the realistic per-frame warm
      // start), before this frame's solve overwrites prevVec. The A/B below
      // re-runs both solvers from this same guess for a representative,
      // criterion-matched per-frame comparison.
      if (f == frames - 1) lastWarmGuess = prevVec;
      double t5 = nowMs();
      std::vector<cd> vec;
      int its = 0;
      if (useGpu) {
#ifdef BOCHNER_WITH_METAL
        const double tb0 = nowMs();
        std::vector<gpu::GaugeLevelData> levels;
        for (const GaugeLattice& L : buildGaugeLevels(lat))
          levels.push_back({L.lx, L.ly, L.lz, L.periodic, L.w, L.tx, L.ty, L.tz});
        const double tb1 = nowMs();
        const int handle = gpu::uploadGauge(levels);
        const double tb2 = nowMs();
        GaugeEigenOptions opts;
        opts.relativeGsDrop = false;  // live path: absolute-drop warm-start early exit
        const gpu::GaugeEigenGpu r =
            gpu::lobpcgSolveGauge(handle, prevVec, /*maxIters=*/100, /*tol=*/1e-4, opts.precCycles,
                                opts.mg.nu1, opts.mg.nu2, opts.mg.coarseSweeps, opts.mg.omega);
        const double tb3 = nowMs();
        if (inWindow) {
          sGpuBuild.add(tb1 - tb0);
          sGpuUpload.add(tb2 - tb1);
          sGpuSolve.add(tb3 - tb2);
        }
        gpu::freeGauge(handle);
        vec = r.vector;
        its = r.iterations;
#endif
      } else {
        GaugeEigenOptions opts;
        opts.relativeGsDrop = false;  // live path (same policy as the viewer's CPU solver)
        opts.tol = 1e-6;
        // Staleness probe: capture the hierarchy ONCE at frame freezeFrame,
        // then precondition every later frame's solve with it (stale transfers
        // + coarse operators; fine operator current -- see above).
        if (freezeFrame >= 0 && f >= freezeFrame) {
          if (frozenLevels.empty()) frozenLevels = buildGaugeLevels(lat);
          opts.frozenPrecLevels = &frozenLevels;
        }
        const GaugeEigenResult r =
            smallestEigenpairGaugeMG(lat, prevVec.empty() ? nullptr : &prevVec, opts);
        vec = r.vector;
        its = r.iterations;
        if (iterLog)
          std::printf("ITER f=%d its=%d res=%.3e conv=%d lam=%.9f frozen=%d\n", f, r.iterations,
                      r.residual, r.converged ? 1 : 0, r.eigenvalue,
                      opts.frozenPrecLevels ? 1 : 0);
      }
      if (!vec.empty()) prevVec = vec;
      double t6 = nowMs();
      const std::vector<double> psi = toInterleaved(vec);
      nFilaments = 0;
      for (auto& fl : linkFilaments(grid, traceZeroSet(grid, psi)))
        if (fl.points.size() >= 4) ++nFilaments;
      double t7 = nowMs();

      if (inWindow) {
        sCfl.add(t1 - t0);
        sAdv.add(t2 - t1);
        sDif.add(t3 - t2);
        sPrj.add(t4 - t3);
        sSim.add(t4 - t0);
        sConn.add(t5 - t4);
        sEig.add(t6 - t5);
        sTrc.add(t7 - t6);
        sExt.add(t7 - t4);
      }
      {
        // Transverse (y) kinetic energy in a downstream slab, normalised by the
        // inflow. Axisymmetric steady wake -> flat; shedding -> oscillates.
        // Diagnostic only (the viewer performs no such pass), so it runs
        // outside the t0..t7 stage brackets and is charged to no table row.
        double ty = 0.0;
        long cnt = 0;
        const int i0 = grid.nx() / 2;  // downstream half
        for (int i = i0; i < grid.nx(); ++i)
          for (int j = 0; j <= grid.ny(); ++j)
            for (int k = 0; k < grid.nz(); ++k) {
              const double vy = u.y[static_cast<std::size_t>(grid.faceYIndex(i, j, k))];
              ty += vy * vy;
              ++cnt;
            }
        transKE = cnt ? std::sqrt(ty / cnt) : 0.0;
      }
      if (probe && (f % 10 == 0 || f == frames - 1))
        std::printf("  [wake] frame %4d  t=%7.3f  rms(u_y)=%.6e  eig its %d\n", f, simTime,
                    transKE, its);
      // Ablation chains run OUTSIDE the t5..t7 stage timers, so the stage
      // table stays clean in ablate/gpuablate modes.
      if (gpuAblate && useGpu) {
#ifdef BOCHNER_WITH_METAL
        // Second GPU chain, hierarchy truncated to the finest level only,
        // warm-started from its OWN previous frame (a faithful independent
        // chain). Charged its own upload, as any per-frame GS-only port would
        // pay it too. Same live stopping policy as the main GPU chain; higher
        // iteration cap since plain smoothing needs more outer steps.
        const double ta0 = nowMs();
        const auto lv1 = buildGaugeLevels(lat);
        std::vector<gpu::GaugeLevelData> lone;
        lone.push_back({lv1[0].lx, lv1[0].ly, lv1[0].lz, lv1[0].periodic, lv1[0].w, lv1[0].tx,
                        lv1[0].ty, lv1[0].tz});
        const int h1 = gpu::uploadGauge(lone);
        GaugeEigenOptions o1;
        o1.relativeGsDrop = false;
        const gpu::GaugeEigenGpu r1 =
            gpu::lobpcgSolveGauge(h1, prevVecGpu1, /*maxIters=*/300, /*tol=*/1e-4, o1.precCycles,
                                  o1.mg.nu1, o1.mg.nu2, o1.mg.coarseSweeps, o1.mg.omega);
        gpu::freeGauge(h1);
        const double ta1 = nowMs();
        if (!r1.vector.empty()) prevVecGpu1 = r1.vector;
        if (!r1.converged) ++gpu1Failures;
        if (inWindow) {
          itsGpuMgSum += its;
          itsGpuMgMax = std::max<long>(itsGpuMgMax, its);
          msGpuMgSum += t6 - t5;
          itsGpu1Sum += r1.iterations;
          itsGpu1Max = std::max<long>(itsGpu1Max, r1.iterations);
          msGpu1Sum += ta1 - ta0;
          ++gablFrames;
        }
        if (f < 8 || (f % 25) == 0) {
          // Double-precision relative eigen-residual of a returned float
          // vector: ||Lv - rho v|| / rho, the acceptance quantity.
          const auto resOf = [&](const std::vector<cd>& v) {
            if (v.empty()) return -1.0;
            const double rho = rayleighOf(lat, v);
            const std::vector<cd> Lv = applyConnectionLaplacian(lat, v);
            double rr = 0, nn = 0;
            for (std::size_t i = 0; i < v.size(); ++i) {
              rr += std::norm(Lv[i] - rho * v[i]);
              nn += std::norm(v[i]);
            }
            return std::sqrt(rr / std::max(1e-300, nn)) / std::max(1e-300, rho);
          };
          const double lamMg = rayleighOf(lat, vec);
          const double lam1 = rayleighOf(lat, prevVecGpu1);
          std::printf("  [gpuabl] frame %3d  GPU MG %3d it/%6.2f ms (res %.1e)  "
                      "GPU 1-level %3d it/%7.2f ms (res %.1e)  dlam %.1e%s\n",
                      f, its, t6 - t5, resOf(vec), r1.iterations, ta1 - ta0,
                      resOf(prevVecGpu1),
                      std::abs(lamMg - lam1) / std::max(1e-30, std::abs(lamMg)),
                      r1.converged ? "" : "  (1-level below acceptance)");
          std::fflush(stdout);
        }
#endif
      }
      if (gpuCheb && useGpu && !chebChains.empty()) {
#ifdef BOCHNER_WITH_METAL
        if (f == 0) {
          // The covMG-vs-cheb comparison is only meaningful if the hierarchy is
          // real: nz = round(0.6*res) need not stay divisible (res 48 -> nz 29,
          // no coarse level at all -- the "covMG" chain would silently be
          // single-level GS). Disclose the depth so a shallow run can't pass
          // as a multigrid data point.
          std::printf("  [gpucheb] lattice %dx%dx%d, gauge hierarchy levels = %zu\n", lat.lx,
                      lat.ly, lat.lz, buildGaugeLevels(lat).size());
          std::fflush(stdout);
        }
        // Chebyshev GPU chains: the finest level packed straight from this
        // frame's lattice -- no buildGaugeLevels, no coarse uploads (the apply
        // is matvecs alone) -- each chain charged its own upload and warm from
        // its OWN previous frame. Same live stopping policy as the main GPU
        // chain; higher outer cap (the polynomial takes more outer steps cold).
        const auto resOf = [&](const std::vector<cd>& v) {
          if (v.empty()) return -1.0;
          const double rho = rayleighOf(lat, v);
          const std::vector<cd> Lv = applyConnectionLaplacian(lat, v);
          double rr = 0, nn = 0;
          for (std::size_t i = 0; i < v.size(); ++i) {
            rr += std::norm(Lv[i] - rho * v[i]);
            nn += std::norm(v[i]);
          }
          return std::sqrt(rr / std::max(1e-300, nn)) / std::max(1e-300, rho);
        };
        const bool sample = f < 8 || (f % 25) == 0;
        if (sample)
          std::printf("  [gpucheb] frame %3d  GPU MG %3d it/%6.2f ms (res %.1e)", f, its, t6 - t5,
                      resOf(vec));
        const double lamMg = rayleighOf(lat, vec);
        for (GpuChebChain& c : chebChains) {
          const double tc0 = nowMs();
          std::vector<gpu::GaugeLevelData> levs;
          if (c.hybrid) {  // full hierarchy (its build is charged, like the main chain)
            for (const GaugeLattice& Lv : buildGaugeLevels(lat))
              levs.push_back({Lv.lx, Lv.ly, Lv.lz, Lv.periodic, Lv.w, Lv.tx, Lv.ty, Lv.tz});
          } else {  // polynomial preconditioner: finest level only
            levs.push_back({lat.lx, lat.ly, lat.lz, lat.periodic, lat.w, lat.tx, lat.ty, lat.tz});
          }
          const int hc = gpu::uploadGauge(levs);
          const GaugeEigenOptions defo;  // precCycles/coarseSweeps defaults of the main chain
          const gpu::GaugeEigenGpu rc =
              c.hybrid ? gpu::lobpcgSolveGaugeHybrid(hc, c.prev, /*maxIters=*/300, /*tol=*/1e-4,
                                                     defo.precCycles, defo.mg.coarseSweeps, c.deg,
                                                     c.ratio)
                       : gpu::lobpcgSolveGaugeCheb(hc, c.prev, /*maxIters=*/300, /*tol=*/1e-4,
                                                   c.deg, c.ratio);
          gpu::freeGauge(hc);
          const double tc1 = nowMs();
          if (!rc.vector.empty()) c.prev = rc.vector;
          if (inWindow) {
            c.itsSum += rc.iterations;
            c.itsMax = std::max<long>(c.itsMax, rc.iterations);
            c.msSum += tc1 - tc0;
            // Both chains' acceptance quantity, symmetrically: the relative
            // eigen-residual recomputed in double (outside every stage timer).
            const double rres = resOf(c.prev);
            c.resSum += rres;
            c.resMax = std::max(c.resMax, rres);
          }
          if (sample) {
            const double lamC = rayleighOf(lat, c.prev);
            std::printf("  %s %d:%.0f %3d it/%7.2f ms (res %.1e, dlam %.1e)",
                        c.hybrid ? "hyb" : "cheb", c.deg, c.ratio,
                        rc.iterations, tc1 - tc0, resOf(c.prev),
                        std::abs(lamMg - lamC) / std::max(1e-30, std::abs(lamMg)));
          }
        }
        if (sample) {
          std::printf("\n");
          std::fflush(stdout);
        }
        if (inWindow) {
          itsGpuMgSum += its;
          itsGpuMgMax = std::max<long>(itsGpuMgMax, its);
          msGpuMgSum += t6 - t5;
          const double rres = resOf(vec);
          resGpuMgSum += rres;
          resGpuMgMax = std::max(resGpuMgMax, rres);
          ++gchebFrames;
        }
#endif
      }
      if (ablate && !useGpu && f == 0) {
        // Isolation: does the covariant transfer differ AT ALL on this lattice?
        const auto lv = buildGaugeLevels(lat);
        std::printf("  [abl] ISOLATION: grid %dx%dx%d, HIERARCHY LEVELS = %zu\n", lat.lx, lat.ly,
                    lat.lz, lv.size());
        std::fflush(stdout);
        if (lv.size() < 2) {
          std::printf("  [abl] ONLY ONE LEVEL: there is no coarse grid, so the V-cycle is pure\n"
                      "        smoothing and the transfer is never applied. The ablation is\n"
                      "        vacuous on this operator.\n");
          std::fflush(stdout);
        }
        const long nfine = static_cast<long>(lat.lx) * lat.ly * lat.lz;
        const long ncrs = lv.size() > 1 ? static_cast<long>(lv[1].lx) * lv[1].ly * lv[1].lz : 0;
        if (lv.size() > 1) {
        std::vector<cd> cv(static_cast<std::size_t>(ncrs));
        for (std::size_t i = 0; i < cv.size(); ++i)
          cv[i] = cd(std::sin(0.7 * double(i)), std::cos(0.3 * double(i)));
        const auto pc = prolongGauge(lat, cv, true);
        const auto pp = prolongGauge(lat, cv, false);
        double dn = 0.0, pn = 0.0;
        for (std::size_t i = 0; i < pc.size(); ++i) {
          dn += std::norm(pc[i] - pp[i]);
          pn += std::norm(pc[i]);
        }
        double thmax = 0.0;
        for (double t : lat.lkx) thmax = std::max(thmax, std::abs(t));
        for (double t : lat.lky) thmax = std::max(thmax, std::abs(t));
        for (double t : lat.lkz) thmax = std::max(thmax, std::abs(t));
        std::printf("  [abl] ISOLATION: grid %dx%dx%d, levels=%zu, fine=%ld coarse=%ld,"
                    " max|theta|=%.4f, ||Pcov-Pplain||/||Pcov||=%.3e\n",
                    lat.lx, lat.ly, lat.lz, lv.size(), nfine, ncrs, thmax,
                    pn > 0 ? std::sqrt(dn / pn) : -1.0);
        std::fflush(stdout);
        }
      }
      // 2026-08-04 revision experiments: BOCHNER_ABL_START skips the ablation
      // chains until the given frame (needed to make the shedding-configuration
      // ablation affordable: the chains then run only over the developed-wake
      // window, each chain warm after its own first solve). Unset = 0 = the
      // published full-run behavior. BOCHNER_ABL_TOL overrides the certified
      // chain tolerance (unset = 1e-8, the published setting).
      static const int kAblStart = [] {
        const char* e = std::getenv("BOCHNER_ABL_START");
        return e ? std::atoi(e) : 0;
      }();
      static const double kAblTol = [] {
        const char* e = std::getenv("BOCHNER_ABL_TOL");
        return e ? std::atof(e) : 1e-8;
      }();
      if (ablate && !useGpu && f >= kAblStart) {
        auto solve = [&](std::vector<cd>& warm, bool cov, int lvl, double* ms) {
          GaugeEigenOptions o;
          o.relativeGsDrop = !liveAbl;
          o.tol = liveAbl ? 1e-6 : kAblTol;
          o.mg.covariantTransfer = cov;
          o.maxLevels = lvl;
          const double t0 = nowMs();
          const GaugeEigenResult r =
              smallestEigenpairGaugeMG(lat, warm.empty() ? nullptr : &warm, o);
          *ms = nowMs() - t0;
          if (!r.vector.empty()) warm = r.vector;
          return r;
        };
        double msC = 0, msP = 0, msS = 0;
        const GaugeEigenResult rc = solve(prevVecCov, true, 0, &msC);
        const GaugeEigenResult rs = solve(prevVecSmooth, true, 1, &msS);
        const int itsC = rc.iterations;
        const GaugeEigenResult rp = solve(prevVecPlain, false, 0, &msP);
        if (!rp.converged) ++plainFailures;
        if (inWindow) {
          itsCovSum += itsC;
          itsPlainSum += rp.iterations;
          itsCovMax = std::max<long>(itsCovMax, itsC);
          itsPlainMax = std::max<long>(itsPlainMax, rp.iterations);
          itsSmoothSum += rs.iterations;
          itsSmoothMax = std::max<long>(itsSmoothMax, rs.iterations);
          msCovSum += msC;
          msPlainSum += msP;
          msSmoothSum += msS;
          ++ablFrames;
        }
        // Per-frame trace: identical counts across two different
        // preconditioners would be an artifact, not a null result, so the
        // per-frame values and the eigenvalues both have to be visible.
        if (f < 8 || (f % 25) == 0) {
          const double lamCov = rayleighOf(lat, prevVecCov);
          const double lamPln = rayleighOf(lat, prevVecPlain);
          std::printf("  [abl] frame %3d  MGcov %3d it/%6.1fms  MGplain %3d it/%6.1fms  "
                      "1-level %3d it/%6.1fms   dlam %.1e\n",
                      f, itsC, msC, rp.iterations, msP, rs.iterations, msS,
                      std::abs(lamCov - lamPln) / std::max(1e-30, std::abs(lamCov)));
          std::fflush(stdout);
        }
      }
      if (f == frames - 1) {
        lastLat = lat;
        lastVec = vec;
        lastIts = its;
      }
      if ((f + 1) % 50 == 0) {
        std::printf("  frame %4d: sim %.1f ms, extract %.1f ms, %zu filaments, eig its %d\n",
                    f + 1, t4 - t0, t7 - t4, nFilaments, its);
        std::fflush(stdout);
      }
    }

    std::printf("\nper-frame stages, median [min..max] over the last %d frames:\n", window);
    printStat("cfl", sCfl);
    printStat("advect", sAdv);
    printStat("diffuse", sDif);
    printStat("project", sPrj);
    printStat("SIM", sSim);
    printStat("connection", sConn);
    printStat("eigensolve", sEig);
    printStat("trace+link", sTrc);
    printStat("EXTRACT", sExt);
    if (useGpu && sGpuSolve.median() > 0) {
      std::printf("  -- eigensolve row split (GPU) --\n");
      printStat("hier build", sGpuBuild);
      printStat("upload", sGpuUpload);
      printStat("lobpcg", sGpuSolve);
    }
    const double frameMs = sSim.median() + sExt.median();
    std::printf("  full loop  %8.2f ms  = %.1f fps (extraction every frame)\n", frameMs,
                1000.0 / frameMs);

    if (ablate && ablFrames > 0) {
      std::printf("\nTRANSFER ABLATION on the live demo operator (%ld frames in window):\n",
                  ablFrames);
      std::printf("  MG covariant : mean %6.1f it  max %3ld   mean %7.1f ms/frame\n",
                  double(itsCovSum) / ablFrames, itsCovMax, msCovSum / ablFrames);
      std::printf("  MG plain-P   : mean %6.1f it  max %3ld   mean %7.1f ms/frame\n",
                  double(itsPlainSum) / ablFrames, itsPlainMax, msPlainSum / ablFrames);
      std::printf("  1 level (GS) : mean %6.1f it  max %3ld   mean %7.1f ms/frame"
                  "   <- what 96x32x19 gets\n",
                  double(itsSmoothSum) / ablFrames, itsSmoothMax, msSmoothSum / ablFrames);
      std::printf("  hierarchy is worth %.2fx in iterations, %.2fx in time;"
                  " covariant vs plain %.2fx\n",
                  itsCovSum > 0 ? double(itsSmoothSum) / double(itsCovSum) : -1.0,
                  msCovSum > 0 ? msSmoothSum / msCovSum : -1.0,
                  itsCovSum > 0 ? double(itsPlainSum) / double(itsCovSum) : -1.0);
      std::printf("  Both chains warm-start from their OWN previous frame, so this is\n");
      std::printf("  the demo as a gauge-unaware implementation would have run it.\n");
    }

    if (gpuAblate && gablFrames > 0) {
      std::printf("\nGPU SINGLE-LEVEL (GS-only) BASELINE, both chains warm, last %ld frames:\n",
                  gablFrames);
      std::printf("  GPU MG (full hierarchy) : mean %5.1f it  max %3ld   mean %7.2f ms/frame"
                  " (build+upload+solve)\n",
                  double(itsGpuMgSum) / gablFrames, itsGpuMgMax, msGpuMgSum / gablFrames);
      std::printf("  GPU 1-level (%d GS)     : mean %5.1f it  max %3ld   mean %7.2f ms/frame"
                  " (build+upload+solve)\n",
                  GaugeEigenOptions{}.mg.coarseSweeps, double(itsGpu1Sum) / gablFrames,
                  itsGpu1Max, msGpu1Sum / gablFrames);
      if (gpu1Failures)
        std::printf("  1-level chain hit the stagnation test without the double-precision\n"
                    "  acceptance on %d frame(s) -- see per-frame dlam lines above.\n",
                    gpu1Failures);
      std::printf("  multigrid on the GPU is worth %.2fx in time, %.2fx in iterations\n",
                  msGpuMgSum > 0 ? msGpu1Sum / msGpuMgSum : -1.0,
                  itsGpuMgSum > 0 ? double(itsGpu1Sum) / double(itsGpuMgSum) : -1.0);
    }

    if (gpuCheb && gchebFrames > 0) {
      std::printf("\nGPU CHEBYSHEV-LOBPCG (zero-setup polynomial port), all chains warm,"
                  " last %ld frames:\n", gchebFrames);
      std::printf("  GPU covMG (full hierarchy) : mean %5.1f it  max %3ld   mean %7.2f ms/frame"
                  " (build+upload+solve)   exit res mean %.1e max %.1e\n",
                  double(itsGpuMgSum) / gchebFrames, itsGpuMgMax, msGpuMgSum / gchebFrames,
                  resGpuMgSum / gchebFrames, resGpuMgMax);
      for (const auto& c : chebChains) {
        std::printf("  GPU %s deg %2d ratio %5.0f: mean %5.1f it  max %3ld   mean %7.2f ms/frame"
                    " (%s)   exit res mean %.1e max %.1e\n",
                    c.hybrid ? "hybrid" : "cheb  ", c.deg, c.ratio,
                    double(c.itsSum) / gchebFrames, c.itsMax, c.msSum / gchebFrames,
                    c.hybrid ? "build+upload+solve, cheb-smoothed V-cycle"
                             : "upload+solve, finest level only",
                    c.resSum / gchebFrames, c.resMax);
        std::printf("    cheb/covMG time ratio %.2fx, iteration ratio %.2fx\n",
                    msGpuMgSum > 0 ? c.msSum / msGpuMgSum : -1.0,
                    itsGpuMgSum > 0 ? double(c.itsSum) / double(itsGpuMgSum) : -1.0);
      }
      std::printf("  (exit res = relative eigen-residual recomputed in double; both solvers stop\n"
                  "   on rho stagnation, so this is the symmetric acceptance quantity.)\n");
    }

    // --- A/B on the final operator: CPU double covMG-LOBPCG and SLEPc Lanczos ---
    // Criterion-matched at the certified relative eigen-residual tol used by the
    // paper's Tables 6-7 (relativeGsDrop=true), so the demonstration carries a
    // headline as rigorous as the rest of the paper. Run single-threaded
    // (OMP_NUM_THREADS=1) for the clean apples-to-apples number; the live-loop
    // absolute-drop covMG line is kept for reference against the demo table.
    // Median-of-BENCH_REPS wall times (the paper-wide statistic), so this
    // per-frame comparison is as rigorous as Tables 6-7. Both solvers start
    // from the same realistic warm guess (lastWarmGuess); repeats are
    // side-effect-free (the guess is const).
    // 2026-08-04 revision experiments: BOCHNER_AB_TOLS (comma-separated list)
    // runs the certified A/B at several tolerances in one process, so the
    // published 1e-6 control and the paper-wide certified 1e-7 (revision-plan
    // §6.5 "Cost" paragraph) are measured against the SAME warm guess and the
    // SAME frozen operator. Unset = "1e-6" = published behavior, bit-identical.
    std::vector<double> abTols;
    if (const char* e = std::getenv("BOCHNER_AB_TOLS")) {
      std::string s(e);
      for (std::size_t p = 0; p < s.size();) {
        std::size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        abTols.push_back(std::atof(s.substr(p, q - p).c_str()));
        p = q + 1;
      }
    }
    if (abTols.empty()) abTols.push_back(1e-6);
    if (const char* dumpPath = std::getenv("BOCHNER_DUMP_LAT")) {
      std::printf("\n[dump] final frame's GaugeLattice (%dx%dx%d, periodic=%d, w=%.6g) -> %s: %s\n",
                  lastLat.lx, lastLat.ly, lastLat.lz, lastLat.periodic ? 1 : 0, lastLat.w,
                  dumpPath, dumpGaugeLattice(lastLat, dumpPath) ? "ok" : "FAILED");
    }
    for (const double abTol : abTols) {
    std::printf("\nA/B on the final frame's operator (%zu filaments traced), tol %.0e"
                " matched, median of %d:\n", nFilaments, abTol, benchstat::reps());
    {
      GaugeEigenOptions opts;
      opts.relativeGsDrop = true;  // certified relative-residual drop (paper Tables 6-7)
      opts.tol = abTol;
      GaugeEigenResult r;
      const double ms = benchstat::medianMs([&] {
        r = smallestEigenpairGaugeMG(lastLat, lastWarmGuess.empty() ? nullptr : &lastWarmGuess, opts);
      });
      std::printf("  CPU double covMG-LOBPCG (warm, certified): %7.1f ms, %d its, lambda %.6f,"
                  " line dist to %s vec %.2e\n",
                  ms, r.iterations, r.eigenvalue, mode.c_str(), lineDistance(r.vector, lastVec));
    }
    // 2026-08-15 survey round: Chebyshev-preconditioned LOBPCG rows (identical
    // outer loop and warm guess, preconditioner swapped to the zero-setup
    // polynomial). BOCHNER_AB_CHEB = comma-separated deg:ratio list. Unset =
    // published behavior, bit-identical.
    if (const char* e = std::getenv("BOCHNER_AB_CHEB")) {
      std::string s(e);
      for (std::size_t p = 0; p < s.size();) {
        std::size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        const std::string tok = s.substr(p, q - p);
        p = q + 1;
        const std::size_t colon = tok.find(':');
        if (colon == std::string::npos) continue;
        GaugeEigenOptions opts;
        opts.relativeGsDrop = true;
        opts.tol = abTol;
        opts.chebDegree = std::atoi(tok.substr(0, colon).c_str());
        opts.chebRatio = std::atof(tok.substr(colon + 1).c_str());
        GaugeEigenResult r;
        const double ms = benchstat::medianMs([&] {
          r = smallestEigenpairGaugeMG(lastLat, lastWarmGuess.empty() ? nullptr : &lastWarmGuess,
                                       opts);
        });
        std::printf("  CPU double cheb-LOBPCG deg %2d ratio %4.0f (warm): %7.1f ms, %d its%s,"
                    " lambda %.6f, res %.1e\n",
                    opts.chebDegree, opts.chebRatio, ms, r.iterations, r.converged ? "" : "!",
                    r.eigenvalue, r.residual);
      }
    }
#ifdef BOCHNER_WITH_METAL
    // gpucheb parity rows: the GPU float Chebyshev solve on the same frozen
    // operator and warm guess as the CPU rows above, one row per chain config.
    // Line distance is to the loop's returned vector (the GPU covMG chain's) --
    // the "same filaments" acceptance quantity.
    if (gpuCheb && useGpu) {
      for (const auto& c : chebChains) {
        std::vector<gpu::GaugeLevelData> levs;
        if (c.hybrid) {
          for (const GaugeLattice& Lv : buildGaugeLevels(lastLat))
            levs.push_back({Lv.lx, Lv.ly, Lv.lz, Lv.periodic, Lv.w, Lv.tx, Lv.ty, Lv.tz});
        } else {
          levs.push_back({lastLat.lx, lastLat.ly, lastLat.lz, lastLat.periodic, lastLat.w,
                          lastLat.tx, lastLat.ty, lastLat.tz});
        }
        const GaugeEigenOptions defo;
        gpu::GaugeEigenGpu rg;
        const double ms = benchstat::medianMs([&] {
          const int hc = gpu::uploadGauge(levs);
          rg = c.hybrid ? gpu::lobpcgSolveGaugeHybrid(hc, lastWarmGuess, /*maxIters=*/300, abTol,
                                                      defo.precCycles, defo.mg.coarseSweeps,
                                                      c.deg, c.ratio)
                        : gpu::lobpcgSolveGaugeCheb(hc, lastWarmGuess, /*maxIters=*/300, abTol,
                                                    c.deg, c.ratio);
          gpu::freeGauge(hc);
        });
        std::printf("  GPU float %s deg %2d ratio %4.0f (warm): %7.1f ms, %d its,"
                    " lambda %.6f, res %.1e, line dist to %s vec %.2e\n",
                    c.hybrid ? "hybrid-LOBPCG" : "cheb-LOBPCG  ", c.deg, c.ratio, ms,
                    rg.iterations, rg.eigenvalue, rg.residual, mode.c_str(),
                    lineDistance(rg.vector, lastVec));
      }
    }
#endif
    {
      GaugeEigenOptions opts;
      opts.relativeGsDrop = false;  // live absolute-drop warm-start early exit (demo table)
      opts.tol = abTol;
      GaugeEigenResult r;
      const double ms = benchstat::medianMs([&] {
        r = smallestEigenpairGaugeMG(lastLat, lastWarmGuess.empty() ? nullptr : &lastWarmGuess, opts);
      });
      std::printf("  CPU double covMG-LOBPCG (warm, live abs-drop): %7.1f ms, %d its, lambda %.6f\n",
                  ms, r.iterations, r.eigenvalue);
    }
    {
      const FaceField thetaL = connectionAngles(grid, u, hbar);
      const CooMatrix E = connectionLaplacian(grid, thetaL);
      const std::vector<double> guess = toInterleaved(lastWarmGuess);
      EigenPair lp;
      const double ms = benchstat::medianMs([&] {
        lp = smallestEigenpairLanczos(E, abTol, lastWarmGuess.empty() ? nullptr : &guess);
      });
      std::printf("  SLEPc Lanczos   (warm): %7.1f ms, %d its, lambda %.6f\n", ms, lp.iterations,
                  lp.value);
    }
    }  // for abTol
    std::printf("  (%s eigensolve at the last frame: %d iterations)\n", mode.c_str(), lastIts);
    benchstat::printTimingSummary();
  }
  SlepcFinalize();
  return 0;
}
