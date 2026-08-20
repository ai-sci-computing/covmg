# PETSc-free tools build unconditionally.

# Linear-solver comparison: gauge-MG V-cycle vs plain CG (no PETSc/SLEPc needed).
add_executable(torus_seed_check torus_seed_check.cpp)
target_link_libraries(torus_seed_check PRIVATE bochner_core)

add_executable(aniso_bench aniso_bench.cpp)
target_link_libraries(aniso_bench PRIVATE bochner_core)

add_executable(ring_gap ring_gap.cpp)
target_link_libraries(ring_gap PRIVATE bochner_core)

add_executable(mg_bench mg_bench.cpp)
target_link_libraries(mg_bench PRIVATE bochner_core)

# Non-abelian transferability demo: SU(2)/SU(3) covariant Laplacian, gauge-MG
# mesh-independence vs CG + covMG-LOBPCG ground state (no PETSc/SLEPc needed).
add_executable(sun_gauge_bench sun_gauge_bench.cpp)
target_link_libraries(sun_gauge_bench PRIVATE bochner_core)

# Wilson-action Monte-Carlo SU(3) configurations (rough but
# CORRELATED fields) -- beta sweep + refinement series for the paper. Links the
# SLEPc Lanczos baseline when PETSc is on (same pattern as sun_gauge_bench).
add_executable(mc_gauge_bench mc_gauge_bench.cpp)
target_link_libraries(mc_gauge_bench PRIVATE bochner_core)

add_executable(ablation_bench ablation_bench.cpp)
target_link_libraries(ablation_bench PRIVATE bochner_core)

# Diagnostic: separates hierarchy (flux aliasing at a coarse level) from outer
# loop (degenerate ground state) as the cause of the SU(3) n=64 iteration jump.
add_executable(flux_alias_diag flux_alias_diag.cpp)
target_link_libraries(flux_alias_diag PRIVATE bochner_core)

add_executable(convergence_factor_bench convergence_factor_bench.cpp)
target_link_libraries(convergence_factor_bench PRIVATE bochner_core)

add_executable(block_eig_bench block_eig_bench.cpp)
target_link_libraries(block_eig_bench PRIVATE bochner_core)

add_executable(seed_slice seed_slice.cpp)
target_link_libraries(seed_slice PRIVATE bochner_core)

# Matrix export for external eigensolver baselines (PyAMG LOBPCG): the U(1)
# uniform-flux torus / smooth SU(d) operator as a complex MatrixMarket file.
add_executable(export_operator export_operator.cpp)
target_link_libraries(export_operator PRIVATE bochner_core)

# The remaining tools use the SLEPc-backed EigenSolver / CF pipeline (PETSc only).
if(BOCHNER_WITH_PETSC)
  # Grid demo: closed CF sim <-> W-P extractor loop on the MAC grid.
  add_executable(ring_demo_grid ring_demo_grid.cpp)
  target_link_libraries(ring_demo_grid PRIVATE bochner_core)

  # Re-quantization demo: Hill's vortex decomposed into nested rings vs hbar.
  add_executable(hill_decomp hill_decomp.cpp)
  target_link_libraries(hill_decomp PRIVATE bochner_core)

  # Phase-3 benchmark: smallest-eigenvector solve methods vs frustration & size.
  add_executable(eig_benchmark eig_benchmark.cpp)
  target_link_libraries(eig_benchmark PRIVATE bochner_core)

  # Phase-3 warm-start study: cold vs warm per-frame eigensolve over a real ring
  # sim sequence (the interactivity lever).
  add_executable(warmstart_bench warmstart_bench.cpp)
  target_link_libraries(warmstart_bench PRIVATE bochner_core)

  # Experiment A: gauge-aware subdivision vs the eigensolver for extraction.
  add_executable(subdivision_demo subdivision_demo.cpp)
  target_link_libraries(subdivision_demo PRIVATE bochner_core)

  # Eigensolver comparison: gauge-MG covMG-LOBPCG vs SLEPc Lanczos.
  add_executable(eig_compare eig_compare.cpp)
  target_link_libraries(eig_compare PRIVATE bochner_core)

  # Eigensolve on the lattice-gauge-solvers setup (periodic torus, uniform flux):
  # gauge-MG covMG-LOBPCG vs SLEPc Lanczos + the backward-iteration estimate.
  add_executable(torus_eig_compare torus_eig_compare.cpp)
  target_link_libraries(torus_eig_compare PRIVATE bochner_core)

  # Genuinely distributed Krylov-Schur baseline: MPIAIJ on COMM_WORLD, run
  # under mpiexec -n R to measure real MPI scaling (an earlier 1->2->4 pass
  # duplicated COMM_SELF solves and measured bandwidth contention).
  add_executable(mpi_baseline_bench mpi_baseline_bench.cpp)
  target_link_libraries(mpi_baseline_bench PRIVATE bochner_core)

  # Same-preconditioner / different-outer-loop control: the covariant V-cycle
  # behind PCSHELL inside SLEPc GD/LOBPCG, plus a Jacobi
  # floor, against covMG-LOBPCG and the AMG rows on the flux torus.
  add_executable(covpc_outer_bench covpc_outer_bench.cpp)
  target_link_libraries(covpc_outer_bench PRIVATE bochner_core)

  # Served-application benchmark: magnetic Schrodinger / linear Ginzburg-Landau
  # lowest eigenpair on an open Neumann box (bounded domain, not FFT-diagonalized).
  add_executable(magnetic_eig_compare magnetic_eig_compare.cpp)
  target_link_libraries(magnetic_eig_compare PRIVATE bochner_core)

  # Chebyshev-filtered Lanczos baseline (SLEPc STFILTER) vs plain Lanczos /
  # Krylov-Schur, on the same U(1)/SU(d) operators as export_operator.
  add_executable(chebyshev_lanczos_bench chebyshev_lanczos_bench.cpp)
  target_link_libraries(chebyshev_lanczos_bench PRIVATE bochner_core)

  # Adaptive gauge-aware coarse space: GAMG-preconditioned LOBPCG
  # with a COMPUTED near-null space (DD-alpha-AMG idea) vs the default constant
  # near-kernel, on the export_operator .mtx files.
  add_executable(gauge_aware_amg_eig gauge_aware_amg_eig.cpp)
  target_link_libraries(gauge_aware_amg_eig PRIVATE bochner_core)

  # Pressure-projection solver study (PC x tolerance on a real scalar Poisson).
  add_executable(pressure_bench pressure_bench.cpp)
  target_link_libraries(pressure_bench PRIVATE bochner_core)

  # Full per-frame pipeline profiler (sim / connection / eigensolve / extraction).
  add_executable(pipeline_profile pipeline_profile.cpp)
  target_link_libraries(pipeline_profile PRIVATE bochner_core)

  # Matrix-free control: Krylov-Schur on a MatShell around our
  # own apply, isolating the data structure from the algorithm.
  add_executable(matrixfree_baseline matrixfree_baseline.cpp)
  target_link_libraries(matrixfree_baseline PRIVATE bochner_core)

  # Start-policy symmetry control: covMG-LOBPCG's default psi=1 start vs the
  # SLEPc baseline's, matched in both directions.
  add_executable(start_symmetry_bench start_symmetry_bench.cpp)
  target_link_libraries(start_symmetry_bench PRIVATE bochner_core)

  # Headless flow-past-obstacle demo profiler (the paper's demo numbers:
  # CPU 1T / CPU nT / GPU per-stage timings + eigensolver A/B).
  add_executable(obstacle_profile obstacle_profile.cpp)
  target_link_libraries(obstacle_profile PRIVATE bochner_core)
endif()
