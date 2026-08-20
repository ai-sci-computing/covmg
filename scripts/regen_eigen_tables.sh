#!/bin/zsh
# Regenerate every benchmark that involves the covMG-LOBPCG eigensolvers, plus
# the demo profiles -- the tables invalidated by the 2026-07-17 C7 (certificate
# floor) + C14 (fixed-cycle preconditioner skips the dead residual matvec)
# changes. Linear-solve tables are NOT regenerated: the linear V-cycle path
# (tol > 0) is untouched by both changes.
#
# Protocol: single-threaded (OMP_NUM_THREADS=1), median of BENCH_REPS=5
# identical runs per solve (tools/BenchTiming.h), matching the certified
# 2026-07-11 record. Demo profiles run at their documented thread counts.
# Tuned-baseline points rerun with -eps_ncv.
#
# Usage: scripts/regen_eigen_tables.sh <output-dir>   (from the build dir)
set -e
out=${1:?output dir required}
mkdir -p "$out"
export BENCH_REPS=5
stage() { echo "=== STAGE $1 start $(date +%H:%M:%S) ==="; }

stage su3            ; OMP_NUM_THREADS=1 ./tools/sun_gauge_bench 3            > "$out/su3.log" 2>&1
stage su3-ncv32      ; OMP_NUM_THREADS=1 ./tools/sun_gauge_bench 3 -eps_ncv 32 > "$out/su3_ncv32.log" 2>&1
stage su2            ; OMP_NUM_THREADS=1 ./tools/sun_gauge_bench 2            > "$out/su2.log" 2>&1
stage su2-ncv32      ; OMP_NUM_THREADS=1 ./tools/sun_gauge_bench 2 -eps_ncv 32 > "$out/su2_ncv32.log" 2>&1
stage ring           ; OMP_NUM_THREADS=1 ./tools/eig_compare                  > "$out/ring.log" 2>&1
stage ring-ncv32     ; OMP_NUM_THREADS=1 ./tools/eig_compare 48 64 -eps_ncv 32 > "$out/ring_ncv32.log" 2>&1
stage ring-ncv48     ; OMP_NUM_THREADS=1 ./tools/eig_compare 64 -eps_ncv 48   > "$out/ring_ncv48.log" 2>&1
stage torus          ; for n in 16 24 32 48 64 96; do OMP_NUM_THREADS=1 ./tools/torus_eig_compare $n 4 >> "$out/torus.log" 2>&1; done
stage torus96-ncv32  ; OMP_NUM_THREADS=1 ./tools/torus_eig_compare 96 4 -eps_ncv 32 > "$out/torus96_ncv32.log" 2>&1
stage warmstart      ; OMP_NUM_THREADS=1 ./tools/warmstart_bench 46 8         > "$out/warmstart.log" 2>&1
stage mc             ; OMP_NUM_THREADS=1 ./tools/mc_gauge_bench all 3         > "$out/mc.log" 2>&1
stage mc48           ; OMP_NUM_THREADS=1 ./tools/mc_gauge_bench refine48 3    > "$out/mc48.log" 2>&1
stage block          ; OMP_NUM_THREADS=1 ./tools/block_eig_bench 3 smooth 16 24 32 > "$out/block.log" 2>&1
stage obstacle-gpu   ; ./tools/obstacle_profile 32 200 gpu                    > "$out/obstacle_gpu.log" 2>&1
stage obstacle-cpu4  ; OMP_NUM_THREADS=4 ./tools/obstacle_profile 32 200 cpu  > "$out/obstacle_cpu4.log" 2>&1
stage obstacle-cpu1  ; OMP_NUM_THREADS=1 ./tools/obstacle_profile 32 200 cpu  > "$out/obstacle_cpu1.log" 2>&1
stage pipeline-4t    ; OMP_NUM_THREADS=4 ./tools/pipeline_profile 46 10       > "$out/pipeline_4t.log" 2>&1
stage pipeline-1t    ; OMP_NUM_THREADS=1 ./tools/pipeline_profile 46 10       > "$out/pipeline_1t.log" 2>&1
stage block64        ; OMP_NUM_THREADS=1 ./tools/block_eig_bench 3 smooth 64  > "$out/block64.log" 2>&1
# --- 2026-08 survey additions: the polynomial (Chebyshev-LOBPCG /
# Chebyshev-Davidson) columns, the hybrid smoother rows, the warm Chebyshev
# chain, and the gauge-dependence campaign of the article's Table 4.
# BOCHNER_CHEB_SWEEP=1 uses the drivers' default degree x interval grid.
stage torus-sweeps   ; for n in 16 32 48 64 96; do BOCHNER_CHEB_SWEEP=1 BOCHNER_CHEBDAV=1 BOCHNER_HYBRID="2:6,3:6" OMP_NUM_THREADS=1 ./tools/torus_eig_compare $n 4 >> "$out/torus_sweeps.log" 2>&1; done
stage ring-sweep     ; BOCHNER_CHEB_SWEEP=1 OMP_NUM_THREADS=1 ./tools/eig_compare 24 32 48 64 > "$out/ring_sweep.log" 2>&1
stage magnetic-sweep ; for n in 16 24 32 48; do BOCHNER_CHEB_SWEEP=1 OMP_NUM_THREADS=1 ./tools/magnetic_eig_compare $n 2; done > "$out/magnetic_sweep.log" 2>&1
stage sun-sweeps     ; for d in 2 3; do BOCHNER_CHEB_SWEEP=1 OMP_NUM_THREADS=1 ./tools/sun_gauge_bench $d > "$out/sun${d}_sweep.log" 2>&1; done
stage warm-cheb      ; BOCHNER_WARM_CHEB=32:300 OMP_NUM_THREADS=1 ./tools/warmstart_bench 46 8 > "$out/warmstart_cheb.log" 2>&1
stage gauge-hurts    ; mkdir -p "$out/gauge-hurts" && cd "$out/gauge-hurts" && \
  OMP_NUM_THREADS=1 $OLDPWD/tools/export_operator u1 16 4 torus16.mtx && \
  BOCHNER_EXPORT_REGAUGE=4242 OMP_NUM_THREADS=1 $OLDPWD/tools/export_operator u1 16 4 torus16_regauge.mtx && \
  OMP_NUM_THREADS=1 $OLDPWD/tools/export_operator u1 32 4 torus32.mtx && \
  BOCHNER_EXPORT_REGAUGE=4242 OMP_NUM_THREADS=1 $OLDPWD/tools/export_operator u1 32 4 torus32_regauge.mtx && \
  for m in *.mtx; do OMP_NUM_THREADS=1 $OLDPWD/tools/gauge_aware_amg_eig $m 4; done > gamg.log 2>&1 && \
  cd "$OLDPWD"
# The wake rows (warm A/B at 1e-7, the device chains of the article's device
# table, and the live transfer switches) regenerate from obstacle_profile:
#   BOCHNER_AB_TOLS=1e-7 BOCHNER_AB_CHEB="16:300,32:1000,64:1000,64:3000,96:3000" ./tools/obstacle_profile 40 200 gpu
#   BOCHNER_GPU_CHEB="16:300,32:1000,64:1000,64:3000,96:3000" BOCHNER_GPU_HYBRID="2:6,3:6" ./tools/obstacle_profile 40 200 gpucheb
#   BOCHNER_ABL_TOL=1e-7 ./tools/obstacle_profile 40 200 ablate
# (shedding: prepend BOCHNER_HBAR=0.061 BOCHNER_R=0.357 BOCHNER_U=2 BOCHNER_NU=0.0025 and run 4000 frames;
#  the frozen-operator rows and the equivariance certificate use
#  BOCHNER_DUMP_LAT + ./tools/flux_alias_diag eigfile with BOCHNER_EIGFILE_{CHEB,REGAUGE,RANDSTART};
#  PRIMME columns: tools/primme_baseline, build recipe in its header.)

echo "=== REGEN COMPLETE $(date +%H:%M:%S) ==="
