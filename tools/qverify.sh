#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# qverify.sh — verification harness for legacy Sirius (gpu_processing).
#
# Runs a query on ours-2gpu (magi legacy build), on DuckDB CPU (truth), and
# optionally on the ORIGINAL single-GPU sirius-main baseline, then compares
# results key-by-key with a float tolerance and reports timings + fallback
# status. Encodes the measurement rules that repeatedly bit us:
#   - results are parsed by row shape + explicit run-boundary markers, never
#     by line windows ([magi] banners pollute stdout);
#   - DOUBLE columns are compared with a tolerance, never hashed/md5'd;
#   - CPU truth = plain SQL on the same binary (never count(*)-wrap — it can
#     misplan/assert);
#   - fallback is checked in BOTH stdout and stderr; FB>0 on the GPU config
#     means the "GPU" numbers are fake (silent CPU fallback) → FAIL.
#
# MUST run on a GPU compute node with >= 4 visible GPUs and enough CPUs:
#   srun --jobid=<JOBID> --overlap -N1 tools/qverify.sh [opts] "SQL"
# (salloc needs -c 48: too few CPUs starves magi host workers.)
#
# Usage:
#   tools/qverify.sh [-n RUNS] [-d DB] [-m] [-C] [-t RTOL] "SQL"
#     -n RUNS   warm repetitions per GPU config (default 3)
#     -d DB     duckdb database (default /dev/shm/tpch_sf50.duckdb, auto-copied
#               from /scratch/diwu/tpch_sf50.duckdb when missing)
#     -m        also run ORIGINAL sirius-main 1-GPU baseline (perf reference)
#     -C        skip the CPU-truth run (e.g. when it is impractically slow)
#     -t RTOL   float tolerance for value compare (default 1e-6)
#   SQL: full query text; use single quotes inside (no double quotes), drop
#        ORDER BY, add ::DOUBLE where DECIMAL is unsupported on legacy-2gpu.
#
# Env overrides:
#   QV_BIN       ours binary   (default <repo>/build/legacy-release-2gpu/duckdb)
#   QV_LIBS      ours LD_LIBRARY_PATH (default /scratch/diwu/sirius/.pixi/envs/default/lib)
#   QV_MAIN_BIN  original main (default /scratch/diwu/sirius-main/build/release/duckdb)
#   QV_MAIN_LIBS main LD_LIBRARY_PATH (default sirius-main25 conda env lib)
#   QV_INIT      ours gpu_buffer_init args (default: 25 GB/20 GB/pinned 60 GB)
#   QV_MAIN_INIT main gpu_buffer_init args (default: 30 GB/45 GB/pinned 80 GB
#                — 20 GB processing OOMs SF50 and silently falls back)
# ─────────────────────────────────────────────────────────────────────────────
set -u
# Re-exec under a LOGIN shell on the executing node: srun forwards the
# submitting host's environment, which on this cluster lacks the compute
# node's CUDA driver library paths — without this, both engines silently
# fall back to CPU (and this script's fallback check flags it, but the run
# is useless). bash -l re-sources the compute node's profile.
if [ -z "${QV_INNER:-}" ]; then
  export QV_INNER=1
  exec bash -l "$0" "$@"
fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS=3; DB=/dev/shm/tpch_sf50.duckdb; DO_MAIN=0; DO_CPU=1; RTOL=1e-6
while getopts "n:d:mCt:" o; do
  case $o in
    n) RUNS=$OPTARG;; d) DB=$OPTARG;; m) DO_MAIN=1;; C) DO_CPU=0;; t) RTOL=$OPTARG;;
    *) echo "bad option"; exit 2;;
  esac
done
shift $((OPTIND-1))
Q=${1:?usage: qverify.sh [-n N] [-d DB] [-m] [-C] [-t RTOL] \"SQL\"}
case $Q in *\"*) echo "FAIL: query must not contain double quotes"; exit 2;; esac

BIN=${QV_BIN:-$REPO/build/legacy-release-2gpu/duckdb}
LIBS=${QV_LIBS:-/scratch/diwu/sirius/.pixi/envs/default/lib}
MAIN_BIN=${QV_MAIN_BIN:-/scratch/diwu/sirius-main/build/release/duckdb}
MAIN_LIBS=${QV_MAIN_LIBS:-/home/diwu/scratch/miniconda3/envs/sirius-main25/lib}
# NB: assign defaults via [ -z ] — an unquoted ${VAR:-"..."} default eats the
# inner double quotes, the init reaches the parser as gpu_buffer_init(25 GB,..)
# → Parser Error → every gpu_processing silently falls back to CPU (bug found
# by this harness's own self-test).
INIT="${QV_INIT:-}"
[ -z "$INIT" ] && INIT='call gpu_buffer_init("25 GB","20 GB", pinned_memory_size = "60 GB");'
MAIN_INIT="${QV_MAIN_INIT:-}"
[ -z "$MAIN_INIT" ] && MAIN_INIT='call gpu_buffer_init("30 GB","45 GB", pinned_memory_size = "80 GB");'
# .print, not a SELECT: the legacy fork binary emits NO plain-SELECT results
# (and can segfault on heavy plain SQL); .print is CLI-level, works on both
# duckdb 1.2.1 (main) and the 1.4.3 fork.
B=".print __QVERIFY_RUN_BOUNDARY__"
T=$(mktemp -d /tmp/qverify.XXXXXX)
KEEP=0
finish() {
  if [ "$KEEP" = 1 ]; then
    echo "artifacts kept at: $T (ours/cpu/main .out/.err + .sql)"
  else
    rm -rf "$T"
  fi
}
trap finish EXIT

command -v nvidia-smi >/dev/null || echo "WARN: no nvidia-smi — are you on a GPU node?"
[ -s "$DB" ] || { echo "copying DB to $DB ..."; cp /scratch/diwu/tpch_sf50.duckdb "$DB" || exit 1; }
[ -x "$BIN" ] || { echo "FAIL: ours binary missing: $BIN"; exit 1; }

# ── ours-2gpu ────────────────────────────────────────────────────────────────
{ echo ".mode csv"; echo ".headers off"; echo "$INIT"
  for _ in $(seq 1 "$RUNS"); do echo "call gpu_processing(\"$Q\");"; echo "$B"; done
} > "$T/ours.sql"
env LD_LIBRARY_PATH="$LIBS" CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_MODULE_LOADING=EAGER \
    MAGI_QUERY_TIME=1 MAGI_PHASE_TIME=1 \
    timeout 900 "$BIN" "$DB" -unsigned < "$T/ours.sql" > "$T/ours.out" 2> "$T/ours.err"
RC_OURS=$?
FB_OURS=$(( $(grep -aci "fallback" "$T/ours.out") + $(grep -aci "fallback to DuckDB" "$T/ours.err") ))
echo "── ours-2gpu (rc=$RC_OURS) ──"
echo "  [query-time]: $(grep -a query-time "$T/ours.err" | grep -oE '[0-9.]+ms' | tr '\n' ' ')"
echo "  fallback=$FB_OURS  shuffle-joins=$(grep -ac magi-join-phase "$T/ours.err")"
grep -a "bcast-decision gpu=0" "$T/ours.err" | sed 's/^/  /' | sort -u | head -6
grep -aiE "WORKER-THROW|GPUassert|illegal|cuda error" "$T/ours.err" | sort -u | head -3 | sed 's/^/  ⚠ /'
if [ "$RC_OURS" != 0 ] || [ "$FB_OURS" -gt 0 ]; then
  KEEP=1
  echo "  ── ours stderr tail ──"; tail -5 "$T/ours.err" | sed 's/^/  │ /'
fi

CONFIGS=()
# ── CPU truth (plain SQL, same binary → DuckDB CPU) ─────────────────────────
if [ "$DO_CPU" = 1 ]; then
  # CPU truth = gpu_processing WITHOUT gpu_buffer_init: the call errors out
  # and takes the DuckDB-CPU fallback, which prints the CPU result. Plain SQL
  # is NOT usable on the fork binary (emits nothing / can segfault on heavy
  # queries), and count(*)-wrapping misplans — never use either as truth.
  { echo ".mode csv"; echo ".headers off"
    echo "call gpu_processing(\"$Q\");"; echo "$B"; } > "$T/cpu.sql"
  env LD_LIBRARY_PATH="$LIBS" timeout 1800 "$BIN" "$DB" -unsigned \
      < "$T/cpu.sql" > "$T/cpu.out" 2> "$T/cpu.err"
  RC_CPU=$?
  grep -aqi "fallback" "$T/cpu.out" "$T/cpu.err" || {
    echo "  ⚠ cpu-truth run did NOT fall back — truth unreliable"; KEEP=1; }
  echo "── cpu truth (rc=$RC_CPU) ──"
  CONFIGS+=("cpu=$T/cpu.out")
fi
CONFIGS+=("ours=$T/ours.out")

# ── original sirius-main 1-GPU (perf baseline) ──────────────────────────────
if [ "$DO_MAIN" = 1 ]; then
  { echo ".mode csv"; echo ".headers off"; echo "$MAIN_INIT"
    for _ in $(seq 1 "$RUNS"); do echo "call gpu_processing(\"$Q\");"; echo "$B"; done
  } > "$T/main.sql"
  rm -rf "$T/mlog"; mkdir -p "$T/mlog"
  env LD_LIBRARY_PATH="$MAIN_LIBS" CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=EAGER \
      SIRIUS_LOG_LEVEL=debug SIRIUS_LOG_DIR="$T/mlog" \
      timeout 900 "$MAIN_BIN" "$DB" -unsigned < "$T/main.sql" > "$T/main.out" 2> "$T/main.err"
  RC_MAIN=$?
  FB_MAIN=$(( $(grep -aci "fallback" "$T/main.out") + $(grep -aci "fallback" "$T/main.err") ))
  MLOG=$(ls -t "$T"/mlog/*.log 2>/dev/null | head -1)
  echo "── main-1gpu (rc=$RC_MAIN) ──"
  echo "  exec-times: $(grep -a 'Execute query time' "${MLOG:-/dev/null}" 2>/dev/null | grep -oE '[0-9.]+ ms' | tr '\n' ' ')"
  echo "  fallback=$FB_MAIN $( [ "$FB_MAIN" -gt 0 ] && echo '⚠ main numbers are CPU, not GPU' )"
  if [ "$RC_MAIN" != 0 ] || [ "$FB_MAIN" -gt 0 ]; then
    KEEP=1
    echo "  ── main stderr tail ──"; tail -5 "$T/main.err" | sed 's/^/  │ /'
  fi
  CONFIGS+=("main=$T/main.out")
fi

# ── compare ──────────────────────────────────────────────────────────────────
echo "── compare (rtol=$RTOL) ──"
python3 "$REPO/tools/qverify_compare.py" --rtol "$RTOL" "${CONFIGS[@]}"
CMP=$?
[ "$CMP" != 0 ] && KEEP=1
if [ "$FB_OURS" -gt 0 ]; then
  echo "VERDICT-OVERRIDE: FAIL — ours-2gpu fell back to CPU ($FB_OURS hits); GPU numbers are fake."
  exit 1
fi
exit $CMP
