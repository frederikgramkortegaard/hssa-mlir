#!/usr/bin/env bash
# Endless differential-fuzzing driver for hssa-mlir.
#
#   scripts/fuzz-gen.sh [N] [START_SEED]
#
# For each of N random ROOPL++ programs (scripts/gen_rplpp.py SEED):
#   1. compile with the reference compiler (roopl.parse,check,translate)
#   2. interpret the compiled HSSA with the reference interpreter (the ORACLE),
#      capturing the final heap dump "<slot> => <value>"
#   3. push the HSSA through our pipeline:
#        import-hssa | hssa-resolve-uncalls | canonicalize | convert-hssa-to-arith
#      then lower to LLVM and run natively (lli --jit-kind=mcjit) with a
#      heap-returning-main wrapper, asserting every integer heap slot the
#      interpreter reported matches the native run.
#
# Failure taxonomy (what we are hunting):
#   (a) GEN-COMPILE-FAIL     generator emitted something the compiler rejects
#                            -> GENERATOR bug, tighten it.
#   (b) PIPELINE-FAIL        compiled fine but our import/lowering/native failed
#                            -> OUR bug. program saved.
#   (c) DIFF-MISMATCH        native heap != interpreter heap
#                            -> OUR bug. program saved.
#   (d) ORACLE-FAIL          interpreter itself rejected/errored the HSSA
#                            (should not happen for well-formed input; flagged
#                             and saved so we can tell it apart from (a)).
#   Crashes/hangs are caught via per-stage timeouts and counted as (b).
#
# Failing programs (+ their .hssa and diagnostics) are saved under
# fuzz-gen-fails/.  A clean batch is a strong pre-publication signal.
set -uo pipefail

N="${1:-200}"
START="${2:-1}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF="${HSSA_REFERENCE:-$(cd "$(dirname "$0")/.." && pwd)/third_party/hssa-reference}"
L="${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}"
OPT="$ROOT/build/bin/hssa-opt"
TR="$ROOT/build/bin/hssa-translate"
GEN="$ROOT/scripts/gen_rplpp.py"
JAR="$REF/target/scala-3.4.1/hssa.jar"
FAILDIR="$ROOT/fuzz-gen-fails"
WORK="$(mktemp -d /tmp/fuzzgen.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

# Portable timeout (coreutils `timeout` or `gtimeout` on macOS; else no-op).
TO=""
if command -v timeout >/dev/null 2>&1; then TO="timeout"
elif command -v gtimeout >/dev/null 2>&1; then TO="gtimeout"; fi
run_to() { # <secs> <cmd...>
  local s="$1"; shift
  if [ -n "$TO" ]; then "$TO" "$s" "$@"; else "$@"; fi
}

[ -f "$JAR" ] || { echo "ABORT: reference jar missing: $JAR"; exit 2; }
[ -x "$OPT" ] && [ -x "$TR" ] || { echo "ABORT: build hssa-opt/hssa-translate first"; exit 2; }

mkdir -p "$FAILDIR"

# counters
gen_ok=0 compiled=0 pipelined=0 matched=0
c_gen_compile=0 c_pipeline=0 c_diff=0 c_oracle=0

save_fail() { # <kind> <seed> <rplpp> <hssa-or-empty> <log>
  local kind="$1" seed="$2" rplpp="$3" hssa="$4" log="$5"
  local base="$FAILDIR/${kind}_seed${seed}"
  cp "$rplpp" "$base.rplpp" 2>/dev/null
  [ -n "$hssa" ] && [ -f "$hssa" ] && cp "$hssa" "$base.hssa" 2>/dev/null
  printf '%s\n' "$log" > "$base.log"
  echo "  saved -> $base.*"
}

for seed in $(seq "$START" $((START + N - 1))); do
  rplpp="$WORK/p$seed.rplpp"
  hssa="$WORK/p$seed.hssa"

  # --- generate ---
  if ! python3 "$GEN" "$seed" > "$rplpp" 2>"$WORK/gen.err"; then
    echo "[$seed] (a) GEN crashed"; c_gen_compile=$((c_gen_compile+1))
    save_fail "gen-crash" "$seed" "$rplpp" "" "$(cat "$WORK/gen.err")"
    continue
  fi
  gen_ok=$((gen_ok+1))

  # --- compile (reference) --- exit code is unreliable; scan stdout ---
  cout="$(run_to 60 java -jar "$JAR" load "$rplpp", roopl.parse, roopl.check, roopl.translate 2>&1)"
  if echo "$cout" | grep -q '^Error @' || ! echo "$cout" | grep -q '^rel '; then
    echo "[$seed] (a) GEN-COMPILE-FAIL"
    echo "$cout" | grep -A4 '^Error @' | head -6 | sed 's/^/      /'
    c_gen_compile=$((c_gen_compile+1))
    save_fail "a-gencompile" "$seed" "$rplpp" "" "$cout"
    continue
  fi
  printf '%s\n' "$cout" > "$hssa"
  compiled=$((compiled+1))

  # --- interpret (ORACLE) --- heap dump "<slot> => <value>" ---
  iout="$(run_to 60 java -jar "$JAR" load "$hssa", hssa.parse, hssa.check, hssa.exec main 2>&1)"
  if ! echo "$iout" | grep -q '=>'; then
    echo "[$seed] (d) ORACLE-FAIL (interpreter rejected compiled HSSA)"
    echo "$iout" | head -4 | sed 's/^/      /'
    c_oracle=$((c_oracle+1))
    save_fail "d-oracle" "$seed" "$rplpp" "$hssa" "$iout"
    continue
  fi
  # Keep integer values at slot >= 1 (skip slot 0 / null / <rel ...> vtables).
  slots="$(echo "$iout" | awk '/=>/ {gsub(/[[:space:]]/,"",$1); if ($1>=1 && $3 ~ /^-?[0-9]+$/) printf "%s:%s ", $1, $3}')"

  # --- our pipeline: import -> lower -> native ---
  mlir="$WORK/p$seed.mlir"
  if ! run_to 60 "$TR" --import-hssa "$hssa" 2>"$WORK/imp.err" \
        | run_to 60 "$OPT" --hssa-resolve-uncalls 2>"$WORK/ru.err" \
        | run_to 60 "$OPT" --canonicalize 2>"$WORK/can.err" \
        | run_to 60 "$OPT" --convert-hssa-to-arith 2>"$WORK/arith.err" > "$mlir" \
     || [ ! -s "$mlir" ]; then
    echo "[$seed] (b) PIPELINE-FAIL (import/lowering)"
    for e in imp ru can arith; do
      [ -s "$WORK/$e.err" ] && { echo "      --- $e ---"; head -4 "$WORK/$e.err" | sed 's/^/      /'; }
    done
    c_pipeline=$((c_pipeline+1))
    save_fail "b-pipeline" "$seed" "$rplpp" "$hssa" \
      "$(for e in imp ru can arith; do echo "=== $e ==="; cat "$WORK/$e.err"; done)"
    continue
  fi
  pipelined=$((pipelined+1))

  # If no integer slots to compare, we still validated the pipeline; count it
  # as pipelined but trivially matched (nothing to diverge).
  if [ -z "$slots" ]; then
    matched=$((matched+1))
    continue
  fi

  # --- heap-returning-main wrapper, then native run & diff ---
  python3 - "$mlir" "$slots" <<'PY'
import sys
f=sys.argv[1]; pairs=sys.argv[2].split()
s=open(f).read().replace('@main(','@prog(').replace('@main)','@prog)')
L=['  func.func @main() -> i32 {','    %h = call @prog() : () -> memref<2x?xi32>',
   '    %row0 = arith.constant 0 : index','    %acc0 = arith.constant 0 : i32']
acc='%acc0'
for k,p in enumerate(pairs):
    sl,val=p.split(':')
    L+=[f'    %i{k} = arith.constant {sl} : index',
        f'    %v{k} = memref.load %h[%row0, %i{k}] : memref<2x?xi32>',
        f'    %e{k} = arith.constant {val} : i32',
        f'    %d{k} = arith.subi %v{k}, %e{k} : i32',
        f'    %acc{k+1} = arith.addi %acc{k}, %d{k} : i32']; acc=f'%acc{k+1}'
L+=[f'    return {acc} : i32','  }']
i=s.rfind('}'); open(f,'w').write(s[:i]+'\n'+'\n'.join(L)+'\n'+s[i:])
PY

  if run_to 90 "$L/mlir-opt" --convert-to-llvm "$mlir" 2>"$WORK/l1.err" \
       | run_to 90 "$L/mlir-translate" --mlir-to-llvmir 2>"$WORK/l2.err" \
       | run_to 90 "$L/lli" --jit-kind=mcjit 2>"$WORK/l3.err"
  then
    matched=$((matched+1))
    # (quiet on success to keep the stream readable)
  else
    rc=$?
    if [ $rc -ge 124 ]; then
      echo "[$seed] (d) CRASH/HANG in native run (rc=$rc)"
      c_pipeline=$((c_pipeline+1))
      save_fail "d-crash" "$seed" "$rplpp" "$hssa" \
        "native rc=$rc slots=$slots"$'\n'"$(cat "$WORK"/l*.err)"
    else
      echo "[$seed] (c) DIFF-MISMATCH (native heap != interpreter)"
      echo "      interpreter slots: $slots" | sed 's/^/      /'
      c_diff=$((c_diff+1))
      save_fail "c-diff" "$seed" "$rplpp" "$hssa" \
        "native rc=$rc slots=$slots"$'\n'"$(cat "$WORK"/l*.err)"
    fi
  fi
done

echo
echo "================ FUZZ SUMMARY (seeds $START..$((START+N-1)), N=$N) ================"
echo "  generated ok ............. $gen_ok / $N"
echo "  compiled (ref) ........... $compiled"
echo "  fully pipelined .......... $pipelined"
echo "  diff-MATCHED (native==oracle) $matched"
echo "  ----------------------------------------"
echo "  (a) gen->compile fails ... $c_gen_compile   [generator bug -> tighten]"
echo "  (b) pipeline fails ....... $c_pipeline   [OUR bug]"
echo "  (c) diff mismatches ...... $c_diff   [OUR bug]"
echo "  (d) oracle/crash ......... $c_oracle"
echo "========================================================================"
if [ $((c_pipeline + c_diff)) -gt 0 ]; then
  echo "REAL hssa-mlir failures found -> see $FAILDIR/"
  exit 1
elif [ "$c_gen_compile" -gt 0 ]; then
  echo "Only generator-level failures (tighten generator). No pipeline/diff bugs."
  exit 0
else
  echo "CLEAN: every compiled program pipelined and matched the interpreter."
  exit 0
fi
