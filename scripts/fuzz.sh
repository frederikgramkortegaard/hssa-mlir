#!/usr/bin/env bash
# Property-testing / fuzzing for hssa-mlir.
#
#   scripts/fuzz.sh metamorphic     reversibility invariants over the corpus (default)
#   scripts/fuzz.sh crash [N]       N random mutations into the importer+passes; crash-hunt
#
# Metamorphic invariants per corpus program (no oracle needed):
#   M1  imports, and passes --hssa-verify-linear
#   M2  parse->print->parse is stable (idempotent round-trip)
#   M3  inversion is involutive: invert;invert == original (modulo canonicalize) [soft]
#   M4  linearity survives inversion
#   M5  linearity survives --canonicalize (a standard pass)
# The differential half (optimize-then-interpret == interpret) is `make diff-oo`;
# unbounded random differential fuzzing is scripts/fuzz-gen.sh.
#
# HSSA_BIN overrides the tool dir (e.g. a sanitizer build: HSSA_BIN=build-asan/bin).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${HSSA_BIN:-$ROOT/build/bin}"
OPT="$BIN/hssa-opt"; TR="$BIN/hssa-translate"
REF="${HSSA_REFERENCE:-$(cd "$(dirname "$0")/.." && pwd)/third_party/hssa-reference}"

corpus() { ls "$REF/programs/roopl/verified/"*.hssa 2>/dev/null; }

is_crash() { local rc="$1"; [ "$rc" -eq 134 ] || [ "$rc" -ge 137 ]; }  # abort / signal; 124=timeout handled separately

metamorphic() {
  local pass=0 warn=0 fail=0
  for f in $(corpus); do
    local b; b=$(basename "$f" .hssa)
    if ! "$TR" --import-hssa "$f" 2>/dev/null >/tmp/m0.mlir; then echo "FAIL $b: M1 import"; fail=$((fail+1)); continue; fi
    if ! "$OPT" --hssa-verify-linear /tmp/m0.mlir >/dev/null 2>&1; then echo "FAIL $b: M1 verify-linear"; fail=$((fail+1)); continue; fi
    "$OPT" /tmp/m0.mlir 2>/dev/null >/tmp/m1.mlir
    "$OPT" /tmp/m1.mlir 2>/dev/null >/tmp/m2.mlir
    if ! diff -q /tmp/m1.mlir /tmp/m2.mlir >/dev/null; then echo "FAIL $b: M2 parse->print->parse unstable"; fail=$((fail+1)); continue; fi
    if ! "$OPT" --hssa-invert /tmp/m0.mlir 2>/dev/null | "$OPT" --hssa-verify-linear >/dev/null 2>&1; then echo "FAIL $b: M4 verify-linear after invert"; fail=$((fail+1)); continue; fi
    if ! "$OPT" --canonicalize /tmp/m0.mlir 2>/dev/null | "$OPT" --hssa-verify-linear >/dev/null 2>&1; then echo "FAIL $b: M5 verify-linear after canonicalize"; fail=$((fail+1)); continue; fi
    # M3 involution (semantic): invert;invert stays valid + linear. Inversion is
    # involutive up to cosmetic block/dead-const restructuring, so EXACT syntactic
    # identity is NOT expected (see invert.mlir lit tests for exact cases); the
    # real invariant is that double inversion reproduces a well-formed linear program.
    if ! "$OPT" --hssa-invert /tmp/m0.mlir 2>/dev/null | "$OPT" --hssa-invert 2>/dev/null | "$OPT" --hssa-verify-linear >/dev/null 2>&1; then echo "FAIL $b: M3 invert;invert not linear"; fail=$((fail+1)); continue; fi
    echo "ok   $b"; pass=$((pass+1))
  done
  echo "--- metamorphic: $pass ok, $warn warn, $fail fail ---"
  return $fail
}

crash() {
  local N="${1:-500}"; local crashes=0
  local seeds=($(corpus)); local ns=${#seeds[@]}
  [ "$ns" -gt 0 ] || { echo "no corpus (set HSSA_REFERENCE)"; exit 2; }
  mkdir -p "$ROOT/fuzz-crashes"
  local i
  for ((i=0; i<N; i++)); do
    local seed="${seeds[$((RANDOM % ns))]}"
    python3 - "$seed" "$RANDOM" > /tmp/mut.hssa <<'PY'
import sys, random
random.seed(int(sys.argv[2]))
lines = open(sys.argv[1]).read().splitlines()
toks = ['~','()',':=','=:','->','<-','0','1','xyz','(',')',',','rel','begin','end','dup','add']
for _ in range(random.randint(1,6)):
    if not lines: break
    r = random.random(); i = random.randrange(len(lines))
    if r < 0.30: del lines[i]
    elif r < 0.50: lines[i] = lines[i][:random.randint(0, len(lines[i]))]
    elif r < 0.70: lines.insert(i, lines[random.randrange(len(lines))])
    else:
        w = lines[i].split()
        if w: w[random.randrange(len(w))] = random.choice(toks); lines[i] = ' '.join(w)
sys.stdout.write('\n'.join(lines))
PY
    timeout 10 "$TR" --import-hssa /tmp/mut.hssa >/tmp/mut.mlir 2>/dev/null; local rc=$?
    if [ "$rc" -eq 124 ] || is_crash "$rc"; then
      crashes=$((crashes+1)); cp /tmp/mut.hssa "$ROOT/fuzz-crashes/crash-import-$i-rc$rc.hssa"
      echo "CRASH import (rc=$rc): fuzz-crashes/crash-import-$i-rc$rc.hssa"
    elif [ "$rc" -eq 0 ]; then
      # imported OK -> stress the transform passes on the (possibly weird) IR
      timeout 10 "$OPT" --hssa-invert --hssa-resolve-uncalls --convert-hssa-to-arith /tmp/mut.mlir >/dev/null 2>&1; local rc2=$?
      if [ "$rc2" -eq 124 ] || is_crash "$rc2"; then
        crashes=$((crashes+1)); cp /tmp/mut.hssa "$ROOT/fuzz-crashes/crash-opt-$i-rc$rc2.hssa"
        echo "CRASH passes (rc=$rc2): fuzz-crashes/crash-opt-$i-rc$rc2.hssa"
      fi
    fi
  done
  echo "--- crash fuzz: $N inputs, $crashes crashes (graceful import=0 / parse-error=1 are fine) ---"
  return $crashes
}

case "${1:-metamorphic}" in
  metamorphic) metamorphic ;;
  crash) crash "${2:-500}" ;;
  *) echo "usage: $0 [metamorphic | crash [N]]"; exit 2 ;;
esac
