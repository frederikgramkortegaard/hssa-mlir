#!/usr/bin/env bash
# Differential test against REFERENCE-AUTHORED expectations.
# The reference .test programs encode expected results as literal-output
# assertions -- e.g. "8 := power 2 =: 3" asserts power(2,3) == 8, written by
# the reference team. We import each, drive its .test relation from a main,
# lower to native, and run: exit 0 means our semantics agree with those
# expectations. No live interpreter needed. (For the live-interpreter diff,
# see difftest-live.sh once the toolkit jar is built.)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REFROOT="$(cd "$(dirname "$0")/.." && pwd)/third_party/hssa-reference"
HSSA="${HSSA_CORPUS:-$REFROOT/programs/hssa}"
L="${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}"
OPT="$ROOT/build/bin/hssa-opt"; TR="$ROOT/build/bin/hssa-translate"
fail=0
check() { # <file> <test-relation>
  local file="$HSSA/$1" rel="$2" tmp; tmp="$(mktemp /tmp/diff.XXXXXX.mlir)"
  if ! "$TR" --import-hssa "$file" 2>/dev/null >"$tmp"; then echo "FAIL $2 (import)"; fail=1; return; fi
  python3 - "$tmp" "$rel" <<'PY'
import sys; tmp,rel=sys.argv[1:3]; s=open(tmp).read()
m='  func.func @main() -> i32 {\n    hssa.call @%s() : () -> ()\n    %%z = arith.constant 0 : i32\n    return %%z : i32\n  }\n'%rel
i=s.rfind('}'); open(tmp,'w').write(s[:i]+m+s[i:])
PY
  if "$OPT" --hssa-resolve-uncalls --canonicalize --convert-hssa-to-arith "$tmp" 2>/dev/null \
     | "$L/mlir-opt" --convert-to-llvm 2>/dev/null \
     | "$L/mlir-translate" --mlir-to-llvmir 2>/dev/null | "$L/lli" 2>/dev/null
  then echo "ok   $2"; else echo "FAIL $2 (reference assertions diverge)"; fail=1; fi
  rm -f "$tmp"
}
check power.hssa power.test
check fibpair.hssa fibpair.test
exit $fail
