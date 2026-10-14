#!/usr/bin/env bash
# LIVE differential test: run programs through the reference interpreter AND our
# pipeline, assert identical results. The interpreter is the independent oracle
# -- this is the only check that validates our *understanding* of HSSA, not just
# internal consistency. Needs the toolkit jar (cd third_party/hssa-reference &&
# sbt assembly) and the LLVM tools.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF="${HSSA_REFERENCE:-$(cd "$(dirname "$0")/.." && pwd)/third_party/hssa-reference}"
JAR="$REF/target/scala-3.4.1/hssa.jar"
HSSA="$REF/programs/hssa"
L="${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}"
OPT="$ROOT/build/bin/hssa-opt"; TR="$ROOT/build/bin/hssa-translate"
[ -f "$JAR" ] || { echo "SKIP: reference jar not built. run: (cd $REF && sbt assembly)"; exit 0; }
fail=0
# A ()->int entry: our native result must equal the interpreter's.
valdiff() { # <file.hssa> <entry-relation>
  local file="$1" entry="$2" ref tmp
  ref=$(cd "$REF" && java -jar "$JAR" load "programs/hssa/$file", hssa.parse, hssa.check, hssa.exec "$entry" 2>/dev/null | grep -vE '^[[:space:]]*$' | tail -1)
  [[ "$ref" =~ ^-?[0-9]+$ ]] || { echo "SKIP $entry (ref result not a scalar int: '$ref')"; return; }
  tmp=$(mktemp /tmp/live.XXXXXX.mlir)
  "$TR" --import-hssa "$HSSA/$file" 2>/dev/null > "$tmp"
  python3 - "$tmp" "$entry" "$ref" <<'PY'
import sys; f,entry,ref=sys.argv[1:4]; s=open(f).read()
m='  func.func @main() -> i32 {\n    %%r = hssa.call @%s() : () -> i32\n    %%e = arith.constant %s : i32\n    %%d = arith.subi %%r, %%e : i32\n    return %%d : i32\n  }\n'%(entry,ref)
i=s.rfind('}'); open(f,'w').write(s[:i]+m+s[i:])
PY
  if "$OPT" --hssa-resolve-uncalls --canonicalize --convert-hssa-to-arith "$tmp" 2>/dev/null \
     | "$L/mlir-opt" --convert-to-llvm 2>/dev/null | "$L/mlir-translate" --mlir-to-llvmir 2>/dev/null | "$L/lli" 2>/dev/null
  then echo "ok   $entry  (ours == interpreter: $ref)"; else echo "FAIL $entry  (ours diverges from interpreter $ref)"; fail=1; fi
  rm -f "$tmp"
}
valdiff power.hssa power.main
exit $fail
