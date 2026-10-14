#!/usr/bin/env bash
# LIVE differential test for OBJECT programs whose `main` returns the heap (not a
# scalar), which the scalar difftest-live cannot handle. We run the reference
# interpreter on `main`, read its final-heap dump (slot => value), and assert our
# native run leaves the SAME integer in every reported slot. The interpreter is
# the independent oracle: this validates that a real OO program COMPUTES the
# right answer natively, not merely that it compiles or that its own reversible
# checks pass.
#
# Slot 0 (our bump cursor) and any non-integer slot (a stored relation, printed
# `<rel ...>`, which we lower to a dispatch index) are skipped. Reliable only
# because the heap is zero-initialized (mmem.new allocates a memref<2x?xi32>
# arena and zeroes both rows), so a never-written cell reads 0 deterministically
# -- matching the interpreter's fresh heap rather than malloc garbage.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF="${HSSA_REFERENCE:-$(cd "$(dirname "$0")/.." && pwd)/third_party/hssa-reference}"
PROGS="$REF/programs/roopl/verified"
L="${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}"
OPT="$ROOT/build/bin/hssa-opt"; TR="$ROOT/build/bin/hssa-translate"
[ -f "$REF/target/scala-3.4.1/hssa.jar" ] || { echo "SKIP: reference jar not built. run: (cd $REF && sbt assembly)"; exit 0; }
JAR="$REF/target/scala-3.4.1/hssa.jar"
fail=0

heapmatch() { # <Program>
  local prog="$1" file="$PROGS/$1.hssa" slots tmp
  [ -f "$file" ] || { echo "SKIP $prog (no source)"; return; }
  # Interpreter heap dump: "<slot> => <value>". Keep integer values at slot >= 1.
  slots=$(cd "$REF" && java -jar "$JAR" load "programs/roopl/verified/$1.hssa", hssa.parse, hssa.check, hssa.exec main 2>/dev/null \
          | awk '/=>/ {gsub(/[[:space:]]/,"",$1); if ($1>=1 && $3 ~ /^-?[0-9]+$/) printf "%s:%s ", $1, $3}')
  [ -n "$slots" ] || { echo "SKIP $prog (no integer heap slots)"; return; }
  tmp=$(mktemp /tmp/oo.XXXXXX.mlir)
  "$TR" --import-hssa "$file" 2>/dev/null \
    | "$OPT" --hssa-resolve-uncalls 2>/dev/null \
    | "$OPT" --canonicalize 2>/dev/null \
    | "$OPT" --convert-hssa-to-arith 2>/dev/null > "$tmp" || { echo "FAIL $prog (lowering)"; fail=1; rm -f "$tmp"; return; }
  # Rename the heap-returning main; append a scalar @main that loads each
  # compared slot and returns the summed difference (0 iff every slot matches).
  python3 - "$tmp" "$slots" <<'PY'
import sys
f=sys.argv[1]; pairs=sys.argv[2].split()
s=open(f).read().replace('@main(','@prog(').replace('@main)','@prog)')
# The heap handle is a single memref<2x?xi32>: row 0 holds cell values. Load the
# compared slots from row 0 (column = slot).
L=['  func.func @main() -> i32 {','    %h = call @prog() : () -> memref<2x?xi32>','    %row0 = arith.constant 0 : index','    %acc0 = arith.constant 0 : i32']
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
  if "$L/mlir-opt" --convert-to-llvm "$tmp" 2>/dev/null \
     | "$L/mlir-translate" --mlir-to-llvmir 2>/dev/null | "$L/lli" 2>/dev/null
  then echo "ok   $prog.main  (native heap == interpreter: $slots)"
  else echo "FAIL $prog.main  (native heap diverges from interpreter: $slots)"; fail=1; fi
  rm -f "$tmp"
}

for p in NestedExpression FibPair Conditional Loop LocalBlock \
         BinaryTree LinkedList DoublyLinkedList RTM; do heapmatch "$p"; done
exit $fail
