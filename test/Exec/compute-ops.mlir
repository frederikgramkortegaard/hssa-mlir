// RUN: hssa-opt --hssa-resolve-uncalls --convert-hssa-to-arith %s | mlir-opt --convert-to-llvm | mlir-translate --mlir-to-llvmir | lli
// mul value, euclidiv round-trip, comparison. Returns 0 iff all correct.
func.func @main() -> i32 {
  %c3 = arith.constant 3 : i32
  %c5 = arith.constant 5 : i32
  %c17 = arith.constant 17 : i32
  %m = hssa.mul(%c3, %c5) : (i32, i32) -> i32
  %c15 = arith.constant 15 : i32
  %d1 = arith.subi %m, %c15 : i32
  %q, %r = hssa.euclidiv(%c3) %c17 : (i32, i32) -> (i32, i32)
  %back = hssa.euclidiv(%c3) %q, %r {inverted} : (i32, i32, i32) -> i32
  %d2 = arith.subi %back, %c17 : i32
  %lt = hssa.less(%c3, %c5) : (i32, i32) -> i32
  %c1 = arith.constant 1 : i32
  %d3 = arith.subi %lt, %c1 : i32
  %e1 = arith.addi %d1, %d2 : i32
  %e2 = arith.addi %e1, %d3 : i32
  return %e2 : i32
}
