// Semantic self-check via native execution: returns 0 iff every identity holds.
// RUN: hssa-opt --hssa-resolve-uncalls --convert-hssa-to-arith %s | mlir-opt --convert-to-llvm | mlir-translate --mlir-to-llvmir | lli
func.func @main() -> i32 {
  %c3 = arith.constant 3 : i32
  %c5 = arith.constant 5 : i32
  %c10 = arith.constant 10 : i32
  // add(3) 5 = 8, then add{inverted} recovers 5
  %a = hssa.add(%c3) %c5 : i32
  %ai = hssa.add(%c3) %a {inverted} : i32
  %d1 = arith.subi %ai, %c5 : i32
  // subfrom(10) is self-inverse: 10-3=7, 10-7=3
  %s = hssa.subfrom(%c10) %c3 : i32
  %si = hssa.subfrom(%c10) %s : i32
  %d2 = arith.subi %si, %c3 : i32
  // xor(5) self-inverse
  %x = hssa.xor(%c5) %c3 : i32
  %xi = hssa.xor(%c5) %x : i32
  %d3 = arith.subi %xi, %c3 : i32
  %e1 = arith.addi %d1, %d2 : i32
  %e2 = arith.addi %e1, %d3 : i32
  return %e2 : i32
}
