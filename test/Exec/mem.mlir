// RUN: hssa-opt --hssa-resolve-uncalls --convert-hssa-to-arith %s | mlir-opt --convert-to-llvm | mlir-translate --mlir-to-llvmir | lli
// Store 42, read it back through the heap, free. Returns 0 iff memory round-trips.
func.func @main() -> i32 {
  %m = hssa.mmem.new : () -> !hssa.mem
  %sz = arith.constant 1 : i32
  %m1, %p = hssa.mmem.allocate(%sz) %m : (i32, !hssa.mem) -> (!hssa.mem, i32)
  %v = arith.constant 42 : i32
  %m2, %old = hssa.mmem.readwrite(%p) %m1, %v : (i32, !hssa.mem, i32) -> (!hssa.mem, i32)
  %u = hssa.unit : !hssa.unit
  %m3, %got = hssa.mmem.readwrite(%p) %m2, %u : (i32, !hssa.mem, !hssa.unit) -> (!hssa.mem, i32)
  hssa.mmem.new %m3 {inverted} : (!hssa.mem) -> ()
  %c42 = arith.constant 42 : i32
  %d = arith.subi %got, %c42 : i32
  return %d : i32
}
