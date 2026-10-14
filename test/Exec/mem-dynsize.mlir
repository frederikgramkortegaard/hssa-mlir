// RUN: hssa-opt --hssa-resolve-uncalls --convert-hssa-to-arith %s | mlir-opt --convert-to-llvm | mlir-translate --mlir-to-llvmir | lli
//
// Dynamic heap sizing (mmem.new K): size the managed heap by a RUNTIME-computed
// K, allocate a block well past the default 1024, round-trip a value through it,
// free, and destroy the heap. Returns 0 iff memory round-trips on the
// dynamically-sized heap. This is the ADDITIVE extension beyond the reference
// (whose mmem.new is argument-less and fixed at 1024), so it has no interpreter
// oracle -- correctness is asserted by the reversible round-trip itself.
func.func @main() -> i32 {
  // K is computed at run time: 1000 + 1500 = 2500 cells (> the 1024 default).
  %a = arith.constant 1000 : i32
  %b = arith.constant 1500 : i32
  %k = arith.addi %a, %b : i32
  // A dynamically-allocated heap of K cells.
  %m = hssa.mmem.new(%k) : (i32) -> !hssa.mem
  // Allocate a 3-cell block and write to a cell near the top of the heap
  // (index only reachable because K > 1024), then read it back.
  %sz = arith.constant 3 : i32
  %m1, %p = hssa.mmem.allocate(%sz) %m : (i32, !hssa.mem) -> (!hssa.mem, i32)
  %c2 = arith.constant 2 : i32
  %hi = arith.addi %p, %c2 : i32
  %v = arith.constant 42 : i32
  %m2, %old = hssa.mmem.readwrite(%hi) %m1, %v : (i32, !hssa.mem, i32) -> (!hssa.mem, i32)
  // Take the value back out (write unit), leaving the cell zero-cleared so free
  // succeeds (the checked deallocator asserts the block is zero).
  %u = hssa.unit : !hssa.unit
  %m3, %got = hssa.mmem.readwrite(%hi) %m2, %u : (i32, !hssa.mem, !hssa.unit) -> (!hssa.mem, i32)
  // The other two cells of the block are already zero; free the block.
  %m4 = hssa.mmem.allocate(%sz) %m3, %p {inverted} : (i32, !hssa.mem, i32) -> !hssa.mem
  // Destroy the heap (K threads to the inverse as a read-only parameter).
  hssa.mmem.new(%k) %m4 {inverted} : (i32, !hssa.mem) -> ()
  // 0 iff the value round-tripped.
  %c42 = arith.constant 42 : i32
  %d = arith.subi %got, %c42 : i32
  return %d : i32
}
