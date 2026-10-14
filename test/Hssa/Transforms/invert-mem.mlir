// RUN: hssa-opt --hssa-invert %s | FileCheck %s
// RUN: hssa-opt --hssa-verify-linear %s
// RUN: hssa-opt --hssa-invert --hssa-invert %s | FileCheck %s --check-prefix=INVOL
// RUN: hssa-opt --hssa-invert %s | FileCheck %s --check-prefix=RW
// RUN: hssa-opt --hssa-verify-linear %s

// Inverting a heap-building function flips new<->~new and allocate<->~allocate
// and reverses their order; the heap token threads through untouched. No
// custom buildInverse is needed -- the generic shape handles it.
// CHECK-LABEL: func @mk(
// CHECK-SAME: %[[SZ:.*]]: i32, %[[M:.*]]: !hssa.mem, %[[P:.*]]: i32)
func.func @mk(%sz: i32) -> (!hssa.mem, i32) attributes {hssa.params = 1 : i64} {
  // CHECK: %[[M0:.*]] = hssa.mmem.allocate(%[[SZ]]) %[[M]], %[[P]] {inverted} : (i32, !hssa.mem, i32) -> !hssa.mem
  // CHECK: hssa.mmem.new %[[M0]] {inverted} : (!hssa.mem) -> ()
  // CHECK: return
  %m = hssa.mmem.new : () -> !hssa.mem
  %m2, %p = hssa.mmem.allocate(%sz) %m : (i32, !hssa.mem) -> (!hssa.mem, i32)
  return %m2, %p : !hssa.mem, i32
}

// Inverting twice restores the forward program.
// INVOL-LABEL: func @mk(
// INVOL: hssa.mmem.new : () -> !hssa.mem
// INVOL: hssa.mmem.allocate(%{{.*}}) %{{.*}} : (i32, !hssa.mem) -> (!hssa.mem, i32)

// -----

// readwrite is self-inverse: inverting a function that swaps a cell yields a
// readwrite that swaps it back. Linearity accepts the heap + cell threading.
// RW-LABEL: func @swap(
// RW: hssa.mmem.readwrite(%{{.*}}) %{{.*}}, %{{.*}} : (i32, !hssa.mem, i32) -> (!hssa.mem, i32)
func.func @swap(%ptr: i32, %m: !hssa.mem, %v: i32) -> (!hssa.mem, i32)
    attributes {hssa.params = 1 : i64} {
  %m2, %old = hssa.mmem.readwrite(%ptr) %m, %v : (i32, !hssa.mem, i32) -> (!hssa.mem, i32)
  return %m2, %old : !hssa.mem, i32
}
