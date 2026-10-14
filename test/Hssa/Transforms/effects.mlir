// Side-effect / trait modeling. The dialect declares memory effects so stock
// MLIR passes reason soundly about reversibility:
//   * trapping, result-less ops (inverted dup, forward tests.assert, tests.fail)
//     must NOT be deleted by DCE/canonicalize;
//   * linear producers (forward dup, compute ops) must NOT be CSE-merged,
//     because aliasing one linear value into two consumers breaks reversibility.
//
// RUN: hssa-opt -canonicalize %s | FileCheck %s --check-prefix=CANON
// RUN: hssa-opt -cse %s | FileCheck %s --check-prefix=CSE

//===----------------------------------------------------------------------===//
// (a) Trapping ops survive a DCE-ish pass. None produces a result, yet each
// declares a write to the abstract control/trap resource, so canonicalize keeps
// all three.
//===----------------------------------------------------------------------===//

// CANON-LABEL: func @traps_survive
func.func @traps_survive(%p: i32, %x: i32) attributes {hssa.params = 1 : i64} {
  // CANON: hssa.dup(%{{.*}}) %{{.*}} {inverted}
  hssa.dup(%p) %x {inverted} : (i32, i32) -> ()
  // CANON: hssa.tests.fail
  hssa.tests.fail
  // CANON: hssa.tests.assert(%{{.*}}) %{{.*}}
  hssa.tests.assert(%p) %x : (i32, i32) -> ()
  // CANON: return
  return
}

//===----------------------------------------------------------------------===//
// (b) CSE does NOT merge two independent forward dups of the same value: that
// would make a single linear value feed two consumers (consumed twice ->
// irreversible). Two distinct dup ops must remain.
//===----------------------------------------------------------------------===//

// CSE-LABEL: func @dup_not_cse_merged
func.func @dup_not_cse_merged(%p: i32) -> (i32, i32) attributes {hssa.params = 1 : i64} {
  // Two separate results: if CSE had merged them, both returns would name %0.
  // CSE: %[[A:.*]] = hssa.dup(%{{.*}})
  // CSE: %[[B:.*]] = hssa.dup(%{{.*}})
  // CSE: return %[[A]], %[[B]]
  %a = hssa.dup(%p) : (i32) -> i32
  %b = hssa.dup(%p) : (i32) -> i32
  return %a, %b : i32, i32
}

// Same guarantee for a compute op (fresh linear value, must not be shared).
// CSE-LABEL: func @compute_not_cse_merged
func.func @compute_not_cse_merged(%lhs: i32, %rhs: i32) -> (i32, i32) attributes {hssa.params = 2 : i64} {
  // CSE: %[[C:.*]] = hssa.mul(%{{.*}}, %{{.*}})
  // CSE: %[[D:.*]] = hssa.mul(%{{.*}}, %{{.*}})
  // CSE: return %[[C]], %[[D]]
  %c = hssa.mul(%lhs, %rhs) : (i32, i32) -> i32
  %d = hssa.mul(%lhs, %rhs) : (i32, i32) -> i32
  return %c, %d : i32, i32
}

//===----------------------------------------------------------------------===//
// (c) Heap ops read+write the abstract heap resource, so CSE never merges two
// identical readwrites and canonicalize never drops a heap op whose produced
// heap is dead.
//===----------------------------------------------------------------------===//

// CSE-LABEL: func @heap_not_cse_merged
func.func @heap_not_cse_merged(%m: !hssa.mem, %p: i32) -> (!hssa.mem, !hssa.mem) {
  // CSE: hssa.mmem.readwrite(%{{.*}}) %{{.*}}
  // CSE: hssa.mmem.readwrite(%{{.*}}) %{{.*}}
  %m1, %old1 = hssa.mmem.readwrite(%p) %m, %p : (i32, !hssa.mem, i32) -> (!hssa.mem, i32)
  %m2, %old2 = hssa.mmem.readwrite(%p) %m, %p : (i32, !hssa.mem, i32) -> (!hssa.mem, i32)
  return %m1, %m2 : !hssa.mem, !hssa.mem
}

// CANON-LABEL: func @heap_op_not_dce
func.func @heap_op_not_dce(%p: i32) {
  // Even with every produced value unused, the heap op stays (it mutates the
  // abstract heap). mmem.new's heap and the readwrite are both kept.
  // CANON: hssa.mmem.new
  // CANON: hssa.mmem.readwrite
  %m = hssa.mmem.new : () -> !hssa.mem
  %m1, %old = hssa.mmem.readwrite(%p) %m, %p : (i32, !hssa.mem, i32) -> (!hssa.mem, i32)
  return
}
