// The repair pass is the write-back dual of --hssa-verify-linear: it restores
// the one-use invariant that standard optimizations break. A value consumed
// more than once is dup'd per extra consumer; a value never consumed is
// discarded. After repair, --hssa-verify-linear accepts the result.
//
// RUN: hssa-opt --hssa-repair-linear %s | FileCheck %s
// RUN: hssa-opt --hssa-repair-linear %s | hssa-opt --hssa-verify-linear

// A produced value with no consumer is a leak -> discard it.
// CHECK-LABEL: func @leak
// CHECK: %[[X:.*]] = hssa.add
// CHECK: hssa.discard %[[X]]
func.func @leak(%p: i32) attributes {hssa.params = 1 : i64} {
  %c = arith.constant 0 : i32
  %x = hssa.add(%p) %c : i32
  return
}

// A value consumed by two ops is fan-out -> one consumer keeps it, the other
// gets a fresh dup'd copy.
// CHECK-LABEL: func @fanout
// CHECK: %[[X:.*]] = hssa.add(%arg0)
// CHECK: %[[COPY:.*]] = hssa.dup(%[[X]]) : (i32) -> i32
// CHECK-DAG: hssa.add(%{{.*}}) %[[X]]
// CHECK-DAG: hssa.add(%{{.*}}) %[[COPY]]
func.func @fanout(%p: i32) attributes {hssa.params = 1 : i64} {
  %c = arith.constant 0 : i32
  %x = hssa.add(%p) %c : i32
  %y = hssa.add(%c) %x : i32
  %z = hssa.add(%c) %x : i32
  hssa.discard %y : i32
  hssa.discard %z : i32
  return
}

// Already linear -> unchanged (no dup, no discard inserted).
// CHECK-LABEL: func @ok
// CHECK-NOT: hssa.dup
// CHECK-NOT: hssa.discard
func.func @ok(%p: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %c = arith.constant 0 : i32
  %x = hssa.add(%p) %c : i32
  return %x : i32
}
