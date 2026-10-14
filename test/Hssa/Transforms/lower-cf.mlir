// RUN: hssa-opt --convert-hssa-to-arith %s | FileCheck %s

// hssa.exit lowers to an unconditional branch carrying the payload.
// CHECK-LABEL: func @straight(
// CHECK-SAME: %[[X:.*]]: i32)
func.func @straight(%x: i32) -> i32 {
  // CHECK: cf.br ^[[NEXT:bb[0-9]+]](%[[X]] : i32)
  hssa.exit ^next(%x : i32)
^next(%v: i32):
  // CHECK: ^[[NEXT]](%[[V:.*]]: i32):
  // CHECK: return %[[V]] : i32
  return %v : i32
}

// Direction regression: selector 0 picks $first, so after lowering the
// TRUTHY successor of cf.cond_br must be $second. Both edges carry the same
// payload (HSSA's conditional exit has a single payload list), so the
// successors are told apart by their bodies: $first has the add, $second
// just returns. Swapping the successors would flip control flow.
// CHECK-LABEL: func @cond(
// CHECK-SAME: %[[S:.*]]: i32, %[[V:.*]]: i32)
func.func @cond(%s: i32, %v: i32) -> i32 {
  // CHECK: %[[C:.*]] = arith.trunci %[[S]] : i32 to i1
  // CHECK: cf.cond_br %[[C]], ^[[SECOND:bb[0-9]+]](%[[V]] : i32), ^[[FIRST:bb[0-9]+]](%[[V]] : i32)
  hssa.cond_exit %s, ^first(%v : i32), ^second(%v : i32)
// CHECK: ^[[FIRST]](%[[X:.*]]: i32):
// CHECK: arith.addi %[[X]], %[[X]]
^first(%x: i32):
  %p = hssa.add(%x) %x : i32
  return %p : i32
// CHECK: ^[[SECOND]](%[[Y:.*]]: i32):
// CHECK: return %[[Y]]
^second(%y: i32):
  return %y : i32
}
