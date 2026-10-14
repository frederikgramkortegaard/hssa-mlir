// RUN: hssa-opt --hssa-resolve-uncalls %s | FileCheck %s

// CHECK-LABEL: func @g(
// CHECK: hssa.add
func.func @g(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %y = hssa.add(%p) %x : i32
  return %y : i32
}

// The uncall is rewritten to a forward call of the generated inverse.
// CHECK-LABEL: func @f(
// CHECK-SAME: %[[P:.*]]: i32, %[[X:.*]]: i32)
func.func @f(%p: i32, %x: i32) -> i32 {
  // CHECK: %[[R:.*]] = hssa.call @g_inverse(%[[P]]) %[[X]] : (i32, i32) -> i32
  // CHECK-NOT: inverted
  // CHECK: return %[[R]] : i32
  %y = hssa.call @g(%p) %x {inverted} : (i32, i32) -> i32
  return %y : i32
}

// CHECK-LABEL: func @g_inverse(
// CHECK-SAME: %[[P:.*]]: i32, %[[X:.*]]: i32)
// CHECK: %[[R:.*]] = hssa.add(%[[P]]) %[[X]] {inverted} : i32
// CHECK: return %[[R]] : i32
