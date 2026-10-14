// RUN: hssa-opt --hssa-invert %s | FileCheck %s

// --hssa-invert silently skips functions it cannot invert, leaving them
// byte-for-byte intact, while still inverting the rest of the module.

// CHECK-LABEL: func @leaky(
// CHECK-SAME: %[[X:.*]]: i32) -> i32
// CHECK: %[[C:.*]] = arith.constant 0 : i32
// CHECK: hssa.discard %[[X]] : i32
// CHECK: return %[[C]] : i32
func.func @leaky(%x: i32) -> i32 attributes {hssa.params = 0 : i64} {
  %c = arith.constant 0 : i32
  hssa.discard %x : i32
  return %c : i32
}

// CHECK-LABEL: func @ok(
// CHECK-SAME: %[[P:.*]]: i32, %[[Y:.*]]: i32)
// CHECK: %[[R:.*]] = hssa.add(%[[P]]) %[[Y]] {inverted} : i32
// CHECK: return %[[R]] : i32
func.func @ok(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %y = hssa.add(%p) %x : i32
  return %y : i32
}
