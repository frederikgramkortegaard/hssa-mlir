// RUN: hssa-opt --hssa-invert %s | FileCheck %s

// Inversion must flip each dup to its opposite shape (forward <-> inverted),
// not just toggle a flag.

// CHECK-LABEL: func @f(
// CHECK-SAME: %[[X:.*]]: i32, %[[IN:.*]]: i32)
func.func @f(%x: i32, %in: i32) -> i32 attributes {hssa.params = 1 : i64} {
  // CHECK: %[[C:.*]] = hssa.dup(%[[X]]) : (i32) -> i32
  // CHECK: hssa.dup(%[[X]]) %[[C]] {inverted} : (i32, i32) -> ()
  // CHECK: %[[Y:.*]] = hssa.add(%[[X]]) %[[IN]] {inverted} : i32
  // CHECK: return %[[Y]] : i32
  %y = hssa.add(%x) %in : i32
  %c = hssa.dup(%x) : (i32) -> i32
  hssa.dup(%x) %c {inverted} : (i32, i32) -> ()
  return %y : i32
}
