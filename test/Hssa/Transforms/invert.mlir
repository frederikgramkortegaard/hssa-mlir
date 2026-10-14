// RUN: hssa-opt --hssa-invert %s | FileCheck %s

// CHECK-LABEL: func @f
// CHECK-SAME: (%[[P:.*]]: i32, %[[Q:.*]]: i32, %[[Z:.*]]: i32)
func.func @f(%p: i32, %q: i32, %x: i32) -> i32 attributes {hssa.params = 2 : i64} {
  // The inverse runs the forward ops bottom-up: the xor is undone first,
  // consuming the last argument (which now carries the forward result).
  // CHECK: %[[Y:.*]] = hssa.xor(%[[Q]]) %[[Z]] {inverted} : i32
  // CHECK: %[[X:.*]] = hssa.add(%[[P]]) %[[Y]] {inverted} : i32
  // CHECK: return %[[X]] : i32
  %y = hssa.add(%p) %x : i32
  %z = hssa.xor(%q) %y : i32
  return %z : i32
}

// id is self-inverse: it flips position but never carries {inverted}.
// CHECK-LABEL: func @id_chain
// CHECK-SAME: (%[[P:.*]]: i32, %[[IN:.*]]: i32)
func.func @id_chain(%p: i32, %in: i32) -> i32 attributes {hssa.params = 1 : i64} {
  // CHECK: %[[Y:.*]] = hssa.id %[[IN]] : i32
  // CHECK-NOT: inverted
  // CHECK: %[[X:.*]] = hssa.add(%[[P]]) %[[Y]] {inverted} : i32
  // CHECK: return %[[X]] : i32
  %y = hssa.add(%p) %in : i32
  %z = hssa.id %y : i32
  return %z : i32
}

// Multi-input/multi-output: hssa.params marks the parameter prefix; the
// inverse's arguments are (params, old results) and it returns the old inputs.
// CHECK-LABEL: func @multi(
// CHECK-SAME: %[[P:.*]]: i32, %[[A:.*]]: i32, %[[B:.*]]: i32)
func.func @multi(%p: i32, %x: i32, %y: i32) -> (i32, i32) attributes {hssa.params = 1} {
  // CHECK: %[[Y:.*]] = hssa.sub(%[[P]]) %[[B]] {inverted} : i32
  // CHECK: %[[X:.*]] = hssa.add(%[[P]]) %[[A]] {inverted} : i32
  // CHECK: return %[[X]], %[[Y]] : i32, i32
  %a = hssa.add(%p) %x : i32
  %b = hssa.sub(%p) %y : i32
  return %a, %b : i32, i32
}
