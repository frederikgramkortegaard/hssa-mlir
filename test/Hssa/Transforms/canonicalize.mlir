// RUN: hssa-opt --canonicalize %s | FileCheck %s

// A self-inverse update applied twice with the same parameter is the identity,
// so the stock --canonicalize pass folds it away via the op's registered
// pattern (x ^ p ^ p == x). Both xors vanish, leaving the original value.
// CHECK-LABEL: func @double_xor
// CHECK-NOT: hssa.xor
// CHECK: return %arg1 : i32
func.func @double_xor(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %y = hssa.xor(%p) %x : i32
  %z = hssa.xor(%p) %y : i32
  return %z : i32
}

// subfrom is self-inverse too: p - (p - x) == x.
// CHECK-LABEL: func @double_subfrom
// CHECK-NOT: hssa.subfrom
// CHECK: return %arg1 : i32
func.func @double_subfrom(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %y = hssa.subfrom(%p) %x : i32
  %z = hssa.subfrom(%p) %y : i32
  return %z : i32
}

// Different parameters are NOT the identity, so nothing folds.
// CHECK-LABEL: func @diff_params
// CHECK: hssa.xor
// CHECK: hssa.xor
func.func @diff_params(%p: i32, %q: i32, %x: i32) -> i32 attributes {hssa.params = 2 : i64} {
  %y = hssa.xor(%p) %x : i32
  %z = hssa.xor(%q) %y : i32
  return %z : i32
}
