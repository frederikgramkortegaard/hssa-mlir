// RUN: hssa-opt --hssa-resolve-uncalls %s | FileCheck %s

// A pre-existing, unrelated function already named @g_inverse must not be
// called by mistake: the generated inverse is uniquified to @g_inverse_0
// and the uncall is redirected there.

func.func @g(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %y = hssa.add(%p) %x : i32
  return %y : i32
}

// CHECK-LABEL: func @g_inverse(
// CHECK-NEXT: hssa.xor
// CHECK-NEXT: return
func.func @g_inverse(%x: i32) -> i32 {
  %y = hssa.xor(%x) %x : i32
  return %y : i32
}

// CHECK-LABEL: func @f(
// CHECK: hssa.call @g_inverse_0(
// CHECK-NOT: inverted
// CHECK: return
func.func @f(%p: i32, %x: i32) -> i32 {
  %y = hssa.call @g(%p) %x {inverted} : (i32, i32) -> i32
  return %y : i32
}

// CHECK-LABEL: func @g_inverse_0(
// CHECK: hssa.add({{.*}}) {{.*}} {inverted} : i32
