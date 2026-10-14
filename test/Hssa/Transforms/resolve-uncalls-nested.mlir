// RUN: hssa-opt --hssa-resolve-uncalls %s | FileCheck %s

// Uncalling @h generates @h_inverse, whose body contains a fresh uncall of
// @g; the pass must resolve that one too (worklist fixpoint).

func.func @g(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %y = hssa.add(%p) %x : i32
  return %y : i32
}

func.func @h(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %y = hssa.call @g(%p) %x : (i32, i32) -> i32
  return %y : i32
}

// CHECK-LABEL: func @top(
// CHECK: hssa.call @h_inverse
// CHECK-NOT: inverted
// CHECK: return
func.func @top(%p: i32, %x: i32) -> i32 {
  %y = hssa.call @h(%p) %x {inverted} : (i32, i32) -> i32
  return %y : i32
}

// CHECK-LABEL: func @h_inverse(
// CHECK: hssa.call @g_inverse
// CHECK-NOT: {inverted} :

// CHECK-LABEL: func @g_inverse(
// CHECK: hssa.add({{.*}}) {{.*}} {inverted} : i32
