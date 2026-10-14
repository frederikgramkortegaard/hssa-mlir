// RUN: hssa-opt --hssa-verify-reversible %s | FileCheck %s

// CHECK-LABEL: func @pure
// CHECK-SAME: hssa.class = "(D,D)"
func.func @pure(%p: i32, %x: i32) -> i32 {
  %y = hssa.add(%p) %x : i32
  return %y : i32
}

// discard introduces backwards-nondeterminism.
// CHECK-LABEL: func @leaky
// CHECK-SAME: hssa.class = "(D,N)"
func.func @leaky(%x: i32) attributes {hssa.params = 0 : i64} {
  hssa.discard %x : i32
  return
}

// Calling a (D,N) function makes the caller (D,N).
// CHECK-LABEL: func @caller
// CHECK-SAME: hssa.class = "(D,N)"
func.func @caller(%x: i32) {
  hssa.call @leaky() %x : (i32) -> ()
  return
}

// Uncalling swaps the directions: uncall of (D,N) is (N,D).
// CHECK-LABEL: func @uncaller
// CHECK-SAME: hssa.class = "(N,D)"
func.func @uncaller() -> i32 {
  %x = hssa.call @leaky() {inverted} : () -> i32
  return %x : i32
}

// Both together: (N,N).
// CHECK-LABEL: func @manytomany
// CHECK-SAME: hssa.class = "(N,N)"
func.func @manytomany(%x: i32) -> i32 {
  hssa.call @leaky() %x : (i32) -> ()
  %y = hssa.call @leaky() {inverted} : () -> i32
  return %y : i32
}
