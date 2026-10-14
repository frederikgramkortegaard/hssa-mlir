// RUN: hssa-opt --hssa-verify-linear %s | FileCheck %s
// The whole reference corpus is linear, including programs that discard:
// discard *consumes* its value, so information loss and linearity are
// orthogonal (collatz is (D,N) yet linear).
// RUN: hssa-translate --import-hssa %S/../Import/Inputs/fibpair.hssa | hssa-opt --hssa-verify-linear
// RUN: hssa-translate --import-hssa %S/../Import/Inputs/collatz.hssa | hssa-opt --hssa-verify-linear

// CHECK-LABEL: func @straight
func.func @straight(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %y = hssa.add(%p) %x : i32
  return %y : i32
}

// A function parameter may be read anywhere and also handed out at the end
// (countdown returns its parameter).
// CHECK-LABEL: func @param_through
func.func @param_through(%p: i32, %x: i32) -> (i32, i32) attributes {hssa.params = 1 : i64} {
  %y = hssa.add(%p) %x : i32
  return %y, %p : i32, i32
}

// Reading the same parameter twice is fine; parameters are non-linear.
// CHECK-LABEL: func @param_twice
func.func @param_twice(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %y = hssa.add(%p) %x : i32
  %z = hssa.add(%p) %y : i32
  return %z : i32
}

// Control flow: the conditional exit consumes its selector and hands the
// payload off once even though it rides both edge lists; the join's
// trailing provenance argument is consumed like any other value.
// CHECK-LABEL: func @diamond
func.func @diamond(%p: i32, %s: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  hssa.cond_exit %s, ^join(%x, %c0 : i32, i32), ^join(%x, %c1 : i32, i32)
^join(%v: i32, %bit: i32):
  hssa.discard %bit : i32
  return %v : i32
}
