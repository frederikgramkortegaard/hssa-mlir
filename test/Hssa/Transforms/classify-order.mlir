// RUN: hssa-opt --hssa-verify-reversible %s | FileCheck %s

// The classification is a fixpoint: nondeterminism introduced two call
// levels down must reach the top even though every caller is defined (and
// visited) before its callee, which forces more than one iteration.

// CHECK-LABEL: func @top
// CHECK-SAME: hssa.class = "(D,N)"
func.func @top(%x: i32) {
  hssa.call @mid() %x : (i32) -> ()
  return
}

// CHECK-LABEL: func @mid
// CHECK-SAME: hssa.class = "(D,N)"
func.func @mid(%x: i32) attributes {hssa.params = 0 : i64} {
  hssa.call @bottom() %x : (i32) -> ()
  return
}

// CHECK-LABEL: func @bottom
// CHECK-SAME: hssa.class = "(D,N)"
func.func @bottom(%x: i32) attributes {hssa.params = 0 : i64} {
  hssa.discard %x : i32
  return
}

// Uncalling through a chain swaps directions exactly once at the uncall.
// CHECK-LABEL: func @untop
// CHECK-SAME: hssa.class = "(N,D)"
func.func @untop() -> i32 {
  %x = hssa.call @mid() {inverted} : () -> i32
  return %x : i32
}
