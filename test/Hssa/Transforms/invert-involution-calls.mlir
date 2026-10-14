// Whole-module --hssa-invert must be COHERENT across function calls. A single
// invert runs BOTH a caller and its callee backward: buildInverse flips each
// call's direction, but because the callee is inverted in place too, that flip
// has to be CANCELLED for internal calls -- a forward call to the now-inverted
// B is already the right inverse composition, and a source uncall ~B stays an
// uncall of the inverted B. Only calls to functions NOT inverted here keep the
// flip. The involution invert;invert must therefore restore the module exactly:
// call directions and bodies flip TOGETHER, not independently. Before the fix
// they flipped independently -- the twice-inverted caller's call no longer
// agreed with its callee's signature, the verifier rejected the module, and
// @main was lost on lowering.

// Involution: invert;invert reproduces the original module byte-for-byte,
// including @a's forward call to @b.
// RUN: hssa-opt %s > %t.once
// RUN: hssa-opt --hssa-invert --hssa-invert %s > %t.twice
// RUN: diff %t.once %t.twice

// A single whole-module invert stays coherent: it still verifies (every call
// agrees with its inverted callee's flipped signature).
// RUN: hssa-opt --hssa-invert %s | hssa-opt | FileCheck %s --check-prefix=INV

// The twice-inverted module lowers keeping @main and runs: @main calls @a
// forward then uncalls it, so it recovers its input and returns 0.
// RUN: hssa-opt --hssa-invert --hssa-invert %s | hssa-opt --hssa-resolve-uncalls --convert-hssa-to-arith | mlir-opt --convert-to-llvm | mlir-translate --mlir-to-llvmir | lli

// B: leaf relation, y = x + p.
func.func @b(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %y = hssa.add(%p) %x : i32
  return %y : i32
}

// A: calls B forward (an internal call); z = b(p, x) * ... just threads it.
func.func @a(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %y = hssa.call @b(%p) %x : (i32, i32) -> i32
  %z = hssa.xor(%p) %y : i32
  return %z : i32
}

// Driver: call @a forward then uncall it; the difference must be 0.
func.func @main() -> i32 {
  %p = arith.constant 3 : i32
  %x = arith.constant 5 : i32
  %y = hssa.call @a(%p) %x : (i32, i32) -> i32
  %x2 = hssa.call @a(%p) %y {inverted} : (i32, i32) -> i32
  %d = arith.subi %x2, %x : i32
  return %d : i32
}

// After one whole-module invert, @a is inverted in place and its internal call
// to the (also inverted) @b stays a plain FORWARD call -- the buildInverse flip
// was cancelled because @b is itself inverted here. An incoherent flip would
// leave {inverted} and fail verification against @b's flipped signature.
// INV-LABEL: func @a
// INV: hssa.call @b({{.*}}) {{.*}} : (i32, i32) -> i32
// INV-NOT: hssa.call @b({{.*}}) {{.*}} {inverted}
