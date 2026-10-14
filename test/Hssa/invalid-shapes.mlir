// RUN: hssa-opt %s -split-input-file -verify-diagnostics

// Verifier branches not exercised by invalid.mlir: the inverted euclidiv
// arity, the three remaining tests.assert direction rules, and the two
// remaining compute-op direction rules.

func.func @euclidiv_inverted_arity(%d: i32, %q: i32) -> i32 {
  // expected-error @+1 {{inverted euclidiv consumes two values and produces one}}
  %x = hssa.euclidiv(%d) %q {inverted} : (i32, i32) -> i32
  return %x : i32
}

// -----
func.func @assert_forward_requires_input(%p: i32) {
  // expected-error @+1 {{forward tests.assert requires an input}}
  hssa.tests.assert(%p) : (i32) -> ()
  return
}
func.func @assert_forward_rejects_output(%p: i32, %x: i32) -> i32 {
  // expected-error @+1 {{forward tests.assert cannot produce an output}}
  %r = hssa.tests.assert(%p) %x : (i32, i32) -> i32
  return %r : i32
}
func.func @assert_inverted_requires_output(%p: i32) {
  // expected-error @+1 {{inverted tests.assert must produce an output}}
  hssa.tests.assert(%p) {inverted} : (i32) -> ()
  return
}

// -----
func.func @compute_forward_requires_output(%a: i32, %b: i32) {
  // expected-error @+1 {{forward compute op must produce an output}}
  hssa.mul(%a, %b) : (i32, i32) -> ()
  return
}
func.func @compute_inverted_requires_input(%a: i32, %b: i32) -> i32 {
  // expected-error @+1 {{inverted compute op requires an input}}
  %r = hssa.mul(%a, %b) {inverted} : (i32, i32) -> i32
  return %r : i32
}
