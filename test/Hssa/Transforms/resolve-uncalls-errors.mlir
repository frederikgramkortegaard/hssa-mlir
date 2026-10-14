// RUN: hssa-opt --hssa-resolve-uncalls --split-input-file --verify-diagnostics %s

// An uncall of a symbol that does not exist is rejected by the verifier
// before any pass runs (the pass keeps its own guard as defense in depth).
func.func @f(%p: i32, %x: i32) -> i32 {
  // expected-error @+1 {{references undefined function 'nosuch'}}
  %y = hssa.call @nosuch(%p) %x {inverted} : (i32, i32) -> i32
  return %y : i32
}

// -----

// An uncall of a function whose body contains a non-invertible op
// (hssa.discard) must be rejected with a diagnostic.
func.func @leaky(%x: i32) attributes {hssa.params = 0 : i64} {
  hssa.discard %x : i32
  return
}
func.func @f() {
  // expected-error @+1 {{cannot uncall 'leaky': function is not invertible}}
  %r = hssa.call @leaky() {inverted} : () -> i32
  hssa.discard %r : i32
  return
}

// -----

// A constant defined in one block and used in another is not invertible: the
// rebuild only clones constants into their own block's inverse, so a
// cross-block use must fail cleanly in the pre-flight rather than crash the
// rewrite with a dangling operand.
func.func @xblock_const(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %c = arith.constant 7 : i32
  hssa.exit ^bb1(%x : i32)
^bb1(%y: i32):
  %z = hssa.add(%c) %y : i32
  return %z : i32
}
func.func @f(%p: i32, %x: i32) {
  // expected-error @+1 {{cannot uncall 'xblock_const': function is not invertible}}
  %r = hssa.call @xblock_const(%p) %x {inverted} : (i32, i32) -> i32
  hssa.discard %r : i32
  return
}
