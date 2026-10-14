// RUN: hssa-opt --hssa-verify-linear --split-input-file --verify-diagnostics %s

// A statement's parameter must survive reverse execution, so it cannot be
// one of the values the same statement consumes (Gail & Meyer 2024, p. 170).
func.func @alias(%p: i32, %x: i32) -> i32 {
  // expected-error @+1 {{uses the same value as parameter and input}}
  %y = hssa.add(%x) %x : i32
  return %y : i32
}

// -----

// Consuming a value twice: the second hand-off has nothing left to take.
func.func @double_consume(%p: i32, %x: i32) {
  %y = hssa.add(%p) %x : i32
  // expected-error @+1 {{consumes a value that is not live in this block}}
  %z = hssa.add(%p) %x : i32
  hssa.discard %y : i32
  hssa.discard %z : i32
  return
}

// -----

// Same thing inside a single statement; operands are counted one at a time.
func.func @double_consume_one_op(%p: i32, %x: i32) -> (i32, i32) {
  %y = hssa.add(%p) %x : i32
  // expected-error @+1 {{consumes a value that is not live in this block}}
  return %y, %y : i32, i32
}

// -----

// Liveness is block-local: an entry argument is dead in every other block.
func.func @cross_block(%p: i32, %x: i32) {
  hssa.discard %x : i32
  hssa.exit ^next
^next:
  // expected-error @+1 {{consumes a value that is not live in this block}}
  hssa.discard %x : i32
  return
}

// -----

// Reading a value after it was consumed. Fig. 8 as printed does not check
// parameter liveness (name scoping covers it in the textual language);
// SSA values have no scoping, so the pass checks it explicitly.
func.func @read_after_consume(%p: i32, %x: i32) {
  %c = hssa.dup(%x) : (i32) -> i32
  hssa.discard %x : i32
  // expected-error @+1 {{reads a value that is not live in this block}}
  %y = hssa.add(%x) %c : i32
  hssa.discard %y : i32
  return
}

// -----

// Every created value must be consumed before the block ends.
func.func @leak(%p: i32, %x: i32) {
  // expected-error @+1 {{produces a value that is never consumed}}
  %y = hssa.add(%p) %x : i32
  return
}

// -----

// Unconsumed inputs are the same error class ("return call f(x)" forgets x).
// expected-error @+1 {{block argument #1 is never consumed}}
func.func @arg_leak(%p: i32, %x: i32) {
  return
}

// -----

// Linearity is only defined over HSSA statements.
func.func @foreign(%p: i32, %x: i32) -> i32 {
  // expected-error @+1 {{is not an HSSA statement; linearity is undefined}}
  %y = arith.addi %p, %x : i32
  return %y : i32
}

// -----

// A relation's parameters (the leading hssa.params entry args) are read-only:
// consuming one -- here %k appears as the add's consumed input -- makes the
// relation non-invertible, so verify-linear rejects it up front with a pointer
// to the offending statement and parameter.
func.func @consume_param(%k: i32, %x: i32) -> i32 attributes {hssa.params = 2 : i64} {
  // expected-error @+1 {{consumes read-only parameter #0 of the relation; a parameter may only be read, never consumed}}
  %y = hssa.add(%x) %k : i32
  return %y : i32
}
