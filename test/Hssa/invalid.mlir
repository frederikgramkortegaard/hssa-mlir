// RUN: hssa-opt %s -split-input-file -verify-diagnostics


func.func @id_type_mismatch(%x: i32) {
  // expected-error @+1 {{result #0 must be HSSA value}} 
%r = "hssa.id"(%x) : (i32) -> i64
return
}


// -----
func.func @sub_missing_parameter(%in: i32) -> i32 {
  // expected-error @+1 {{expected 2 operands}}
  %out = "hssa.sub"(%in) : (i32) -> i32
  return %out : i32
}
func.func @sub_rejects_i64_input(%p: i32, %in: i64) -> i32 {
  // expected-error @+1 {{operand #1 must be HSSA integer}} 
  %out = "hssa.sub"(%p, %in) : (i32, i64) -> i32
  return %out : i32
}

// -----
func.func @xor_missing_parameter(%in: i32) -> i32 {
  // expected-error @+1 {{expected 2 operands}}
  %out = "hssa.xor"(%in) : (i32) -> i32
  return %out : i32
}
func.func @xor_rejects_i64_input(%p: i32, %in: i64) -> i32 {
  // expected-error @+1 {{operand #1 must be HSSA integer}} 
  %out = "hssa.xor"(%p, %in) : (i32, i64) -> i32
  return %out : i32
}

// -----
func.func @add_missing_parameter(%in: i32) -> i32 {
  // expected-error @+1 {{expected 2 operands}}
  %out = "hssa.add"(%in) : (i32) -> i32
  return %out : i32
}
func.func @add_rejects_i64_input(%p: i32, %in: i64) -> i32 {
  // expected-error @+1 {{operand #1 must be HSSA integer}}
  %out = "hssa.add"(%p, %in) : (i32, i64) -> i32
  return %out : i32
}

// -----
// The two type-equality rules in DupOp::verify() have no reject case here:
// dup is polymorphic over any HSSA value, so a type outside the value domain
// (e.g. i64) is rejected by the generated verifier ("must be HSSA value")
// before verify() ever runs; the type-equality checks need a well-typed op.
func.func @dup_forward_rejects_input(%x: i32, %c: i32) -> i32 {
  // expected-error @+1 {{forward dup cannot have an input}}
  %out = hssa.dup(%x) %c : (i32, i32) -> i32
  return %out : i32
}
func.func @dup_forward_requires_output(%x: i32) {
  // expected-error @+1 {{forward dup must produce an output}}
  hssa.dup(%x) : (i32) -> ()
  return
}
func.func @dup_inverted_rejects_output(%x: i32, %c: i32) -> i32 {
  // expected-error @+1 {{inverted dup cannot produce an output}}
  %out = hssa.dup(%x) %c {inverted} : (i32, i32) -> i32
  return %out : i32
}
func.func @dup_inverted_requires_input(%x: i32) {
  // expected-error @+1 {{inverted dup requires an input}}
  hssa.dup(%x) {inverted} : (i32) -> ()
  return
}

// -----
func.func @compute_forward_rejects_input(%a: i32, %b: i32, %x: i32) -> i32 {
  // expected-error @+1 {{forward compute op cannot have an input}}
  %m = hssa.mul(%a, %b) %x : (i32, i32, i32) -> i32
  return %m : i32
}
func.func @compute_inverted_rejects_output(%a: i32, %b: i32, %x: i32) -> i32 {
  // expected-error @+1 {{inverted compute op cannot produce an output}}
  %m = hssa.mul(%a, %b) %x {inverted} : (i32, i32, i32) -> i32
  return %m : i32
}
func.func @euclidiv_forward_arity(%d: i32, %x: i32) -> i32 {
  // expected-error @+1 {{forward euclidiv consumes one value and produces two}}
  %q = hssa.euclidiv(%d) %x : (i32, i32) -> i32
  return %q : i32
}
func.func @assert_inverted_rejects_input(%p: i32, %x: i32) {
  // expected-error @+1 {{inverted tests.assert cannot have an input}}
  hssa.tests.assert(%p) %x {inverted} : (i32, i32) -> ()
  return
}

// -----

// hssa.call checks operand/result types against the callee, not just counts
// (func.func is not constrained to i32).
func.func private @f64(%p: i64, %x: i64) -> i64 attributes {hssa.params = 1 : i64} {
  return %x : i64
}
func.func @call_type_mismatch(%p: i32, %x: i32) -> i32 {
  // expected-error @+1 {{parameter 0 type does not match 'f64'}}
  %y = hssa.call @f64(%p) %x : (i32, i32) -> i32
  return %y : i32
}

// -----

// A conditional exit has a single payload; its two successor lists may differ
// only in the trailing provenance constant.
func.func @cond_exit_divergent(%s: i32, %a: i32, %b: i32) {
  // expected-error @+1 {{must pass the same payload to both successors}}
  hssa.cond_exit %s, ^x(%a : i32), ^y(%b : i32)
^x(%u: i32):
  hssa.discard %u : i32
  return
^y(%v: i32):
  hssa.discard %v : i32
  return
}

// -----

// hssa.params is an i64 count bounded by the argument list; a value past 2^32
// must not wrap through a narrowing cast.
// expected-error @+1 {{'hssa.params' exceeds the number of function arguments}}
func.func private @params_overflow(%p: i32, %x: i32) -> i32
    attributes {hssa.params = 4294967296 : i64} {
  return %x : i32
}

// -----

// hssa.class is a function-level attribute.
func.func @class_on_op(%x: i32) -> i32 {
  // expected-error @+1 {{'hssa.class' only applies to functions}}
  %y = hssa.id %x {hssa.class = "(N,N)"} : i32
  return %y : i32
}

// -----

// hssa.reversible is a function-level attribute.
func.func @reversible_on_op(%x: i32) -> i32 {
  // expected-error @+1 {{'hssa.reversible' only applies to functions}}
  %y = hssa.id %x {hssa.reversible} : i32
  return %y : i32
}

// -----

// The parameter/input split is semantic; a callee must declare hssa.params so
// the call can be checked (and so the callee is invertible).
func.func private @no_frame(%a: i32, %b: i32) -> i32 {
  return %a : i32
}
func.func @call_without_frame(%p: i32, %x: i32) -> i32 {
  // expected-error @+1 {{callee 'no_frame' lacks the 'hssa.params' attribute}}
  %y = hssa.call @no_frame(%p) %x : (i32, i32) -> i32
  return %y : i32
}

// -----

// A forward mmem.new produces a heap and takes no input.
func.func @mem_new_forward_input(%m: !hssa.mem) {
  // expected-error @+1 {{forward mmem.new produces a heap and consumes none}}
  hssa.mmem.new %m : (!hssa.mem) -> ()
  return
}

// -----

// A forward mmem.allocate must produce a pointer.
func.func @mem_alloc_forward_no_ptr(%sz: i32, %m: !hssa.mem) -> !hssa.mem {
  // expected-error @+1 {{forward mmem.allocate produces a pointer and consumes none}}
  %r = hssa.mmem.allocate(%sz) %m : (i32, !hssa.mem) -> !hssa.mem
  return %r : !hssa.mem
}
