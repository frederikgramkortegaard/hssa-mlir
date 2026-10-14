// RUN: hssa-opt --convert-hssa-to-arith --verify-diagnostics %s

// Inverted calls have no direct lowering: --hssa-resolve-uncalls must run
// first, and a leftover uncall makes legalization fail rather than being
// silently lowered as a forward call.
func.func private @g(i32, i32) -> i32 attributes {hssa.params = 1 : i64}
func.func @f(%p: i32, %x: i32) -> i32 {
  // expected-error @+1 {{failed to legalize operation 'hssa.call' that was explicitly marked illegal}}
  %y = hssa.call @g(%p) %x {inverted} : (i32, i32) -> i32
  return %y : i32
}
