// RUN: hssa-opt %s > %t.once
// RUN: hssa-opt --hssa-invert --hssa-invert %s > %t.twice
// RUN: diff %t.once %t.twice

func.func @f(%p: i32, %q: i32, %x: i32) -> i32 attributes {hssa.params = 2 : i64} {
  // An involution tests that the a program / function is its own inverse
  // i.e. (-(-x)) = x or (x^q)^q = x. This tests that our inversion pass is correct
  // by ensuring a double-inversion results in the same function as we started with
  %y = hssa.add(%p) %x : i32
  %z = hssa.xor(%p) %y : i32
  return %z : i32
}
