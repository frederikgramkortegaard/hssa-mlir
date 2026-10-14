// f(p, q)(x) = (x + p) ^ q
func.func @f(%p: i32, %q: i32, %x: i32) -> i32 attributes {hssa.params = 2 : i64} {
  %y = hssa.add(%p) %x : i32
  %z = hssa.xor(%q) %y : i32
  return %z : i32
}

// Calls f forward (5 -> 14), then uncalls it (14 -> 5). Exit code: 5.
func.func @main() -> i32 {
  %p = arith.constant 3 : i32
  %q = arith.constant 6 : i32
  %x = arith.constant 5 : i32
  %y = hssa.call @f(%p, %q) %x : (i32, i32, i32) -> i32
  %x2 = hssa.call @f(%p, %q) %y {inverted} : (i32, i32, i32) -> i32
  return %x2 : i32
}
