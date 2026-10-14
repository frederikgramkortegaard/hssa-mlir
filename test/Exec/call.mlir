// RUN: hssa-opt --hssa-resolve-uncalls --convert-hssa-to-arith %s | mlir-opt --convert-to-llvm | mlir-translate --mlir-to-llvmir | lli
// Call f forward then uncall it; must recover the original input. Returns 0 iff so.
func.func @f(%p: i32, %x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  %y = hssa.add(%p) %x : i32
  return %y : i32
}
func.func @main() -> i32 {
  %p = arith.constant 3 : i32
  %x = arith.constant 5 : i32
  %y = hssa.call @f(%p) %x : (i32, i32) -> i32
  %x2 = hssa.call @f(%p) %y {inverted} : (i32, i32) -> i32
  %d = arith.subi %x2, %x : i32
  return %d : i32
}
