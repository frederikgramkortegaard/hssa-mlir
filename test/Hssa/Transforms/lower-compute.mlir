// RUN: hssa-opt --convert-hssa-to-arith %s | FileCheck %s --implicit-check-not=hssa.

// The BinArithLowering / CmpLowering instantiations not exercised by
// lower.mlir: div, mod, or, and, lessequal, greater, greaterequal, equal,
// notequal, plus the inverted check-and-consume form of a comparison.

// CHECK-LABEL: func @bin(
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32)
func.func @bin(%a: i32, %b: i32) -> (i32, i32, i32, i32) {
  // CHECK: %[[D:.*]] = arith.divsi %[[A]], %[[B]] : i32
  %d = hssa.div(%a, %b) : (i32, i32) -> i32
  // CHECK: %[[M:.*]] = arith.remsi %[[A]], %[[B]] : i32
  %m = hssa.mod(%a, %b) : (i32, i32) -> i32
  // CHECK: %[[O:.*]] = arith.ori %[[A]], %[[B]] : i32
  %o = hssa.or(%a, %b) : (i32, i32) -> i32
  // CHECK: %[[N:.*]] = arith.andi %[[A]], %[[B]] : i32
  %n = hssa.and(%a, %b) : (i32, i32) -> i32
  // CHECK: return %[[D]], %[[M]], %[[O]], %[[N]]
  return %d, %m, %o, %n : i32, i32, i32, i32
}

// CHECK-LABEL: func @cmp(
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32)
func.func @cmp(%a: i32, %b: i32) -> (i32, i32, i32, i32, i32) {
  // CHECK: %[[LE:.*]] = arith.cmpi sle, %[[A]], %[[B]] : i32
  // CHECK: arith.extui %[[LE]] : i1 to i32
  %le = hssa.lessequal(%a, %b) : (i32, i32) -> i32
  // CHECK: %[[GT:.*]] = arith.cmpi sgt, %[[A]], %[[B]] : i32
  // CHECK: arith.extui %[[GT]] : i1 to i32
  %gt = hssa.greater(%a, %b) : (i32, i32) -> i32
  // CHECK: %[[GE:.*]] = arith.cmpi sge, %[[A]], %[[B]] : i32
  // CHECK: arith.extui %[[GE]] : i1 to i32
  %ge = hssa.greaterequal(%a, %b) : (i32, i32) -> i32
  // CHECK: %[[EQ:.*]] = arith.cmpi eq, %[[A]], %[[B]] : i32
  // CHECK: arith.extui %[[EQ]] : i1 to i32
  %eq = hssa.equal(%a, %b) : (i32, i32) -> i32
  // CHECK: %[[NE:.*]] = arith.cmpi ne, %[[A]], %[[B]] : i32
  // CHECK: arith.extui %[[NE]] : i1 to i32
  %ne = hssa.notequal(%a, %b) : (i32, i32) -> i32
  return %le, %gt, %ge, %eq, %ne : i32, i32, i32, i32, i32
}

// An inverted comparison recomputes and check-and-consumes.
// CHECK-LABEL: func @less_inverted(
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32, %[[X:.*]]: i32)
func.func @less_inverted(%a: i32, %b: i32, %x: i32) {
  // CHECK: %[[BIT:.*]] = arith.cmpi slt, %[[A]], %[[B]] : i32
  // CHECK: %[[W:.*]] = arith.extui %[[BIT]] : i1 to i32
  // CHECK: %[[OK:.*]] = arith.cmpi eq, %[[X]], %[[W]] : i32
  // CHECK: cf.assert %[[OK]], "inverted compute op: value does not match"
  hssa.less(%a, %b) %x {inverted} : (i32, i32, i32) -> ()
  return
}
