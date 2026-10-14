// RUN: hssa-opt --convert-hssa-to-arith %s | FileCheck %s --implicit-check-not=hssa.

// CHECK-LABEL: func @add(
// CHECK-SAME: %[[P:.*]]: i32, %[[X:.*]]: i32)
func.func @add(%p: i32, %x: i32) -> i32 {
  // CHECK: %[[R:.*]] = arith.addi %[[X]], %[[P]] : i32
  // CHECK: return %[[R]] : i32
  %y = hssa.add(%p) %x : i32
  return %y : i32
}

// CHECK-LABEL: func @add_inverted(
// CHECK-SAME: %[[P:.*]]: i32, %[[X:.*]]: i32)
func.func @add_inverted(%p: i32, %x: i32) -> i32 {
  // CHECK: %[[R:.*]] = arith.subi %[[X]], %[[P]] : i32
  // CHECK: return %[[R]] : i32
  %y = hssa.add(%p) %x {inverted} : i32
  return %y : i32
}

// CHECK-LABEL: func @sub(
// CHECK-SAME: %[[P:.*]]: i32, %[[X:.*]]: i32)
func.func @sub(%p: i32, %x: i32) -> i32 {
  // CHECK: %[[R:.*]] = arith.subi %[[X]], %[[P]] : i32
  // CHECK: return %[[R]] : i32
  %y = hssa.sub(%p) %x : i32
  return %y : i32
}

// CHECK-LABEL: func @sub_inverted(
// CHECK-SAME: %[[P:.*]]: i32, %[[X:.*]]: i32)
func.func @sub_inverted(%p: i32, %x: i32) -> i32 {
  // CHECK: %[[R:.*]] = arith.addi %[[X]], %[[P]] : i32
  // CHECK: return %[[R]] : i32
  %y = hssa.sub(%p) %x {inverted} : i32
  return %y : i32
}

// xor lowers identically in both directions.
// CHECK-LABEL: func @xor(
// CHECK-SAME: %[[P:.*]]: i32, %[[X:.*]]: i32)
func.func @xor(%p: i32, %x: i32) -> i32 {
  // CHECK: %[[R:.*]] = arith.xori %[[X]], %[[P]] : i32
  // CHECK: return %[[R]] : i32
  %y = hssa.xor(%p) %x : i32
  return %y : i32
}

// CHECK-LABEL: func @xor_inverted(
// CHECK-SAME: %[[P:.*]]: i32, %[[X:.*]]: i32)
func.func @xor_inverted(%p: i32, %x: i32) -> i32 {
  // CHECK: %[[R:.*]] = arith.xori %[[X]], %[[P]] : i32
  // CHECK: return %[[R]] : i32
  %y = hssa.xor(%p) %x {inverted} : i32
  return %y : i32
}

// id lowers to nothing: uses are forwarded to the input.
// CHECK-LABEL: func @id(
// CHECK-SAME: %[[X:.*]]: i32)
func.func @id(%x: i32) -> i32 {
  // CHECK: return %[[X]] : i32
  %y = hssa.id %x : i32
  return %y : i32
}

// Forward dup lowers to nothing: the copy is the parameter.
// CHECK-LABEL: func @dup(
// CHECK-SAME: %[[P:.*]]: i32)
func.func @dup(%p: i32) -> i32 {
  // CHECK: return %[[P]] : i32
  %c = hssa.dup(%p) : (i32) -> i32
  return %c : i32
}

// CHECK-LABEL: func @dup_inverted(
// CHECK-SAME: %[[P:.*]]: i32, %[[X:.*]]: i32)
func.func @dup_inverted(%p: i32, %x: i32) {
  // CHECK: %[[EQ:.*]] = arith.cmpi eq, %[[X]], %[[P]] : i32
  // CHECK: cf.assert %[[EQ]], "inverted dup: value does not match parameter"
  // CHECK: return
  hssa.dup(%p) %x {inverted} : (i32, i32) -> ()
  return
}

func.func private @callee(i32, i32) -> i32 attributes {hssa.params = 1 : i64}

// CHECK-LABEL: func @caller(
// CHECK-SAME: %[[P:.*]]: i32, %[[X:.*]]: i32)
func.func @caller(%p: i32, %x: i32) -> i32 {
  // CHECK: %[[R:.*]] = call @callee(%[[P]], %[[X]]) : (i32, i32) -> i32
  // CHECK: return %[[R]] : i32
  %y = hssa.call @callee(%p) %x : (i32, i32) -> i32
  return %y : i32
}

// CHECK-LABEL: func @mul(
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32)
func.func @mul(%a: i32, %b: i32) -> i32 {
  // CHECK: %[[M:.*]] = arith.muli %[[A]], %[[B]] : i32
  // CHECK: return %[[M]] : i32
  %m = hssa.mul(%a, %b) : (i32, i32) -> i32
  return %m : i32
}

// CHECK-LABEL: func @mul_inverted(
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32, %[[X:.*]]: i32)
func.func @mul_inverted(%a: i32, %b: i32, %x: i32) {
  // CHECK: %[[M:.*]] = arith.muli %[[A]], %[[B]] : i32
  // CHECK: %[[EQ:.*]] = arith.cmpi eq, %[[X]], %[[M]] : i32
  // CHECK: cf.assert %[[EQ]]
  hssa.mul(%a, %b) %x {inverted} : (i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: func @less(
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32)
func.func @less(%a: i32, %b: i32) -> i32 {
  // CHECK: %[[BIT:.*]] = arith.cmpi slt, %[[A]], %[[B]] : i32
  // CHECK: %[[E:.*]] = arith.extui %[[BIT]] : i1 to i32
  // CHECK: return %[[E]] : i32
  %c = hssa.less(%a, %b) : (i32, i32) -> i32
  return %c : i32
}

// CHECK-LABEL: func @euclidiv(
// CHECK-SAME: %[[D:.*]]: i32, %[[X:.*]]: i32)
func.func @euclidiv(%d: i32, %x: i32) -> (i32, i32) {
  // CHECK: %[[Q:.*]] = arith.divsi %[[X]], %[[D]] : i32
  // CHECK: %[[R:.*]] = arith.remsi %[[X]], %[[D]] : i32
  // CHECK: return %[[Q]], %[[R]] : i32, i32
  %q, %r = hssa.euclidiv(%d) %x : (i32, i32) -> (i32, i32)
  return %q, %r : i32, i32
}

// CHECK-LABEL: func @euclidiv_inverted(
// CHECK-SAME: %[[D:.*]]: i32, %[[Q:.*]]: i32, %[[R:.*]]: i32)
func.func @euclidiv_inverted(%d: i32, %q: i32, %r: i32) -> i32 {
  // CHECK: %[[P:.*]] = arith.muli %[[Q]], %[[D]] : i32
  // CHECK: %[[X:.*]] = arith.addi %[[P]], %[[R]] : i32
  // CHECK: return %[[X]] : i32
  %x = hssa.euclidiv(%d) %q, %r {inverted} : (i32, i32, i32) -> i32
  return %x : i32
}

// CHECK-LABEL: func @subfrom_op(
// CHECK-SAME: %[[P:.*]]: i32, %[[X:.*]]: i32)
func.func @subfrom_op(%p: i32, %x: i32) -> i32 {
  // CHECK: %[[R:.*]] = arith.subi %[[P]], %[[X]] : i32
  // CHECK: return %[[R]] : i32
  %r = hssa.subfrom(%p) %x : i32
  return %r : i32
}

// CHECK-LABEL: func @discard(
func.func @discard(%x: i32) {
  // CHECK-NEXT: return
  hssa.discard %x : i32
  return
}

// CHECK-LABEL: func @assert_fwd(
// CHECK-SAME: %[[P:.*]]: i32, %[[X:.*]]: i32)
func.func @assert_fwd(%p: i32, %x: i32) {
  // CHECK: %[[EQ:.*]] = arith.cmpi eq, %[[X]], %[[P]] : i32
  // CHECK: cf.assert %[[EQ]], "tests.assert failed"
  hssa.tests.assert(%p) %x : (i32, i32) -> ()
  return
}

// CHECK-LABEL: func @assert_inv(
// CHECK-SAME: %[[P:.*]]: i32)
func.func @assert_inv(%p: i32) -> i32 {
  // CHECK: return %[[P]] : i32
  %v = hssa.tests.assert(%p) {inverted} : (i32) -> i32
  return %v : i32
}

// CHECK-LABEL: func @fail(
func.func @fail() {
  // CHECK: %[[F:.*]] = arith.constant false
  // CHECK: cf.assert %[[F]], "tests.fail"
  hssa.tests.fail
  return
}
