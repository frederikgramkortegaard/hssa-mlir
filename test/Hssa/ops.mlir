// RUN: hssa-opt %s | hssa-opt | FileCheck %s

// CHECK-LABEL: func @id_round_trip
// CHECK-SAME: (%[[X:.*]]: i32)
func.func @id_round_trip(%x: i32) -> i32 {
  // CHECK: %[[R:.*]] = hssa.id %[[X]] : i32
  %res = hssa.id %x : i32
  // CHECK: return %[[R]] : i32
  return %res : i32
}


// CHECK-LABEL: func @add(
// CHECK-SAME: %[[P:.*]]: i32, %[[IN:.*]]: i32)
func.func @add(%p: i32, %in: i32) -> i32 {
  // CHECK: %[[OUT:.*]] = hssa.add(%[[P]]) %[[IN]] : i32
  %out = hssa.add(%p) %in : i32
  // CHECK: return %[[OUT]] : i32
  return %out : i32
}

// CHECK-LABEL: func @add_inverted(
// CHECK-SAME: %[[P:.*]]: i32, %[[IN:.*]]: i32)
func.func @add_inverted(%p: i32, %in: i32) -> i32 {
  // CHECK: %[[OUT:.*]] = hssa.add(%[[P]]) %[[IN]] {inverted} : i32
  %out = hssa.add(%p) %in {inverted} : i32
  // CHECK: return %[[OUT]] : i32
  return %out : i32
}

// CHECK-LABEL: func @add_constant_parameter(
// CHECK-SAME: %[[IN:.*]]: i32)
func.func @add_constant_parameter(%in: i32) -> i32 {
  // CHECK: %[[C:.*]] = arith.constant 1 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: %[[OUT:.*]] = hssa.add(%[[C]]) %[[IN]] : i32
  %out = hssa.add(%c1) %in : i32
  // CHECK: return %[[OUT]] : i32
  return %out : i32
}

// CHECK-LABEL: func @sub(
// CHECK-SAME: %[[P:.*]]: i32, %[[IN:.*]]: i32)
func.func @sub(%p: i32, %in: i32) -> i32 {
  // CHECK: %[[OUT:.*]] = hssa.sub(%[[P]]) %[[IN]] : i32
  %out = hssa.sub(%p) %in : i32
  // CHECK: return %[[OUT]] : i32
  return %out : i32
}

// CHECK-LABEL: func @sub_inverted(
// CHECK-SAME: %[[P:.*]]: i32, %[[IN:.*]]: i32)
func.func @sub_inverted(%p: i32, %in: i32) -> i32 {
  // CHECK: %[[OUT:.*]] = hssa.sub(%[[P]]) %[[IN]] {inverted} : i32
  %out = hssa.sub(%p) %in {inverted} : i32
  // CHECK: return %[[OUT]] : i32
  return %out : i32
}

// CHECK-LABEL: func @sub_constant_parameter(
// CHECK-SAME: %[[IN:.*]]: i32)
func.func @sub_constant_parameter(%in: i32) -> i32 {
  // CHECK: %[[C:.*]] = arith.constant 1 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: %[[OUT:.*]] = hssa.sub(%[[C]]) %[[IN]] : i32
  %out = hssa.sub(%c1) %in : i32
  // CHECK: return %[[OUT]] : i32
  return %out : i32
}
// CHECK-LABEL: func @xor(
// CHECK-SAME: %[[P:.*]]: i32, %[[IN:.*]]: i32)
func.func @xor(%p: i32, %in: i32) -> i32 {
  // CHECK: %[[OUT:.*]] = hssa.xor(%[[P]]) %[[IN]] : i32
  %out = hssa.xor(%p) %in : i32
  // CHECK: return %[[OUT]] : i32
  return %out : i32
}

// CHECK-LABEL: func @xor_inverted(
// CHECK-SAME: %[[P:.*]]: i32, %[[IN:.*]]: i32)
func.func @xor_inverted(%p: i32, %in: i32) -> i32 {
  // CHECK: %[[OUT:.*]] = hssa.xor(%[[P]]) %[[IN]] {inverted} : i32
  %out = hssa.xor(%p) %in {inverted} : i32
  // CHECK: return %[[OUT]] : i32
  return %out : i32
}

// CHECK-LABEL: func @xor_constant_parameter(
// CHECK-SAME: %[[IN:.*]]: i32)
func.func @xor_constant_parameter(%in: i32) -> i32 {
  // CHECK: %[[C:.*]] = arith.constant 1 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: %[[OUT:.*]] = hssa.xor(%[[C]]) %[[IN]] : i32
  %out = hssa.xor(%c1) %in : i32
  // CHECK: return %[[OUT]] : i32
  return %out : i32
}

// CHECK-LABEL: func @dup(
// CHECK-SAME: %[[X:.*]]: i32)
func.func @dup(%x: i32) -> i32 {
  // CHECK: %[[C:.*]] = hssa.dup(%[[X]]) : (i32) -> i32
  %copy = hssa.dup(%x) : (i32) -> i32
  // CHECK: return %[[C]] : i32
  return %copy : i32
}

// CHECK-LABEL: func @dup_inverted(
// CHECK-SAME: %[[X:.*]]: i32, %[[C:.*]]: i32)
func.func @dup_inverted(%x: i32, %c: i32) -> i32 {
  // CHECK: hssa.dup(%[[X]]) %[[C]] {inverted} : (i32, i32) -> ()
  hssa.dup(%x) %c {inverted} : (i32, i32) -> ()
  // CHECK: return %[[X]] : i32
  return %x : i32
}


// CHECK-LABEL: func @subfrom(
// CHECK-SAME: %[[P:.*]]: i32, %[[IN:.*]]: i32)
func.func @subfrom(%p: i32, %in: i32) -> i32 {
  // CHECK: %[[OUT:.*]] = hssa.subfrom(%[[P]]) %[[IN]] : i32
  %out = hssa.subfrom(%p) %in : i32
  // CHECK: return %[[OUT]] : i32
  return %out : i32
}

// CHECK-LABEL: func @compute_and_check(
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32, %[[X:.*]]: i32)
func.func @compute_and_check(%a: i32, %b: i32, %x: i32) -> i32 {
  // CHECK: %[[M:.*]] = hssa.mul(%[[A]], %[[B]]) : (i32, i32) -> i32
  %m = hssa.mul(%a, %b) : (i32, i32) -> i32
  // CHECK: hssa.mul(%[[A]], %[[B]]) %[[M]] {inverted} : (i32, i32, i32) -> ()
  hssa.mul(%a, %b) %m {inverted} : (i32, i32, i32) -> ()
  // CHECK: %[[C:.*]] = hssa.less(%[[A]], %[[B]]) : (i32, i32) -> i32
  %c = hssa.less(%a, %b) : (i32, i32) -> i32
  // CHECK: hssa.discard %[[C]] : i32
  hssa.discard %c : i32
  // CHECK: %[[QR:.*]]:2 = hssa.euclidiv(%[[B]]) %[[X]] : (i32, i32) -> (i32, i32)
  %q, %r = hssa.euclidiv(%b) %x : (i32, i32) -> (i32, i32)
  // CHECK: %[[X2:.*]] = hssa.euclidiv(%[[B]]) %[[QR]]#0, %[[QR]]#1 {inverted} : (i32, i32, i32) -> i32
  %x2 = hssa.euclidiv(%b) %q, %r {inverted} : (i32, i32, i32) -> i32
  // CHECK: hssa.tests.assert(%[[A]]) %[[X2]] : (i32, i32) -> ()
  hssa.tests.assert(%a) %x2 : (i32, i32) -> ()
  // CHECK: %[[P:.*]] = hssa.tests.assert(%[[A]]) {inverted} : (i32) -> i32
  %p = hssa.tests.assert(%a) {inverted} : (i32) -> i32
  // CHECK: hssa.tests.fail
  hssa.tests.fail
  // CHECK: return %[[P]] : i32
  return %p : i32
}
