// RUN: hssa-opt %s | hssa-opt | FileCheck %s

// The managed heap is an opaque, single-threaded token.
// CHECK-LABEL: func @heap
func.func @heap(%sz: i32) -> i32 attributes {hssa.params = 1 : i64} {
  // CHECK: %[[M:.*]] = hssa.mmem.new : () -> !hssa.mem
  %m = hssa.mmem.new : () -> !hssa.mem
  // CHECK: %[[M2:.*]], %[[P:.*]] = hssa.mmem.allocate(%{{.*}}) %[[M]] : (i32, !hssa.mem) -> (!hssa.mem, i32)
  %m2, %p = hssa.mmem.allocate(%sz) %m : (i32, !hssa.mem) -> (!hssa.mem, i32)
  // CHECK: %[[M3:.*]] = hssa.mmem.allocate(%{{.*}}) %[[M2]], %[[P]] {inverted} : (i32, !hssa.mem, i32) -> !hssa.mem
  %m3 = hssa.mmem.allocate(%sz) %m2, %p {inverted} : (i32, !hssa.mem, i32) -> !hssa.mem
  // CHECK: hssa.mmem.new %[[M3]] {inverted} : (!hssa.mem) -> ()
  hssa.mmem.new %m3 {inverted} : (!hssa.mem) -> ()
  return %sz : i32
}

// A cell swap: write a value, read the old one back. Self-inverse, so no flag.
// CHECK-LABEL: func @readwrite
func.func @readwrite(%ptr: i32, %m: !hssa.mem, %v: i32) -> (!hssa.mem, i32)
    attributes {hssa.params = 1 : i64} {
  // CHECK: %{{.*}}, %{{.*}} = hssa.mmem.readwrite(%{{.*}}) %{{.*}}, %{{.*}} : (i32, !hssa.mem, i32) -> (!hssa.mem, i32)
  %m2, %old = hssa.mmem.readwrite(%ptr) %m, %v : (i32, !hssa.mem, i32) -> (!hssa.mem, i32)
  return %m2, %old : !hssa.mem, i32
}

// The unit "take": write unit to empty a cell, read its int out. Unit is
// constant-like, so it needs no separate finalization.
// CHECK-LABEL: func @take
func.func @take(%ptr: i32, %m: !hssa.mem) -> (!hssa.mem, i32)
    attributes {hssa.params = 1 : i64} {
  // CHECK: %[[U:.*]] = hssa.unit : !hssa.unit
  %u = hssa.unit : !hssa.unit
  // CHECK: hssa.mmem.readwrite(%{{.*}}) %{{.*}}, %[[U]] : (i32, !hssa.mem, !hssa.unit) -> (!hssa.mem, i32)
  %m2, %old = hssa.mmem.readwrite(%ptr) %m, %u : (i32, !hssa.mem, !hssa.unit) -> (!hssa.mem, i32)
  return %m2, %old : !hssa.mem, i32
}
