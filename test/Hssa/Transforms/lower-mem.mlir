// RUN: hssa-opt --convert-hssa-to-arith %s | FileCheck %s

// The heap is a single dynamically-sized memref<2x?xi32> handle: row 0 = cell
// values, row 1 = the free/allocated flag array (0 = free/`null`, 1 =
// allocated), allocated and zeroed per mmem.new. Encoding both rows in one
// handle keeps MemType a single threaded SSA value while carrying the
// dynamically-sized flag array with the heap. allocate is first-fit (it reuses
// freed slots, matching the interpreter), and its loop bounds depend only on
// the requested size -- never on the heap length -- so it works at any size.
// CHECK-DAG: func.func private @__hssa_new(%{{.*}}: index) -> memref<2x?xi32>
// CHECK-DAG: func.func private @__hssa_malloc(%{{.*}}: memref<2x?xi32>, %{{.*}}: i32) -> i32
// CHECK-DAG: func.func private @__hssa_free(%{{.*}}: memref<2x?xi32>, %{{.*}}: i32, %{{.*}}: i32)

// CHECK-LABEL: func @mem(
func.func @mem(%sz: i32, %v: i32) -> i32 attributes {hssa.params = 2 : i64} {
  // new allocates+zeroes a fresh heap (default 1024 cells when no size given);
  // allocate is a first-fit call over that heap returning a ptr.
  // CHECK: %[[HEAP:.*]] = call @__hssa_new(%{{.*}}) : (index) -> memref<2x?xi32>
  %m = hssa.mmem.new : () -> !hssa.mem
  // CHECK: %[[P:.*]] = call @__hssa_malloc(%[[HEAP]], %{{.*}}) : (memref<2x?xi32>, i32) -> i32
  %m1, %p = hssa.mmem.allocate(%sz) %m : (i32, !hssa.mem) -> (!hssa.mem, i32)
  // readwrite is a load+store on row 0 (the value row) at the pointer.
  // CHECK: %[[I:.*]] = arith.index_cast %[[P]]
  // CHECK: memref.load %[[HEAP]][%{{.*}}, %[[I]]]
  // CHECK: memref.store %{{.*}}, %[[HEAP]][%{{.*}}, %[[I]]]
  %m2, %old = hssa.mmem.readwrite(%p) %m1, %v : (i32, !hssa.mem, i32) -> (!hssa.mem, i32)
  // inverted allocate frees the block (checked deallocator); inverted new frees
  // the arena (memref.dealloc).
  // CHECK: call @__hssa_free(%[[HEAP]], %[[P]], %{{.*}}) : (memref<2x?xi32>, i32, i32) -> ()
  %m3 = hssa.mmem.allocate(%sz) %m2, %p {inverted} : (i32, !hssa.mem, i32) -> !hssa.mem
  // CHECK: memref.dealloc %[[HEAP]]
  hssa.mmem.new %m3 {inverted} : (!hssa.mem) -> ()
  return %old : i32
}

// A dynamically-sized heap: mmem.new K with a runtime K sizes the allocation.
// CHECK-LABEL: func @dynsize(
func.func @dynsize(%k: i32) -> i32 attributes {hssa.params = 1 : i64} {
  // CHECK: %[[N:.*]] = arith.index_cast %{{.*}} : i32 to index
  // CHECK: call @__hssa_new(%[[N]]) : (index) -> memref<2x?xi32>
  %m = hssa.mmem.new(%k) : (i32) -> !hssa.mem
  %sz = arith.constant 1 : i32
  %m1, %p = hssa.mmem.allocate(%sz) %m : (i32, !hssa.mem) -> (!hssa.mem, i32)
  %v = arith.constant 7 : i32
  %m2, %old = hssa.mmem.readwrite(%p) %m1, %v : (i32, !hssa.mem, i32) -> (!hssa.mem, i32)
  %m3 = hssa.mmem.allocate(%sz) %m2, %p {inverted} : (i32, !hssa.mem, i32) -> !hssa.mem
  hssa.mmem.new(%k) %m3 {inverted} : (i32, !hssa.mem) -> ()
  return %old : i32
}

// The unit value lowers to the i32 sentinel 0 (the heap stays internal).
// CHECK-LABEL: func @unit
// CHECK: arith.constant 0 : i32
func.func @unit() -> i32 attributes {hssa.params = 0 : i64} {
  %m = hssa.mmem.new : () -> !hssa.mem
  %sz = arith.constant 1 : i32
  %m1, %p = hssa.mmem.allocate(%sz) %m : (i32, !hssa.mem) -> (!hssa.mem, i32)
  %u = hssa.unit : !hssa.unit
  %m2, %old = hssa.mmem.readwrite(%p) %m1, %u : (i32, !hssa.mem, !hssa.unit) -> (!hssa.mem, i32)
  %m3 = hssa.mmem.allocate(%sz) %m2, %p {inverted} : (i32, !hssa.mem, i32) -> !hssa.mem
  hssa.mmem.new %m3 {inverted} : (!hssa.mem) -> ()
  return %old : i32
}
