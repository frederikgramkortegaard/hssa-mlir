// RUN: hssa-opt --convert-hssa-to-arith %s | FileCheck %s

// A vtable-shaped relation: a relation reference (!hssa.rel) is produced by a
// dup and threaded through an internal block as a block argument before being
// returned. Lowering must (a) collapse !hssa.rel to its i32 dispatch index,
// (b) convert the function signature, AND (c) convert the INTERNAL block
// argument types -- not just the entry. This guards the full-region signature
// conversion; before it, the internal !hssa.rel block arg blocked legalization.

// CHECK-LABEL: func @vtable
// CHECK-SAME: (%{{.*}}: i32) -> i32
// rel_ref @callee lowers to its dispatch index constant...
// CHECK: %[[ID:.*]] = arith.constant {{.*}} : i32
// ...threaded through the internal block as an i32, not !hssa.rel.
// CHECK: cf.br ^[[L1:.*]](%[[ID]] : i32)
// CHECK: ^[[L1]](%[[R:.*]]: i32):
// CHECK: return %[[R]] : i32
// CHECK-NOT: hssa.
// CHECK-NOT: !hssa.rel
func.func @vtable(%method: i32) -> !hssa.rel attributes {hssa.params = 1 : i64} {
  %r = hssa.rel_ref @callee : !hssa.rel
  %d = hssa.dup(%r) : (!hssa.rel) -> !hssa.rel
  hssa.exit ^bb1(%d : !hssa.rel)
^bb1(%res: !hssa.rel):
  hssa.discard %method : i32
  return %res : !hssa.rel
}

func.func @callee(%x: i32) -> i32 attributes {hssa.params = 1 : i64} {
  return %x : i32
}
