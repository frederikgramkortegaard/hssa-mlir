//===- HssaTypes.td - Hssa dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HSSA_TYPES
#define HSSA_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Hssa/HssaDialect.td"

//===----------------------------------------------------------------------===//
// Hssa type definitions
//===----------------------------------------------------------------------===//

// A scalar integer -- the HSSA value kind arithmetic, pointers, sizes, and
// branch selectors operate on. Just builtin i32 (integers are integers), named
// so an integer-only position reads as "integer", not "any value".
def Hssa_Int : TypeAlias<I32, "HSSA integer">;

class Hssa_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Hssa_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// The managed-memory heap, threaded single-assignment through memory ops
// (mmem.new produces it, every mmem op consumes one version and produces the
// next). Opaque: a program never inspects it, only hands it along.
def Hssa_MemType : Hssa_Type<"Mem", "mem"> {
  let summary = "A managed-memory heap handle.";
}

// The unit value: a value carrying no information. A heap cell can hold a unit,
// which is how reversible memory marks a slot "empty" -- a readwrite swaps the
// stored value out and a unit in to "take" it.
def Hssa_UnitType : Hssa_Type<"Unit", "unit"> {
  let summary = "The HSSA unit value.";
}

// A first-class relation value: a reference to a relation. A vtable dups
// relations into values, stores them in the heap, and calls them indirectly
// (virtual dispatch). Opaque here; lowering assigns each a dispatch index.
def Hssa_RelType : Hssa_Type<"Rel", "rel"> {
  let summary = "A relation reference (first-class relation value).";
}

// An HSSA value of any kind -- an integer, the heap handle, a unit, or a
// relation reference. This is the dialect's form of the reference interpreter's
// Value sum type
// (https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/interpretation/Value.scala).
// It carries the plain name "value"; an integer-only position uses Hssa_Int.
// dup is polymorphic over it, edge payloads carry it, and a call threads it.
def Hssa_Value : AnyTypeOf<[Hssa_Int, Hssa_MemType, Hssa_UnitType, Hssa_RelType],
                           "HSSA value">;

// A stored cell value: an integer, a unit, or a relation reference -- any value
// that fits in a heap slot, i.e. everything but the heap handle itself. This is
// a static-typing refinement the dialect adds: the reference stores an untyped
// Value in every cell and makes readwrite polymorphic via a type metavariable
// (ManagedMemory.scala), whereas we type each cell by its demanded kind.
def Hssa_Cell : AnyTypeOf<[Hssa_Int, Hssa_UnitType, Hssa_RelType],
                          "HSSA heap cell">;

#endif // HSSA_TYPES
