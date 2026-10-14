//===- HssaOps.h - Hssa dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HSSA_HSSAOPS_H
#define HSSA_HSSAOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Hssa/HssaOpInterfaces.h"
#include "Hssa/HssaTypes.h"

// The SelfInverse trait (an involution marker; see HssaOpInterfaces.td) must be
// declared before the generated op classes that carry it. It lives directly in
// mlir::hssa (not a nested OpTrait namespace) so it does not shadow the global
// mlir::OpTrait that the dialect's C++ queries unqualified.
namespace mlir::hssa {
template <typename ConcreteType>
class SelfInverse
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SelfInverse> {};

// Abstract side-effect resources for the dialect. Memory effects are declared
// against a named resource rather than the DefaultResource so the intent is
// explicit and so heap effects and trap effects are kept in distinct resource
// classes (a pass that reasons about one does not conflate it with the other).

// The managed heap (!hssa.mem). mmem.new / mmem.allocate / mmem.readwrite and
// their inverses read and write this single abstract resource, so DCE/CSE treat
// every heap op conservatively: none is dead (it mutates shared state) and no
// two are interchangeable (they are not reordered or deduplicated).
struct HssaHeapResource
    : public ::mlir::SideEffects::Resource::Base<HssaHeapResource> {
  ::llvm::StringRef getName() final { return "HssaHeap"; }
};

// Abstract control / linearity barrier. Two related jobs, both "this op is not
// freely movable, removable, or duplicable":
//   * Trapping ops (inverted dup, forward tests.assert, inverted compute,
//     tests.fail) may abort the program and produce no SSA result -- without a
//     declared effect a generic DCE would delete them.
//   * Linear producers (forward dup, forward compute) DO produce a result, but
//     HSSA values are linear: CSE/value-numbering must NOT merge two identical
//     ones into a single shared value (that would consume one value twice and
//     break reversibility), and a dead one is a linearity error to report, not
//     to silently DCE. An empty effect list would make them look pure and thus
//     mergeable/removable -- the opposite of what linearity requires.
// A Write to this single abstract resource expresses both: every op that writes
// it is kept and is never deduplicated against another.
struct HssaTrapResource
    : public ::mlir::SideEffects::Resource::Base<HssaTrapResource> {
  ::llvm::StringRef getName() final { return "HssaControl"; }
};
} // namespace mlir::hssa

// The generated op classes reference the dialect types and interfaces above, so
// the GET_OP_CLASSES include must come last. The #define also acts as a barrier
// that stops clang-format from sorting the .inc up into the block above it.
#define GET_OP_CLASSES
#include "Hssa/HssaOps.h.inc"

#endif // HSSA_HSSAOPS_H
