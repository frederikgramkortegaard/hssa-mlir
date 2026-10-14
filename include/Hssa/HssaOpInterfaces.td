//===- HssaOpInterfaces.td - Hssa op interfaces ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HSSA_OP_INTERFACES
#define HSSA_OP_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Side-effect resources (see HssaOps.h for the matching C++ Resource classes).
//===----------------------------------------------------------------------===//

// The abstract managed heap (!hssa.mem). Memory ops read/write it so DCE/CSE
// keep them conservative; see mmem.* and the getEffects implementations.
def HssaHeapResource : Resource<"::mlir::hssa::HssaHeapResource">;

// Abstract control/trap state written by result-less assertion ops so they are
// not treated as dead code; see tests.fail and the getEffects implementations.
def HssaTrapResource : Resource<"::mlir::hssa::HssaTrapResource">;

def HssaInvertibleOpInterface : OpInterface<"HssaInvertible"> {
  let cppNamespace = "::mlir::hssa";
  let methods = [
    InterfaceMethod<"Whether the op runs backward.", "bool", "getInverted">,
    InterfaceMethod<"Set the direction.", "void", "setInverted", (ins "bool":$value)>,
    InterfaceMethod<"Create this op's inverse and return it.",
    "::mlir::Operation *", "buildInverse",
    (ins "::mlir::OpBuilder &":$builder,
         "::llvm::ArrayRef<::mlir::Value>":$newParams,
         "::llvm::ArrayRef<::mlir::Value>":$newIns),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{
      return
  ::mlir::hssa::detail::buildInverseDefault($_op.getOperation(),
                                                       builder,
  newParams, newIns);
    }]>,
  ];
}

// Marks an op that is its own inverse (an involution): running it twice with the
// same parameters is the identity, so its `inverted` direction is the same op.
// The importer uses this to drop the meaningless `~` flag, and self-inverse
// update ops canonicalize their double application away.
def SelfInverse : NativeOpTrait<"SelfInverse"> {
  let cppNamespace = "::mlir::hssa";
}

// Each op declares which of its operands are read (parameters) vs consumed
// (inputs), and which results are produced. Ops whose shape depends on the
// direction (dup, call) answer from their own structure.
def HssaGenericShapeOpInterface : OpInterface<"HssaGenericShape"> {
  let cppNamespace = "::mlir::hssa";
  let methods = [
    InterfaceMethod<"Get the operands which the Op consumes","::llvm::SmallVector<mlir::Value>","getConsumed">,
    InterfaceMethod<"Get the results which the Op produces","::llvm::SmallVector<mlir::Value>","getProduced">,
    InterfaceMethod<"Get the operands which the Op uses as parameters","::llvm::SmallVector<mlir::Value>","getParameters">,
  ];
}

#endif // HSSA_OP_INTERFACES
