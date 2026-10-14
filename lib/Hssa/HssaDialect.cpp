//===- HssaDialect.cpp - Hssa dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Hssa/HssaDialect.h"
#include "Hssa/HssaOps.h"
#include "Hssa/HssaTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::hssa;

#include "Hssa/HssaOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Hssa dialect.
//===----------------------------------------------------------------------===//

void HssaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Hssa/HssaOps.cpp.inc"
      >();
  registerTypes();
}

// Rebuild the constants the UnitOp/RelRefOp folders produce (both ConstantLike).
Operation *HssaDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  if (isa<UnitType>(type) && isa<UnitAttr>(value))
    return UnitOp::create(builder, loc, type);
  if (isa<RelType>(type))
    if (auto sym = dyn_cast<FlatSymbolRefAttr>(value))
      return RelRefOp::create(builder, loc, type, sym);
  return nullptr;
}

::llvm::LogicalResult
mlir::hssa::HssaDialect::verifyOperationAttribute(::mlir::Operation *op,
                                                  ::mlir::NamedAttribute attr) {
  if (attr.getName() == "hssa.params") {
    auto value = ::llvm::dyn_cast<::mlir::IntegerAttr>(attr.getValue());
    if (!value || !value.getType().isSignlessInteger(64) || value.getInt() < 0)
      return op->emitError(
          "'hssa.params' must be a non-negative i64 integer attribute");
    auto fn = ::llvm::dyn_cast<::mlir::FunctionOpInterface>(op);
    if (!fn)
      return op->emitError("'hssa.params' only applies to functions");
    if (value.getInt() > static_cast<int64_t>(fn.getNumArguments()))
      return op->emitError(
          "'hssa.params' exceeds the number of function arguments");
    return ::mlir::success();
  }
  if (attr.getName() == "hssa.reversible") {
    if (!::llvm::isa<::mlir::UnitAttr>(attr.getValue()))
      return op->emitError("'hssa.reversible' must be a unit attribute");
    if (!::llvm::isa<::mlir::FunctionOpInterface>(op))
      return op->emitError("'hssa.reversible' only applies to functions");
  }
  if (attr.getName() == "hssa.class") {
    auto value = ::llvm::dyn_cast<::mlir::StringAttr>(attr.getValue());
    if (!value || !::llvm::is_contained({"(D,D)", "(D,N)", "(N,D)", "(N,N)"},
                                        value.strref()))
      return op->emitError(
          "'hssa.class' must be one of \"(D,D)\", \"(D,N)\", \"(N,D)\", "
          "\"(N,N)\"");
    if (!::llvm::isa<::mlir::FunctionOpInterface>(op))
      return op->emitError("'hssa.class' only applies to functions");
  }
  return ::mlir::success();
}
