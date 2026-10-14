//===- HssaOpInterfaces.h - Hssa op interfaces ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HSSA_HSSAOPINTERFACES_H
#define HSSA_HSSAOPINTERFACES_H

#include "mlir/IR/OpDefinition.h"
namespace mlir::hssa::detail {
Operation *buildInverseDefault(Operation *op, OpBuilder &builder,
                               ArrayRef<Value> newParams,
                               ArrayRef<Value> newIns);
LogicalResult verifyComputeOp(Operation *op, bool inverted);
} // namespace mlir::hssa::detail

#include "Hssa/HssaOpInterfaces.h.inc"

#endif
