//===- HssaTypes.cpp - Hssa dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Hssa/HssaTypes.h"

#include "Hssa/HssaDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::hssa;

#define GET_TYPEDEF_CLASSES
#include "Hssa/HssaOpsTypes.cpp.inc"

void HssaDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Hssa/HssaOpsTypes.cpp.inc"
      >();
}
