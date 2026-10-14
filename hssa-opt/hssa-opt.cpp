//===- hssa-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Hssa/HssaDialect.h"
#include "Hssa/HssaPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::hssa::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::hssa::HssaDialect>();
  // All core dialects, so the tool can re-parse its own lowering output
  // (arith, cf, func, llvm, ...).
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Hssa optimizer driver\n", registry));
}
