//===- hssa-cap-demo.c - Simple demo of C-API -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: capi-test
// RUN: hssa-capi-test 2>&1 | FileCheck %s

#include <stdio.h>

#include "Hssa-c/Dialects.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/IR.h"

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__arith__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__hssa__(), ctx);

  MlirModule module = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString("%0 = arith.constant 2 : i32\n"
                                          "%1 = hssa.id %0 : i32\n"));
  if (mlirModuleIsNull(module)) {
    printf("ERROR: Could not parse.\n");
    mlirContextDestroy(ctx);
    return 1;
  }
  MlirOperation op = mlirModuleGetOperation(module);

  // CHECK: %[[C:.*]] = arith.constant 2 : i32
  // CHECK: hssa.id %[[C]] : i32
  mlirOperationDump(op);

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
  return 0;
}
