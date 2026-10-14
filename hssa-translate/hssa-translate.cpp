//===- hssa-translate.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "Hssa/HssaImporter.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/SourceMgr.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();

  mlir::TranslateToMLIRRegistration importHssa(
      "import-hssa", "Import textual HSSA into the hssa dialect",
      [](llvm::SourceMgr &sourceMgr,
         mlir::MLIRContext *context) -> mlir::OwningOpRef<mlir::Operation *> {
        return mlir::hssa::importHssa(sourceMgr, context)
            .release()
            .getOperation();
      });

  return failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
