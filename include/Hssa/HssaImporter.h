//===- HssaImporter.h - textual HSSA -> hssa dialect ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HSSA_HSSAIMPORTER_H
#define HSSA_HSSAIMPORTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir::hssa {

/// Parse the textual HSSA program in `sourceMgr`'s main buffer into a module
/// of hssa-dialect functions. Returns nullptr after emitting diagnostics on
/// parse errors or constructs outside the supported subset.
OwningOpRef<ModuleOp> importHssa(llvm::SourceMgr &sourceMgr,
                                 MLIRContext *context);

} // namespace mlir::hssa

#endif // HSSA_HSSAIMPORTER_H
