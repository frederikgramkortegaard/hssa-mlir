//===- HssaDialect.td - Hssa dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HSSA_DIALECT
#define HSSA_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Hssa dialect definition.
//===----------------------------------------------------------------------===//

def Hssa_Dialect : Dialect {
    let name = "hssa";
    let summary = "The HSSA out-of-tree MLIR dialect.";
    let description = [{
        HSSA (Hybrid SSA) is a reversible-computing intermediate representation
        (Gail & Meyer, Reversible Computation 2024). This dialect hosts it in
        MLIR so HSSA programs can be imported, checked for linearity and
        reversibility, inverted, and lowered to LLVM.

        The authoritative semantics is the reference interpreter -- Lukas Gail /
        THM's Scala HSSA toolkit (https://github.com/Leridon/hssa), pinned to
        commit 6738274 for the source permalinks used throughout these files.
        "The reference interpreter" always means that toolkit, and non-obvious
        design choices cite the specific Scala source they match.
    }];
    let cppNamespace = "::mlir::hssa";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
    let hasOperationAttrVerify = 1;
    let hasConstantMaterializer = 1;
}

#endif // HSSA_DIALECT
