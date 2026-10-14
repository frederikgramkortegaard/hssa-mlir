//===- HssaPasses.h - Hssa passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef HSSA_HSSAPASSES_H
#define HSSA_HSSAPASSES_H

#include "Hssa/HssaDialect.h"
#include "Hssa/HssaOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace hssa {
#define GEN_PASS_DECL
#include "Hssa/HssaPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Hssa/HssaPasses.h.inc"
} // namespace hssa
} // namespace mlir

#endif
