//===- HssaExtension.cpp - Extension module -------------------------===//
//
// This is the nanobind version of the example module. There is also a pybind11
// example in HssaExtensionPybind11.cpp.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Hssa-c/Dialects.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

NB_MODULE(_hssaDialectsNanobind, m) {
  //===--------------------------------------------------------------------===//
  // hssa dialect
  //===--------------------------------------------------------------------===//
  auto hssaM = m.def_submodule("hssa");

  hssaM.def(
      "register_dialects",
      [](mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
             context,
         bool load) {
        MlirDialectHandle arithHandle = mlirGetDialectHandle__arith__();
        MlirDialectHandle hssaHandle = mlirGetDialectHandle__hssa__();
        MlirContext context_ = context.get()->get();
        mlirDialectHandleRegisterDialect(arithHandle, context_);
        mlirDialectHandleRegisterDialect(hssaHandle, context_);
        if (load) {
          mlirDialectHandleLoadDialect(arithHandle, context_);
          mlirDialectHandleLoadDialect(hssaHandle, context_);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);

  hssaM.def("print_fp_type",
            [](const mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyF16Type &,
               nb::handle stderr_obj) {
              nb::print("this is a fp16 type", /*end=*/nb::handle(),
                        stderr_obj);
            });
  hssaM.def("print_fp_type",
            [](const mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyF32Type &,
               nb::handle stderr_obj) {
              nb::print("this is a fp32 type", /*end=*/nb::handle(),
                        stderr_obj);
            });
  hssaM.def("print_fp_type",
            [](const mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyF64Type &,
               nb::handle stderr_obj) {
              nb::print("this is a fp64 type", /*end=*/nb::handle(),
                        stderr_obj);
            });
}
