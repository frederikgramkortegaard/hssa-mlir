# RUN: %python %s 2>&1 | FileCheck %s
import sys

# CHECK: Testing mlir_hssa package
print("Testing mlir_hssa package", file=sys.stderr)

import mlir_hssa.ir
from mlir_hssa.dialects import hssa_nanobind as hssa_d

with mlir_hssa.ir.Context():
    hssa_d.register_dialects()
    hssa_module = mlir_hssa.ir.Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = hssa.id %0 : i32
    """
    )
    # CHECK: %[[C2:.*]] = arith.constant 2 : i32
    # CHECK: hssa.id %[[C2]] : i32
    print(str(hssa_module), file=sys.stderr)

    # CHECK: this is a fp16 type
    hssa_d.print_fp_type(mlir_hssa.ir.F16Type.get(), sys.stderr)
    # CHECK: this is a fp32 type
    hssa_d.print_fp_type(mlir_hssa.ir.F32Type.get(), sys.stderr)
    # CHECK: this is a fp64 type
    hssa_d.print_fp_type(mlir_hssa.ir.F64Type.get(), sys.stderr)


# CHECK: Testing mlir package
print("Testing mlir package", file=sys.stderr)

from mlir.ir import *

# CHECK-NOT: RuntimeWarning: nanobind: type '{{.*}}' was already registered!
