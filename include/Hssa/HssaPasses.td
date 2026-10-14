//===- HssaPasses.td - Hssa dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HSSA_PASS
#define HSSA_PASS

include "mlir/Pass/PassBase.td"

def HssaInvert: Pass<"hssa-invert", "::mlir::ModuleOp"> {
  let summary = "Invert every invertible function in place";
  let dependentDialects = ["arith::ArithDialect"];
}

 def ConvertHssaToArith : Pass<"convert-hssa-to-arith", "::mlir::ModuleOp"> {
    let summary = "Lower HSSA ops to arith/cf/memref.";
    let description = [{
      Lowers all of HSSA to standard dialects via one type-converting dialect
      conversion. Scalar ops become arith/cf. The managed heap mirrors the
      reference interpreter's first-fit allocator: !hssa.mem converts to a
      memref<?xi32> value array and !hssa.unit to i32 (the informationless
      sentinel 0). A second flag array distinguishes free from allocated cells,
      so mmem.allocate hands out the lowest free run (zeroing it) and its inverse
      frees a zero-cleared block -- reusing freed slots exactly as the interpreter
      does. mmem.readwrite is a load+store. Ready for --convert-to-llvm.
    }];
    let dependentDialects = ["arith::ArithDialect", "cf::ControlFlowDialect",
                             "func::FuncDialect", "memref::MemRefDialect"];
  }

def HssaResolveUncalls : Pass<"hssa-resolve-uncalls", "::mlir::ModuleOp"> {
  let summary = "Materialize inverse functions for inverted calls and rewrite "
                "them to forward calls.";
  let dependentDialects = ["arith::ArithDialect"];
}

def HssaVerifyLinear : Pass<"hssa-verify-linear", "::mlir::func::FuncOp"> {
  let summary = "Check block-local linearity: every value is created once "
                "and consumed exactly once.";
  let description = [{
    Implements the consistency check from Gail & Meyer 2024 (fig. 8, sect.
    3.4): each block is walked top-down with a live set: inputs may only
    consume live values, parameters may only read them, and nothing may
    survive the terminator. Function parameters (the "hssa.params" prefix)
    and constants are exempt, like the relation signature and literals in
    the paper. Two deliberate deviations from the figure as printed:
    parameter liveness is checked explicitly (the paper delegates it to
    name scoping, which SSA values do not have), and a block is consistent
    when its live set ends empty.
  }];
}

def HssaRepairLinear : Pass<"hssa-repair-linear", "::mlir::func::FuncOp"> {
  let summary = "Restore linearity: a `dup` per extra consumer, a `discard` "
                "per never-consumed value.";
  let description = [{
    The write-back dual of --hssa-verify-linear (Gail & Meyer 2024, sect. 4.1):
    standard optimizations break HSSA's one-use invariant by SHARING a value
    (e.g. CSE) or ORPHANING one (e.g. DCE). This pass repairs both -- a value
    consumed more than once is `dup`'d so each consumer gets its own copy, and a
    value never consumed is `discard`ed. After it runs, --hssa-verify-linear
    passes. Note `dup` is reversible but `discard` is not, so repairing fan-out
    keeps the program reversible while repairing a leak costs reversibility
    (observable with --hssa-verify-reversible). v1 is block-local; it skips
    values used across blocks or by a cond_exit's shared payload.
  }];
}

def HssaVerifyReversible : Pass<"hssa-verify-reversible", "::mlir::ModuleOp"> {
  let summary = "Classify every function as (D,D)/(D,N)/(N,D)/(N,N) and "
                "reject hssa.reversible functions outside (D,D).";
  let description = [{
    Implements the fixpoint classification from Gail & Meyer 2024 (sect. 3.5):
    discard introduces backwards-nondeterminism, its inverse (the oracle)
    forwards-nondeterminism, and calls propagate the callee's class (swapped
    for uncalls). The result is recorded as a "hssa.class" attribute; any
    function carrying the "hssa.reversible" unit attribute whose class is not
    (D,D) is an error.

    Strict-RSSA mode (the `strict` option) treats the whole module as the
    purely-reversible fragment: every function is held to (D,D) as if it
    carried "hssa.reversible", so any function classified (D,N)/(N,D)/(N,N) is
    an error. This is the machine check behind the paper's claim that the one
    dialect hosts the whole reversible-SSA family while the RSSA fragment
    (HSSA's (D,D) subset) is enforced on demand -- no re-import or per-function
    annotation needed, just the switch.
  }];
  let options = [
    Option<"strict", "strict", "bool", /*default=*/"false",
           "Enforce the reversible RSSA fragment: error on any function not "
           "classified (D,D), as if every function were hssa.reversible.">,
  ];
}
#endif // HSSA_PASS
