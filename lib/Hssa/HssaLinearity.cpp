//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Well-formedness: the linearity checker (--hssa-verify-linear, Gail & Meyer
// fig. 8), the reversibility classifier (--hssa-verify-reversible), and
// linearity repair (--hssa-repair-linear, the write-back dual of the checker).
// Part of the dialect/RC core.
//
//===----------------------------------------------------------------------===//
#include "Hssa/HssaTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringMap.h"

#include "Hssa/HssaOps.h"
#include "Hssa/HssaPasses.h"

namespace mlir::hssa {
#define GEN_PASS_DEF_HSSAVERIFYLINEAR
#define GEN_PASS_DEF_HSSAVERIFYREVERSIBLE
#define GEN_PASS_DEF_HSSAREPAIRLINEAR
#include "Hssa/HssaPasses.h.inc"

namespace {
// The Gail & Meyer sect. 3.5 fixpoint: every function starts (D,D); discard
// taints the backward direction, an oracle would taint the forward one, and
// calls join the callee's class (swapped for uncalls), since running f
// backward trades the directions. Unknown callees are assumed (D,D).
class HssaVerifyReversible
    : public impl::HssaVerifyReversibleBase<HssaVerifyReversible> {
public:
  using impl::HssaVerifyReversibleBase<
      HssaVerifyReversible>::HssaVerifyReversibleBase;

  struct Cls {
    bool fwdN = false, bwdN = false;
    bool operator==(const Cls &o) const {
      return fwdN == o.fwdN && bwdN == o.bwdN;
    }
  };

  void runOnOperation() final {
    SmallVector<func::FuncOp> funcs;
    getOperation()->walk([&](func::FuncOp fn) { funcs.push_back(fn); });

    llvm::StringMap<Cls> classes;
    bool changed = true;
    while (changed) {
      changed = false;
      for (func::FuncOp fn : funcs) {
        Cls c;
        fn.walk([&](Operation *op) {
          if (isa<DiscardOp>(op))
            c.bwdN = true;
          if (auto call = dyn_cast<CallOp>(op)) {
            Cls callee = classes.lookup(call.getCallee());
            if (call.getInverted())
              std::swap(callee.fwdN, callee.bwdN);
            c.fwdN |= callee.fwdN;
            c.bwdN |= callee.bwdN;
          }
        });
        if (!(classes[fn.getSymName()] == c)) {
          classes[fn.getSymName()] = c;
          changed = true;
        }
      }
    }

    for (func::FuncOp fn : funcs) {
      Cls c = classes.lookup(fn.getSymName());
      std::string cls = std::string("(") + (c.fwdN ? "N" : "D") + "," +
                        (c.bwdN ? "N" : "D") + ")";
      fn->setAttr("hssa.class", StringAttr::get(&getContext(), cls));
      if (!(c.fwdN || c.bwdN))
        continue;
      // Strict-RSSA mode holds every function to the reversible (D,D) fragment,
      // exactly as if it carried hssa.reversible -- so the switch enforces the
      // whole module as RSSA without touching the source. Outside strict mode
      // only explicitly-marked functions are enforced (unchanged behaviour).
      if (strict)
        fn.emitError("function is not in the reversible (D,D) fragment: "
                     "classified " +
                     cls);
      else if (fn->hasAttr("hssa.reversible"))
        fn.emitError("function is marked hssa.reversible but is classified " +
                     cls);
      else
        continue;
      signalPassFailure();
    }
  }
};

// === Linearity: shared vocabulary for the checker and the repair pass ===
// A value is EXEMPT from linearity (freely read or ignored): a unit (carries no
// information), a function parameter (the leading hssa.params entry args,
// readable anywhere), or a constant (consuming one is a runtime value-check,
// not a hand-off). Everything else is created once and consumed exactly once.
static bool isLinearExempt(Value v, Block *entry, unsigned numParams) {
  if (isa<UnitType>(v.getType()))
    return true;
  if (auto arg = dyn_cast<BlockArgument>(v))
    return arg.getOwner() == entry && arg.getArgNumber() < numParams;
  Operation *def = v.getDefiningOp();
  return def && def->hasTrait<OpTrait::ConstantLike>();
}

// The count of leading read-only parameters (the hssa.params attribute). The
// numArgs-1 fallback is a lenient guess, fine for the read-only checks here;
// invertFunctionBody deliberately does NOT use it -- inverting under a wrong
// guess builds a broken inverse, so it requires the attribute.
static unsigned numLinearParams(func::FuncOp fn) {
  if (auto attr = fn->getAttrOfType<IntegerAttr>("hssa.params"))
    return attr.getInt();
  unsigned n = fn.getNumArguments();
  return n > 0 ? n - 1 : 0;
}

// Split an HSSA statement into its linearity roles: values it reads as
// parameters (reusable), consumes (once), and produces. Returns false for a
// non-HSSA op. cond_exit is special: its two edge payloads carry the same
// values and only one edge runs, so a value in both edges counts as one
// consumption.
static bool hssaLinearRoles(Operation *op, SmallVectorImpl<Value> &params,
                            SmallVectorImpl<Value> &consumed,
                            SmallVectorImpl<Value> &produced) {
  if (auto c = dyn_cast<CondExitOp>(op)) {
    DenseSet<Value> inFirst(c.getFirstOperands().begin(),
                            c.getFirstOperands().end());
    consumed.push_back(c.getSelector());
    consumed.append(c.getFirstOperands().begin(), c.getFirstOperands().end());
    for (Value v : c.getSecondOperands())
      if (!inFirst.contains(v))
        consumed.push_back(v);
    return true;
  }
  if (auto shape = dyn_cast<HssaGenericShape>(op)) {
    llvm::append_range(params, shape.getParameters());
    llvm::append_range(consumed, shape.getConsumed());
    llvm::append_range(produced, shape.getProduced());
    return true;
  }
  if (isa<func::ReturnOp>(op)) {
    llvm::append_range(consumed, op->getOperands());
    return true;
  }
  return false;
}

class HssaVerifyLinear : public impl::HssaVerifyLinearBase<HssaVerifyLinear> {
public:
  using impl::HssaVerifyLinearBase<HssaVerifyLinear>::HssaVerifyLinearBase;

  void runOnOperation() final {
    func::FuncOp fn = getOperation();
    if (fn.isExternal())
      return;
    unsigned numParams = numLinearParams(fn);
    Block *entry = &fn.getBody().front();
    auto exempt = [&](Value v) { return isLinearExempt(v, entry, numParams); };

    for (Block &b : fn.getBody()) {
      DenseSet<Value> live;
      for (BlockArgument arg : b.getArguments())
        if (!exempt(arg))
          live.insert(arg);

      for (Operation &op : b) {
        if (op.hasTrait<OpTrait::ConstantLike>())
          continue;
        SmallVector<Value> params, consumed, produced;
        if (!hssaLinearRoles(&op, params, consumed, produced)) {
          op.emitError("is not an HSSA statement; linearity is undefined");
          return signalPassFailure();
        }

        for (Value p : params)
          if (!exempt(p) && llvm::is_contained(consumed, p)) {
            op.emitError("uses the same value as parameter and input");
            return signalPassFailure();
          }
        // A relation's parameters are read-only: a statement may read one as a
        // parameter operand any number of times, but consuming one (taking it
        // in an input position) makes the relation non-invertible. Returning a
        // parameter is not a consumption -- it hands the read-only value back
        // out -- so func.return is exempt from this check.
        for (Value v : consumed) {
          if (auto arg = dyn_cast<BlockArgument>(v))
            if (!isa<func::ReturnOp>(op) && arg.getOwner() == entry &&
                arg.getArgNumber() < numParams) {
              op.emitError("consumes read-only parameter #")
                  << arg.getArgNumber()
                  << " of the relation; a parameter may only be read, never "
                     "consumed -- dup it into a copy, consume the copy, then "
                     "uncompute (e.g. kc := dup _k =: ()  ...  () := ~dup _k =: "
                     "kc2)";
              return signalPassFailure();
            }
          if (exempt(v))
            continue;
          if (!live.erase(v)) {
            op.emitError("consumes a value that is not live in this block");
            return signalPassFailure();
          }
        }
        for (Value p : params) {
          if (exempt(p))
            continue;
          if (!live.contains(p)) {
            op.emitError("reads a value that is not live in this block");
            return signalPassFailure();
          }
        }
        for (Value v : produced)
          if (!exempt(v))
            live.insert(v);
      }

      if (live.empty())
        continue;
      // Report the first leaked value in program order.
      for (BlockArgument arg : b.getArguments())
        if (live.contains(arg)) {
          emitError(arg.getLoc()) << "block argument #" << arg.getArgNumber()
                                  << " is never consumed";
          return signalPassFailure();
        }
      for (Operation &op : b)
        for (Value r : op.getResults())
          if (live.contains(r)) {
            op.emitError("produces a value that is never consumed");
            return signalPassFailure();
          }
    }
  }
};

// The write-back dual of HssaVerifyLinear: where the checker reports a value
// consumed more than once or never consumed, repair fixes it -- a `dup` per
// extra consumer, a `discard` per leak. v1 is block-local; it skips any value
// used across blocks or by a cond_exit (its shared payload makes one logical
// consumption look like two operands).
class HssaRepairLinear : public impl::HssaRepairLinearBase<HssaRepairLinear> {
public:
  using impl::HssaRepairLinearBase<HssaRepairLinear>::HssaRepairLinearBase;

  void runOnOperation() final {
    func::FuncOp fn = getOperation();
    if (fn.isExternal())
      return;
    Block *entry = &fn.getBody().front();
    unsigned numParams = numLinearParams(fn);
    OpBuilder b(fn.getContext());

    for (Block &blk : fn.getBody()) {
      // Non-exempt values defined in this block, in definition order.
      SmallVector<Value> owned;
      for (BlockArgument a : blk.getArguments())
        if (!isLinearExempt(a, entry, numParams))
          owned.push_back(a);
      for (Operation &op : blk)
        if (!op.hasTrait<OpTrait::ConstantLike>())
          for (Value r : op.getResults())
            if (!isLinearExempt(r, entry, numParams))
              owned.push_back(r);

      for (Value v : owned) {
        // Collect this value's consuming operands (not its parameter reads).
        // Bail on anything v1 does not handle: a use in another block, or a
        // cond_exit use (shared payload).
        SmallVector<OpOperand *> consumed;
        bool skip = false;
        for (OpOperand &use : v.getUses()) {
          Operation *o = use.getOwner();
          if (o->getBlock() != &blk || isa<CondExitOp>(o)) {
            skip = true;
            break;
          }
          SmallVector<Value> ps, cs, prod;
          if (!hssaLinearRoles(o, ps, cs, prod))
            continue; // a read by a non-HSSA op; not a consumption
          if (!llvm::is_contained(ps, v) && llvm::is_contained(cs, v))
            consumed.push_back(&use);
        }
        if (skip)
          continue;

        if (consumed.empty()) {
          // Leak: consume it with a discard just before the terminator.
          b.setInsertionPoint(blk.getTerminator());
          DiscardOp::create(b, v.getLoc(), TypeRange{}, ValueRange{v});
        } else if (consumed.size() >= 2) {
          // Fan-out: keep the first consumer on v; give each of the rest its
          // own copy, dup'd right after v is defined (so the copy and the
          // read of v are both live before v's surviving consumption).
          if (isa<BlockArgument>(v))
            b.setInsertionPointToStart(&blk);
          else
            b.setInsertionPointAfter(v.getDefiningOp());
          for (OpOperand *use : llvm::drop_begin(consumed)) {
            Value copy = DupOp::create(b, v.getLoc(), TypeRange{v.getType()},
                                       ValueRange{v})
                             .getResult(0);
            use->set(copy);
          }
        }
      }
    }
  }
};
} // namespace
} // namespace mlir::hssa
