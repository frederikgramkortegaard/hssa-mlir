//===- HssaOps.cpp - Hssa dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Hssa/HssaOps.h"
#include "Hssa/HssaDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "Hssa/HssaOpInterfaces.cpp.inc"
#include "Hssa/HssaOps.cpp.inc"

using namespace mlir;
using namespace mlir::hssa;

// ConstantLike ops must fold to a single constant attribute. The unit value
// carries no information, so it folds to a plain unit attribute.
OpFoldResult UnitOp::fold(FoldAdaptor) { return UnitAttr::get(getContext()); }

// A rel_ref is just the symbol it names, so it folds to its own callee attr.
OpFoldResult RelRefOp::fold(FoldAdaptor) { return getCalleeAttr(); }

// The optional slots must agree with the direction:
//   forward:  x := dup y =: ()   output present, no input
//   inverted: () := dup y =: x   input present, no output
LogicalResult DupOp::verify() {
  if (!getInverted()) {
    if (getInput()) {
      return emitOpError("forward dup cannot have an input (nothing is "
                         "consumed)");
    } else if (!getOutput()) {
      return emitOpError("forward dup must produce an output");
    } else if (getOutput().getType() != getParameter().getType()) {
      return emitOpError(
          "forward dup requires output and parameter to be of the same type");
    }
  } else {
    if (getOutput()) {
      return emitOpError("inverted dup cannot produce an output");
    } else if (!getInput()) {
      return emitOpError("inverted dup requires an input");
    } else if (getInput().getType() != getParameter().getType()) {
      return emitOpError(
          "inverted dup requires input and parameter to be of the same type");
    }
  }

  return success();
}

// Both directions write the control/linearity barrier resource, so dup is never
// DCE'd and two dups are never CSE-merged. Inverted dup is a check-and-consume
// that traps on mismatch (the classic trap case). Forward dup produces a fresh
// copy, which is semantically effect-free -- but under LINEARITY that copy is a
// distinct linear value that must not be aliased with another dup's result, and
// a dead copy is a linearity error, not dead code. Reporting an empty effect
// list here would mark forward dup pure and let CSE/DCE merge or delete it,
// breaking reversibility; the barrier Write is what keeps it conservative.
void DupOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       hssa::HssaTrapResource::get());
}

// Not buildInverseDefault: that drops the callee attribute, and
// operandSegmentSizes must be recomputed for the flipped shape. The typed
// builder handles both.
Operation *CallOp::buildInverse(OpBuilder &builder, ArrayRef<Value> newParams,
                                ArrayRef<Value> newIns) {
  SmallVector<Type> resultTypes;
  for (Value v : getConsumed())
    resultTypes.push_back(v.getType());
  auto newOp = CallOp::create(builder, getLoc(), resultTypes, getCalleeAttr(),
                              newParams, newIns);
  newOp.setInverted(!getInverted());
  return newOp;
}

// newParams[0] is the (remapped) callee relation value -- the same value, run
// backward; the rest are the read parameters. The flipped shape consumes what
// the forward produced and produces what it consumed.
Operation *CallIndirectOp::buildInverse(OpBuilder &builder,
                                        ArrayRef<Value> newParams,
                                        ArrayRef<Value> newIns) {
  SmallVector<Type> resultTypes;
  for (Value v : getConsumed())
    resultTypes.push_back(v.getType());
  Value callee = newParams.front();
  auto newOp = CallIndirectOp::create(builder, getLoc(), resultTypes, callee,
                                      newParams.drop_front(), newIns);
  newOp.setInverted(!getInverted());
  return newOp;
}

LogicalResult EuclidivOp::verify() {
  if (!getInverted()) {
    if (getIns().size() != 1 || getOuts().size() != 2)
      return emitOpError(
          "forward euclidiv consumes one value and produces two");
  } else {
    if (getIns().size() != 2 || getOuts().size() != 1)
      return emitOpError(
          "inverted euclidiv consumes two values and produces one");
  }
  return success();
}

LogicalResult TestsAssertOp::verify() {
  if (!getInverted()) {
    if (!getInput())
      return emitOpError("forward tests.assert requires an input");
    if (getOutput())
      return emitOpError("forward tests.assert cannot produce an output");
  } else {
    if (getInput())
      return emitOpError("inverted tests.assert cannot have an input");
    if (!getOutput())
      return emitOpError("inverted tests.assert must produce an output");
  }
  return success();
}

// Forward tests.assert is a result-less check-and-consume that traps on
// mismatch: it writes the abstract trap resource so it is not DCE-able.
// Inverted it is a generator (produces the parameter) and is side-effect-free.
void TestsAssertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!getInverted())
    effects.emplace_back(MemoryEffects::Write::get(),
                         hssa::HssaTrapResource::get());
}

LogicalResult mlir::hssa::detail::verifyComputeOp(Operation *op,
                                                  bool inverted) {
  unsigned numIns = op->getNumOperands() - 2;
  unsigned numOuts = op->getNumResults();
  if (!inverted) {
    if (numIns != 0)
      return op->emitOpError("forward compute op cannot have an input");
    if (numOuts != 1)
      return op->emitOpError("forward compute op must produce an output");
  } else {
    if (numIns != 1)
      return op->emitOpError("inverted compute op requires an input");
    if (numOuts != 0)
      return op->emitOpError("inverted compute op cannot produce an output");
  }
  return success();
}

Operation *mlir::hssa::detail::buildInverseDefault(Operation *op,
                                                   OpBuilder &builder,
                                                   ArrayRef<Value> newParams,
                                                   ArrayRef<Value> newIns) {
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(newParams);
  state.addOperands(newIns);
  for (Value v : cast<HssaGenericShape>(op).getConsumed())
    state.addTypes(v.getType());
  // An op with AttrSizedOperandSegments (e.g. mmem.new, whose optional size and
  // optional heap are two segments) needs its operand-segment attribute set, or
  // verification trips. The default inverse rebuilds operands as parameters then
  // inputs, so the two segments are exactly those counts.
  if (op->hasTrait<OpTrait::AttrSizedOperandSegments>())
    state.addAttribute(
        OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
        builder.getDenseI32ArrayAttr(
            {static_cast<int32_t>(newParams.size()),
             static_cast<int32_t>(newIns.size())}));
  Operation *newOp = builder.create(state);
  cast<HssaInvertible>(newOp).setInverted(
      !cast<HssaInvertible>(op).getInverted());
  return newOp;
}

LogicalResult MmemNewOp::verify() {
  // Forward makes a heap from nothing; inverted consumes one back to nothing.
  if (getInverted()) {
    if (!getInput() || getOutput())
      return emitOpError("inverted mmem.new consumes a heap and produces none");
  } else {
    if (getInput() || !getOutput())
      return emitOpError("forward mmem.new produces a heap and consumes none");
  }
  return success();
}

// All heap ops model their effect on one abstract heap resource. We report both
// Read and Write unconditionally (rather than Allocate/Free): the goal is that
// DCE/CSE treat every heap op conservatively -- none is dead, no two are
// reordered or deduplicated -- regardless of direction. Modelling Allocate/Free
// instead would let MLIR's allocation-aware DCE delete a "dangling" heap, which
// is exactly the reversible bookkeeping we must preserve. Threading the heap
// SSA value already orders them; the resource effect adds the don't-touch-me.

// mmem.new: forward creates the heap, inverted destroys it. Either way it
// mutates the abstract heap, so write (plus read to stay conservative).
void MmemNewOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), hssa::HssaHeapResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       hssa::HssaHeapResource::get());
}

LogicalResult MmemAllocateOp::verify() {
  // Forward produces a pointer; inverted consumes one to free the block.
  if (getInverted()) {
    if (!getPtrIn() || getPtrOut())
      return emitOpError("inverted mmem.allocate consumes a pointer and "
                         "produces none");
  } else {
    if (getPtrIn() || !getPtrOut())
      return emitOpError("forward mmem.allocate produces a pointer and "
                         "consumes none");
  }
  return success();
}

// mmem.allocate: forward carves a block, inverted frees it; both touch the
// allocator bookkeeping inside the heap. Read+Write on the heap resource.
void MmemAllocateOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), hssa::HssaHeapResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       hssa::HssaHeapResource::get());
}

// mmem.readwrite: reads the cell's old value and writes the new one (self-
// inverse). Read+Write on the heap resource.
void MmemReadwriteOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), hssa::HssaHeapResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       hssa::HssaHeapResource::get());
}

SuccessorOperands ExitOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "exit has one successor");
  return SuccessorOperands(getValuesMutable());
}

SuccessorOperands CondExitOp::getSuccessorOperands(unsigned index) {
  return SuccessorOperands(index == 0 ? getFirstOperandsMutable()
                                      : getSecondOperandsMutable());
}

// Every case receives the same payload (HSSA hands one payload to the selected
// successor; the operands are shared, stored once).
SuccessorOperands SwitchOp::getSuccessorOperands(unsigned index) {
  return SuccessorOperands(getPayloadMutable());
}

LogicalResult SwitchOp::verify() {
  if (getCases().size() < 2)
    return emitOpError("switch needs at least two successors");
  return success();
}

LogicalResult CondExitOp::verify() {
  if (getODSOperandIndexAndLength(0).second != 1)
    return emitOpError("selector segment must contain exactly one operand");
  llvm::APInt value;
  if (matchPattern(getSelector(), m_ConstantInt(&value)) && value.ugt(1))
    return emitOpError("constant selector must be 0 or 1");
  // A conditional exit carries a single HSSA payload; both successor operand
  // lists must agree, modulo the trailing provenance constant an edge into a
  // two-predecessor block appends (0 from one side, 1 from the other).
  auto payload = [](OperandRange ops) {
    SmallVector<Value> v(ops.begin(), ops.end());
    if (!v.empty() && v.back().getDefiningOp() &&
        v.back().getDefiningOp()->hasTrait<OpTrait::ConstantLike>())
      v.pop_back();
    return v;
  };
  if (payload(getFirstOperands()) != payload(getSecondOperands()))
    return emitOpError("must pass the same payload to both successors");
  return success();
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto fn =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
  if (!fn)
    return emitOpError("references undefined function '") << getCallee() << "'";
  auto paramsAttr = fn->getAttrOfType<IntegerAttr>("hssa.params");
  if (!paramsAttr)
    return emitOpError("callee '")
           << getCallee() << "' lacks the 'hssa.params' attribute";
  unsigned numParams = paramsAttr.getInt();
  if (numParams > fn.getNumArguments())
    return success(); // malformed attribute; rejected by the dialect hook
  unsigned calleeIns = fn.getNumArguments() - numParams;
  unsigned calleeOuts = fn.getNumResults();
  unsigned wantIns = getInverted() ? calleeOuts : calleeIns;
  unsigned wantOuts = getInverted() ? calleeIns : calleeOuts;
  if (getParams().size() != numParams)
    return emitOpError("expects ")
           << numParams << " parameter(s) for '" << getCallee() << "'";
  if (getIns().size() != wantIns)
    return emitOpError("consumes ")
           << wantIns << " value(s) for '" << getCallee() << "'";
  if (getOuts().size() != wantOuts)
    return emitOpError("produces ")
           << wantOuts << " value(s) for '" << getCallee() << "'";
  // Counts line up; the operand/result types must match the callee signature
  // too (func.func is not constrained to i32, so a non-i32 callee would
  // otherwise verify clean and produce invalid IR on lowering). Inversion
  // preserves the parameter prefix and swaps inputs with results.
  ArrayRef<Type> argTypes = fn.getArgumentTypes();
  ArrayRef<Type> resTypes = fn.getResultTypes();
  ArrayRef<Type> calleeInTypes = argTypes.drop_front(numParams);
  ArrayRef<Type> wantInTypes = getInverted() ? resTypes : calleeInTypes;
  ArrayRef<Type> wantOutTypes = getInverted() ? calleeInTypes : resTypes;
  for (unsigned i = 0; i < numParams; ++i)
    if (getParams()[i].getType() != argTypes[i])
      return emitOpError("parameter ")
             << i << " type does not match '" << getCallee() << "'";
  for (unsigned i = 0; i < wantIns; ++i)
    if (getIns()[i].getType() != wantInTypes[i])
      return emitOpError("input ")
             << i << " type does not match '" << getCallee() << "'";
  for (unsigned i = 0; i < wantOuts; ++i)
    if (getOuts()[i].getType() != wantOutTypes[i])
      return emitOpError("result ")
             << i << " type does not match '" << getCallee() << "'";
  return success();
}

namespace {
// A self-inverse update applied twice with the same parameter is the identity:
//   op(p)(op(p)(x)) = x    (x^p^p = x;  p-(p-x) = x).
// Fold the outer op away, recovering the inner op's input (and erase the inner,
// whose result then has no users -- its input is consumed exactly once again,
// keeping the program linear).
template <typename OpT>
struct CancelSelfInverseUpdate : OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    if (op.getInverted() || !op.getInput().hasOneUse())
      return failure();
    auto inner = op.getInput().template getDefiningOp<OpT>();
    if (!inner || inner.getInverted() ||
        inner.getParameter() != op.getParameter())
      return failure();
    rewriter.replaceOp(op, inner.getInput());
    rewriter.eraseOp(inner);
    return success();
  }
};
} // namespace

void XorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<CancelSelfInverseUpdate<XorOp>>(context);
}

void SubfromOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<CancelSelfInverseUpdate<SubfromOp>>(context);
}
