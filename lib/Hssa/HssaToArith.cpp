//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Execution lowering (--convert-hssa-to-arith): HSSA -> arith/cf/memref (+ the
// managed-memory runtime), ready for --convert-to-llvm. This is the path used
// to RUN programs and differentially validate against the reference interpreter
// -- an execution/validation layer, separate from the dialect and RC passes.
//
//===----------------------------------------------------------------------===//
#include "Hssa/HssaTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringMap.h"

#include "Hssa/HssaOps.h"
#include "Hssa/HssaPasses.h"

namespace mlir::hssa {
#define GEN_PASS_DEF_CONVERTHSSATOARITH
#include "Hssa/HssaPasses.h.inc"

namespace {
// Update ops whose two directions are a pair of arith ops.
template <typename OpT, typename FwdArith, typename InvArith>
struct UpdateLowering : OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    if (op.getInverted())
      rewriter.replaceOpWithNewOp<InvArith>(op, op.getInput(),
                                            op.getParameter());
    else
      rewriter.replaceOpWithNewOp<FwdArith>(op, op.getInput(),
                                            op.getParameter());
    return success();
  }
};
struct XorLowering : OpRewritePattern<XorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(XorOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, op.getInput(),
                                               op.getParameter());
    return success();
  }
};
// id/dup/discard are polymorphic over any HSSA value, so they lower through the
// type converter (a mem/rel/unit operand has already become memref/i32/i32).
struct IdLowering : OpConversionPattern<IdOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(IdOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOp(op, adaptor.getInput());
    return success();
  }
};
// A reversibility guard: at run time assert `got == expected` (the
// check-and-consume an inverted op performs) and produce nothing. Shared by
// every inverted op whose forward direction created a value that must now be
// recovered-and-checked (inverted dup, compute ops, tests.assert).
static void emitEqAssert(OpBuilder &b, Location loc, Value got, Value expected,
                         StringRef msg) {
  Value eq =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, got, expected);
  cf::AssertOp::create(b, loc, eq, msg);
}

struct DupLowering : OpConversionPattern<DupOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(DupOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    if (op.getInverted()) {
      emitEqAssert(r, op.getLoc(), adaptor.getInput(), adaptor.getParameter(),
                   "inverted dup: value does not match parameter");
      r.eraseOp(op);
    } else {
      r.replaceOp(op, adaptor.getParameter());
    }
    return success();
  }
};
struct SubfromLowering : OpRewritePattern<SubfromOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SubfromOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SubIOp>(op, op.getParameter(),
                                               op.getInput());
    return success();
  }
};

// Forward compute ops become the arith computation; inverted ones recompute
// and check-and-consume (cmpi + cf.assert), producing nothing.
template <typename OpT>
static LogicalResult rewriteComputeOp(OpT op, Value computed,
                                      PatternRewriter &rewriter) {
  if (!op.getInverted()) {
    rewriter.replaceOp(op, computed);
    return success();
  }
  emitEqAssert(rewriter, op.getLoc(), op.getInput(), computed,
               "inverted compute op: value does not match");
  rewriter.eraseOp(op);
  return success();
}

template <typename OpT, typename ArithT>
struct BinArithLowering : OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    Value computed =
        ArithT::create(rewriter, op.getLoc(), op.getLhs(), op.getRhs());
    return rewriteComputeOp(op, computed, rewriter);
  }
};

template <typename OpT, arith::CmpIPredicate pred>
struct CmpLowering : OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    Value bit = arith::CmpIOp::create(rewriter, op.getLoc(), pred, op.getLhs(),
                                      op.getRhs());
    Value computed = arith::ExtUIOp::create(rewriter, op.getLoc(),
                                            rewriter.getI32Type(), bit);
    return rewriteComputeOp(op, computed, rewriter);
  }
};

struct EuclidivLowering : OpRewritePattern<EuclidivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(EuclidivOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (!op.getInverted()) {
      Value dividend = op.getIns().front();
      Value q =
          arith::DivSIOp::create(rewriter, loc, dividend, op.getDivisor());
      Value r =
          arith::RemSIOp::create(rewriter, loc, dividend, op.getDivisor());
      rewriter.replaceOp(op, {q, r});
    } else {
      Value prod =
          arith::MulIOp::create(rewriter, loc, op.getIns()[0], op.getDivisor());
      Value dividend =
          arith::AddIOp::create(rewriter, loc, prod, op.getIns()[1]);
      rewriter.replaceOp(op, dividend);
    }
    return success();
  }
};

struct DiscardLowering : OpConversionPattern<DiscardOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(DiscardOp op, OpAdaptor,
                                ConversionPatternRewriter &r) const override {
    r.eraseOp(op);
    return success();
  }
};

struct TestsAssertLowering : OpRewritePattern<TestsAssertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TestsAssertOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInverted()) {
      rewriter.replaceOp(op, op.getParameter());
      return success();
    }
    emitEqAssert(rewriter, op.getLoc(), op.getInput(), op.getParameter(),
                 "tests.assert failed");
    rewriter.eraseOp(op);
    return success();
  }
};

struct TestsFailLowering : OpRewritePattern<TestsFailOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TestsFailOp op,
                                PatternRewriter &rewriter) const override {
    Value falseVal = arith::ConstantOp::create(rewriter, op.getLoc(),
                                               rewriter.getBoolAttr(false));
    cf::AssertOp::create(rewriter, op.getLoc(), falseVal, "tests.fail");
    rewriter.eraseOp(op);
    return success();
  }
};

// A relation reference lowers to its dispatch index: a stable integer id per
// relation symbol, stored in a vtable slot and switched on by an indirect call
// (a direct call never inspects it). The id map is built from the module's
// relation order before conversion.
// The dispatch index of a relation. Every function in the module is registered
// (see the relIds walk), so a lookup miss means a dangling reference -- an
// upstream bug, not a value to default silently.
static int64_t relDispatchId(const llvm::StringMap<int64_t> &ids,
                             StringRef name) {
  auto it = ids.find(name);
  assert(it != ids.end() && "dispatch target is not a known relation");
  return it == ids.end() ? 0 : it->second; // keep release builds safe
}

struct RelRefLowering : OpConversionPattern<RelRefOp> {
  RelRefLowering(const TypeConverter &conv, MLIRContext *ctx,
                 const llvm::StringMap<int64_t> &ids)
      : OpConversionPattern(conv, ctx), ids(ids) {}
  LogicalResult matchAndRewrite(RelRefOp op, OpAdaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<arith::ConstantOp>(
        op, r.getI32IntegerAttr(relDispatchId(ids, op.getCallee())));
    return success();
  }
  const llvm::StringMap<int64_t> &ids;
};

// Indirect dispatch: a call_indirect was resolved (by --hssa-resolve-uncalls)
// into an index-keyed candidate set -- hssa.dispatch_keys are the candidate
// relations whose dispatch indices the runtime value is matched against, and
// hssa.dispatch_targets are the functions to call (the candidates themselves
// forward; their inverses for a resolved uncall). Lower to a cf.switch on the
// index: each case calls one target and branches to a continuation carrying the
// results. The last candidate doubles as the (always-valid) default.
struct CallIndirectLowering : OpConversionPattern<CallIndirectOp> {
  CallIndirectLowering(const TypeConverter &conv, MLIRContext *ctx,
                       const llvm::StringMap<int64_t> &ids)
      : OpConversionPattern(conv, ctx), ids(ids) {}
  LogicalResult matchAndRewrite(CallIndirectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    auto keys = op->getAttrOfType<ArrayAttr>("hssa.dispatch_keys");
    auto targets = op->getAttrOfType<ArrayAttr>("hssa.dispatch_targets");
    if (!keys || !targets || keys.empty())
      return failure(); // not resolved -- run --hssa-resolve-uncalls first
    Location loc = op.getLoc();
    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();
    // Each candidate receives the parameters followed by the consumed inputs.
    SmallVector<Value> operands(adaptor.getParams());
    operands.append(adaptor.getIns().begin(), adaptor.getIns().end());

    // The continuation holds the op's results as block arguments; the switch
    // goes at the end of the original (now terminator-less) block.
    Block *entry = op->getBlock();
    Block *cont = r.splitBlock(entry, Block::iterator(op));
    SmallVector<Location> argLocs(resultTypes.size(), loc);
    cont->addArguments(resultTypes, argLocs);

    // One block per candidate: call the target, then branch to the merge.
    SmallVector<Block *> caseBlocks;
    for (Attribute tgt : targets) {
      Block *cb = r.createBlock(cont);
      auto call = func::CallOp::create(r, loc, cast<FlatSymbolRefAttr>(tgt),
                                       resultTypes, operands);
      cf::BranchOp::create(r, loc, cont, call.getResults());
      caseBlocks.push_back(cb);
    }

    // The dispatch switch: case relId(key) -> that candidate's block.
    SmallVector<APInt> caseVals;
    SmallVector<Block *> caseDests;
    SmallVector<ValueRange> caseOperands;
    for (unsigned k = 0; k + 1 < caseBlocks.size(); ++k) {
      caseVals.push_back(APInt(
          32, relDispatchId(ids, cast<FlatSymbolRefAttr>(keys[k]).getValue())));
      caseDests.push_back(caseBlocks[k]);
      caseOperands.push_back({});
    }
    r.setInsertionPointToEnd(entry);
    cf::SwitchOp::create(r, loc, adaptor.getCallee(), caseBlocks.back(),
                         ValueRange{}, caseVals, caseDests, caseOperands);
    r.replaceOp(op, cont->getArguments());
    return success();
  }
  const llvm::StringMap<int64_t> &ids;
};

// Edge payloads carry any HSSA value, so these terminators lower through the
// converter (converted operands, and target block args converted to match).
struct ExitLowering : OpConversionPattern<ExitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ExitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(), adaptor.getValues());
    return success();
  }
};

struct CondExitLowering : OpConversionPattern<CondExitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(CondExitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    // selector is 0 or 1 in valid programs; 0 picks $first, so the truthy
    // branch of cf.cond_br is $second.
    Value cond = arith::TruncIOp::create(r, op.getLoc(), r.getI1Type(),
                                         adaptor.getSelector());
    r.replaceOpWithNewOp<cf::CondBranchOp>(
        op, cond, op.getSecond(), adaptor.getSecondOperands(), op.getFirst(),
        adaptor.getFirstOperands());
    return success();
  }
};

struct SwitchLowering : OpConversionPattern<SwitchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SwitchOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    // cf.switch wants a default plus explicit cases. The selector is a valid
    // 0-based index in a reversible program, so the last successor doubles as
    // the (never-taken-otherwise) default and cases 0..K-2 are explicit.
    SmallVector<Block *> cases(op.getCases().begin(), op.getCases().end());
    ValueRange payload = adaptor.getPayload();
    SmallVector<APInt> caseValues;
    SmallVector<Block *> caseDests;
    SmallVector<ValueRange> caseOperands;
    for (unsigned i = 0; i + 1 < cases.size(); ++i) {
      caseValues.push_back(APInt(32, i));
      caseDests.push_back(cases[i]);
      caseOperands.push_back(payload);
    }
    r.replaceOpWithNewOp<cf::SwitchOp>(op, adaptor.getSelector(), cases.back(),
                                       payload, caseValues, caseDests,
                                       caseOperands);
    return success();
  }
};

struct CallLowering : OpConversionPattern<CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(CallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    // Inverted calls have no direct lowering; --hssa-resolve-uncalls must
    // rewrite them first. Declining here makes legalization fail on them.
    if (op.getInverted())
      return failure();
    // The callee's arguments are its parameters followed by its consumed
    // inputs, matching the operand order here.
    SmallVector<Value> operands(adaptor.getParams());
    operands.append(adaptor.getIns().begin(), adaptor.getIns().end());
    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();
    r.replaceOpWithNewOp<func::CallOp>(op, op.getCalleeAttr(), resultTypes,
                                       operands);
    return success();
  }
};
// === Managed memory -> memref ===
// The managed heap is ONE dynamically-sized memref<2x?xi32> handle, allocated
// per mmem.new and threaded single-assignment through every memory op. The two
// rows model the reference interpreter's managed heap FAITHFULLY (the first-fit
// allocator, UnallocationError, and content.tail.sliding scan cited below are
// all from
// https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/ManagedMemory.scala):
//   row 0 -- cell values.
//   row 1 -- allocation flags: 0 = free (the interpreter's `null`), 1 =
//            allocated. `null` and `Int(0)` are distinct states, so the flag
//            row is needed; an i32 value alone cannot say whether a cell is
//            unallocated or allocated-holding-zero.
// Encoding both rows in a single handle keeps MemType lowering to ONE SSA value
// (it threads through block args, calls and returns like any value) while still
// carrying the dynamically-sized flag array with the heap.
//
// mmem.new K allocates memref.alloc(K) and zeroes both rows (a fresh heap of
// nulls); ~mmem.new frees it (memref.dealloc). The heap SIZE is a runtime value,
// so the allocator/free helpers take the heap as a parameter (no module global)
// and work for ANY size -- their loop bounds come from the requested allocation
// `size`, never from the heap length.
//
// allocate is FIRST-FIT: it hands out the lowest pointer p>=1 with `size`
// consecutive free cells, marks them allocated, and zeroes their values (the
// interpreter sets them to Int(0)). deallocate (inverted allocate) requires the
// block already zero-cleared and frees it. This matches the interpreter's reuse
// of freed slots exactly -- a bump cursor would diverge (different pointers,
// and stale data in reused cells) on any program that frees then reallocates.
static constexpr int64_t kDefaultHeapSize = 1024;
static constexpr const char *kNewFn = "__hssa_new";
static constexpr const char *kMallocFn = "__hssa_malloc";
static constexpr const char *kFreeFn = "__hssa_free";

static MemRefType heapType(MLIRContext *ctx) {
  return MemRefType::get({2, ShapedType::kDynamic}, IntegerType::get(ctx, 32));
}

// Managed-memory runtime, emitted once per module as plain cf+arith+memref
// functions (no scf, so any --convert-*-to-llvm pipeline lowers them).
//   @__hssa_new(n)       -- allocate a memref<2x?xi32> of n cells and zero both
//                           rows (a fresh heap of nulls, like the interpreter's
//                           fresh array).
//   @__hssa_malloc(h,sz) -- scan row 1 of h for the lowest free run of `sz`
//                           cells (mirroring the interpreter's
//                           content.tail.sliding(sz).find(all null)), mark it
//                           allocated, zero its values, return the pointer.
//   @__hssa_free(h,p,sz) -- assert the block is zero-cleared (the interpreter's
//                           UnallocationError) and free it.
// Row indices are the constants c0r (values) and c1r (flags); the column is the
// cell index. Loop bounds depend only on `sz`/`n`, never on the heap length.
static const char *kMemRuntime = R"mlir(
  func.func private @__hssa_new(%n: index) -> memref<2x?xi32> {
    %h = memref.alloc(%n) : memref<2x?xi32>
    %c0i = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    cf.br ^zrow(%c0 : index)
  ^zrow(%r: index):
    %rdone = arith.cmpi sge, %r, %c2 : index
    cf.cond_br %rdone, ^ret, ^zcolinit(%r : index)
  ^zcolinit(%r2: index):
    cf.br ^zcol(%r2, %c0 : index, index)
  ^zcol(%r3: index, %c: index):
    %cdone = arith.cmpi sge, %c, %n : index
    cf.cond_br %cdone, ^rnext(%r3 : index), ^zbody(%r3, %c : index, index)
  ^zbody(%r4: index, %c2b: index):
    memref.store %c0i, %h[%r4, %c2b] : memref<2x?xi32>
    %cn = arith.addi %c2b, %c1 : index
    cf.br ^zcol(%r4, %cn : index, index)
  ^rnext(%r5: index):
    %rn = arith.addi %r5, %c1 : index
    cf.br ^zrow(%rn : index)
  ^ret:
    return %h : memref<2x?xi32>
  }
  func.func private @__hssa_malloc(%heap: memref<2x?xi32>, %size: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c0r = arith.constant 0 : index
    %c1r = arith.constant 1 : index
    cf.br ^scan(%c1 : i32)
  ^scan(%p: i32):
    cf.br ^chk(%c0, %p : i32, i32)
  ^chk(%k: i32, %pp: i32):
    %done = arith.cmpi sge, %k, %size : i32
    cf.cond_br %done, ^found(%pp : i32), ^test(%k, %pp : i32, i32)
  ^test(%k2: i32, %pp2: i32):
    %idx = arith.addi %pp2, %k2 : i32
    %ii = arith.index_cast %idx : i32 to index
    %f = memref.load %heap[%c1r, %ii] : memref<2x?xi32>
    %free = arith.cmpi eq, %f, %c0 : i32
    cf.cond_br %free, ^next(%k2, %pp2 : i32, i32), ^advance(%pp2 : i32)
  ^next(%k3: i32, %pp3: i32):
    %kn = arith.addi %k3, %c1 : i32
    cf.br ^chk(%kn, %pp3 : i32, i32)
  ^advance(%pp4: i32):
    %pnext = arith.addi %pp4, %c1 : i32
    cf.br ^scan(%pnext : i32)
  ^found(%fp: i32):
    cf.br ^mark(%c0, %fp : i32, i32)
  ^mark(%mk: i32, %fp2: i32):
    %mdone = arith.cmpi sge, %mk, %size : i32
    cf.cond_br %mdone, ^ret(%fp2 : i32), ^mstep(%mk, %fp2 : i32, i32)
  ^mstep(%mk2: i32, %fp3: i32):
    %midx = arith.addi %fp3, %mk2 : i32
    %mii = arith.index_cast %midx : i32 to index
    memref.store %c1, %heap[%c1r, %mii] : memref<2x?xi32>
    memref.store %c0, %heap[%c0r, %mii] : memref<2x?xi32>
    %mkn = arith.addi %mk2, %c1 : i32
    cf.br ^mark(%mkn, %fp3 : i32, i32)
  ^ret(%rp: i32):
    return %rp : i32
  }
  func.func private @__hssa_free(%heap: memref<2x?xi32>, %ptr: i32, %size: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c0r = arith.constant 0 : index
    %c1r = arith.constant 1 : index
    cf.br ^loop(%c0 : i32)
  ^loop(%k: i32):
    %done = arith.cmpi sge, %k, %size : i32
    cf.cond_br %done, ^ret, ^body(%k : i32)
  ^body(%k2: i32):
    %idx = arith.addi %ptr, %k2 : i32
    %ii = arith.index_cast %idx : i32 to index
    %v = memref.load %heap[%c0r, %ii] : memref<2x?xi32>
    %ok = arith.cmpi eq, %v, %c0 : i32
    cf.assert %ok, "mmem: deallocating non-zero-cleared memory"
    memref.store %c0, %heap[%c1r, %ii] : memref<2x?xi32>
    %kn = arith.addi %k2, %c1 : i32
    cf.br ^loop(%kn : i32)
  ^ret:
    return
  }
)mlir";

struct UnitLowering : OpConversionPattern<UnitOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(UnitOp op, OpAdaptor,
                                ConversionPatternRewriter &r) const override {
    // Unit carries no information; lower it to the sentinel 0.
    r.replaceOpWithNewOp<arith::ConstantOp>(op, r.getI32IntegerAttr(0));
    return success();
  }
};

struct MmemNewLowering : OpConversionPattern<MmemNewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(MmemNewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    MemRefType memTy = heapType(getContext());
    if (op.getInverted()) {
      // Free the arena. The threaded-in heap handle is the memref to release.
      memref::DeallocOp::create(r, loc, adaptor.getInput());
      r.eraseOp(op);
      return success();
    }
    // Allocate and zero a fresh heap of `size` cells (default 1024 when the
    // importer supplied no size -- reference-faithful). The size is a runtime
    // i32 value (literal or computed); cast it to the index the allocator wants.
    Value size = adaptor.getSize();
    if (!size)
      size = arith::ConstantOp::create(r, loc, r.getI32IntegerAttr(
                                                   kDefaultHeapSize))
                 .getResult();
    Value n = arith::IndexCastOp::create(r, loc, r.getIndexType(), size);
    Value heap = func::CallOp::create(r, loc, TypeRange{memTy}, kNewFn,
                                      ValueRange{n})
                     .getResult(0);
    r.replaceOp(op, heap);
    return success();
  }
};

struct MmemAllocateLowering : OpConversionPattern<MmemAllocateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(MmemAllocateOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    Value heap = adaptor.getMemIn();
    Type i32 = r.getI32Type();
    if (op.getInverted()) {
      // Free: hand heap + pointer + size to the checked deallocator.
      func::CallOp::create(
          r, loc, TypeRange{}, kFreeFn,
          ValueRange{heap, adaptor.getPtrIn(), adaptor.getSize()});
      r.replaceOp(op, heap);
    } else {
      // Allocate: first-fit search over this heap returns the pointer.
      Value ptr = func::CallOp::create(r, loc, TypeRange{i32}, kMallocFn,
                                       ValueRange{heap, adaptor.getSize()})
                      .getResult(0);
      r.replaceOp(op, {heap, ptr});
    }
    return success();
  }
};

struct MmemReadwriteLowering : OpConversionPattern<MmemReadwriteOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(MmemReadwriteOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    Value heap = adaptor.getMemIn();
    // Row 0 is the cell-value array; the pointer is the column.
    Value row0 =
        arith::ConstantOp::create(r, loc, r.getIndexAttr(0)).getResult();
    Value idx = arith::IndexCastOp::create(r, loc, r.getIndexType(),
                                           adaptor.getPointer());
    Value old = memref::LoadOp::create(r, loc, heap, ValueRange{row0, idx});
    memref::StoreOp::create(r, loc, adaptor.getWriteVal(), heap,
                            ValueRange{row0, idx});
    r.replaceOp(op, {heap, old});
    return success();
  }
};

// Convert a function's signature AND every block's argument types -- not just
// the entry. A convertible payload (the heap, a relation reference) can thread
// through internal blocks (e.g. a vtable carrying a relation value from one
// dispatch block to the next), so those block arguments must convert too;
// populateFunctionOpInterfaceTypeConversionPattern only handles the entry.
struct FuncSignatureLowering : OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor,
                                ConversionPatternRewriter &r) const override {
    const TypeConverter *tc = getTypeConverter();
    FunctionType fnTy = op.getFunctionType();
    TypeConverter::SignatureConversion sig(fnTy.getNumInputs());
    SmallVector<Type> newResults;
    if (failed(tc->convertSignatureArgs(fnTy.getInputs(), sig)) ||
        failed(tc->convertTypes(fnTy.getResults(), newResults)) ||
        failed(r.convertRegionTypes(&op.getBody(), *tc, &sig)))
      return failure();
    r.modifyOpInPlace(op, [&] {
      op.setFunctionType(
          r.getFunctionType(sig.getConvertedTypes(), newResults));
    });
    return success();
  }
};

class ConvertHssaToArith
    : public impl::ConvertHssaToArithBase<ConvertHssaToArith> {
public:
  using impl::ConvertHssaToArithBase<
      ConvertHssaToArith>::ConvertHssaToArithBase;
  void runOnOperation() final {
    MLIRContext *ctx = &getContext();

    // Materialize the heap arena once. kMemRuntime is the single source of
    // truth for both zero-initialized globals (the value array and the
    // free/allocated flag array; see the MmemAllocate lowering) and the
    // first-fit allocate/free helpers -- clone it wholesale. One arena
    // suffices: a program creates it once.
    ModuleOp module = getOperation();
    if (!module.lookupSymbol(kNewFn)) {
      OwningOpRef<ModuleOp> rt =
          parseSourceString<ModuleOp>(kMemRuntime, ParserConfig(ctx));
      if (!rt) {
        module.emitError("failed to parse managed-memory runtime");
        return signalPassFailure();
      }
      // Clone to the top of the module, in order (globals then helpers), so the
      // arena leads and the user functions follow.
      OpBuilder b(ctx);
      b.setInsertionPointToStart(module.getBody());
      for (Operation &op : rt->getBody()->getOperations())
        b.setInsertionPointAfter(b.clone(op));
    }

    // The managed heap becomes a memref; unit and relation references become
    // i32 (unit = sentinel 0, relation = dispatch index).
    TypeConverter conv;
    conv.addConversion([](Type t) { return t; });
    conv.addConversion([ctx](MemType) { return heapType(ctx); });
    conv.addConversion(
        [ctx](UnitType) -> Type { return IntegerType::get(ctx, 32); });
    conv.addConversion(
        [ctx](RelType) -> Type { return IntegerType::get(ctx, 32); });
    // Converting a block argument's type (a mem/rel payload threaded across
    // blocks, e.g. a vtable's relation value) needs a materialization to bridge
    // the old and new types at the boundary; the leftover casts reconcile away
    // once both sides are converted.
    auto materialize = [](OpBuilder &b, Type resultType, ValueRange inputs,
                          Location loc) -> Value {
      if (inputs.size() != 1)
        return Value();
      return UnrealizedConversionCastOp::create(b, loc, resultType, inputs)
          .getResult(0);
    };
    conv.addSourceMaterialization(materialize);
    conv.addTargetMaterialization(materialize);

    // A stable dispatch index per relation, shared by rel_ref lowering (and,
    // once it exists, indirect-call dispatch): the relation's position in the
    // module. Deterministic across a run.
    llvm::StringMap<int64_t> relIds;
    int64_t nextId = 0;
    getOperation()->walk([&](func::FuncOp fn) {
      // The injected memory-runtime helpers are not HSSA relations; keep them
      // out of the dispatch-index space.
      if (!fn.getSymName().starts_with("__hssa_"))
        relIds[fn.getSymName()] = nextId++;
    });

    RewritePatternSet patterns(ctx);
    // i32-only ops: operands never change type, so plain rewrites suffice.
    patterns
        .add<UpdateLowering<AddOp, arith::AddIOp, arith::SubIOp>,
             UpdateLowering<SubOp, arith::SubIOp, arith::AddIOp>, XorLowering,
             SubfromLowering, EuclidivLowering, TestsAssertLowering,
             TestsFailLowering, BinArithLowering<MulOp, arith::MulIOp>,
             BinArithLowering<DivOp, arith::DivSIOp>,
             BinArithLowering<ModOp, arith::RemSIOp>,
             BinArithLowering<OrOp, arith::OrIOp>,
             BinArithLowering<AndOp, arith::AndIOp>,
             CmpLowering<LessOp, arith::CmpIPredicate::slt>,
             CmpLowering<LessEqualOp, arith::CmpIPredicate::sle>,
             CmpLowering<GreaterOp, arith::CmpIPredicate::sgt>,
             CmpLowering<GreaterEqualOp, arith::CmpIPredicate::sge>,
             CmpLowering<EqualOp, arith::CmpIPredicate::eq>,
             CmpLowering<NotEqualOp, arith::CmpIPredicate::ne>>(ctx);
    // Ops whose operands/results carry convertible types go through the
    // converter so mem/unit/rel become memref/i32/i32.
    patterns.add<IdLowering, DupLowering, DiscardLowering, CallLowering,
                 ExitLowering, CondExitLowering, SwitchLowering, UnitLowering,
                 MmemNewLowering, MmemAllocateLowering, MmemReadwriteLowering>(
        conv, ctx);
    patterns.add<RelRefLowering, CallIndirectLowering>(conv, ctx, relIds);
    // Convert function signatures (and all block args), calls, returns, and
    // branches so the heap and relation references thread across call and block
    // boundaries.
    patterns.add<FuncSignatureLowering>(conv, ctx);
    populateCallOpTypeConversionPattern(patterns, conv);
    populateReturnOpTypeConversionPattern(patterns, conv);
    populateBranchOpInterfaceTypeConversionPattern(patterns, conv);

    ConversionTarget target(*ctx);
    target.addIllegalDialect<HssaDialect>();
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                           func::FuncDialect, memref::MemRefDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return conv.isSignatureLegal(op.getFunctionType()) &&
             conv.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return conv.isLegal(op); });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return conv.isLegal(op); });
    target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
        [&](Operation *op) { return conv.isLegal(op); });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    // Fold away the boundary casts inserted while converting block-argument
    // types; a consistent 1:1 conversion leaves none behind.
    SmallVector<UnrealizedConversionCastOp> casts;
    getOperation()->walk(
        [&](UnrealizedConversionCastOp c) { casts.push_back(c); });
    reconcileUnrealizedCasts(casts);
    // With the HSSA ops gone, the HSSA-level function metadata is dead: strip
    // it so the lowered module is clean arith/cf/func.
    getOperation()->walk([](func::FuncOp fn) {
      fn->removeAttr("hssa.params");
      fn->removeAttr("hssa.class");
      fn->removeAttr("hssa.reversible");
    });
  }
};
} // namespace
} // namespace mlir::hssa
