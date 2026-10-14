//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reversibility: invert a function body (run it backwards), the --hssa-invert
// pass, and --hssa-resolve-uncalls (materialize inverse relations for inverted
// direct/indirect calls). Part of the dialect/RC core.
//
//===----------------------------------------------------------------------===//
#include "Hssa/HssaTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include "Hssa/HssaOps.h"
#include "Hssa/HssaPasses.h"

namespace mlir::hssa {
#define GEN_PASS_DEF_HSSAINVERT
#define GEN_PASS_DEF_HSSARESOLVEUNCALLS
#include "Hssa/HssaPasses.h.inc"

namespace {
// Invert `fn` in place. Each block inverts like a miniature function (its
// entry arguments are what it consumes, its terminator payload is what it
// produces), and the CFG reverses: entries and exits trade places, with the
// provenance constants on the edges determining successor order (index
// semantics: 0 = first label, 1 = second). The leading "hssa.params"
// arguments are read-only parameters; the inverse's signature is
// (params..., old results...) -> (old inputs...). Fails without modifying
// anything if the function contains a non-invertible op or an unsupported
// shape.
LogicalResult invertFunctionBody(func::FuncOp fn) {
  if (fn.isExternal())
    return failure();
  Region &region = fn.getBody();
  MLIRContext *ctx = fn.getContext();
  Type i32 = IntegerType::get(ctx, 32);

  unsigned numArgs = fn.getNumArguments();
  // The parameter/input split is semantic, not guessable: hssa.params is
  // required. A function without it is not invertible (rather than inverted
  // under a numArgs-1 guess that is wrong whenever ins != numArgs-1).
  auto paramsAttr = fn->getAttrOfType<IntegerAttr>("hssa.params");
  if (!paramsAttr)
    return failure();
  unsigned numParams = paramsAttr.getInt();
  if (numParams > numArgs)
    return failure();

  Block *entry = &region.front();
  SmallVector<Block *> oldBlocks;
  func::ReturnOp ret;
  for (Block &b : region) {
    if (b.empty())
      return failure();
    oldBlocks.push_back(&b);
    Operation *term = b.getTerminator();
    if (auto r = dyn_cast<func::ReturnOp>(term)) {
      if (ret)
        return failure();
      ret = r;
    } else if (!isa<ExitOp, CondExitOp, SwitchOp>(term)) {
      return failure();
    }
    for (Operation &op : b.without_terminator())
      if (!op.hasTrait<OpTrait::ConstantLike>() && !isa<HssaInvertible>(op))
        return failure();
  }
  if (!ret)
    return failure();
  Block *endBlock = ret->getBlock();

  DenseMap<Block *, unsigned> predCount;
  for (Block *b : oldBlocks) {
    Operation *t = b->getTerminator();
    for (unsigned i = 0; i < t->getNumSuccessors(); ++i)
      predCount[t->getSuccessor(i)]++;
  }
  if (predCount.lookup(entry) != 0)
    return failure();
  // A merge (two or more predecessors) carries a trailing provenance argument:
  // which predecessor it was entered from (a bit for 2, a 0-based index for K).
  auto hasProvenanceArg = [&](Block *b) { return predCount.lookup(b) >= 2; };
  for (Block *b : oldBlocks)
    if (hasProvenanceArg(b) && b->getNumArguments() == 0)
      return failure();

  // Strip the trailing provenance constant from an edge into a
  // two-predecessor block; `bit` receives its value (-1 if none).
  auto stripProvenanceBit = [&](Block *target, ValueRange operands,
                                int64_t &bit) -> FailureOr<SmallVector<Value>> {
    SmallVector<Value> payload(operands.begin(), operands.end());
    bit = -1;
    if (hasProvenanceArg(target)) {
      if (payload.empty())
        return failure();
      auto cst = payload.back().getDefiningOp<arith::ConstantOp>();
      if (!cst)
        return failure();
      bit = cast<IntegerAttr>(cst.getValue()).getInt();
      payload.pop_back();
    }
    return payload;
  };

  // The values a block hands off at its terminator, provenance constant
  // stripped
  // (`bit` receives the stripped index, -1 if none). An exit/cond_exit strips
  // the edge into its target; a switch's successors are merge blocks carrying
  // no such constant, so its payload passes through. For the single-payload
  // callers only -- a cond_exit's two edges carry the same values. (The inEdges
  // scan above walks every edge separately and does not use this.)
  auto producedPayload = [&](Operation *t,
                             int64_t &bit) -> FailureOr<SmallVector<Value>> {
    if (auto e = dyn_cast<ExitOp>(t))
      return stripProvenanceBit(e.getDest(), e.getValues(), bit);
    if (auto c = dyn_cast<CondExitOp>(t))
      return stripProvenanceBit(c.getFirst(), c.getFirstOperands(), bit);
    bit = -1;
    return SmallVector<Value>(cast<SwitchOp>(t).getPayload());
  };

  struct InEdge {
    Block *pred;
    unsigned succIndex; // which successor of pred's terminator this edge is
    int64_t bit;        // provenance index stripped at the merge (-1 if none)
  };
  DenseMap<Block *, SmallVector<InEdge, 2>> inEdges;
  for (Block *b : oldBlocks) {
    Operation *t = b->getTerminator();
    int64_t bit;
    if (auto e = dyn_cast<ExitOp>(t)) {
      if (failed(stripProvenanceBit(e.getDest(), e.getValues(), bit)))
        return failure();
      inEdges[e.getDest()].push_back({b, 0, bit});
    } else if (auto c = dyn_cast<CondExitOp>(t)) {
      if (failed(stripProvenanceBit(c.getFirst(), c.getFirstOperands(), bit)))
        return failure();
      inEdges[c.getFirst()].push_back({b, 0, bit});
      if (failed(stripProvenanceBit(c.getSecond(), c.getSecondOperands(), bit)))
        return failure();
      inEdges[c.getSecond()].push_back({b, 1, bit});
    } else if (auto s = dyn_cast<SwitchOp>(t)) {
      // Every case shares one payload; successors are single-predecessor
      // blocks (no provenance to strip), so bit = -1 for each.
      for (auto [i, succ] : llvm::enumerate(s.getCases())) {
        if (failed(stripProvenanceBit(succ, s.getPayload(), bit)))
          return failure();
        inEdges[succ].push_back({b, (unsigned)i, bit});
      }
    }
  }
  for (Block *b : oldBlocks) {
    auto &edges = inEdges[b];
    if (edges.empty() && b != entry)
      return failure();
    // A K-predecessor merge's incoming provenance indices must be a
    // permutation of 0..K-1, so the inverse branch orders its cases by index.
    if (edges.size() >= 2) {
      SmallVector<bool> seen(edges.size(), false);
      for (const InEdge &e : edges) {
        if (e.bit < 0 || (size_t)e.bit >= edges.size() || seen[e.bit])
          return failure();
        seen[e.bit] = true;
      }
    }
  }

  // A conditional exit must pass the same payload along both edges (modulo
  // the stripped provenance constants); the inverse is reconstructed from
  // the first list alone.
  for (Block *b : oldBlocks) {
    auto c = dyn_cast<CondExitOp>(b->getTerminator());
    if (!c)
      continue;
    int64_t bit;
    auto first = stripProvenanceBit(c.getFirst(), c.getFirstOperands(), bit);
    auto second = stripProvenanceBit(c.getSecond(), c.getSecondOperands(), bit);
    if (*first != *second)
      return failure();
  }

  // Pre-flight recoverability check, before any mutation: simulate the
  // backward reconstruction symbolically so that unsupported shapes become
  // a clean failure instead of an IRMapping assertion mid-rewrite. Each
  // block must consume exactly what it defines: walking its ops in reverse,
  // every produced value must still be pending, and at the end the pending
  // set must be exactly the block's consumable inputs. Parameters may only
  // be function parameters or constants (cloned per block).
  auto isFnParam = [&](Value v) {
    auto arg = dyn_cast<BlockArgument>(v);
    return arg && arg.getOwner() == entry && arg.getArgNumber() < numParams;
  };
  // Unit values carry no information: they can be created and dropped freely
  // (the interpreter's "empty" marker). Like the linearity checker, inversion
  // treats them as non-linear -- a produced unit need not be consumed, and a
  // consumed one is materialized fresh in the inverse rather than recovered.
  auto isUnit = [](Value v) { return isa<UnitType>(v.getType()); };
  for (Block *b : oldBlocks) {
    DenseSet<Value> pending;
    Operation *t = b->getTerminator();
    int64_t bit;
    // A constant is a pass-through only when it is cloned into this same
    // block's inverse, which the rebuild does per block. A constant defined
    // in another block has no clone here, so it is not exempt; it falls
    // through to the clean "not invertible" failure below instead of leaving
    // a dangling operand for the rebuild to crash on.
    auto isLocalConst = [&](Value v) {
      Operation *def = v.getDefiningOp();
      return def && def->hasTrait<OpTrait::ConstantLike>() &&
             def->getBlock() == b;
    };
    // Function parameters and constants in payloads are pass-throughs, not
    // values the block must recover.
    auto seedIf = [&](Value v) {
      if (!isLocalConst(v) && !isFnParam(v) && !isUnit(v))
        pending.insert(v);
    };
    if (b == endBlock) {
      for (Value v : ret.getOperands())
        seedIf(v);
    } else {
      // Bind the returned SmallVector to a named local: iterating directly over
      // *producedPayload(...) would range over a temporary that dies at the end
      // of the range-init expression (a dangling stack buffer).
      auto payload = producedPayload(t, bit);
      for (Value v : *payload)
        seedIf(v);
      // A branch also spends a selector, which the inverse merge must recover.
      if (auto c = dyn_cast<CondExitOp>(t))
        seedIf(c.getSelector());
      else if (auto s = dyn_cast<SwitchOp>(t))
        seedIf(s.getSelector());
    }
    for (Operation &op : llvm::reverse(b->without_terminator())) {
      if (op.hasTrait<OpTrait::ConstantLike>())
        continue;
      auto shape = cast<HssaGenericShape>(&op);
      for (Value v : shape.getProduced())
        if (!isUnit(v) && !pending.erase(v))
          return failure(); // produced but never consumed afterwards
      // Parameters are checked before this op's own consumed values are
      // registered, so a value that is both parameter and input of the same
      // statement (which backward reconstruction cannot order) is rejected.
      for (Value p : shape.getParameters())
        if (!isFnParam(p) && !isLocalConst(p) && !isUnit(p) &&
            !pending.contains(p))
          return failure();
      for (Value v : shape.getConsumed())
        if (!isUnit(v) && !isLocalConst(v) && !pending.insert(v).second)
          return failure(); // consumed twice
    }
    // What remains pending must be exactly the block's own inputs.
    unsigned argBase = (b == entry) ? numParams : 0;
    for (unsigned i = argBase; i < b->getNumArguments(); ++i)
      if (!pending.erase(b->getArgument(i)))
        return failure(); // dead block argument
    if (!pending.empty())
      return failure(); // value used from another block
  }

  SmallVector<Type> consumedTypes(fn.getArgumentTypes().drop_front(numParams));
  SmallVector<Type> oldResultTypes(fn.getResultTypes());
  Location fl = fn.getLoc();

  // One inverse block per block. The end block's inverse is the new entry:
  // it receives the parameters and the old results.
  DenseMap<Block *, Block *> inv;
  for (Block *b : oldBlocks) {
    Block *nb = new Block();
    region.push_back(nb);
    inv[b] = nb;
  }
  for (unsigned i = 0; i < numParams; ++i)
    inv[endBlock]->addArgument(fn.getArgumentTypes()[i], fl);
  for (Type t : oldResultTypes)
    inv[endBlock]->addArgument(t, fl);
  for (Block *b : oldBlocks) {
    if (b == endBlock)
      continue;
    Operation *t = b->getTerminator();
    int64_t bit;
    auto payload = producedPayload(t, bit);
    for (Value v : *payload)
      inv[b]->addArgument(v.getType(), fl);
    // A branch (cond_exit/switch) inverts to a merge of its successors, which
    // needs the provenance argument distinguishing them.
    if (isa<CondExitOp, SwitchOp>(t))
      inv[b]->addArgument(i32, fl);
  }

  IRMapping paramMap;
  for (unsigned i = 0; i < numParams; ++i)
    paramMap.map(entry->getArgument(i), inv[endBlock]->getArgument(i));

  for (Block *b : oldBlocks) {
    Block *nb = inv[b];
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(nb);
    IRMapping map = paramMap;
    // A unit operand the inverse needs is materialized fresh (units are
    // interchangeable), rather than recovered from the forward value map.
    auto makeUnit = [&]() -> Value {
      return UnitOp::create(builder, fl, UnitType::get(ctx));
    };

    // Seed: the block's produced values (its terminator payload) arrive as
    // the inverse block's arguments. A constant in the payload becomes a
    // check backward: the arriving value must equal it.
    SmallVector<std::pair<Value, Value>> constChecks;
    auto isConstant = [](Value v) {
      return v.getDefiningOp() &&
             v.getDefiningOp()->hasTrait<OpTrait::ConstantLike>();
    };
    auto seed = [&](ArrayRef<Value> payload, unsigned argBase) {
      for (auto [i, v] : llvm::enumerate(payload)) {
        Value arg = nb->getArgument(argBase + i);
        if (isConstant(v))
          constChecks.push_back({v, arg});
        else
          map.map(v, arg);
      }
    };
    Operation *t = b->getTerminator();
    int64_t bit;
    if (b == endBlock) {
      SmallVector<Value> payload(ret.getOperands());
      seed(payload, numParams);
    } else {
      auto payload = producedPayload(t, bit);
      seed(*payload, 0);
      // cond_exit (2 cases) and switch (K cases) both spend a selector to reach
      // the selected successor; on inversion the merge rebinds it from the
      // provenance argument that follows the payload.
      Value sel;
      if (auto c = dyn_cast<CondExitOp>(t))
        sel = c.getSelector();
      else if (auto s = dyn_cast<SwitchOp>(t))
        sel = s.getSelector();
      if (sel) {
        Value selArg = nb->getArgument(payload->size());
        if (isConstant(sel))
          constChecks.push_back({sel, selArg});
        else
          map.map(sel, selArg);
      }
    }

    SmallVector<Operation *> constants, oldOps;
    for (Operation &op : b->without_terminator()) {
      if (op.hasTrait<OpTrait::ConstantLike>())
        constants.push_back(&op);
      else
        oldOps.push_back(&op);
    }
    for (Operation *op : constants) {
      Operation *clone = builder.clone(*op);
      map.map(op->getResults(), clone->getResults());
    }
    for (auto [cv, arg] : constChecks) {
      auto check = DupOp::create(builder, cv.getLoc(), TypeRange{},
                                 ValueRange{map.lookup(cv), arg});
      check.setInverted(true);
    }

    for (Operation *op : llvm::reverse(oldOps)) {
      auto shape = cast<HssaGenericShape>(op);
      SmallVector<Value> newParams;
      for (Value p : shape.getParameters())
        newParams.push_back(isUnit(p) ? makeUnit() : map.lookupOrDefault(p));
      SmallVector<Value> newIns;
      for (Value i : shape.getProduced())
        newIns.push_back(isUnit(i) ? makeUnit() : map.lookup(i));
      auto newOp =
          cast<HssaInvertible>(op).buildInverse(builder, newParams, newIns);
      // A consumed constant re-emerges backward as an assertion: the
      // recovered value must equal it, and is consumed by the check.
      SmallVector<Value> producedNew(
          cast<HssaGenericShape>(newOp).getProduced());
      for (auto [i, cv] : llvm::enumerate(shape.getConsumed())) {
        if (isUnit(cv)) {
          continue; // a produced unit in the inverse is simply dropped
        } else if (isConstant(cv)) {
          auto check =
              DupOp::create(builder, cv.getLoc(), TypeRange{},
                            ValueRange{map.lookup(cv), producedNew[i]});
          check.setInverted(true);
        } else {
          map.map(cv, producedNew[i]);
        }
      }
    }

    // The inverse terminator mirrors the block's forward entry.
    if (b == entry) {
      SmallVector<Value> returns;
      for (unsigned i = numParams; i < b->getNumArguments(); ++i) {
        Value a = b->getArgument(i);
        returns.push_back(isUnit(a) ? makeUnit() : map.lookup(a));
      }
      func::ReturnOp::create(builder, fl, returns);
      continue;
    }
    unsigned nPayload = b->getNumArguments() - (hasProvenanceArg(b) ? 1 : 0);
    auto payloadOperands = [&]() {
      SmallVector<Value> operands;
      for (unsigned i = 0; i < nPayload; ++i) {
        Value a = b->getArgument(i);
        operands.push_back(isUnit(a) ? makeUnit() : map.lookup(a));
      }
      return operands;
    };
    // An inverse edge into inv[pred]: the payload, plus -- when pred's forward
    // terminator was a branch (so inv[pred] is a merge) -- the index telling
    // that merge which of its predecessors this edge is (pred's own succIndex).
    auto edgeOperands = [&](const InEdge &edge) {
      SmallVector<Value> operands = payloadOperands();
      if (isa<CondExitOp, SwitchOp>(edge.pred->getTerminator()))
        operands.push_back(arith::ConstantOp::create(
            builder, fl, IntegerAttr::get(i32, edge.succIndex)));
      return operands;
    };
    auto &edges = inEdges[b];
    if (edges.size() == 1) {
      ExitOp::create(builder, fl, edgeOperands(edges[0]), inv[edges[0].pred]);
    } else {
      // Order successors by provenance index: case k targets the predecessor
      // whose forward edge carried index k.
      SmallVector<const InEdge *> ordered(edges.size(), nullptr);
      for (const InEdge &e : edges)
        ordered[e.bit] = &e;
      Value sel = map.lookup(b->getArgument(nPayload));
      if (edges.size() == 2) {
        CondExitOp::create(builder, fl, sel, edgeOperands(*ordered[0]),
                           edgeOperands(*ordered[1]), inv[ordered[0]->pred],
                           inv[ordered[1]->pred]);
      } else {
        // N-ary: one shared payload handed to the selected successor. (The
        // generated merges have single-predecessor cases, so no case needs its
        // own provenance; a case that did would show up as a verifier error.)
        SmallVector<Block *> cases;
        for (const InEdge *e : ordered)
          cases.push_back(inv[e->pred]);
        SwitchOp::create(builder, fl, sel, payloadOperands(), cases);
      }
    }
  }

  for (Block *b : oldBlocks)
    b->dropAllReferences();
  for (Block *b : oldBlocks)
    b->erase();
  Block *newEntry = inv[endBlock];
  region.getBlocks().splice(region.begin(), region.getBlocks(),
                            Region::iterator(newEntry));

  SmallVector<Type> newArgTypes(fn.getArgumentTypes().take_front(numParams));
  newArgTypes.append(oldResultTypes);
  fn.setType(FunctionType::get(ctx, newArgTypes, consumedTypes));
  // The inverse is self-describing even if numParams came from the
  // all-but-last fallback: without this, re-inverting a multi-result
  // function would split its arguments wrongly.
  fn->setAttr("hssa.params",
              IntegerAttr::get(IntegerType::get(ctx, 64), numParams));
  return success();
}

// A call_indirect applies a relation value, so its candidate targets are the
// relations whose signature it could be invoking (an over-approximation by
// signature -- sound, since the runtime index selects among exactly these).
// Forward: the callee f has args == params++ins and results == outs. Inverted:
// it applies f's inverse, so f has args == params++outs and results == ins.
// Generated inverses (named <x>_inverse here) are excluded -- only source
// relations are dispatch targets.
static SmallVector<func::FuncOp>
dispatchCandidates(ModuleOp module, CallIndirectOp op,
                   const llvm::StringSet<> &generatedNames) {
  SmallVector<Type> args(op.getParams().getTypes().begin(),
                         op.getParams().getTypes().end());
  SmallVector<Type> res;
  if (op.getInverted()) {
    llvm::append_range(args, op.getResultTypes());
    llvm::append_range(res, op.getIns().getTypes());
  } else {
    llvm::append_range(args, op.getIns().getTypes());
    llvm::append_range(res, op.getResultTypes());
  }
  SmallVector<func::FuncOp> cands;
  for (auto f : module.getOps<func::FuncOp>()) {
    if (!f->hasAttr("hssa.params") || generatedNames.contains(f.getSymName()))
      continue;
    if (f.getFunctionType().getInputs() == ArrayRef<Type>(args) &&
        f.getFunctionType().getResults() == ArrayRef<Type>(res))
      cands.push_back(f);
  }
  return cands;
}

class HssaResolveUncalls
    : public impl::HssaResolveUncallsBase<HssaResolveUncalls> {
public:
  using impl::HssaResolveUncallsBase<
      HssaResolveUncalls>::HssaResolveUncallsBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    SymbolTable symbolTable(module);

    // Resolving an uncall materializes an inverse function; resolving an
    // indirect uncall materializes an inverse per candidate. Either can expose
    // further (in)direct uncalls inside the new inverse, so both are worklists
    // drained to a fixpoint. `generated` dedups by the ORIGINAL callee (not by
    // name), so a pre-existing <callee>_inverse is never called by mistake.
    SmallVector<CallOp> callWork;
    SmallVector<CallIndirectOp> indirectWork;
    llvm::StringMap<func::FuncOp> generated;
    llvm::StringSet<> generatedNames;
    bool bad = false;

    auto getInverse = [&](StringRef name, Operation *diag) -> func::FuncOp {
      auto it = generated.find(name);
      if (it != generated.end())
        return it->second;
      auto callee = symbolTable.lookup<func::FuncOp>(name);
      if (!callee) {
        diag->emitError("cannot uncall '") << name << "': no such function";
        bad = true;
        return nullptr;
      }
      func::FuncOp inverse = callee.clone();
      inverse.setSymName((name + "_inverse").str());
      symbolTable.insert(inverse); // uniquifies on collision
      generated[name] = inverse;
      generatedNames.insert(inverse.getSymName());
      if (failed(invertFunctionBody(inverse))) {
        diag->emitError("cannot uncall '")
            << name << "': function is not invertible";
        bad = true;
        return nullptr;
      }
      inverse.walk([&](CallOp n) {
        if (n.getInverted())
          callWork.push_back(n);
      });
      inverse.walk([&](CallIndirectOp n) {
        if (n.getInverted())
          indirectWork.push_back(n);
      });
      return inverse;
    };

    module.walk([&](CallOp c) {
      if (c.getInverted())
        callWork.push_back(c);
    });
    module.walk([&](CallIndirectOp c) {
      if (c.getInverted())
        indirectWork.push_back(c);
    });

    while (!bad && (!callWork.empty() || !indirectWork.empty())) {
      while (!bad && !callWork.empty()) {
        CallOp c = callWork.pop_back_val();
        func::FuncOp inverse = getInverse(c.getCallee(), c);
        if (!inverse)
          break;
        OpBuilder b(c);
        auto fwd =
            CallOp::create(b, c.getLoc(), c.getResultTypes(),
                           FlatSymbolRefAttr::get(ctx, inverse.getSymName()),
                           c.getParams(), c.getIns());
        c.replaceAllUsesWith(fwd.getResults());
        c.erase();
      }
      while (!bad && !indirectWork.empty()) {
        CallIndirectOp c = indirectWork.pop_back_val();
        if (!c.getInverted())
          continue;
        // An inverted indirect call applies the inverse of whichever candidate
        // the runtime index names: resolve each candidate's inverse and record
        // the index-keyed dispatch (key = the forward candidate, whose index
        // the switch tests; target = its inverse, what to call). Then the op is
        // an ordinary forward dispatch.
        auto cands = dispatchCandidates(module, c, generatedNames);
        if (cands.empty()) {
          c.emitError("cannot uncall indirectly: no relation matches the "
                      "callee signature");
          bad = true;
          break;
        }
        SmallVector<Attribute> keys, targets;
        for (func::FuncOp f : cands) {
          func::FuncOp inv = getInverse(f.getSymName(), c);
          if (!inv)
            break;
          keys.push_back(FlatSymbolRefAttr::get(ctx, f.getSymName()));
          targets.push_back(FlatSymbolRefAttr::get(ctx, inv.getSymName()));
        }
        if (bad)
          break;
        c->setAttr("hssa.dispatch_keys", ArrayAttr::get(ctx, keys));
        c->setAttr("hssa.dispatch_targets", ArrayAttr::get(ctx, targets));
        c.setInverted(false);
      }
    }
    if (bad)
      return signalPassFailure();

    // Forward indirect calls dispatch straight to their candidates (key ==
    // target), so convert-hssa-to-arith can emit the index switch.
    module.walk([&](CallIndirectOp c) {
      if (bad || c->hasAttr("hssa.dispatch_keys"))
        return;
      auto cands = dispatchCandidates(module, c, generatedNames);
      if (cands.empty()) {
        c.emitError("indirect call: no relation matches the callee signature");
        bad = true;
        return;
      }
      SmallVector<Attribute> syms;
      for (func::FuncOp f : cands)
        syms.push_back(FlatSymbolRefAttr::get(ctx, f.getSymName()));
      c->setAttr("hssa.dispatch_keys", ArrayAttr::get(ctx, syms));
      c->setAttr("hssa.dispatch_targets", ArrayAttr::get(ctx, syms));
    });
    if (bad)
      return signalPassFailure();
  }
};
class HssaInvert : public impl::HssaInvertBase<HssaInvert> {
public:
  using impl::HssaInvertBase<HssaInvert>::HssaInvertBase;
  void runOnOperation() final {
    ModuleOp module = getOperation();
    llvm::StringSet<> inverted;
    module.walk([&](func::FuncOp fn) {
      // Silent skip of non-invertible functions, until inversion becomes
      // demand-driven via uncall resolution.
      if (succeeded(invertFunctionBody(fn)))
        inverted.insert(fn.getSymName());
    });
    // A call's direction (its `inverted` flag) is relative to its callee's
    // body. buildInverse unconditionally flipped the flag as part of running
    // the caller backward, which is correct ONLY when the callee is left
    // untouched: `~B` then runs the original B backward. But whole-module
    // inversion also inverts B in place, so B now computes B^-1; the flipped
    // `~B` would run B^-1 backward = forward B -- the wrong composition. Where
    // the callee was itself inverted the two flips cancel: toggle the site
    // back so a forward call to the now-inverted B stays forward (and a source
    // uncall of B stays an uncall of the inverted B). Calls to functions NOT
    // inverted here (e.g. the external, self-inverse mmem.read) keep the flip,
    // so `~B` still means "run the untouched B backward".
    module.walk([&](CallOp c) {
      if (inverted.contains(c.getCallee()))
        c.setInverted(!c.getInverted());
    });
    // call_indirect targets a runtime relation VALUE, not a named symbol, so
    // the direct toggle above cannot apply. A first-class relation keeps its
    // SYMBOL identity across whole-module inversion (`rel_ref @m` still names
    // @m), and the call's flag only chooses forward/backward application of
    // that value -- so flipping the flag (what buildInverse already did) is the
    // right local move, and it is its own inverse, so invert;invert restores
    // every indirect site exactly. We deliberately do NOT cancel the flip here:
    // unlike a direct callee, @m's in-place inversion also changed its
    // signature, and the surrounding ~m vtable-uncall / forward-lookup pair is
    // reconstructed by --hssa-resolve-uncalls materializing m's inverse, not by
    // re-pointing the live dispatch. (A single whole-module invert of an
    // indirect-dispatch program is therefore not independently lowerable -- the
    // inverse of first-class dispatch is uncall resolution, as designed; the
    // coherent, lowerable result is the involution invert;invert.)
  }
};
} // namespace
} // namespace mlir::hssa
