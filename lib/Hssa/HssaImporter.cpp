//===- HssaImporter.cpp - textual HSSA -> hssa dialect ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Supported subset: multi-block relations with binary and N-ary control flow,
// the scalar builtins, calls to relations in the same file, first-class
// relations (!hssa.rel), and managed memory (mmem.new/allocate/readwrite) with
// !hssa.mem / i32 / !hssa.unit / !hssa.rel values whose types are inferred per
// relation (anchored by the memory primitives). Pair syntax is flattened to
// multiple values; a literal in an output or entry position becomes a
// check-and-consume (inverted dup). Entries are blocks: a multi-label entry's
// trailing block argument is the provenance value, fed the 0-based label index
// by each incoming edge. Cross-relation type inference runs as a callee-first
// signature pre-pass (so the heap and relation references thread across call
// boundaries), and the cell-polymorphic heap reader (mmem.read) is
// monomorphized per demanded cell type. Arrays, expression-level applications,
// wildcards, and variables bound to whole pairs are rejected.
//
//===----------------------------------------------------------------------===//

#include "Hssa/HssaImporter.h"
#include "Hssa/HssaDialect.h"
#include "Hssa/HssaOps.h"
#include "HssaSyntax.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include <limits>

using namespace mlir;
using namespace mlir::hssa;

// Strict mode: stamp every imported relation hssa.reversible, so
// --hssa-verify-reversible rejects anything outside the (D,D) fragment,
// i.e. treat the input as RSSA rather than full hybrid HSSA.
static llvm::cl::opt<bool> strictReversible(
    "hssa-strict-reversible",
    llvm::cl::desc("Mark every imported relation hssa.reversible"),
    llvm::cl::init(false));

namespace {

struct Line {
  StringRef text;
  unsigned number;
};

struct Importer {
  MLIRContext *context;
  StringRef filename;
  ModuleOp module;
  OpBuilder builder;
  Type i32;
  Type memType;
  Type unitType;
  Type relType;

  llvm::StringMap<Value> values;
  llvm::StringSet<> relNames;
  llvm::StringSet<> usedExitLabels;
  // Variable names carrying the !hssa.mem heap, inferred per relation.
  llvm::StringSet<> memVars;
  // Variable names carrying a !hssa.rel relation reference (output of a dup of
  // a relation name -- the vtable's first-class-relation mechanism).
  llvm::StringSet<> relVars;

  // Cross-file import state: the imported file contents (kept alive so the
  // relations' StringRefs stay valid) and the set of files already merged (by
  // path), so a diamond import reads each file once.
  SmallVector<std::unique_ptr<llvm::MemoryBuffer>> ownedBuffers;
  llvm::StringSet<> visitedFiles;

  // Each relation's inferred signature: the types of its parameter leaves, its
  // input leaves (the begin block's entry pattern) and its output leaves (the
  // end block's exit pattern). Computed once up front, in call-graph order, so
  // a call site can type its arguments and results from the callee -- the heap
  // and relation references thread across call boundaries this way.
  struct RelSig {
    SmallVector<Type> params, inputs, outputs;
    bool operator==(const RelSig &o) const {
      return params == o.params && inputs == o.inputs && outputs == o.outputs;
    }
  };
  llvm::StringMap<RelSig> relSigs;

  // A relation is cell-polymorphic when a caller reads a value through it whose
  // type (a relation reference from a vtable slot) differs from the relation's
  // generic signature. Each distinct demanded signature becomes a monomorphic
  // specialization: a clone of the relation's body rebuilt with the cell-typed
  // variables forced to the demanded types.
  struct Spec {
    StringRef callee; // the generic relation being specialized
    RelSig demanded;  // the demanded signature (callee's forward frame)
    std::string name; // the specialization's symbol
  };
  SmallVector<Spec> specs;
  // While building a specialization, the cell-typed variables are forced here;
  // typeOf consults it before the per-relation inference.
  llvm::StringMap<Type> forcedTypes;

  // The specialization of `callee` for a demanded signature, if one exists.
  // specs holds only a handful of entries, so a scan beats maintaining a
  // separate index keyed on a stringified signature.
  const Spec *findSpec(StringRef callee, const RelSig &demand) const {
    for (const Spec &s : specs)
      if (s.callee == callee && s.demanded == demand)
        return &s;
    return nullptr;
  }

  Importer(MLIRContext *context, StringRef filename, ModuleOp module)
      : context(context), filename(filename), module(module), builder(context),
        i32(IntegerType::get(context, 32)), memType(MemType::get(context)),
        unitType(UnitType::get(context)), relType(RelType::get(context)) {}

  // The type a named variable carries: a forced type while building a
  // specialization, else the heap/relation if inferred so, else i32.
  Type typeOf(StringRef name) {
    auto f = forcedTypes.find(name);
    if (f != forcedTypes.end())
      return f->second;
    if (relVars.contains(name))
      return relType;
    return memVars.contains(name) ? memType : i32;
  }

  // The type an output/input token denotes: unit for "()", i32 for a literal,
  // otherwise the variable's inferred type.
  Type tokenType(StringRef token) {
    if (token == "()")
      return unitType;
    if (isInteger(token))
      return i32;
    return typeOf(token);
  }

  Location loc(const Line &line) {
    return FileLineColLoc::get(context, filename, line.number, 1);
  }

  InFlightDiagnostic error(const Line &line, const Twine &message) {
    return emitError(loc(line), message);
  }

  // Flatten a pattern into its scalar leaves. "()" contributes nothing;
  // "(a, (b, c))" and the paren-less "a, b" both contribute their scalars
  // (pairs are comma-folds; parentheses only group).
  // keepUnit: treat "()" as a unit-valued leaf rather than nothing. Used for
  // the cell value of mmem.readwrite, where "()" denotes the unit value; every
  // other position reads "()" as the empty pattern.
  LogicalResult flattenPattern(const Line &line, StringRef text,
                               SmallVectorImpl<StringRef> &leaves,
                               bool keepUnit = false) {
    text = text.trim();
    if (text == "()") {
      if (keepUnit)
        leaves.push_back(text);
      return success();
    }
    if (text.empty())
      return success();
    SmallVector<StringRef> parts = splitTop(text);
    if (parts.size() > 1) {
      for (StringRef part : parts) {
        if (part.empty())
          return error(line, "malformed pattern '" + text + "'");
        if (failed(flattenPattern(line, part, leaves, keepUnit)))
          return failure();
      }
      return success();
    }
    if (text.starts_with("(") && text.ends_with(")"))
      return flattenPattern(line, text.drop_front().drop_back(), leaves,
                            keepUnit);
    if (!isInteger(text) && !llvm::all_of(text, [](char c) {
          return llvm::isAlnum(c) || c == '_' || c == '.';
        }))
      return error(line, "malformed pattern '" + text + "'");
    leaves.push_back(text);
    return success();
  }

  LogicalResult defineValue(const Line &line, StringRef name, Value value) {
    if (!values.insert({name, value}).second)
      return error(line, "variable '" + name + "' is defined twice");
    return success();
  }

  FailureOr<Value> useValue(const Line &line, StringRef token) {
    if (token == "()")
      return UnitOp::create(builder, loc(line), unitType).getResult();
    // A relation used as a value (e.g. dup's parameter in a vtable)
    // materializes as a typed relation reference; lowering turns it into a
    // dispatch index.
    if (relNames.contains(token))
      return RelRefOp::create(builder, loc(line), relType,
                              FlatSymbolRefAttr::get(context, token))
          .getResult();
    if (isInteger(token)) {
      int64_t constant;
      (void)token.getAsInteger(10, constant);
      if (constant > std::numeric_limits<int32_t>::max() ||
          constant < std::numeric_limits<int32_t>::min())
        return error(line, "literal '" + token + "' does not fit in i32"),
               failure();
      return arith::ConstantOp::create(builder, loc(line),
                                       IntegerAttr::get(i32, constant))
          .getResult();
    }
    auto it = values.find(token);
    if (it == values.end())
      return error(line, "unknown variable '" + token + "'"), failure();
    return it->second;
  }

  // Bind op results to the output pattern. A literal output is an assertion:
  // check-and-consume the result against the constant (inverted dup).
  LogicalResult bindOutputs(const Line &line, ArrayRef<StringRef> outTokens,
                            ValueRange results) {
    for (auto [token, result] : llvm::zip_equal(outTokens, results)) {
      // A "()" output means the result is unit: it carries no information and
      // is exempt from linearity, so it is simply left unconsumed.
      if (token == "()")
        continue;
      if (!isInteger(token)) {
        if (failed(defineValue(line, token, result)))
          return failure();
        continue;
      }
      auto constant = useValue(line, token);
      if (failed(constant))
        return failure();
      auto check = DupOp::create(builder, loc(line), TypeRange{},
                                 ValueRange{*constant, result});
      check.setInverted(true);
    }
    return success();
  }

  LogicalResult parseStatement(const Line &line) {
    StmtSplit s = splitStmt(line.text);
    if (!s.ok)
      return error(line, "statement is missing its ':=' / '=:' separators");
    StringRef lhs = s.lhs, inText = s.inText, opName = s.opName;
    bool inverted = s.inverted;
    SmallVector<StringRef> paramTokens = splitTop(s.paramExpr);

    // readwrite's cell value may be the unit value, written textually as "()";
    // everywhere else "()" is the empty pattern.
    bool keepUnit = opName == "mmem.readwrite";
    SmallVector<StringRef> outTokens, inTokens;
    if (failed(flattenPattern(line, lhs.trim(), outTokens, keepUnit)) ||
        failed(flattenPattern(line, inText.trim(), inTokens, keepUnit)))
      return failure();

    // dup and id are polymorphic over any value, and the reference writes a
    // multi-value copy or identity as a single op over a pair (e.g. dup a, b).
    // We realize it as a per-leaf fan-out: dup/id each leaf independently. A
    // pair handed to dup is a group of read-only parameters that the program
    // reuses (toffoli later runs ~dup a, b), so each leaf must stay an
    // independent read rather than be bundled into one consumed value. Matches
    // the interpreter leaf-for-leaf.
    if (opName == "dup" || opName == "id") {
      SmallVector<StringRef> paramLeaves;
      if (failed(flattenPattern(line, s.paramExpr, paramLeaves)))
        return failure();
      unsigned copies = opName == "dup"
                            ? (inverted ? inTokens.size() : outTokens.size())
                            : outTokens.size();
      if (copies > 1 || paramLeaves.size() > 1)
        return emitNaryInfoOp(line, opName, inverted, paramLeaves, inTokens,
                              outTokens);
    }

    // mmem.new carries an OPTIONAL size parameter K (an ADDITIVE extension
    // beyond the reference, whose mmem.new is argument-less and fixed at 1024).
    // Its builtin shape declares 0 params, so the generic arity check below
    // cannot see K; handle it here. K may be a literal or a computed i32 value.
    // Absent -> default to a 1024 constant, reproducing the reference heap and
    // keeping the lowering's single size-driven path.
    if (opName == "mmem.new")
      return emitMmemNew(line, inverted, paramTokens, inTokens, outTokens);

    // Params may be pair-structured: a multi-parameter method call emits nested
    // pairs, e.g. `f _this, ((p, osize), counter), csize`. Flatten to scalar
    // leaves -- pairs are recursive structure (not data), so this matches both
    // the relation definition's own flattened parameter pattern and the
    // input/output handling above. (splitTop alone leaves `((p, osize), counter)`
    // as one token, which useValue cannot resolve.)
    SmallVector<StringRef> paramLeaves;
    if (failed(flattenPattern(line, s.paramExpr, paramLeaves)))
      return failure();
    SmallVector<Value> params;
    for (StringRef token : paramLeaves) {
      auto value = useValue(line, token);
      if (failed(value))
        return failure();
      params.push_back(*value);
    }
    SmallVector<Value> ins;
    for (StringRef token : inTokens) {
      auto value = useValue(line, token);
      if (failed(value))
        return failure();
      ins.push_back(*value);
    }

    if (auto shape = builtinShape(opName))
      return emitBuiltin(line, opName, *shape, inverted, params, ins,
                         outTokens);
    if (relNames.contains(opName))
      return emitCall(line, opName, inverted, params, ins, outTokens);
    // An indirect call: the operator is a local variable holding a relation
    // value (produced by dup, or read from a vtable), applied as a first-class
    // relation. This is the virtual-dispatch mechanism.
    if (Value callee = values.lookup(opName))
      if (isa<RelType>(callee.getType()))
        return emitCallIndirect(line, callee, inverted, params, ins, outTokens);
    return error(line, "unsupported operation '" + opName + "'");
  }

  // Emit a multi-value dup/id as a per-leaf fan-out (see parseStatement): each
  // leaf becomes an independent 1-ary op, pairing parameters/inputs with
  // outputs by position. bindOutputs still handles a literal output leaf as a
  // check-and-consume.
  LogicalResult emitNaryInfoOp(const Line &line, StringRef opName, bool inverted,
                               ArrayRef<StringRef> paramLeaves,
                               ArrayRef<StringRef> inTokens,
                               ArrayRef<StringRef> outTokens) {
    if (opName == "id") {
      if (inTokens.size() != outTokens.size())
        return error(line, "id maps N inputs to N outputs");
      for (auto [inTok, outTok] : llvm::zip(inTokens, outTokens)) {
        auto in = useValue(line, inTok);
        if (failed(in))
          return failure();
        Value out = IdOp::create(builder, loc(line), (*in).getType(), *in);
        if (failed(bindOutputs(line, {outTok}, {out})))
          return failure();
      }
      return success();
    }
    // forward: out_i := dup param_i =: ()   inverted: () := dup param_i =: in_i
    if (!inverted) {
      if (paramLeaves.size() != outTokens.size())
        return error(line, "dup copies N parameters to N outputs");
      for (auto [pTok, outTok] : llvm::zip(paramLeaves, outTokens)) {
        auto p = useValue(line, pTok);
        if (failed(p))
          return failure();
        Value out = DupOp::create(builder, loc(line), TypeRange{(*p).getType()},
                                  ValueRange{*p})
                        .getOutput();
        if (failed(bindOutputs(line, {outTok}, {out})))
          return failure();
      }
      return success();
    }
    if (paramLeaves.size() != inTokens.size())
      return error(line, "inverted dup checks N inputs against N parameters");
    for (auto [pTok, inTok] : llvm::zip(paramLeaves, inTokens)) {
      auto p = useValue(line, pTok);
      auto in = useValue(line, inTok);
      if (failed(p) || failed(in))
        return failure();
      DupOp::create(builder, loc(line), TypeRange{}, ValueRange{*p, *in})
          .setInverted(true);
    }
    return success();
  }

  LogicalResult emitCallIndirect(const Line &line, Value callee, bool inverted,
                                 ArrayRef<Value> params, ArrayRef<Value> ins,
                                 ArrayRef<StringRef> outTokens) {
    SmallVector<Type> resultTypes;
    for (StringRef token : outTokens)
      resultTypes.push_back(tokenType(token));
    auto op = CallIndirectOp::create(builder, loc(line), resultTypes, callee,
                                     params, ins);
    op.setInverted(inverted);
    return bindOutputs(line, outTokens, op.getOuts());
  }

  // Whether a builtin op is self-inverse, read from its SelfInverse trait so
  // there is no duplicated name list to drift from the op definitions.
  bool isSelfInverse(StringRef opName) {
    if (auto name =
            RegisteredOperationName::lookup(("hssa." + opName).str(), context))
      return name->hasTrait<SelfInverse>();
    return false;
  }

  // mmem.new, with its optional size parameter K. Forward: `m := mmem.new K`
  // (K optional) produces a heap of K cells. Inverted: `() := ~mmem.new K =: m`
  // frees it. K threads unchanged to the inverse as a read-only parameter. When
  // K is absent it defaults to a 1024 constant, so every program behaves exactly
  // as the reference (fixed 1024 heap) and the lowering has one size-driven path.
  LogicalResult emitMmemNew(const Line &line, bool inverted,
                            ArrayRef<StringRef> paramTokens,
                            ArrayRef<StringRef> inTokens,
                            ArrayRef<StringRef> outTokens) {
    if (paramTokens.size() > 1)
      return error(line, "mmem.new takes at most one size parameter");
    Value size;
    if (paramTokens.empty()) {
      size = arith::ConstantOp::create(builder, loc(line),
                                       IntegerAttr::get(i32, 1024))
                 .getResult();
    } else {
      auto s = useValue(line, paramTokens.front());
      if (failed(s))
        return failure();
      if (!(*s).getType().isInteger(32))
        return error(line, "mmem.new size must be an integer");
      size = *s;
    }
    unsigned wantIns = inverted ? 1 : 0;
    unsigned wantOuts = inverted ? 0 : 1;
    if (inTokens.size() != wantIns)
      return error(line, (inverted ? "~" : "") + Twine("mmem.new consumes ") +
                             Twine(wantIns) + " value(s)");
    if (outTokens.size() != wantOuts)
      return error(line, (inverted ? "~" : "") + Twine("mmem.new produces ") +
                             Twine(wantOuts) + " value(s)");
    Value input;
    if (inverted) {
      auto v = useValue(line, inTokens.front());
      if (failed(v))
        return failure();
      input = *v;
    }
    SmallVector<Type> resultTypes;
    for (StringRef token : outTokens)
      resultTypes.push_back(tokenType(token));
    // Build explicitly through OperationState so the AttrSizedOperandSegments
    // bookkeeping (size present, input present-iff-inverted) is set correctly.
    OperationState state(loc(line), MmemNewOp::getOperationName());
    state.addOperands(size);
    if (input)
      state.addOperands(input);
    state.addTypes(resultTypes);
    state.addAttribute(
        MmemNewOp::getOperandSegmentSizeAttr(),
        builder.getDenseI32ArrayAttr({1, input ? 1 : 0}));
    if (inverted)
      state.addAttribute("inverted", UnitAttr::get(context));
    Operation *op = builder.create(state);
    return bindOutputs(line, outTokens, op->getResults());
  }

  LogicalResult emitBuiltin(const Line &line, StringRef opName,
                            const BuiltinShape &shape, bool inverted,
                            ArrayRef<Value> params, ArrayRef<Value> ins,
                            ArrayRef<StringRef> outTokens) {
    if (inverted && !shape.invertible)
      return error(line, "'~" + opName + "' is not executable");
    if (params.size() != shape.numParams)
      return error(line, opName + " takes " + Twine(shape.numParams) +
                             " parameter(s)");
    unsigned wantIns = inverted ? shape.invIns : shape.fwdIns;
    unsigned wantOuts = inverted ? shape.invOuts : shape.fwdOuts;
    if (ins.size() != wantIns)
      return error(line, (inverted ? "~" : "") + opName + " consumes " +
                             Twine(wantIns) + " value(s)");
    if (outTokens.size() != wantOuts)
      return error(line, (inverted ? "~" : "") + opName + " produces " +
                             Twine(wantOuts) + " value(s)");

    // A self-inverse op (id, subfrom, xor, tests.fail, mmem.readwrite) is its
    // own inverse, so its textual ~ is meaningless and it carries no usable
    // flag. Queried from the op's SelfInverse trait, not a duplicated name list.
    bool flag = inverted && !isSelfInverse(opName);

    SmallVector<Type> resultTypes;
    for (StringRef token : outTokens)
      resultTypes.push_back(tokenType(token));

    OperationState state(loc(line), "hssa." + opName.str());
    state.addOperands(params);
    state.addOperands(ins);
    state.addTypes(resultTypes);
    if (flag)
      state.addAttribute("inverted", UnitAttr::get(context));
    Operation *op = builder.create(state);
    return bindOutputs(line, outTokens, op->getResults());
  }

  LogicalResult emitCall(const Line &line, StringRef callee, bool inverted,
                         ArrayRef<Value> params, ArrayRef<Value> ins,
                         ArrayRef<StringRef> outTokens) {
    SmallVector<Type> resultTypes;
    for (StringRef token : outTokens)
      resultTypes.push_back(tokenType(token));
    // Route to a monomorphic specialization when this call site demands a
    // signature other than the callee's generic one -- a value of relation type
    // read through a cell-polymorphic helper (mmem.read). The demanded sig is
    // the callee's forward frame; an inverted call swaps the input/output
    // roles.
    RelSig demand;
    for (Value p : params)
      demand.params.push_back(p.getType());
    SmallVector<Type> insT;
    for (Value v : ins)
      insT.push_back(v.getType());
    demand.inputs = inverted ? resultTypes : insT;
    demand.outputs = inverted ? insT : SmallVector<Type>(resultTypes);
    StringRef target = callee;
    if (const Spec *s = findSpec(callee, demand))
      target = s->name;
    auto op =
        CallOp::create(builder, loc(line), resultTypes,
                       FlatSymbolRefAttr::get(context, target), params, ins);
    op.setInverted(inverted);
    return bindOutputs(line, outTokens, op.getOuts());
  }

  // One basic block of a relation, as text plus parse results.
  struct BlockText {
    Line entry, exit;
    SmallVector<Line> stmts;
    SmallVector<StringRef> entryLabels, entryLeaves;
    StringRef entrySel;
    SmallVector<StringRef> exitLabels, exitLeaves;
    StringRef exitSel;
    Block *block = nullptr;
    // A one-label block threads its selector as a trailing block argument when
    // the selector carries information (a variable to bind, or a value to check
    // against the entry's literal) rather than the trivial unconditional 0.
    // This is the degenerate N-ary dispatch form the vtable uses.
    bool threadsSel = false;
    // A merge binds a provenance value: which of its K predecessors it was
    // entered from (a bit for K=2, a 0-based index for K>2).
    bool isMerge() const { return entryLabels.size() >= 2; }
  };

  LogicalResult parseBlockTexts(StringRef name, ArrayRef<Line> lines,
                                SmallVectorImpl<BlockText> &blocks) {
    enum { WantEntry, WantExit } state = WantEntry;
    for (const Line &line : lines) {
      bool isEntry = line.text.ends_with("<-") &&
                     (line.text.contains(":=") || line.text.contains("=:"));
      bool isExit = line.text.starts_with("->");
      if (state == WantEntry) {
        if (!isEntry)
          return error(line, "expected a block entry 'pattern, sel := "
                             "label(s) <-'");
        blocks.push_back(BlockText{});
        blocks.back().entry = line;
        state = WantExit;
        continue;
      }
      if (isEntry)
        return error(line, "block entry before the previous block's exit");
      if (!isExit) {
        blocks.back().stmts.push_back(line);
        continue;
      }
      blocks.back().exit = line;
      state = WantEntry;
    }
    if (state == WantExit)
      return error(blocks.back().entry, "block is missing its exit line");
    if (blocks.empty())
      return emitError(FileLineColLoc::get(context, filename, 0, 0),
                       "relation '" + name + "' has no body");

    for (BlockText &bt : blocks) {
      bool entryHasDelim;
      auto [lhs, rhs] = splitDelim(bt.entry.text, entryHasDelim);
      StringRef labelsText = rhs.trim().rtrim("<-").trim();
      bt.entryLabels = splitTop(labelsText);
      if (bt.entryLabels.empty())
        return error(bt.entry, "an entry needs at least one label");
      SmallVector<StringRef> parts = splitTop(lhs);
      if (parts.size() < 2)
        return error(bt.entry, "expected entry pattern 'values, sel'");
      bt.entrySel = parts.back();
      for (StringRef part : ArrayRef<StringRef>(parts).drop_back())
        if (failed(flattenPattern(bt.entry, part, bt.entryLeaves)))
          return failure();
      // A merge entry (two or more labels) binds the provenance value, so its
      // selector must be a variable. A one-label entry's selector may be the
      // trivial 0, a variable (bound), or a literal (checked), the threaded
      // selector form.
      if (bt.isMerge() && isInteger(bt.entrySel))
        return error(bt.entry, "merge entries bind a selector variable");
      if (bt.entryLabels == ArrayRef<StringRef>{"begin"} && bt.entrySel != "0")
        return error(bt.entry, "the 'begin' entry takes selector 0");

      StringRef exitText = bt.exit.text;
      (void)exitText.consume_front("->");
      bool exitHasDelim;
      auto [labels, rhsText] = splitDelim(exitText, exitHasDelim);
      if (!exitHasDelim)
        return error(bt.exit, "exit is missing its ':=' / '=:' separator");
      bt.exitLabels = splitTop(labels.trim());
      if (bt.exitLabels.empty())
        return error(bt.exit, "an exit needs at least one label");
      SmallVector<StringRef> rparts = splitTop(rhsText);
      if (rparts.size() < 2)
        return error(bt.exit, "expected exit values 'values, sel'");
      bt.exitSel = rparts.back();
      for (StringRef part : ArrayRef<StringRef>(rparts).drop_back())
        if (failed(flattenPattern(bt.exit, part, bt.exitLeaves)))
          return failure();
      // A multi-label exit consumes a selector to pick the branch; a one-label
      // exit sends a selector value along the edge (0 when unconditional).
      if (bt.exitLabels.size() >= 2 && bt.exitSel == "0")
        return error(bt.exit, "multi-label exits consume a selector variable");
      if (bt.exitLabels == ArrayRef<StringRef>{"end"} && bt.exitSel != "0")
        return error(bt.exit, "the 'end' exit takes selector 0");
    }
    return success();
  }

  // The heap is the first scalar on either side of a memory statement.
  static StringRef memAnchor(StringRef side) {
    SmallVector<StringRef> parts = splitTop(side.trim());
    if (parts.empty())
      return StringRef();
    StringRef p = parts[0].trim();
    while (p.starts_with("(") && p.ends_with(")"))
      p = p.drop_front().drop_back().trim();
    parts = splitTop(p);
    return parts.empty() ? StringRef() : parts[0].trim();
  }

  // The inferred types of a pattern's scalar leaves (units skipped, matching
  // how relation signatures are flattened).
  SmallVector<Type> leafTypes(StringRef patternText) {
    SmallVector<StringRef> leaves;
    collectLeaves(patternText, leaves);
    SmallVector<Type> types;
    for (StringRef l : leaves)
      types.push_back(tokenType(l));
    return types;
  }

  static StringRef firstLeaf(StringRef patternText) {
    SmallVector<StringRef> leaves;
    collectLeaves(patternText, leaves);
    return leaves.empty() ? StringRef() : leaves.front();
  }

  // Record that `callee` is applied with a signature other than its generic
  // one, so a monomorphic specialization must be built. Deduplicated by the
  // demanded signature.
  void recordSpec(StringRef callee, const RelSig &demand) {
    if (findSpec(callee, demand))
      return;
    specs.push_back(
        {callee, demand, (callee + ".spec" + Twine(specs.size())).str()});
  }

  // Scan a relation's calls for cell-polymorphic use: a direct call whose
  // argument/result types (as this caller infers them, including use-based
  // relation typing) differ from the callee's generic signature needs a
  // specialization. Run with this caller's inference active.
  void discoverSpecs(ArrayRef<BlockText> blocks) {
    for (const BlockText &bt : blocks)
      for (const Line &line : bt.stmts) {
        StmtSplit s = splitStmt(line.text);
        if (!s.ok)
          continue;
        auto genIt = relSigs.find(s.opName);
        if (genIt == relSigs.end())
          continue;
        RelSig demand;
        demand.params = leafTypes(s.paramExpr);
        demand.inputs = leafTypes(s.inverted ? s.lhs : s.inText);
        demand.outputs = leafTypes(s.inverted ? s.inText : s.lhs);
        if (!(demand == genIt->second))
          recordSpec(s.opName, demand);
      }
  }

  // Force the cell-typed variables of a specialization: seed the demanded
  // signature onto the parameter/input/output leaves, then propagate across
  // same-type ops (dup/id) to a fixpoint so an internal cell carried by a dup
  // (e.g. mmem.read's _readvalue/_readcopy) takes the demanded type too.
  void computeForcedTypes(const BlockText &begin, const BlockText &end,
                          const RelSig &d, ArrayRef<StringRef> relParams,
                          ArrayRef<BlockText> blocks) {
    forcedTypes.clear();
    auto seed = [&](ArrayRef<StringRef> leaves, ArrayRef<Type> types) {
      if (leaves.size() != types.size())
        return;
      for (auto [l, t] : llvm::zip(leaves, types))
        if (!isInteger(l) && l != "()")
          forcedTypes[l] = t;
    };
    seed(relParams, d.params);
    seed(begin.entryLeaves, d.inputs);
    seed(end.exitLeaves, d.outputs);
    SmallVector<std::pair<StringRef, StringRef>> sameType;
    for (const BlockText &bt : blocks)
      for (const Line &line : bt.stmts) {
        StmtSplit s = splitStmt(line.text);
        if (!s.ok)
          continue;
        if (s.opName == "dup") {
          StringRef a = firstLeaf(s.paramExpr);
          StringRef b = firstLeaf(s.inverted ? s.inText : s.lhs);
          if (!a.empty() && !b.empty())
            sameType.push_back({a, b});
        } else if (s.opName == "id") {
          StringRef a = firstLeaf(s.inText), b = firstLeaf(s.lhs);
          if (!a.empty() && !b.empty())
            sameType.push_back({a, b});
        }
      }
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto [a, b] : sameType) {
        Type ta = forcedTypes.lookup(a), tb = forcedTypes.lookup(b);
        if (ta && !tb) {
          forcedTypes[b] = ta;
          changed = true;
        } else if (tb && !ta) {
          forcedTypes[a] = tb;
          changed = true;
        }
      }
    }
  }

  // Mark each leaf of a pattern whose matching signature slot carries the heap
  // or a relation reference. Arity mismatches are left for the build to report.
  void anchorByTypes(StringRef patternText, ArrayRef<Type> types) {
    SmallVector<StringRef> leaves;
    collectLeaves(patternText, leaves);
    if (leaves.size() != types.size())
      return;
    for (auto [leaf, ty] : llvm::zip(leaves, types)) {
      if (isInteger(leaf))
        continue;
      if (ty == memType)
        memVars.insert(leaf);
      else if (ty == relType)
        relVars.insert(leaf);
    }
  }

  // Anchor the inference: a variable in the heap position of a memory
  // primitive (new/allocate/readwrite) carries !hssa.mem; the output of a dup
  // of a relation name carries !hssa.rel; and a call's arguments/results take
  // their types from the callee's signature (relSigs, filled in call-graph
  // order beforehand) so the heap and relation values thread across calls.
  void inferMemVars(ArrayRef<BlockText> blocks) {
    memVars.clear();
    relVars.clear();
    for (const BlockText &bt : blocks)
      for (const Line &line : bt.stmts) {
        StmtSplit s = splitStmt(line.text);
        if (!s.ok)
          continue;
        if (isMemoryOp(s.opName)) {
          for (StringRef side : {s.lhs, s.inText}) {
            StringRef a = memAnchor(side);
            if (!a.empty() && a != "()" && !isInteger(a))
              memVars.insert(a);
          }
          continue;
        }
        // dup of a relation name produces a relation value: anchor the output.
        if (s.opName == "dup") {
          if (relNames.contains(s.paramExpr)) {
            StringRef out = memAnchor(s.lhs);
            if (!out.empty() && out != "()" && !isInteger(out))
              relVars.insert(out);
          }
          continue;
        }
        // A direct call to a user relation: type its argument and result leaves
        // from the callee's signature. An inverted call (~f) swaps the input
        // and output roles. Indirect calls (callee is a value, not a name) are
        // not anchored here; their result type is inferred where they produce.
        auto sigIt = relSigs.find(s.opName);
        if (sigIt != relSigs.end()) {
          const RelSig &sig = sigIt->second;
          anchorByTypes(s.paramExpr, sig.params);
          anchorByTypes(s.inText, s.inverted ? sig.outputs : sig.inputs);
          anchorByTypes(s.lhs, s.inverted ? sig.inputs : sig.outputs);
          continue;
        }
        // An indirect call: the operator is neither a builtin nor a named
        // relation, so it is a local variable holding a relation value. Only a
        // relation can be applied, so the operator carries !hssa.rel (use-based
        // inference). The result's type follows from how the result is used.
        if (!s.opName.empty() && !builtinShape(s.opName) &&
            !relNames.contains(s.opName))
          relVars.insert(s.opName);
      }
  }

  // The user relations a relation calls directly (the op of a statement is a
  // relation name). Used to order the signature pre-pass callees-first. A dup
  // of a relation references it but does not call it, so it is not an edge.
  void collectCallees(ArrayRef<Line> body, SmallVectorImpl<StringRef> &out) {
    for (const Line &line : body) {
      StmtSplit s = splitStmt(line.text);
      if (s.ok && relNames.contains(s.opName))
        out.push_back(s.opName);
    }
  }

  // The begin block (entering from 'begin') is always the first; locate the
  // unique block exiting through 'end'. Shared by the signature pre-pass and
  // the body builder so their structural checks never diverge; since any
  // failure here aborts the whole import, the error is emitted only once.
  FailureOr<BlockText *> findBeginEnd(StringRef name,
                                      SmallVectorImpl<BlockText> &blocks) {
    BlockText &begin = blocks.front();
    if (begin.entryLabels != ArrayRef<StringRef>{"begin"})
      return error(begin.entry, "the first block must enter from 'begin'"),
             failure();
    BlockText *end = nullptr;
    for (BlockText &bt : blocks) {
      if (bt.exitLabels != ArrayRef<StringRef>{"end"})
        continue;
      if (end)
        return error(bt.exit, "'end' is used more than once"), failure();
      end = &bt;
    }
    if (!end)
      return emitError(loc(begin.entry),
                       "relation '" + name + "' never exits through 'end'"),
             failure();
    return end;
  }

  // Compute a relation's signature (parameter/input/output leaf types). Run for
  // every relation in call-graph order before any body is built, so each call
  // site can type itself from the callee. Emits nothing beyond findBeginEnd's
  // structural errors (which abort the import), so re-parsing in parseRelation
  // never double-reports.
  LogicalResult computeRelationSig(StringRef name,
                                   ArrayRef<StringRef> relParams,
                                   ArrayRef<Line> lines) {
    SmallVector<BlockText, 1> blocks;
    if (failed(parseBlockTexts(name, lines, blocks)))
      return failure();
    auto end = findBeginEnd(name, blocks);
    if (failed(end))
      return failure();
    inferMemVars(blocks);
    RelSig sig;
    for (StringRef p : relParams)
      sig.params.push_back(typeOf(p));
    for (StringRef leaf : blocks.front().entryLeaves)
      sig.inputs.push_back(typeOf(leaf));
    for (StringRef leaf : (*end)->exitLeaves)
      sig.outputs.push_back(typeOf(leaf));
    relSigs[name] = std::move(sig);
    discoverSpecs(blocks);
    return success();
  }

  // forced: when non-null, build a monomorphic specialization -- the cell-typed
  // variables are forced to the demanded signature and the func is emitted
  // under `emitName`. Otherwise builds the generic relation `name`.
  LogicalResult parseRelation(StringRef name, ArrayRef<StringRef> relParams,
                              ArrayRef<Line> lines,
                              const RelSig *forced = nullptr,
                              StringRef emitName = StringRef()) {
    if (emitName.empty())
      emitName = name;
    SmallVector<BlockText, 1> blocks;
    if (failed(parseBlockTexts(name, lines, blocks)))
      return failure();

    auto endOr = findBeginEnd(name, blocks);
    if (failed(endOr))
      return failure();
    BlockText &begin = blocks.front();
    BlockText *end = *endOr;

    forcedTypes.clear();
    if (forced)
      computeForcedTypes(begin, *end, *forced, relParams, blocks);

    // Labels name edges: each occurs once in an entry and once in an exit.
    llvm::StringMap<std::pair<BlockText *, unsigned>> labelTargets;
    for (BlockText &bt : blocks) {
      if (&bt == &begin)
        continue;
      for (auto [pos, label] : llvm::enumerate(bt.entryLabels)) {
        if (label == "begin" || label == "end")
          return error(bt.entry, "'" + label + "' is reserved");
        if (!labelTargets.insert({label, {&bt, (unsigned)pos}}).second)
          return error(bt.entry,
                       "label '" + label + "' targeted more than once");
      }
    }

    // A one-label block threads a selector value iff its entry selector is a
    // variable (to bind) or some incoming one-label exit sends a non-zero
    // selector (to check). The trivial unconditional 0->0 edge is dropped.
    llvm::StringMap<StringRef> labelSel;
    for (BlockText &bt : blocks)
      if (bt.exitLabels.size() == 1)
        labelSel[bt.exitLabels[0]] = bt.exitSel;
    for (BlockText &bt : blocks) {
      if (&bt == &begin || bt.isMerge())
        continue;
      StringRef in = labelSel.lookup(bt.entryLabels[0]);
      bt.threadsSel = bt.entrySel != "0" || (!in.empty() && in != "0");
    }

    // Infer which variables carry the heap so block arguments, parameters and
    // results get the right types; everything else is i32.
    inferMemVars(blocks);

    SmallVector<Type> argTypes;
    for (StringRef p : relParams)
      argTypes.push_back(typeOf(p));
    for (StringRef leaf : begin.entryLeaves)
      argTypes.push_back(typeOf(leaf));
    SmallVector<Type> resultTypes;
    for (StringRef leaf : end->exitLeaves)
      resultTypes.push_back(typeOf(leaf));
    auto fn =
        func::FuncOp::create(builder, loc(begin.entry), emitName,
                             FunctionType::get(context, argTypes, resultTypes));
    fn->setAttr("hssa.params", builder.getI64IntegerAttr(relParams.size()));
    if (strictReversible)
      fn->setAttr("hssa.reversible", builder.getUnitAttr());

    usedExitLabels.clear();
    begin.block = fn.addEntryBlock();
    for (BlockText &bt : blocks) {
      if (&bt == &begin)
        continue;
      bt.block = new Block();
      fn.getBody().push_back(bt.block);
      // A block argument's type follows the variable it binds; the trailing
      // provenance bit (two-predecessor merge) or threaded selector is i32.
      for (StringRef leaf : bt.entryLeaves)
        bt.block->addArgument(typeOf(leaf), loc(bt.entry));
      if (bt.isMerge() || bt.threadsSel)
        bt.block->addArgument(i32, loc(bt.entry));
    }

    for (BlockText &bt : blocks) {
      builder.setInsertionPointToStart(bt.block);
      values.clear();
      // Relation parameters are entry-block arguments; they dominate every
      // block and are the only values visible across blocks.
      for (auto [i, paramName] : llvm::enumerate(relParams))
        if (failed(
                defineValue(bt.entry, paramName, begin.block->getArgument(i))))
          return failure();
      unsigned leafBase = (&bt == &begin) ? relParams.size() : 0;
      for (auto [i, leaf] : llvm::enumerate(bt.entryLeaves)) {
        Value arg = bt.block->getArgument(leafBase + i);
        // A literal in an entry pattern is an assertion: check-and-consume
        // the arriving value against it.
        if (isInteger(leaf)) {
          auto constant = useValue(bt.entry, leaf);
          if (failed(constant))
            return failure();
          auto check = DupOp::create(builder, loc(bt.entry), TypeRange{},
                                     ValueRange{*constant, arg});
          check.setInverted(true);
          continue;
        }
        if (failed(defineValue(bt.entry, leaf, arg)))
          return failure();
      }
      if (bt.isMerge()) {
        if (failed(defineValue(bt.entry, bt.entrySel,
                               bt.block->getArguments().back())))
          return failure();
      } else if (bt.threadsSel) {
        // The threaded selector: a literal entry selector checks the arriving
        // value (like a literal payload); a variable binds it.
        Value selArg = bt.block->getArguments().back();
        if (isInteger(bt.entrySel)) {
          auto constant = useValue(bt.entry, bt.entrySel);
          if (failed(constant))
            return failure();
          DupOp::create(builder, loc(bt.entry), TypeRange{},
                        ValueRange{*constant, selArg})
              .setInverted(true);
        } else if (failed(defineValue(bt.entry, bt.entrySel, selArg))) {
          return failure();
        }
      }

      for (const Line &line : bt.stmts)
        if (failed(parseStatement(line)))
          return failure();

      if (failed(emitExit(bt, labelTargets)))
        return failure();
    }

    for (const auto &entry : labelTargets)
      if (!usedExitLabels.contains(entry.getKey()))
        return error(entry.getValue().first->entry,
                     "label '" + entry.getKey() + "' is never jumped to");

    forcedTypes.clear();
    builder.setInsertionPointToEnd(module.getBody());
    return success();
  }

  LogicalResult
  emitExit(BlockText &bt,
           llvm::StringMap<std::pair<BlockText *, unsigned>> &labelTargets) {
    SmallVector<Value> payload;
    for (StringRef leaf : bt.exitLeaves) {
      auto value = useValue(bt.exit, leaf);
      if (failed(value))
        return failure();
      payload.push_back(*value);
    }

    if (bt.exitLabels == ArrayRef<StringRef>{"end"}) {
      func::ReturnOp::create(builder, loc(bt.exit), payload);
      return success();
    }

    // Build the operand list for one edge: the payload, plus the provenance
    // constant if the target block has a conditional entry.
    auto edgeOperands = [&](StringRef label)
        -> FailureOr<std::pair<Block *, SmallVector<Value>>> {
      if (label == "end")
        return error(bt.exit, "'end' must be an exit's only label"), failure();
      auto it = labelTargets.find(label);
      if (it == labelTargets.end())
        return error(bt.exit, "unknown label '" + label + "'"), failure();
      if (!usedExitLabels.insert(label).second)
        return error(bt.exit,
                     "label '" + label + "' is jumped to more than once"),
               failure();
      auto [target, pos] = it->second;
      SmallVector<Value> operands(payload);
      if (target->isMerge()) {
        // Merge target: append this edge's 0-based label index as the
        // provenance value (a bit for a 2-predecessor merge, an index for K).
        operands.push_back(arith::ConstantOp::create(
            builder, loc(bt.exit), IntegerAttr::get(i32, pos)));
      } else if (target->threadsSel) {
        // Threaded selector: send this edge's selector value along.
        auto sel = useValue(bt.exit, bt.exitSel);
        if (failed(sel))
          return failure();
        operands.push_back(*sel);
      }
      if (operands.size() != target->block->getNumArguments())
        return error(bt.exit, "label '" + label + "' expects " +
                                  Twine(target->block->getNumArguments()) +
                                  " value(s)"),
               failure();
      return std::pair<Block *, SmallVector<Value>>(target->block, operands);
    };

    if (bt.exitLabels.size() == 1) {
      auto edge = edgeOperands(bt.exitLabels[0]);
      if (failed(edge))
        return failure();
      ExitOp::create(builder, loc(bt.exit), edge->second, edge->first);
      return success();
    }

    auto selector = useValue(bt.exit, bt.exitSel);
    if (failed(selector))
      return failure();

    if (bt.exitLabels.size() == 2) {
      auto first = edgeOperands(bt.exitLabels[0]);
      auto second = edgeOperands(bt.exitLabels[1]);
      if (failed(first) || failed(second))
        return failure();
      CondExitOp::create(builder, loc(bt.exit), *selector, first->second,
                         second->second, first->first, second->first);
      return success();
    }

    // Three or more labels: an N-ary switch on the 0-based label index. Each
    // successor is a single-predecessor block that receives exactly the
    // payload (the switch hands the one payload to the selected successor).
    SmallVector<Block *> cases;
    for (StringRef label : bt.exitLabels) {
      auto edge = edgeOperands(label);
      if (failed(edge))
        return failure();
      if (edge->second.size() != payload.size())
        return error(bt.exit, "switch target '" + label +
                                  "' must be a single-predecessor block");
      cases.push_back(edge->first);
    }
    SwitchOp::create(builder, loc(bt.exit), *selector, payload, cases);
    return success();
  }

  // A relation gathered from the source(s), tagged with the file it came from
  // so diagnostics point at the right one after eager cross-file merging.
  struct Relation {
    StringRef name;
    SmallVector<StringRef> params;
    SmallVector<Line> body;
    StringRef file;
  };

  // Resolve an import name to a file path, matching the reference
  // (Modular.scala): each run of N dots in the name becomes "/" followed by
  // (N-1) copies of "../" -- so a single dot is a subdirectory separator and a
  // leading ".." climbs one directory -- then ".hssa", resolved against the
  // importing file's directory.
  static std::string resolveImport(StringRef fromFile, StringRef name) {
    std::string rel;
    for (size_t i = 0; i < name.size();) {
      if (name[i] == '.') {
        unsigned n = 0;
        while (i < name.size() && name[i] == '.') {
          ++n;
          ++i;
        }
        rel.push_back('/');
        for (unsigned k = 1; k < n; ++k)
          rel += "../";
      } else {
        rel.push_back(name[i++]);
      }
    }
    SmallString<128> path(llvm::sys::path::parent_path(fromFile));
    llvm::sys::path::append(path, StringRef(rel).ltrim('/'));
    path += ".hssa";
    llvm::sys::path::remove_dots(path, /*remove_dot_dot=*/true);
    return std::string(path);
  }

  // Collect one source file's relations into `relations`, eagerly merging every
  // imported file first (Option A: one flat module -- MLIR has no file-level
  // import, so this is a front-end concern). Imports resolve relative to this
  // file and are deduplicated by path, so a diamond import reads each file once;
  // a relation name colliding across files is a redefinition error (relNames).
  LogicalResult collectSource(StringRef source, StringRef fileLabel,
                              SmallVectorImpl<Relation> &relations) {
    StringRef savedFile = filename;
    filename = fileLabel;

    SmallVector<Line> lines;
    unsigned number = 0;
    for (StringRef raw : llvm::split(source, '\n')) {
      ++number;
      StringRef text = raw.trim();
      if (!text.empty() && !text.starts_with("//"))
        lines.push_back({text, number});
    }

    // Imports first: merge each referenced file before this file's relations.
    for (const Line &line : lines) {
      if (!line.text.starts_with("import ") &&
          !line.text.starts_with("import\t"))
        continue;
      StringRef target = line.text.drop_front(6).trim();
      if (target.empty())
        return error(line, "import needs a name");
      std::string path = resolveImport(fileLabel, target);
      if (!visitedFiles.insert(path).second)
        continue; // already merged
      auto bufOr = llvm::MemoryBuffer::getFile(path);
      if (!bufOr)
        return error(line,
                     "cannot read import '" + target + "' (" + path + ")");
      std::unique_ptr<llvm::MemoryBuffer> buf = std::move(*bufOr);
      StringRef content = buf->getBuffer();
      StringRef label = buf->getBufferIdentifier();
      ownedBuffers.push_back(std::move(buf));
      if (failed(collectSource(content, label, relations)))
        return failure();
    }

    // This file's own relations (the recursion above restored filename to us).
    bool sawRel = false;
    for (const Line &line : lines) {
      if (line.text.starts_with("import ") ||
          line.text.starts_with("import\t"))
        continue;
      if (line.text.starts_with("rel ") || line.text.starts_with("rel\t")) {
        if (!line.text.ends_with(":"))
          return error(line, "expected 'rel name [params]:'");
        // "rel name parameter:". A relation has a single parameter, which is
        // one expression (a pattern). The name is the first token; the rest is
        // that expression, flattened like any other pattern (so a pair such as
        // "_this, value" yields its leaves, consistently with commas elsewhere
        // meaning pairs).
        StringRef header = line.text.drop_front(4).drop_back(1).trim();
        StringRef name =
            header.take_until([](char c) { return c == ' ' || c == '\t'; });
        if (name.empty())
          return error(line, "relation is missing a name");
        Relation rel;
        rel.name = name;
        rel.file = fileLabel;
        if (failed(flattenPattern(line, header.drop_front(name.size()).trim(),
                                  rel.params)))
          return failure();
        for (StringRef param : rel.params)
          if (isInteger(param))
            return error(line, "relation parameter '" + param +
                                   "' cannot be a number");
        if (!relNames.insert(rel.name).second)
          return error(line, "relation '" + rel.name + "' is defined twice");
        relations.push_back(std::move(rel));
        sawRel = true;
        continue;
      }
      if (!sawRel)
        return error(line, "expected 'rel name:' before statements");
      relations.back().body.push_back(line);
    }

    filename = savedFile;
    return success();
  }

  LogicalResult run(StringRef source) {
    SmallVector<Relation> relations;
    visitedFiles.insert(filename);
    if (failed(collectSource(source, filename, relations)))
      return failure();

    // Signature pre-pass: fix every relation's parameter/input/output types
    // before any body is built, so a call site types its arguments and results
    // from the callee. Visit callees before callers (post-order DFS over the
    // static call graph); the heap and relation references are anchored locally
    // in the leaves and propagate outward along this order -- no fixpoint. A
    // cycle's back edge is skipped for ordering; such a relation still gets its
    // types from its local anchors.
    llvm::StringMap<unsigned> relIndex;
    for (auto [i, rel] : llvm::enumerate(relations))
      relIndex[rel.name] = i;
    SmallVector<uint8_t> state(relations.size(), 0); // 0 unseen 1 active 2 done
    SmallVector<unsigned> order, stack;
    for (unsigned root = 0; root < relations.size(); ++root) {
      if (state[root])
        continue;
      stack.push_back(root);
      while (!stack.empty()) {
        unsigned i = stack.back();
        if (state[i] == 0) {
          state[i] = 1;
          SmallVector<StringRef> callees;
          collectCallees(relations[i].body, callees);
          for (StringRef c : callees) {
            auto it = relIndex.find(c);
            if (it != relIndex.end() && state[it->second] == 0)
              stack.push_back(it->second);
          }
        } else {
          stack.pop_back();
          if (state[i] == 1) {
            state[i] = 2;
            order.push_back(i);
          }
        }
      }
    }
    for (unsigned i : order) {
      filename = relations[i].file;
      if (failed(computeRelationSig(relations[i].name, relations[i].params,
                                    relations[i].body)))
        return failure();
    }

    builder.setInsertionPointToEnd(module.getBody());
    for (const Relation &rel : relations) {
      filename = rel.file;
      if (failed(parseRelation(rel.name, rel.params, rel.body)))
        return failure();
    }

    // Build the monomorphic specializations discovered during the pre-pass:
    // clones of cell-polymorphic relations (the heap readers) with the demanded
    // cell types forced. Calls were already routed to these by emitCall.
    for (const Spec &spec : specs) {
      auto it = relIndex.find(spec.callee);
      if (it == relIndex.end())
        continue;
      const Relation &rel = relations[it->second];
      filename = rel.file;
      if (failed(parseRelation(rel.name, rel.params, rel.body, &spec.demanded,
                               spec.name)))
        return failure();
    }
    return success();
  }
};

} // namespace

OwningOpRef<ModuleOp> mlir::hssa::importHssa(llvm::SourceMgr &sourceMgr,
                                             MLIRContext *context) {
  context->loadDialect<HssaDialect, func::FuncDialect, arith::ArithDialect>();

  const llvm::MemoryBuffer *buffer =
      sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef filename = buffer->getBufferIdentifier();
  OwningOpRef<ModuleOp> module =
      ModuleOp::create(FileLineColLoc::get(context, filename, 0, 0));

  Importer importer(context, filename, *module);
  if (failed(importer.run(buffer->getBuffer())))
    return nullptr;
  if (failed(verify(*module)))
    return nullptr;
  return module;
}
