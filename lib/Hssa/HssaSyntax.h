//===- HssaSyntax.h - HSSA textual-syntax helpers ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stateless helpers for the HSSA textual format, shared by the importer and its
// analysis scans: the builtin-op catalog (arities per direction) and the
// lexing / pattern-splitting primitives. Pure StringRef processing -- no IR
// state -- which is why they live apart from the stateful Importer.
//
//===----------------------------------------------------------------------===//
#ifndef HSSA_SYNTAX_H
#define HSSA_SYNTAX_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <utility>

namespace mlir::hssa {
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringRef;

// Operand/result arity of a builtin, per direction.
struct BuiltinShape {
  unsigned numParams;
  unsigned fwdIns, fwdOuts;
  unsigned invIns, invOuts;
  bool invertible = true;
};

inline std::optional<BuiltinShape> builtinShape(StringRef name) {
  if (name == "add" || name == "sub" || name == "xor" || name == "subfrom")
    return BuiltinShape{1, 1, 1, 1, 1};
  if (name == "mul" || name == "div" || name == "mod" || name == "or" ||
      name == "and" || name == "less" || name == "lessequal" ||
      name == "greater" || name == "greaterequal" || name == "equal" ||
      name == "notequal")
    return BuiltinShape{2, 0, 1, 1, 0};
  if (name == "euclidiv")
    return BuiltinShape{1, 1, 2, 2, 1};
  if (name == "id")
    return BuiltinShape{0, 1, 1, 1, 1};
  if (name == "dup")
    return BuiltinShape{1, 0, 1, 1, 0};
  if (name == "discard")
    return BuiltinShape{0, 1, 0, 0, 0, /*invertible=*/false};
  if (name == "tests.assert")
    return BuiltinShape{1, 1, 0, 0, 1};
  if (name == "tests.fail")
    return BuiltinShape{0, 0, 0, 0, 0};
  // Managed memory. The heap is threaded single-assignment; sizes/pointers are
  // integers; readwrite's cell value is int-or-unit and it is self-inverse.
  if (name == "mmem.new")
    return BuiltinShape{0, 0, 1, 1, 0};
  if (name == "mmem.allocate")
    return BuiltinShape{1, 1, 2, 2, 1};
  if (name == "mmem.readwrite")
    return BuiltinShape{1, 2, 2, 2, 2};
  return std::nullopt;
}

// Memory ops whose operand/result types are not all i32.
inline bool isMemoryOp(StringRef name) {
  return name == "mmem.new" || name == "mmem.allocate" ||
         name == "mmem.readwrite";
}

inline bool isInteger(StringRef token) {
  int64_t value;
  return !token.getAsInteger(10, value);
}

// Split at the first ':=' or '=:'. The reference grammar treats the two as
// interchangeable separators (asgn_delim = ASGN | NGSA,
// https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/parsing/Parsing.scala#L63),
// so an assignment is
// `out <delim> op param <delim> in` and an exit `-> labels <delim> in` with
// either token at either position -- RTM, for instance, writes ':=' throughout
// where the other programs write '=:' for inputs. `found` reports whether a
// separator was present; when absent, the whole string is returned as `before`.
inline std::pair<StringRef, StringRef> splitDelim(StringRef s, bool &found) {
  size_t a = s.find(":=");
  size_t b = s.find("=:");
  size_t p = a < b ? a : b; // npos (SIZE_MAX) loses to any real index
  found = p != StringRef::npos;
  if (!found)
    return {s, StringRef()};
  return {s.substr(0, p), s.substr(p + 2)};
}

// A statement "out := op param =: in" split into its pieces (the parser and
// every analysis scan need the same split). `ok` is false for a non-statement
// line (no two separators). `inverted` is the leading '~'; `opName` is the op;
// `paramExpr` is the text between the op and the input separator.
struct StmtSplit {
  StringRef lhs, opName, paramExpr, inText;
  bool inverted = false;
  bool ok = false;
};
inline StmtSplit splitStmt(StringRef lineText) {
  StmtSplit s;
  bool found;
  auto [lhs, rest] = splitDelim(lineText, found);
  auto [opText, inText] = splitDelim(rest, found);
  if (!found)
    return s;
  s.lhs = lhs;
  s.inText = inText;
  StringRef op = opText.trim();
  s.inverted = op.consume_front("~");
  op = op.trim();
  s.opName = op.take_until([](char c) { return c == ' ' || c == '\t'; });
  s.paramExpr = op.drop_front(s.opName.size()).trim();
  s.ok = true;
  return s;
}

// Split on commas that are not nested inside parentheses.
inline SmallVector<StringRef> splitTop(StringRef text) {
  SmallVector<StringRef> parts;
  text = text.trim();
  if (text.empty())
    return parts;
  int depth = 0;
  size_t start = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    if (c == '(')
      ++depth;
    else if (c == ')')
      --depth;
    else if (c == ',' && depth == 0) {
      parts.push_back(text.slice(start, i).trim());
      start = i + 1;
    }
  }
  parts.push_back(text.drop_front(start).trim());
  return parts;
}

// Best-effort flatten of a pattern into scalar leaves, for type inference
// (no diagnostics; a malformed pattern is reported by flattenPattern during
// the build). "()" and empty contribute nothing; pairs/parens are folded.
inline void collectLeaves(StringRef text, SmallVectorImpl<StringRef> &leaves) {
  text = text.trim();
  if (text.empty() || text == "()")
    return;
  SmallVector<StringRef> parts = splitTop(text);
  if (parts.size() > 1) {
    for (StringRef part : parts)
      collectLeaves(part, leaves);
    return;
  }
  if (text.starts_with("(") && text.ends_with(")"))
    return collectLeaves(text.drop_front().drop_back(), leaves);
  leaves.push_back(text);
}

} // namespace mlir::hssa

#endif // HSSA_SYNTAX_H
