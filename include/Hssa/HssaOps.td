//===- HssaOps.td - Hssa dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HSSA_OPS
#define HSSA_OPS

include "Hssa/HssaTypes.td"
include "Hssa/HssaOpInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"




//===----------------------------------------------------------------------===//
// Base hssa operation definitions.
//===----------------------------------------------------------------------===//

// Every hssa op carries HssaGenericShapeOpInterface (its parameter / consumed /
// produced role split); invertible ops additionally carry
// HssaInvertibleOpInterface.
class Hssa_Op<string mnemonic, list<Trait> traits = []> :
        Op<Hssa_Dialect, mnemonic, traits # [HssaGenericShapeOpInterface]>;

class Hssa_InvertibleOp<string mnemonic, list<Trait> traits = []> :
        Hssa_Op<mnemonic, traits # [HssaInvertibleOpInterface]>;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// INFORMATION_OPS, SEE https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/Information.scala
def Hssa_IdOp : Hssa_InvertibleOp<"id", [SameOperandsAndResultType, SelfInverse]> {
    let summary = "Identity operation.";
    let description = [{
        The `hssa.id` operation passes the value through unchanged. The input is consumed and the result
        is a new value that must itself be consumed exactly once.

        Although HSSA allows `x := id =: 2`, the inverse of that would be `2 := id =: x` i.e. is a constant on the output side,
        which is an assertion. An MLIR result is always a new value and can't represent an expected constant, so in this dialect,
        constants appear only as parameters, and creating from or checking against a constant is `dup` in this dialect.

        The importer will rewrite `x := id =: c` into `x := dup c =: ()`

        Further, id is its own inverse, thus there is no need for an inverted flag.

        DELIBERATELY NOT Pure: like the arithmetic ops, id is a pure function of
        its operand, but under HSSA linearity (each value consumed exactly once)
        CSE merging two id's would alias one value into two consumers, breaking
        reversibility. Leaving it effect-unmodelled keeps MLIR conservative.

        Shape:
            () / T -> T

        HSSA:
            x.1 := id =: x.0

        Example:

        ```mlir
        %x1 = hssa.id %x0 : i32
        ```
    }];

    let arguments = (ins Hssa_Value:$input);
    let results = (outs Hssa_Value:$output);

    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getConsumed() { return {getInput()}; }
        ::llvm::SmallVector<::mlir::Value> getProduced() { return {getOutput()}; }
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {}; }
        bool getInverted() { return false; }
        void setInverted(bool) {}
    }];

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

// INFORMATION_OPS, SEE https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/Information.scala
//
// dup is deliberately NOT Pure. Effects are computed in C++ (getEffects); both
// directions write the abstract control/linearity barrier resource so dup is
// never DCE'd and two dups are never CSE-merged. Inverted dup (`~dup`) is a
// check-and-consume that TRAPS unless its input equals the parameter -- a
// result-less op that must stay live. Forward dup copies a value out of
// nothing; it is semantically side-effect-free, but under LINEARITY the copy is
// a distinct linear value that must not be aliased or dropped. See
// DupOp::getEffects in HssaOps.cpp.
def Hssa_DupOp : Hssa_InvertibleOp<"dup",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Copy the parameter; inverted, check-and-consume against it.";
    let description = [{
        Forward, dup creates a copy of its parameter out of nothing: copying is
        free and nothing is consumed. Inverted, it consumes a value and traps
        unless it equals the parameter -- undoing a copy is where the cost lives.
        The reference toolkit implements it as produce/consume of the same
        value
        (https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/Information.scala#L28),
        and the assignment rules make a literal in any
        pattern mean exactly this check, which is how the importer desugars
        literal outputs and entry-pattern literals.

        Shape:
            T / () -> T        forward
            T / T  -> ()       inverted

        HSSA:
            x.1 := dup x.0 =: ()
            ()  := ~dup x.0 =: x.1

        Example:

        ```mlir
        %c = hssa.dup(%x) : (i32) -> i32
        hssa.dup(%x) %c {inverted} : (i32, i32) -> ()
        ```
    }];

    // dup is polymorphic (the interpreter's dup has a type metavariable): it
    // copies any HSSA value -- an int, a unit, or a relation reference (the
    // vtable dup's relations). All three positions share that one type.
    let arguments = (ins Hssa_Value:$parameter, Optional<Hssa_Value>:$input, UnitAttr:$inverted);
    let results = (outs Optional<Hssa_Value>:$output);

    let assemblyFormat = [{ `(` $parameter `)` ($input^)? attr-dict `:` functional-type(operands, results) }];
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getConsumed()   { if (auto v = getInput()) {return {v};} else {return {};}}
        ::llvm::SmallVector<::mlir::Value> getProduced()   { if (auto v = getOutput()) {return {v};} else {return {};}}
        ::llvm::SmallVector<::mlir::Value> getParameters()   { return {getParameter()};}
    }];
}


// Reversible in-place updates (add/sub and the self-inverse xor/subfrom).
// DELIBERATELY NOT Pure, for the same linearity reason as the compute ops: even
// though `x += p` is a pure function of its operands, CSE merging two identical
// updates would alias one linear value into two consumers (consumed twice ->
// non-reversible), and a "dead" update result is a linearity error to report,
// not to silently DCE. No modelled effect == MLIR keeps and never merges them.
class Hssa_UpdateOp<string mnemonic, list<Trait> traits = []> : Hssa_InvertibleOp<mnemonic, traits # [SameOperandsAndResultType]> {
    let arguments = (ins Hssa_Int:$parameter, Hssa_Int:$input, UnitAttr:$inverted);
    let results = (outs Hssa_Int:$output);
    let assemblyFormat = [{ `(` $parameter `)` $input attr-dict `:` type($input) }];
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getConsumed() { return {getInput()}; }
        ::llvm::SmallVector<::mlir::Value> getProduced() { return {getOutput()}; }
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {getParameter()}; }
    }];
}

// ARITHMETIC_OPS, SEE https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/Arithmetic.scala
def Hssa_AddOp : Hssa_UpdateOp<"add"> {
    let summary = "Reversible update: output = input + parameter.";
    let description = [{
        The reversible update x += p. The parameter survives the statement, so
        the inverse is x -= p. Arithmetic wraps on i32, matching the reference
        interpreter (Lukas Gail's Scala HSSA toolkit,
        https://github.com/Leridon/hssa), whose add is plain `_ + _` on Int:
        https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/Arithmetic.scala#L51
        Addition mod 2^32 is a bijection in x, which is why add may consume
        its input while mul may not.

        Shape:
            (int) / int -> int      forward
            (int) / int -> int      inverted

        HSSA:
            x.1 := add p =: x.0
            x.0 := ~add p =: x.1

        Example:
        ```mlir
        %x1 = hssa.add(%p) %x0 : i32
        %x0 = hssa.add(%p) %x1 {inverted} : i32
        ```
    }];
}

def Hssa_SubOp : Hssa_UpdateOp<"sub"> {
    let summary = "Reversible update: output = input - parameter.";
    let description = [{
        x -= p, the inverse of add. The paper spells this relation "-" (a
        synonym for the inverse of +); the toolkit names it sub.

        Shape:
            (int) / int -> int      forward
            (int) / int -> int      inverted

        HSSA:
            x.1 := sub p =: x.0
            x.0 := ~sub p =: x.1

        Example:
        ```mlir
        %x1 = hssa.sub(%p) %x0 : i32
        %x0 = hssa.sub(%p) %x1 {inverted} : i32
        ```
    }];
}

def Hssa_XorOp : Hssa_UpdateOp<"xor", [SelfInverse]> {
    let hasCanonicalizer = 1;
    let summary = "Reversible update: output = input ^ parameter. Self-inverse.";
    let description = [{
        x ^= p. Its own inverse, so the inverted flag changes nothing about
        its meaning or its lowering.

        Shape:
            (int) / int -> int      (self-inverse)

        HSSA:
            x.1 := xor p =: x.0
            x.0 := ~xor p =: x.1    (identical to the forward direction)

        Example:
        ```mlir
        %x1 = hssa.xor(%p) %x0 : i32
        ```
    }];
}

def Hssa_SubfromOp : Hssa_UpdateOp<"subfrom", [SelfInverse]> {
    let hasCanonicalizer = 1;
    let summary = "Reverse subtraction: output = parameter - input. Self-inverse.";
    let description = [{
        "Subtract from": out = p - in. Not sub with the arguments swapped --
        the roles differ, since the consumed value is on the right of the
        minus. An involution, because p - (p - x) = x; the paper calls this
        relation sub, and the stdlib builds negation from it (subfrom 0).

        Shape:
            (int) / int -> int      (self-inverse)

        HSSA:
            x.1 := subfrom p =: x.0
            x.0 := ~subfrom p =: x.1    (identical to the forward direction)

        Example:
        ```mlir
        %x1 = hssa.subfrom(%p) %x0 : i32
        ```
    }];
}

// HSSA spells uncall as `[~f p in]`, the same `~` as inverted builtins, so
// call and uncall are one op distinguished by the inverted flag. $ins holds
// whatever the op consumes (forward: the callee's inputs; inverted: its
// outputs), so operand/result counts match the callee signature
// direction-dependently.
def Hssa_CallOp
    : Hssa_InvertibleOp<"call", [AttrSizedOperandSegments,
                                 DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Apply a relation (function); with {inverted}, uncall it.";
    let description = [{
        Relation application is an ordinary statement: outputs := relation
        parameters =: inputs. HSSA spells uncall with the same ~ marker as
        inverted builtins, so call and uncall are one op distinguished by the
        flag. The operand slots are role-based: $ins holds whatever this op
        consumes (forward: the callee's inputs; inverted: its outputs). The
        callee is a symbol attribute, not an operand -- there is no %f.

        Shape (counts follow the callee signature, so they flip with the flag):
            (params...) / ins...  -> outs...     forward
            (params...) / outs... -> ins...      inverted

        HSSA:
            y := f p =: x
            x := ~f p =: y

        Example:

        ```mlir
        %y = hssa.call @f(%p) %x : (i32, i32) -> i32
        %x2 = hssa.call @f(%p) %y {inverted} : (i32, i32) -> i32
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee,
                         Variadic<Hssa_Value>:$params,
                         Variadic<Hssa_Value>:$ins,
                         UnitAttr:$inverted);
    let results = (outs Variadic<Hssa_Value>:$outs);

    let assemblyFormat = [{
        $callee `(` $params `)` $ins attr-dict `:` functional-type(operands, results)
    }];

    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return ::llvm::to_vector(getParams()); }
        ::llvm::SmallVector<::mlir::Value> getConsumed()   { return ::llvm::to_vector(getIns()); }
        ::llvm::SmallVector<::mlir::Value> getProduced()   { return ::llvm::to_vector_of<::mlir::Value>(getOuts()); }
        ::mlir::Operation *buildInverse(::mlir::OpBuilder &builder,
                                        ::llvm::ArrayRef<::mlir::Value> newParams,
                                        ::llvm::ArrayRef<::mlir::Value> newIns);
    }];
}

def Hssa_CallIndirectOp
    : Hssa_InvertibleOp<"call_indirect", [AttrSizedOperandSegments]> {
    let summary = "Apply a relation value; with {inverted}, uncall it.";
    let description = [{
        The indirect form of hssa.call: the callee is a first-class relation
        VALUE (an !hssa.rel produced by dup or read from a vtable), not a static
        symbol. This is HSSA's one Application mechanism -- the interpreter
        evaluates every call's relation sub-expression to a Value.Relation and
        applies it; hssa.call is just the case where that expression is a static
        name. Virtual dispatch produces the callee at run time (read the
        object's vtable slot, apply it to the method index), so the target is
        not statically known. Mirrors func.call_indirect.

        The callee is a read-only parameter, not consumed: a virtual call site
        reads the method value, applies it, then uncomputes it via the inverse
        vtable lookup. $ins holds what the application consumes (forward: the
        callee's inputs; inverted: its outputs). Inversion flips the flag and
        swaps ins/outs; the SAME relation value runs backward (rel.flipped).

        HSSA:
            y := m p =: x        // m is a relation value
            x := ~m p =: y

        Example:

        ```mlir
        %y = hssa.call_indirect %m(%p) %x : (!hssa.rel, i32, i32) -> i32
        ```
    }];

    let arguments = (ins Hssa_RelType:$callee,
                         Variadic<Hssa_Value>:$params,
                         Variadic<Hssa_Value>:$ins,
                         UnitAttr:$inverted);
    let results = (outs Variadic<Hssa_Value>:$outs);

    let assemblyFormat = [{
        $callee `(` $params `)` $ins attr-dict `:` functional-type(operands, results)
    }];

    let extraClassDeclaration = [{
        // The callee relation value is read (a parameter), not consumed; it is
        // uncomputed separately, so inversion threads it like any parameter.
        ::llvm::SmallVector<::mlir::Value> getParameters() {
          ::llvm::SmallVector<::mlir::Value> p;
          p.push_back(getCallee());
          ::llvm::append_range(p, getParams());
          return p;
        }
        ::llvm::SmallVector<::mlir::Value> getConsumed() { return ::llvm::to_vector(getIns()); }
        ::llvm::SmallVector<::mlir::Value> getProduced() { return ::llvm::to_vector_of<::mlir::Value>(getOuts()); }
        ::mlir::Operation *buildInverse(::mlir::OpBuilder &builder,
                                        ::llvm::ArrayRef<::mlir::Value> newParams,
                                        ::llvm::ArrayRef<::mlir::Value> newIns);
    }];
}

// mul/div/mod/or/and and the comparisons can't be update ops: arithmetic
// wraps (i32, matching the JVM reference interpreter:
// https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/Arithmetic.scala#L37
// ), so they lose information. HSSA makes them compute-new ops
// `(int, int) / () -> int`: forward computes a fresh value from the two
// parameters; inverted re-computes it and consumes the input, trapping on
// mismatch. Shape mirrors dup: the optional slots must agree with the flag.
//
// DELIBERATELY NOT Pure. These ops compute a mathematical function of their
// operands, so a casual reading says "mark them Pure for free CSE/DCE". That is
// unsound HERE: HSSA values are LINEAR (each created once, consumed once; see
// --hssa-verify-linear). CSE/value-numbering would merge two syntactically
// identical compute ops into one SSA result shared by both consumers -- turning
// one linear value into a value consumed twice, which breaks reversibility (the
// inverse can no longer un-produce it). DCE would likewise delete a
// forward-direction compute op whose result is dead, but under linearity a dead
// result is a verifier error to be reported, not silently erased. The inverted
// direction additionally traps. Rather than leave the ops effect-UNmodelled
// (which keeps them conservative but says nothing about WHY), they declare the
// MemoryEffectsOpInterface and, in BOTH directions, write the abstract
// control/linearity barrier resource via getEffects (below): that is what keeps
// MLIR from merging or deleting them, and it states the intent explicitly.
// This is the dialect-design point for the paper.
class Hssa_ComputeOp<string mnemonic, list<Trait> traits = []>
    : Hssa_InvertibleOp<mnemonic, traits # [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let arguments = (ins Hssa_Int:$lhs, Hssa_Int:$rhs,
                         Optional<Hssa_Int>:$input, UnitAttr:$inverted);
    let results = (outs Optional<Hssa_Int>:$output);
    let assemblyFormat = [{
        `(` $lhs `,` $rhs `)` ($input^)? attr-dict `:` functional-type(operands, results)
    }];
    let description = [{
        A lossy two-argument function made reversible by parameterization:
        both operands are read as parameters and the result is created fresh.
        The inverse cannot un-compute a lossy function; instead it recomputes
        f(lhs, rhs), checks that the consumed value matches, and traps
        otherwise -- check-and-consume, ~dup with arithmetic inside.
        Comparisons produce 0 or 1 as i32, as in the reference toolkit's
        Arithmetic.scala:
        https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/Arithmetic.scala#L13

        Shape:
            (lhs, rhs) / () -> int      forward
            (lhs, rhs) / int -> ()      inverted

        HSSA (shown for `less`; identical shape for every compute op --
        mul/div/mod/or/and and the six comparisons):
            c  := less a, b =: ()
            () := ~less a, b =: c

        Example:
        ```mlir
        %c = hssa.less(%a, %b) : (i32, i32) -> i32
        hssa.less(%a, %b) %c {inverted} : (i32, i32, i32) -> ()
        ```
    }];
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {getLhs(), getRhs()}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed()   { if (auto v = getInput()) { return {v}; } return {}; }
        ::llvm::SmallVector<::mlir::Value> getProduced()   { if (auto v = getOutput()) { return {v}; } return {}; }
    }];
    let extraClassDefinition = [{
        ::llvm::LogicalResult $cppClass::verify() {
            return ::mlir::hssa::detail::verifyComputeOp(getOperation(), getInverted());
        }
        // Both directions write the control/linearity barrier so no compute op
        // is DCE'd or CSE-merged. Inverted is check-and-consume: it traps unless
        // the input re-computes to f(lhs, rhs). Forward produces a fresh value
        // that is semantically side-effect-free, but under LINEARITY it is a
        // distinct linear value -- CSE merging two identical compute ops would
        // alias one result into two consumers (consumed twice -> irreversible),
        // and a dead result is a linearity error, not dead code. See the matching
        // comment on DupOp::getEffects in HssaOps.cpp.
        void $cppClass::getEffects(
            ::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<
                ::mlir::MemoryEffects::Effect>> &effects) {
          effects.emplace_back(::mlir::MemoryEffects::Write::get(),
                               ::mlir::hssa::HssaTrapResource::get());
        }
    }];
}

def Hssa_MulOp : Hssa_ComputeOp<"mul"> { let summary = "output = lhs * rhs (wrapping)."; }
def Hssa_DivOp : Hssa_ComputeOp<"div"> { let summary = "output = lhs / rhs (truncating)."; }
def Hssa_ModOp : Hssa_ComputeOp<"mod"> { let summary = "output = lhs % rhs."; }
def Hssa_OrOp  : Hssa_ComputeOp<"or">  { let summary = "output = lhs | rhs."; }
def Hssa_AndOp : Hssa_ComputeOp<"and"> { let summary = "output = lhs & rhs."; }
def Hssa_LessOp         : Hssa_ComputeOp<"less">         { let summary = "output = lhs < rhs, as 0/1."; }
def Hssa_LessEqualOp    : Hssa_ComputeOp<"lessequal">    { let summary = "output = lhs <= rhs, as 0/1."; }
def Hssa_GreaterOp      : Hssa_ComputeOp<"greater">      { let summary = "output = lhs > rhs, as 0/1."; }
def Hssa_GreaterEqualOp : Hssa_ComputeOp<"greaterequal"> { let summary = "output = lhs >= rhs, as 0/1."; }
def Hssa_EqualOp        : Hssa_ComputeOp<"equal">        { let summary = "output = lhs == rhs, as 0/1."; }
def Hssa_NotEqualOp     : Hssa_ComputeOp<"notequal">     { let summary = "output = lhs != rhs, as 0/1."; }

// Bijective between one value and a (quotient, remainder) pair, so it
// consumes in both directions and its arity flips with the flag:
// forward 1 -> 2, inverted 2 -> 1.
def Hssa_EuclidivOp : Hssa_InvertibleOp<"euclidiv"> {
    let summary = "Euclidean division by the parameter; consumes both ways.";
    let description = [{
        Division by the parameter is a bijection between a dividend and its
        (quotient, remainder) pair, so euclidiv genuinely consumes in both
        directions and its arity flips with the flag: no runtime check is
        needed backward, q * d + r simply reconstructs the dividend. This is
        the paper's div; mul is officially a synonym for its inverse.

        DELIBERATELY NOT Pure (linearity, as for the other arithmetic ops): its
        quotient/remainder results are linear and must not be CSE-shared. No
        trap, so no modelled effect -- MLIR keeps it conservatively.

        Shape:
            int / int -> (int, int)      forward
            int / (int, int) -> int      inverted

        HSSA:
            (q, r) := euclidiv d =: x
            x := ~euclidiv d =: (q, r)

        Example:
        ```mlir
        %q, %r = hssa.euclidiv(%d) %x : (i32, i32) -> (i32, i32)
        %x2 = hssa.euclidiv(%d) %q, %r {inverted} : (i32, i32, i32) -> i32
        ```
    }];
    let arguments = (ins Hssa_Int:$divisor, Variadic<Hssa_Int>:$ins,
                         UnitAttr:$inverted);
    let results = (outs Variadic<Hssa_Int>:$outs);
    let assemblyFormat = [{
        `(` $divisor `)` $ins attr-dict `:` functional-type(operands, results)
    }];
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {getDivisor()}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed()   { return ::llvm::to_vector(getIns()); }
        ::llvm::SmallVector<::mlir::Value> getProduced()   { return ::llvm::to_vector_of<::mlir::Value>(getOuts()); }
    }];
}

//===----------------------------------------------------------------------===//
// Managed memory
//===----------------------------------------------------------------------===//

// The managed heap is threaded single-assignment: mmem.new produces the first
// version, and every memory op consumes one version and produces the next, so
// heap liveness is checked by the ordinary linearity rules.
def Hssa_MmemNewOp : Hssa_InvertibleOp<"mmem.new",
        [AttrSizedOperandSegments,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Create (inverted: destroy) a managed-memory heap.";
    let description = [{
        Produces a fresh heap from nothing; its inverse consumes an empty heap
        back to nothing
        (https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/ManagedMemory.scala).

        An OPTIONAL `size` parameter K sizes the heap to K cells. This is a
        deliberate, ADDITIVE divergence from the reference, whose `mmem.new`
        takes no argument and is fixed at memorySize = 1024 (ManagedMemory.scala).
        K may be any i32 value -- a literal OR a runtime-computed SSA value --
        so the heap can be sized dynamically. When `size` is absent the heap
        defaults to 1024 cells, reproducing the reference behavior exactly; a
        program that passes K is NOT runnable in the reference interpreter and
        loses the differential oracle (intentional). As a read-only parameter,
        K threads unchanged to the inverse, which frees the same-sized heap.

        Shape:
            (size?) () -> mem      forward
            (size?) mem -> ()      inverted

        HSSA:
            m := mmem.new =: ()
            m := mmem.new K =: ()
            () := ~mmem.new =: m
            () := ~mmem.new K =: m

        Example:
        ```mlir
        %m = hssa.mmem.new : () -> !hssa.mem
        %m = hssa.mmem.new(%k) : (i32) -> !hssa.mem
        hssa.mmem.new %m {inverted} : (!hssa.mem) -> ()
        ```
    }];
    let arguments = (ins Optional<Hssa_Int>:$size,
                         Optional<Hssa_MemType>:$input, UnitAttr:$inverted);
    let results = (outs Optional<Hssa_MemType>:$output);
    let assemblyFormat = [{
        (`(` $size^ `)`)? ($input^)? attr-dict `:` functional-type(operands, results)
    }];
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { if (auto s = getSize()) { return {s}; } return {}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed()   { if (auto v = getInput()) { return {v}; } return {}; }
        ::llvm::SmallVector<::mlir::Value> getProduced()   { if (auto v = getOutput()) { return {v}; } return {}; }
    }];
}

def Hssa_MmemAllocateOp : Hssa_InvertibleOp<"mmem.allocate",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Allocate (inverted: free) a block in the heap.";
    let description = [{
        Allocates a block of `size` cells, threading the heap and producing a
        pointer; its inverse consumes the pointer to free the block
        (https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/ManagedMemory.scala).
        Only logically reversible unless the allocator is deterministic (buddy
        memory): the heap's internal bookkeeping is not guaranteed bit-restored.

        Shape:
            (size) mem -> (mem, ptr)     forward
            (size) mem, ptr -> mem       inverted

        HSSA:
            m', p := mmem.allocate size =: m
            m' := ~mmem.allocate size =: m, p

        Example:
        ```mlir
        %m2, %p = hssa.mmem.allocate(%size) %m : (i32, !hssa.mem) -> (!hssa.mem, i32)
        %m3 = hssa.mmem.allocate(%size) %m2, %p {inverted} : (i32, !hssa.mem, i32) -> !hssa.mem
        ```
    }];
    let arguments = (ins Hssa_Int:$size, Hssa_MemType:$memIn,
                         Optional<Hssa_Int>:$ptrIn, UnitAttr:$inverted);
    let results = (outs Hssa_MemType:$memOut, Optional<Hssa_Int>:$ptrOut);
    let assemblyFormat = [{
        `(` $size `)` $memIn (`,` $ptrIn^)? attr-dict `:`
        functional-type(operands, results)
    }];
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {getSize()}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed() {
            ::llvm::SmallVector<::mlir::Value> v{getMemIn()};
            if (auto p = getPtrIn()) { v.push_back(p); }
            return v;
        }
        ::llvm::SmallVector<::mlir::Value> getProduced() {
            ::llvm::SmallVector<::mlir::Value> v{getMemOut()};
            if (auto p = getPtrOut()) { v.push_back(p); }
            return v;
        }
    }];
}

// The unit value. Carries no information, so like a literal it is
// constant-like: materializing it is free and it is exempt from linearity.
// The importer emits it where the textual form writes `()` as a value (e.g.
// the value handed to mmem.readwrite to empty a cell).
def Hssa_UnitOp : Hssa_Op<"unit", [ConstantLike, Pure]> {
    let summary = "Materialize the unit value.";
    let description = [{
        The unit value -- a value carrying no information. Like a constant it is
        exempt from linearity (free to create, never needs consuming); the
        importer emits it wherever the textual form writes `()` as a value, e.g.
        the value handed to mmem.readwrite to empty a cell.

        Shape:
            () / () -> unit

        Example:
        ```mlir
        %u = hssa.unit : !hssa.unit
        ```
    }];
    let results = (outs Hssa_UnitType:$result);
    let assemblyFormat = [{ attr-dict `:` type($result) }];
    let hasFolder = 1;
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed()   { return {}; }
        ::llvm::SmallVector<::mlir::Value> getProduced()   { return {}; }
    }];
}

// A first-class relation value: a reference to a named relation, materialized
// as a value that can be dup'd, stored in the heap, and applied indirectly (the
// vtable dispatch mechanism). Constant-like -- it is just a name, so it is
// exempt from linearity -- and lowering assigns each relation a dispatch index.
def Hssa_RelRefOp : Hssa_Op<"rel_ref", [ConstantLike, Pure]> {
    let summary = "Materialize a reference to a relation as a value.";
    let description = [{
        A first-class relation value: a reference to a named relation, as a value
        that can be dup'd, stored in the heap, and applied indirectly through
        call_indirect -- the virtual-dispatch mechanism. Constant-like (it is just
        a name, so exempt from linearity); lowering assigns each relation a
        dispatch index.

        Shape:
            () / () -> rel

        Example:
        ```mlir
        %f = hssa.rel_ref @some_relation : !hssa.rel
        ```
    }];
    let arguments = (ins FlatSymbolRefAttr:$callee);
    let results = (outs Hssa_RelType:$result);
    let assemblyFormat = [{ $callee attr-dict `:` type($result) }];
    let hasFolder = 1;
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed()   { return {}; }
        ::llvm::SmallVector<::mlir::Value> getProduced()   { return {}; }
    }];
}

def Hssa_MmemReadwriteOp : Hssa_InvertibleOp<"mmem.readwrite",
        [SelfInverse, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Swap a heap cell's value with a provided one.";
    let description = [{
        Reads the cell at the pointer, writes the provided value, and returns
        the old value, threading the heap
        (https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/ManagedMemory.scala).
        Self-inverse: running it again with the old value restores the cell, so
        it carries no inverted flag. The stored value is a cell (int or unit);
        the written and read-back values are typed independently, matching the
        reference relation's two independent type metavariables.

        Shape:
            (ptr) / (mem, val) -> (mem, oldval)     (self-inverse)

        HSSA:
            m', old := mmem.readwrite ptr =: m, val

        Example:
        ```mlir
        %m2, %old = hssa.mmem.readwrite(%p) %m, %v : (i32, !hssa.mem, i32) -> (!hssa.mem, i32)
        ```
    }];
    let arguments = (ins Hssa_Int:$pointer, Hssa_MemType:$memIn,
                         Hssa_Cell:$writeVal);
    let results = (outs Hssa_MemType:$memOut, Hssa_Cell:$oldVal);
    let assemblyFormat = [{
        `(` $pointer `)` $memIn `,` $writeVal attr-dict `:`
        functional-type(operands, results)
    }];
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {getPointer()}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed()   { return {getMemIn(), getWriteVal()}; }
        ::llvm::SmallVector<::mlir::Value> getProduced()   { return {getMemOut(), getOldVal()}; }

        bool getInverted() { return false; }
        void setInverted(bool) {}
    }];
}

// Explicit information destruction (the oracle, run forward). Deliberately
// not invertible: a function containing a discard has no inverse.
def Hssa_DiscardOp : Hssa_Op<"discard"> {
    let summary = "Consume and drop a value.";
    let description = [{
        Explicit information destruction -- the only source of information
        loss in HSSA, which is what makes "where does this program forget?"
        a syntactic question. Its inverse (the oracle) has no runtime
        semantics, so discard deliberately does not implement HssaInvertible:
        a function containing one has no inverse, and uncalling it is a
        compile error.

        Shape:
            () / T -> ()        (not invertible -- there is no inverse direction)

        HSSA:
            () := discard () =: x

        Example:
        ```mlir
        hssa.discard %x : i32
        ```
    }];
    let arguments = (ins Hssa_Value:$input);
    let assemblyFormat = [{ $input attr-dict `:` type($input) }];
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed()   { return {getInput()}; }
        ::llvm::SmallVector<::mlir::Value> getProduced()   { return {}; }
    }];
}

// dup's mirror: forward consumes the input and traps unless it equals the
// parameter; inverted produces the parameter. Effects are direction-dependent
// (getEffects): forward is result-less and traps -- it declares a write to the
// abstract trap resource so it is not DCE-able -- while inverted produces a
// value and is otherwise side-effect-free.
def Hssa_TestsAssertOp : Hssa_InvertibleOp<"tests.assert",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Check-and-consume against the parameter.";
    let description = [{
        dup mirrored across the flag: forward it consumes the input and traps
        unless it equals the parameter; backward it produces the parameter --
        an assertion run in reverse is a generator. Reference implementation:
        https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/Testing.scala#L15

        Shape:
            T / T -> ()        forward
            T / () -> T        inverted

        HSSA:
            () := tests.assert p =: x
            x := ~tests.assert p =: ()

        Example:
        ```mlir
        hssa.tests.assert(%p) %x : (i32, i32) -> ()
        %x2 = hssa.tests.assert(%p) {inverted} : (i32) -> i32
        ```
    }];
    let arguments = (ins Hssa_Int:$parameter, Optional<Hssa_Int>:$input,
                         UnitAttr:$inverted);
    let results = (outs Optional<Hssa_Int>:$output);
    let assemblyFormat = [{
        `(` $parameter `)` ($input^)? attr-dict `:` functional-type(operands, results)
    }];
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {getParameter()}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed()   { if (auto v = getInput()) { return {v}; } return {}; }
        ::llvm::SmallVector<::mlir::Value> getProduced()   { if (auto v = getOutput()) { return {v}; } return {}; }
    }];
}

// Control flow. HSSA jumps consume their payload (and, for conditional
// exits, the selector bit); entries produce them. Entries need no op: a
// block's arguments are the entry pattern, and a two-predecessor block's
// trailing argument is the provenance bit, fed the 0-based label index as
// a constant by each incoming edge (0 via the first label, 1 via the
// second; index semantics verified against the toolkit corpus, though the
// paper's prose says the opposite).
def Hssa_ExitOp : Hssa_Op<"exit",
    [Terminator, DeclareOpInterfaceMethods<BranchOpInterface>]> {
    let summary = "Unconditional jump, consuming its payload.";
    let description = [{
        The exit half of HSSA's control flow: finalizes the values it passes
        and hands them to the target block's entry pattern. Block-local
        liveness means these payloads are the only way values cross blocks.

        Shape:
            () / (values...) -> ()      (payload consumed, then jump to dest)

        HSSA:
            -> L1 =: x, 0

        Example:
        ```mlir
        hssa.exit ^bb1(%x, %sel : i32, i32)
        ```
    }];
    let arguments = (ins Variadic<Hssa_Value>:$values);
    let successors = (successor AnySuccessor:$dest);
    let assemblyFormat = [{
        $dest (`(` $values^ `:` type($values) `)`)? attr-dict
    }];
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed()   { return ::llvm::to_vector(getValues()); }
        ::llvm::SmallVector<::mlir::Value> getProduced()   { return {}; }
    }];
}

// selector == 0 jumps to $first, 1 to $second (the 0-based label index);
// other values are reversibility violations. The two operand lists differ
// only in trailing edge constants for conditional-entry targets.
def Hssa_CondExitOp
    : Hssa_Op<"cond_exit", [Terminator, AttrSizedOperandSegments,
                            DeclareOpInterfaceMethods<BranchOpInterface>]> {
    let summary = "Conditional jump, consuming payload and selector bit.";
    let hasVerifier = 1;
    let description = [{
        Spends the selector to make a control decision: 0 jumps to the first
        successor, 1 to the second (the 0-based label index, verified against
        the toolkit corpus; the paper's prose states the opposite). The merge
        side needs no op -- a two-predecessor block's trailing argument
        receives the index as a per-edge constant, so branches consume the
        bit and merges produce it.

        Shape:
            () / (selector, payload...) -> ()   (0 -> first successor, 1 -> second)

        HSSA:
            -> L1,L2 =: x, c

        Example:
        ```mlir
        hssa.cond_exit %sel, ^bb1(%x : i32), ^bb2(%x : i32)
        ```
    }];
    let arguments = (ins Hssa_Int:$selector,
                         Variadic<Hssa_Value>:$firstOperands,
                         Variadic<Hssa_Value>:$secondOperands);
    let successors = (successor AnySuccessor:$first, AnySuccessor:$second);
    let assemblyFormat = [{
        $selector `,`
        $first (`(` $firstOperands^ `:` type($firstOperands) `)`)? `,`
        $second (`(` $secondOperands^ `:` type($secondOperands) `)`)?
        attr-dict
    }];
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed() {
            ::llvm::SmallVector<::mlir::Value> v{getSelector()};
            v.append(getFirstOperands().begin(), getFirstOperands().end());
            v.append(getSecondOperands().begin(), getSecondOperands().end());
            return v;
        }
        ::llvm::SmallVector<::mlir::Value> getProduced()   { return {}; }
    }];
}

// The N-ary generalization of cond_exit: the selector is a 0-based label index
// picking one of K successors. The merge side needs no op -- a K-predecessor
// block's trailing argument receives the index as a per-edge constant.
def Hssa_SwitchOp
    : Hssa_Op<"switch", [Terminator,
                         DeclareOpInterfaceMethods<BranchOpInterface>]> {
    let summary = "N-ary jump, consuming payload and the label-index selector.";
    let hasVerifier = 1;
    let description = [{
        The N-ary generalization of cond_exit (K >= 2 successors): the selector
        is a 0-based label index that picks one successor; other values are
        reversibility violations. The same payload is handed to every successor
        (each is a single-predecessor block), and the K-predecessor merge on the
        far side rebinds the index as its provenance argument -- cond_exit's bit
        widened to an index, so a branch consumes the index and the merge
        produces it. Used for multi-method vtable dispatch.

        Shape:
            () / (selector, payload...) -> ()   (selector = 0-based index, picks one of K successors)

        HSSA:
            -> L0,L2,L4 =: x, i

        Example:
        ```mlir
        hssa.switch %i, (%x) [^bb0, ^bb2, ^bb4] : i32
        ```
    }];
    let arguments = (ins Hssa_Int:$selector, Variadic<Hssa_Value>:$payload);
    let successors = (successor VariadicSuccessor<AnySuccessor>:$cases);
    let assemblyFormat = [{
        $selector `,` `(` $payload `)` `[` $cases `]` attr-dict
        (`:` type($payload)^)?
    }];
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed() {
            ::llvm::SmallVector<::mlir::Value> v{getSelector()};
            v.append(getPayload().begin(), getPayload().end());
            return v;
        }
        ::llvm::SmallVector<::mlir::Value> getProduced() { return {}; }
    }];
}

// Unconditionally traps in both directions, so (unlike dup / tests.assert) its
// effect does not depend on a flag: a pure TableGen trait suffices. The write to
// the abstract trap resource makes the intent explicit AND keeps the op out of
// reach of DCE/canonicalize -- a result-less op with no modelled effect would
// otherwise be deleted.
def Hssa_TestsFailOp : Hssa_Op<"tests.fail",
        [SelfInverse, MemoryEffects<[MemWrite<HssaTrapResource>]>]> {
    let summary = "Unconditional failure in either direction.";
    let description = [{
        Raises in both directions, so it carries no inverted flag
        (https://github.com/Leridon/hssa/blob/6738274/src/main/scala/de/thm/mni/hybridcomputing/hssa/plugin/Testing.scala#L13).
        Lowers to an assert on false.

        Shape:
            () / () -> ()       (aborts; consumes and produces nothing)

        HSSA:
            tests.fail

        Example:
        ```mlir
        hssa.tests.fail
        ```
    }];
    let assemblyFormat = [{ attr-dict }];
    let extraClassDeclaration = [{
        ::llvm::SmallVector<::mlir::Value> getParameters() { return {}; }
        ::llvm::SmallVector<::mlir::Value> getConsumed()   { return {}; }
        ::llvm::SmallVector<::mlir::Value> getProduced()   { return {}; }
    }];
}

#endif // HSSA_OPS
