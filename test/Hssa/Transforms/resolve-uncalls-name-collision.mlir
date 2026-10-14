// RUN: hssa-translate --import-hssa %s | hssa-opt --hssa-resolve-uncalls | FileCheck %s

// A program may define a relation whose name collides with the one the pass
// synthesizes for an inverse (<callee>_inverse). resolve-uncalls must not clobber
// or accidentally call the user's relation: SymbolTable::insert uniquifies the
// generated inverse (g_inverse -> g_inverse_0), and the pass dedups by the
// ORIGINAL callee (not by name match), so the uncall is rewritten to call the
// synthesized inverse, never the user's same-named relation.

// The user's own g_inverse survives untouched (it computes xor, not g's inverse);
// the synthesized inverse of g is uniquified to g_inverse_0; and caller's `~g` is
// rewritten to call g_inverse_0, NOT the user's g_inverse. (Order-independent.)
// CHECK-DAG: func.func @g_inverse(
// CHECK-DAG: hssa.xor
// CHECK-DAG: func.func @g_inverse_0(
// CHECK-DAG: hssa.call @g_inverse_0(

rel g p:
  _in, 0 := begin <-
   _out := add p =: _in
           -> end =: _out, 0

rel g_inverse p:
  _in, 0 := begin <-
   _out := xor p =: _in
           -> end =: _out, 0

rel caller p:
  _out, 0 := begin <-
   _in := ~g p =: _out
           -> end =: _in, 0
