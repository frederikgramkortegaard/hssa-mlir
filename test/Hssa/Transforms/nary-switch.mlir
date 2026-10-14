// RUN: hssa-translate --import-hssa %s | FileCheck %s --check-prefix=IMPORT
// The N-ary switch type-checks, is linear, inverts (and is involutive), and lowers.
// RUN: hssa-translate --import-hssa %s | hssa-opt --hssa-verify-linear
// RUN: hssa-translate --import-hssa %s | hssa-opt --hssa-invert | hssa-opt --hssa-verify-linear
// RUN: hssa-translate --import-hssa %s | hssa-opt --hssa-invert | hssa-opt --hssa-invert | hssa-opt --hssa-verify-linear
// RUN: hssa-translate --import-hssa %s | hssa-opt --convert-hssa-to-arith | FileCheck %s --check-prefix=LOWER

// The multi-method vtable's dispatch shape: a selector picks one of three
// labels (the N-ary branch), each arm produces a value, and a three-predecessor
// merge rebinds the selector as its provenance index before uncomputing it.
// This is cond_exit widened from a provenance bit to a provenance index.

// IMPORT-LABEL: func @pick
// IMPORT: hssa.switch %{{.*}}, () [^[[L0:.*]], ^[[L2:.*]], ^[[L4:.*]]]
// The three-predecessor merge binds payload + the provenance index.
// IMPORT: ^{{.*}}(%{{.*}}: i32, %{{.*}}: i32):

// The switch lowers to cf.switch.
// LOWER-LABEL: func @pick
// LOWER: cf.switch %{{.*}} : i32, [
// LOWER-NOT: hssa.
rel pick sel:
  (), 0 := begin <-
    i := dup sel =: ()
         -> L0,L2,L4 =: (), i
  (), 0 := L0 <-
    r := add 10 =: 0
         -> L1 =: r, 0
  (), 0 := L2 <-
    r := add 20 =: 0
         -> L3 =: r, 0
  (), 0 := L4 <-
    r := add 30 =: 0
         -> L5 =: r, 0
  r, i := L1,L3,L5 <-
    () := ~dup sel =: i
         -> end =: r, 0
