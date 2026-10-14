// Corpus regression: the reference programs the importer supports must
// keep importing and lowering end-to-end. Inputs/ holds verbatim copies
// (plus a provenance header) of Leridon/hssa programs/hssa/*.hssa (MIT).
// --convert-hssa-to-arith marks the hssa dialect illegal, so a zero exit
// code also proves no hssa op survived lowering.

// RUN: hssa-translate --import-hssa %S/Inputs/fibpair.hssa | hssa-opt --hssa-resolve-uncalls --convert-hssa-to-arith | FileCheck %s --check-prefix=FIBPAIR
// FIBPAIR-LABEL: func @fibpair(
// FIBPAIR-SAME: %{{.*}}: i32) -> (i32, i32)
// FIBPAIR-LABEL: func @fibpair.test()
// FIBPAIR: call @fibpair(

// RUN: hssa-translate --import-hssa %S/Inputs/collatz.hssa | hssa-opt --hssa-resolve-uncalls --convert-hssa-to-arith | FileCheck %s --check-prefix=COLLATZ
// COLLATZ-LABEL: func @collatz_step(
// COLLATZ-SAME: %{{.*}}: i32) -> i32

// RUN: hssa-translate --import-hssa %S/Inputs/power.hssa | hssa-opt --hssa-resolve-uncalls --convert-hssa-to-arith | FileCheck %s --check-prefix=POWER
// POWER-LABEL: func @l4mul(
// POWER-SAME: %{{.*}}: i32, %{{.*}}: i32) -> i32
// POWER-LABEL: func @power(
// POWER-SAME: %{{.*}}: i32, %{{.*}}: i32) -> i32
// POWER-LABEL: func @power.test()
// POWER-LABEL: func @power.main()

// RUN: hssa-translate --import-hssa %S/Inputs/diss.hssa | hssa-opt --hssa-resolve-uncalls --convert-hssa-to-arith | FileCheck %s --check-prefix=DISS
// DISS-LABEL: func @add_one(
// DISS-SAME: %{{.*}}: i32) -> i32
// DISS-LABEL: func @abs(
// DISS-SAME: %{{.*}}: i32) -> (i32, i32)
