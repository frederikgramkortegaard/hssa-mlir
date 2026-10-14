// Strict-RSSA mode: --hssa-verify-reversible=strict holds every function to the
// reversible (D,D) fragment -- as if each carried hssa.reversible -- so the one
// dialect hosts the whole HSSA family while its RSSA subset is machine-enforced.
//
// (a)+(b) In strict mode the (D,D) program passes with no error while the
//         discard-using (D,N) program is REJECTED with a clear diagnostic.
// RUN: hssa-opt --hssa-verify-reversible=strict --split-input-file \
// RUN:   --verify-diagnostics %s
//
// (c) The SAME file in NON-strict mode: both functions are merely classified,
//     no error -- so strict mode is the only thing that rejects @leaky. Checks
//     the stamped classes to prove the (D,N) function was processed, not an
//     error.
// RUN: hssa-opt --hssa-verify-reversible --split-input-file %s | \
// RUN:   FileCheck %s

// Only reversible ops (dup, add, euclidiv -- no discard, no oracle): classified
// (D,D), so strict mode is satisfied with no error.
// CHECK-LABEL: func @reversible
// CHECK-SAME: hssa.class = "(D,D)"
func.func @reversible(%p: i32, %x: i32) -> i32 {
  %c = hssa.dup(%x) : (i32) -> i32
  %s = hssa.add(%p) %x : i32
  %q, %r = hssa.euclidiv(%p) %s : (i32, i32) -> (i32, i32)
  return %c : i32
}

// -----

// discard taints the backward direction -> (D,N). Strict mode errors here; the
// diagnostic names the fragment and the class. In non-strict mode the same
// function is just stamped (D,N).
// CHECK-LABEL: func @leaky
// CHECK-SAME: hssa.class = "(D,N)"
// expected-error @+1 {{function is not in the reversible (D,D) fragment: classified (D,N)}}
func.func @leaky(%x: i32) attributes {hssa.params = 0 : i64} {
  hssa.discard %x : i32
  return
}
