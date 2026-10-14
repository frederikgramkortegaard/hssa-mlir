# hssa-mlir

An out-of-tree [MLIR](https://mlir.llvm.org/) dialect for HSSA (Hybrid SSA), a reversible-computing IR.

Reversible-computing projects each tend to reimplement their own IR, parser, interpreter, and passes, with nothing shared between them. The hope is that hssa-mlir can help build a shared foundation for reversible-computing languages, compilers, and optimization passes: one textual format, one pass framework, and the LLVM backend, for any reversible front-end that emits HSSA (or RSSA, the reversible SSA form it builds on).

HSSA was designed by Lukas Gail and Uwe Meyer at Technische Hochschule Mittelhessen, building on Torben Æ. Mogensen's reversible SSA form (RSSA). This repository is an independent MLIR implementation of their work, not a replacement for it.

- Imports the HSSA text format, including the Roopl++ object-oriented profile the reference compiler emits (virtual dispatch, dynamic memory, recursion).
- Reversibility as MLIR passes: linearity checking and repair, per-function inversion and uncall resolution, and classification into the reversibility fragments (with a strict purely-reversible (D,D) mode).
- Lowers through `arith`/`cf`/`memref` to LLVM and native code, differentially tested against the reference interpreter with metamorphic and random-program fuzzing.
- A reversible buddy-memory allocator in `stdlib/`.

- 📌 Paper (in progress)
- 📌 [HSSA (Gail & Meyer, RC 2024)](https://doi.org/10.1007/978-3-031-62076-8_11)
- 📌 [RSSA (Mogensen, PSI 2015)](https://doi.org/10.1007/978-3-319-41579-6_16)
- 📌 [Contributing](CONTRIBUTING.md)

---

## Example

HSSA is reversible: every function has a mechanically-derived inverse. Take `f(p, q)(x) = (x + p) ^ q`, with read-only parameters `p`, `q` and a single input `x` (`examples/pipeline.mlir`):

```mlir
func.func @f(%p: i32, %q: i32, %x: i32) -> i32 attributes {hssa.params = 2 : i64} {
  %y = hssa.add(%p) %x : i32
  %z = hssa.xor(%q) %y : i32
  return %z : i32
}
```

`hssa-opt --hssa-invert` turns it into its inverse. The body runs bottom-up with each op flipped, so the result recovers `x` from `(x + p) ^ q`:

```mlir
func.func @f(%p: i32, %q: i32, %z: i32) -> i32 attributes {hssa.params = 2 : i64} {
  %y = hssa.xor(%q) %z {inverted} : i32
  %x = hssa.add(%p) %y {inverted} : i32
  return %x : i32
}
```

A caller reuses that inverse through `uncall` (`~f`), so running `f` forward and then uncalling it is the identity.

A front-end has two ways in: emit the HSSA text format and parse it with `hssa-translate --import-hssa`, or build the dialect directly through the MLIR C or Python bindings. The text format looks like this. Parameters are read-only, so this reversible double duplicates `rhs` and uncomputes the copy instead of consuming it:

```
rel checked_double rhs:
lhs, 0 := begin <-
 c   := dup rhs    =: ()
 t   := add c      =: lhs
 ()  := ~dup rhs   =: c
 out := xor 0      =: t
      -> end =: out, 0
```

That one-use discipline (every value created once and consumed once) is what `--hssa-verify-linear` enforces, and what keeps the program invertible.

## Usage

`examples/roundtrip.hssa` is self-contained: `main` seeds the value 42, applies `inc`, then uncalls it (inc's inverse), recovering 42. Import it to MLIR, lower it through LLVM, and run it, and the process exits 42:

```sh
hssa-translate --import-hssa examples/roundtrip.hssa \
  | hssa-opt --hssa-verify-linear --hssa-resolve-uncalls --canonicalize --convert-hssa-to-arith \
  | mlir-opt --convert-to-llvm \
  | mlir-translate --mlir-to-llvmir \
  | lli --jit-kind=mcjit
echo $?   # 42
```

Stop after the first stage to watch the HSSA text become the dialect:

```sh
hssa-translate --import-hssa examples/roundtrip.hssa
```

Passes (`hssa-opt --help` for the full list):

- `--hssa-verify-linear`: linearity check. Every value is created once and consumed once.
- `--hssa-invert`: replace each function with its inverse.
- `--hssa-resolve-uncalls`: materialise inverse functions for uncalls (`~f`).
- `--hssa-verify-reversible`: classify functions `(D,D)`/`(D,N)`/etc. `=strict` rejects anything outside the reversible `(D,D)` fragment.
- `--hssa-repair-linear`: restore linearity (a `dup` per extra consumer, a `discard` per leak).
- `--convert-hssa-to-arith`: lower to `arith`/`cf`/`memref`.

Reversibility is checked against the reference interpreter with `make diff` / `make diff-oo`.

## Scope

This covers the subset of HSSA that the reference ROOPL++ compiler emits: the object-oriented profile of virtual dispatch, arrays, dynamic memory, and recursion, not the entire HSSA surface. Hand-written programs that use constructs outside that profile may not import yet.

Correctness is pinned by differential testing against the reference interpreter. The importer, inverter, and lowering run on the nine verified ROOPL++ programs, and the native results are compared to the interpreter slot for slot (`make diff-oo`).

## Development

Once built (see below), day-to-day work goes through the `Makefile`:

```sh
make test     # lit test suite
make diff-oo  # differential tests against the reference interpreter
make all      # test + demo + run
```

- `stdlib/`: HSSA standard library (e.g. `buddy.hssa`, a reversible buddy-memory allocator).
- `test/Hssa/Import`, `test/Hssa/Transforms`, `test/Exec`: importer, pass, and native-execution tests.

Adding a pass: see [CONTRIBUTING.md](CONTRIBUTING.md). Property and fuzz testing: `make fuzz`, `scripts/fuzz-gen.sh`.

The differential tests (`make diff*`) and `scripts/fuzz-gen.sh` use the reference interpreter, pinned as a submodule. Set it up with:

```sh
git submodule update --init
(cd third_party/hssa-reference && sbt assembly)
```

Override its location with `HSSA_REFERENCE`.

## Building - Component Build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit \
    -DMLIR_INCLUDE_TESTS=ON
cmake --build . --target check-hssa
```
With Homebrew LLVM (`brew install llvm lit`), the test suite requires
`-DMLIR_INCLUDE_TESTS=ON` and Homebrew's clang
(`-DCMAKE_C_COMPILER=$(brew --prefix llvm)/bin/clang
-DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++`); day-to-day builds
and runs then go through the `Makefile` (`make test`, `make demo`,
`make run`).
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

## Building - Monolithic Build

This setup assumes that you build the project as part of a monolithic LLVM build via the `LLVM_EXTERNAL_PROJECTS` mechanism.
To build LLVM, MLIR, the example and launch the tests run
```sh
mkdir build && cd build
cmake -G Ninja $LLVM_SRC_DIR/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=hssa-dialect -DLLVM_EXTERNAL_HSSA_DIALECT_SOURCE_DIR=../
cmake --build . --target check-hssa
```
Here, `$LLVM_SRC_DIR` needs to point to the root of the monorepo.
