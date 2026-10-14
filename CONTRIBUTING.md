# Contributing

hssa-mlir is meant to be built on. A new reversible-computing optimization is an ordinary MLIR pass; a new reversible front-end just emits the HSSA text format.

## Adding a pass

Passes are declared in `include/Hssa/HssaPasses.td` and implemented in `lib/Hssa/`.

1. Declare the pass in `HssaPasses.td` (name, summary, options).
2. Implement it in `lib/Hssa/`, as a `Pass` or a `RewritePatternSet` wired into one.
3. Add a lit test under `test/Hssa/Transforms/` (a `// RUN:` line plus `FileCheck`).

```sh
cmake --build build
make test
```

## The one hard rule: don't break reversibility

Every value is linear: created once, consumed once. Every function has a direction. A transform that ignores this produces wrong code.

- Keep `--hssa-verify-linear` passing. Standard SSA assumptions don't hold here. A value can't be freely shared, so CSE or GVN style merging is unsound: merge two `dup`s and one result gets consumed twice. The ops declare memory effects so stock passes stay honest. Don't mark things `Pure` to get around that.
- If a transform can leave linearity broken, it has to be fixable by `--hssa-repair-linear`, or it shouldn't run.
- Check it against the oracle. `make diff` and `make diff-oo` run the result natively and compare slot-for-slot to the reference HSSA interpreter.

These are necessary conditions, not a proof of correctness. A pass that breaks linearity, stops inverting, or diverges from the interpreter is wrong; passing all three does not make it right.

## Testing

- Dialect and pass behaviour: lit tests in `test/Hssa/` (`FileCheck`).
- Execution and reversibility: `make diff`, `make diff-oo` (differential against the interpreter).
- Properties and fuzzing:
  - `make fuzz`: metamorphic invariants over the corpus (imports, verify-linear, round-trip, `invert;invert` stays linear, linearity survives `--canonicalize`).
  - `scripts/fuzz.sh crash N`: mutate HSSA into the importer and passes; it must never crash.
  - `scripts/fuzz-gen.sh N`: random ROOPL++ through the reference compiler, then the full pipeline, then diff against the interpreter.
