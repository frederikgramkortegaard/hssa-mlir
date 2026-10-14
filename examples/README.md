# Examples

Two small programs, one per surface. Both are real inputs to the tools, not
placeholders.

- `basic.hssa`: the textual HSSA surface. Three relations: `add_one`,
  `checked_double`, and `main`. `checked_double` shows the reversible-double
  idiom: `dup` the read-only parameter into a scratch value, add the copy to
  produce the result, then `~dup` to uncompute the copy so nothing leaks. `main`
  calls `add_one` forward and then `~add_one`, which cancels it back to the
  input.

- `pipeline.mlir`: the same idea after import, in the Hssa MLIR dialect.
  `@f(p, q)(x)` computes `(x + p) ^ q`. `@main` calls it forward (5 to 14) and
  then calls it again with `{inverted}` (14 back to 5), so the exit code is 5.
  This is the call-then-uncall round-trip that `hssa-opt` and the invert pass
  operate on.
