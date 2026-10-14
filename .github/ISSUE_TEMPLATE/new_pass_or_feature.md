---
name: New pass or feature
about: Propose a new pass or a dialect feature
title: ''
labels: enhancement
assignees: ''
---

## The transform

What it does. Input shape, output shape, which ops it rewrites.

## Why it stays reversible

Every value is created once and consumed once, and every function has a
direction. Explain why the transform preserves that: what it does to `dup`,
to consumes, and to the inverse of each function it touches. If it can leave
linearity broken, say how `--hssa-repair-linear` fixes it up.

## How it gets tested

- The lit test under `test/Hssa/Transforms/` (what it checks).
- `--hssa-verify-linear` still passes on the output.
- `invert` still round-trips.
- `make diff` / `make diff-oo` still agree with the interpreter.
