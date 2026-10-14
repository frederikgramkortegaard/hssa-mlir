## What this changes

<!-- One or two lines. -->

## Reversibility checklist

The one hard rule is: don't break reversibility. Before you ask for review:

- [ ] `--hssa-verify-linear` passes on the output. No op was marked `Pure` to
      get around linearity.
- [ ] A transform that can leave linearity broken is fixable by
      `--hssa-repair-linear`, or it doesn't run.
- [ ] Inversion still round-trips (`invert;invert` stays linear).
- [ ] Checked against the interpreter: `make diff` and `make diff-oo` agree
      slot-for-slot.
- [ ] A lit test under `test/Hssa/` covers the change.
