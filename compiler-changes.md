# Compiler changes this repository has handled

A change to the compiler that changes what TypeNN or its docs describe is agreed first as a
proposal in the compiler's `changes/` directory (the compiler's `changes/README.md` explains the
process). Each proposal lists, under `downstream`, the work it will owe the repositories it names.

When a pull request moves the compiler pin (`typeshade` in `package.json`) past a proposal that
names `typenn`, `scripts/downstream-impact.ts` of the compiler
(`bun <typeshade>/scripts/downstream-impact.ts --repo typenn`) lists the proposal until its work is
done on that branch and its id is recorded below. Record one list item per proposal: the id first,
then the pull request that did the work.

- 0071, pull request #24: `Device` is built on `createDevice()` of `typeshade/runtime`. The pool,
  the single-device guard and the limits cast are gone. The proposal names `typenn` in its expected
  work, not under `downstream`; the registration of `typenn` as a downstream repository is
  typeshade/typeshade#681.

- 0138, pull request #27 (milestone 4 slice 1): tensor and dataset storage use
  `Buffer<f32>` and `Buffer<u32>`, `set` and `dispose`, with typed-array reads.
  Stale parameter assignment reads its device value before replacing it, as the Buffer
  contract requires. The pin is the slice candidate `58e8187a`; the legacy device fuse
  scope remains until milestone 4 slice 5. This records slice 1 only, not the later graph migration.

The pull request #28 moved the pin from `58e8187a` to `f149b403`. The ids below were implemented
at or before that pin and named `typenn` under `downstream`. Each item says what the pin move did
or found.

- 0074, pull request #28: nothing changed. No kernel function iterates inside a dispatch, and
  TypeNN does not read the per-plan time of a chunked `fuse`.
- 0075, pull request #28: nothing changed. The change is additive, and TypeNN decides every
  dispatch on the host.
- 0083, pull request #28: device creation and its diagnostics are unchanged. The tests of
  `tests/plan.test.mjs`, `tests/api.test.mjs` and `tests/smoke.test.mjs` pass at the new pin.
  `BackendInfo` does not read the shared runtime information yet. This is open library work, not
  a requirement of the pin.
- 0108, pull request #28: nothing changed. The generated host views now carry the JSDoc of the
  kernel modules.
- 0109, pull request #28: nothing changed. No kernel module exports a class.
- 0110, pull request #28: nothing changed. No kernel module is written as a class.
- 0112, pull request #26: the five `pow` calls of `src/kernels/binary.shade.ts` are `a ** b`
  (decision D1). Recorded at pull request #28, the first pin move past the implementation.
- 0116, pull request #28: nothing changed. The library build runs `tsc` over the host code and
  the generated views at the new pin, and it passes.
- 0118, pull request #28: nothing changed. No test or host builds a `FuseReport` literal. The
  tests read the groups of a report through `outline` and `stated` in `tests/_plans.mjs`, which
  select fields.
- 0119, pull request #28: nothing changed. No kernel function under `src/kernels/` has an
  integer reduction or a scatter fold, so no kernel gets the subgroup variant.
- 0121, pull request #28: nothing changed, for the reason of 0118.
- 0122, pull request #28: nothing changed. No test or document holds a fusion origin, a report
  snapshot or a bundle snapshot.
- 0128, pull request #28: nothing changed, for the reason of 0118. No report aggregation reads
  the scratch metrics.
- 0129, pull request #28: the tests of failed plans pass at the new pin: "a plan that fails leaves
  no array behind, and the same read runs again" (`tests/resources.test.mjs`), "a plan that fails
  leaves the parameters and the moments as they were, and the step runs again" and "a failed
  Trainer.fit() still leaves the steps it made in the layer" (`tests/optim-lifecycle.test.mjs`),
  and "a failed plan with alsoRead leaves the parameters as they were"
  (`tests/buffer-pool.test.mjs`).
- 0130, pull request #28: nothing changed. No file of `src/` has a stage call or a stage
  attribute.
