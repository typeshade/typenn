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
