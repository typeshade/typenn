import test, { after, before } from "node:test";
import assert from "node:assert/strict";
import { elementwiseCases, reductionAndMatmulCases } from "./_cases.mjs";
import { closeNn, loadNn } from "./_nn.mjs";
import { difference } from "./_reference.mjs";

// Every case of `_cases.mjs` on the CPU tier, against its plain-JavaScript reference. The CPU tier
// runs the same kernel functions as WebGPU, so a wrong rewrite of an operation fails here.
// TYPENN_FAST=1 leaves out the case of 2^24 + 1 elements.

const nn = await loadNn();
let device;
before(async () => {
  device = await nn.device({ device: "cpu" });
});
after(async () => {
  await device.dispose();
  await closeNn();
});

for (const testCase of [...elementwiseCases(), ...reductionAndMatmulCases()]) {
  if (testCase.cpuOnly && process.env.TYPENN_FAST) continue;
  test(testCase.name, async () => {
    const { actual, expected } = await testCase.run(nn, device);
    assert.equal(difference(actual, expected, testCase.tolerance), undefined);
  });
}
