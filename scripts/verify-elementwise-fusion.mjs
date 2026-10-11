import assert from "node:assert/strict";
import { device, inspectElementwiseFusion } from "../dist/index.js";

const runtime = await device("cpu");

function values() {
  return [
    runtime.tensor([1, 2, 3, 4], { shape: [4] }),
    runtime.tensor([5, -3, 2, -8], { shape: [4] }),
    runtime.tensor([2, 1, -2, 0.5], { shape: [4] }),
  ];
}

const [a, b, c] = values();
const add = a.add(b);
const fusedOutput = add.mul(c).relu();
assert.deepEqual(inspectElementwiseFusion(fusedOutput), {
  status: "fused",
  operations: 3,
  dispatches: 1,
});
assert.deepEqual([...(await fusedOutput.data())], [12, 0, 0, 0]);
assert.deepEqual([...(await add.data())], [6, -1, 5, -4]);

const [branchA, branchB, branchC] = values();
const branchedAdd = branchA.add(branchB);
const addBranch = branchedAdd.neg();
const branchedOutput = branchedAdd.mul(branchC).relu();
assert.equal(addBranch.shape[0], 4);
assert.deepEqual(inspectElementwiseFusion(branchedOutput), {
  status: "refused",
  code: "observable-intermediate",
  reason:
    "The add and multiply intermediates must each have exactly one live consumer.",
});

const [mulBranchA, mulBranchB, mulBranchC] = values();
const branchedProduct = mulBranchA.add(mulBranchB).mul(mulBranchC);
const productBranch = branchedProduct.neg();
const mulBranchedOutput = branchedProduct.relu();
assert.equal(productBranch.shape[0], 4);
assert.equal(
  inspectElementwiseFusion(mulBranchedOutput).code,
  "observable-intermediate",
);

const [materializedA, materializedB, materializedC] = values();
const materializedAdd = materializedA.add(materializedB);
await materializedAdd.data();
const materializedOutput = materializedAdd.mul(materializedC).relu();
assert.equal(
  inspectElementwiseFusion(materializedOutput).code,
  "materialized-intermediate",
);

const [aliasA, , aliasC] = values();
const aliasedOutput = aliasA.add(aliasA).mul(aliasC).relu();
assert.equal(inspectElementwiseFusion(aliasedOutput).code, "aliased-leaf");

const [shapeA, , shapeC] = values();
const short = runtime.tensor([1], { shape: [1] });
const shapeMismatch = shapeA.add(short).mul(shapeC).relu();
assert.equal(inspectElementwiseFusion(shapeMismatch).code, "shape-mismatch");

const [unsupportedA, unsupportedB, unsupportedC] = values();
const unsupported = unsupportedA.sub(unsupportedB).mul(unsupportedC).relu();
assert.equal(inspectElementwiseFusion(unsupported).code, "unsupported-chain");

const [parameterA, parameterB, parameterC] = values();
const parameter = runtime.createParameter(new Float32Array([2, 2, 2, 2]), {
  shape: [4],
});
const parameterOutput = parameterA.add(parameterB).mul(parameter).relu();
assert.equal(inspectElementwiseFusion(parameterOutput).code, "requires-grad");
assert.deepEqual(
  [...(await parameterA.mul(parameterB).data())],
  [5, -6, 6, -32],
);

await runtime.dispose();
process.stdout.write("Elementwise fusion CPU and decision checks passed.\n");
