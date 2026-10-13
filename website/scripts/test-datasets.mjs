// Unit tests for the pure helpers of src/lab/datasets.ts, on tiny synthetic splits.
// Run with: npm run test:datasets (Node 22.6+ strips the TypeScript types).
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  bagOfWordsBatch,
  concatBytes,
  imageBatch,
  labels,
  tabularBatch,
  typed,
  verifyPart,
} from "../src/lab/datasets.ts";

let passed = 0;
const pending = [];
function test(name, fn) {
  pending.push(
    Promise.resolve()
      .then(fn)
      .then(() => {
        passed += 1;
        console.log(`ok - ${name}`);
      }),
  );
}

test("imageBatch scales uint8 pixels to [0, 1] in the requested order", () => {
  const split = {
    count: 3,
    x: new Uint8Array([0, 255, 51, 102, 204, 153, 10, 20, 30, 40, 50, 60]),
    y: new Uint8Array([0, 1, 2]),
  };
  // Each image is 2x2x1 (4 values); pick images 2 and 0.
  const out = imageBatch(split, [2, 0], new Float32Array(8));
  // Image 2 is bytes 8..11 (30, 40, 50, 60); image 0 is bytes 0..3 (0, 255, 51, 102).
  // The output is Float32, so the expected values are rounded the same way.
  const expected = Float32Array.from([
    30 / 255,
    40 / 255,
    50 / 255,
    60 / 255,
    0,
    1,
    51 / 255,
    102 / 255,
  ]);
  assert.deepEqual(Array.from(out), Array.from(expected));
});

test("bagOfWordsBatch sets 1 for in-vocabulary ids and ignores the rest", () => {
  const split = {
    count: 2,
    x: new Uint16Array([1, 5, 2, 9, 1]),
    y: new Uint8Array([0, 1]),
    offsets: new Uint32Array([0, 3, 5]),
  };
  const out = bagOfWordsBatch(split, [0, 1], 4, new Float32Array(8).fill(7));
  assert.deepEqual(Array.from(out), [0, 1, 1, 0, 0, 1, 0, 0]);
});

test("bagOfWordsBatch refuses a split without offsets", () => {
  const split = { count: 1, x: new Uint8Array([1]), y: new Uint8Array([0]) };
  assert.throws(
    () => bagOfWordsBatch(split, [0], 4, new Float32Array(4)),
    /offsets/,
  );
});

test("tabularBatch normalises with the train mean and std, and a zero std keeps the value centred", () => {
  const split = {
    count: 2,
    x: new Float32Array([1, 10, 3, 10]),
    y: new Float32Array([0.5, 1.5]),
  };
  const out = tabularBatch(split, [1, 0], [2, 10], [1, 0], new Float32Array(4));
  assert.deepEqual(Array.from(out), [1, 0, -1, 0]);
});

test("labels keeps the split's label type", () => {
  const classes = { count: 2, x: new Uint8Array(2), y: new Uint8Array([3, 7]) };
  assert.ok(labels(classes, [1, 0]) instanceof Uint8Array);
  assert.deepEqual(Array.from(labels(classes, [1, 0])), [7, 3]);
  const reg = {
    count: 2,
    x: new Float32Array(2),
    y: new Float32Array([0.25, 4]),
  };
  assert.ok(labels(reg, [1]) instanceof Float32Array);
  assert.deepEqual(Array.from(labels(reg, [1])), [4]);
});

test("typed decodes little-endian uint16 and float32", () => {
  assert.deepEqual(
    Array.from(typed("uint16", new Uint8Array([1, 2, 0, 1]).buffer)),
    [513, 256],
  );
  const f = new Uint8Array(new Float32Array([1.5]).buffer);
  assert.equal(typed("float32", f.buffer)[0], 1.5);
  assert.throws(
    () => typed("uint16", new Uint8Array([1, 2, 3]).buffer),
    /multiple/,
  );
});

test("concatBytes joins parts in order", () => {
  const joined = new Uint8Array(
    concatBytes([new Uint8Array([1, 2]).buffer, new Uint8Array([3]).buffer]),
  );
  assert.deepEqual(Array.from(joined), [1, 2, 3]);
});

test("verifyPart checks length and SHA-256", async () => {
  const data = new TextEncoder().encode("abc").buffer;
  const sha256 = createHash("sha256").update("abc").digest("hex");
  await verifyPart(data, { file: "a.bin", bytes: 3, sha256 });
  await assert.rejects(
    verifyPart(data, { file: "a.bin", bytes: 4, sha256 }),
    /expected 4 bytes/,
  );
  await assert.rejects(
    verifyPart(data, { file: "a.bin", bytes: 3, sha256: "0".repeat(64) }),
    /SHA-256 mismatch/,
  );
});

await Promise.all(pending);
console.log(`${passed} datasets helper tests passed`);
