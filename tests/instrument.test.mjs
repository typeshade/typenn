import test from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import {
  cpSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// A test that never fails proves nothing, so this one plants a fault in a copy of the sources and
// runs the other tests on the copy (`_nn.mjs` loads the sources through Vite). Each fault must be
// caught by the test that is meant to catch it, and the copy without a fault must pass.
//
//   a value:      `add` subtracts. Only `numerics.test.mjs` can see it, since the plan tests read
//                 no value.
//   the plan:     `relu` reads an element of the array it writes, at another index. The proof of
//                 TypeShade Rule 8.22 refuses the loop (R4, TS8070), so `relu` runs on the CPU
//                 tier, in a group of its own. The values do not change, so only
//                 `plan.test.mjs`, which asserts the tier, the dispatches and the groups, can
//                 see it.

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

/** A copy of the sources under `.typenn-build/` with `from` replaced by `to` in a kernel file. */
function plant(name, file, from, to) {
  const copy = join(root, ".typenn-build", "mutants", name);
  rmSync(copy, { recursive: true, force: true });
  mkdirSync(copy, { recursive: true });
  cpSync(join(root, "src"), join(copy, "src"), {
    recursive: true,
    filter: (path) => !path.endsWith(".shade.typeshade.ts"),
  });
  for (const config of ["vite.config.ts", "tsconfig.json", "package.json"])
    cpSync(join(root, config), join(copy, config));
  if (from === undefined) return copy;
  const path = join(copy, "src", "kernels", file);
  const source = readFileSync(path, "utf8");
  assert.equal(source.split(from).length, 2, `${file} holds ${from} once`);
  writeFileSync(path, source.replace(from, to));
  return copy;
}

/** Start a test file of this directory on the sources in `source`; it resolves to how it ended. */
function run(file, source) {
  const env = { ...process.env, TYPENN_SOURCE: source, TYPENN_FAST: "1" };
  // A test run inside a test run must not think it is a child of the outer one.
  delete env.NODE_TEST_CONTEXT;
  return new Promise((done) => {
    const child = spawn(
      process.execPath,
      ["--test", "--test-reporter=tap", join(root, "tests", file)],
      { env },
    );
    let output = "";
    child.stdout.on("data", (chunk) => (output += chunk));
    child.stderr.on("data", (chunk) => (output += chunk));
    child.on("close", (status) => done({ status, output }));
  });
}

const control = plant("control");
const wrongValue = plant(
  "wrong-value",
  "binary.shade.ts",
  "out[i] = a[i] + b[i];",
  "out[i] = a[i] - b[i];",
);
const refusedLoop = plant(
  "refused-loop",
  "map.shade.ts",
  "out[i] = max(x[i], 0.0);",
  "out[i] = max(x[i], 0.0) + 0.0 * out[(i + 1) % out.length];",
);

// The runs one after another: each loads and compiles the kernel functions through Vite, and four
// at a time they take longer than the same four in a row.
test("the sources without a fault pass the numerics and the plan tests", async () => {
  for (const file of ["numerics.test.mjs", "plan.test.mjs"]) {
    const { status, output } = await run(file, control);
    assert.equal(
      status,
      0,
      `${file} fails on a copy with no fault:\n${output}`,
    );
  }
});

test("a kernel function that computes a wrong value fails the numerics tests", async () => {
  const numerics = await run("numerics.test.mjs", wrongValue);
  assert.notEqual(numerics.status, 0);
  assert.match(numerics.output, /not ok \d+ - binary add, same shape/);
  assert.match(numerics.output, /not ok \d+ - graph add-mul-relu/);
  // The plan tests read no value: they do not see it.
  assert.equal((await run("plan.test.mjs", wrongValue)).status, 0);
});

test("a kernel function the proof refuses fails the plan tests, and no value changes", async () => {
  // The device is required to run every call on WebGPU, so the refused `relu` is an error. On a
  // device that may fall back, it is a group of its own: either way the report is not the one
  // asserted.
  const plan = await run("plan.test.mjs", refusedLoop);
  assert.notEqual(plan.status, 0);
  assert.match(
    plan.output,
    /not ok \d+ - a\.add\(b\)\.mul\(c\)\.relu\(\) is one dispatch/,
  );
  assert.match(
    plan.output,
    /relu\(\): no tier it may use can run it \(webgpu: it runs on the CPU/,
  );
  // `relu` runs on the CPU tier and gives the values it gave, so the numerics tests pass.
  assert.equal((await run("numerics.test.mjs", refusedLoop)).status, 0);
});
