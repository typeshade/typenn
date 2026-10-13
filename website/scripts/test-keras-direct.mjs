// Parity test for src/lab/keras.ts: decode the Keras source files that the browser downloads and
// hold every decoded array to scripts/keras-expected.json (the SHA-256 and the length of each
// array). The fixture comes from the mirror that scripts/fetch_datasets.py built, which equals
// Keras 3.15.1 load_data() element for element. Run with:
//   npm run test:keras
// Source files: from TYPENN_DATASET_CACHE (default .dataset-cache) when a file is there, else from
// the Keras storage (about 60 MB) into a temporary directory. Each file is checked against the
// pinned SHA-256 either way.
// Mirror mode: when public/data (or TYPENN_DATASET_MIRROR) holds the five mirrored datasets, the
// arrays, the vocabulary and the manifest fields are also compared with the mirror directly.
// Fixture: `npm run test:keras -- --write-fixture` rebuilds keras-expected.json from the mirror.
// The script exits with status 1 on the first dataset that differs, and names each mismatch.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  KERAS_IDS,
  KERAS_SOURCES,
  decodeKeras,
  kerasCardManifest,
  kerasUrl,
  verifySource,
} from "../src/lab/keras.ts";

const here = dirname(fileURLToPath(import.meta.url));
const website = join(here, "..");
const fixturePath = join(here, "keras-expected.json");
const cacheDir =
  process.env.TYPENN_DATASET_CACHE ?? join(website, ".dataset-cache");
const mirrorDir =
  process.env.TYPENN_DATASET_MIRROR ?? join(website, "public", "data");
const writeFixture = process.argv.includes("--write-fixture");

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const bytesOf = (typed) =>
  Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength);
const DTYPES = {
  Uint8Array: "uint8",
  Uint16Array: "uint16",
  Uint32Array: "uint32",
  Float32Array: "float32",
};

const mirrorManifest = (id) => {
  const file = join(mirrorDir, id, "manifest.json");
  return existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : null;
};
const mirrorAvailable = KERAS_IDS.every((id) => mirrorManifest(id) !== null);

/** The mirror's field: its parts concatenated, each part checked against its manifest entry. */
function mirrorField(id, field) {
  const pieces = field.parts.map((part) => {
    const bytes = readFileSync(join(mirrorDir, id, part.file));
    assert.equal(bytes.byteLength, part.bytes, `${id}/${part.file} length`);
    assert.equal(sha(bytes), part.sha256, `${id}/${part.file} sha256`);
    return bytes;
  });
  return Buffer.concat(pieces);
}

if (writeFixture) {
  if (!mirrorAvailable) {
    console.error(`--write-fixture needs the mirror in ${mirrorDir}`);
    process.exit(1);
  }
  const datasets = {};
  for (const id of KERAS_IDS) {
    const manifest = mirrorManifest(id);
    const entry = { splits: {} };
    for (const split of ["train", "test"]) {
      const ref = manifest.splits[split];
      const out = { count: ref.count };
      for (const field of ["x", "y", "offsets"]) {
        if (!ref[field]) continue;
        const bytes = mirrorField(id, ref[field]);
        out[field] = {
          dtype: ref[field].dtype,
          shape: ref[field].shape,
          bytes: bytes.byteLength,
          sha256: sha(bytes),
        };
      }
      entry.splits[split] = out;
    }
    if (manifest.vocabularyFile) {
      const words = JSON.parse(
        readFileSync(join(mirrorDir, id, manifest.vocabularyFile.file), "utf8"),
      );
      entry.vocabulary = {
        length: words.length,
        sha256: sha(JSON.stringify(words)),
      };
    }
    if (manifest.normalization) entry.normalization = manifest.normalization;
    datasets[id] = entry;
  }
  const fixture = {
    comment:
      "Expected decoded arrays of src/lab/keras.ts: byte length and SHA-256 of each array (little-endian typed array bytes). Built from the mirror, which equals Keras 3.15.1 load_data(). Rebuild with: npm run test:keras -- --write-fixture",
    sources: KERAS_SOURCES,
    datasets,
  };
  writeFileSync(fixturePath, JSON.stringify(fixture, null, 2) + "\n");
  console.log(`wrote ${fixturePath}`);
  process.exit(0);
}

const fixture = JSON.parse(readFileSync(fixturePath, "utf8"));

// Source files: the cache directory when it has the file, else the Keras storage.
let tempDir = null;
async function sourceBytes(name) {
  const local = join(cacheDir, name);
  if (existsSync(local)) return new Uint8Array(readFileSync(local));
  tempDir ??= mkdtempSync(join(tmpdir(), "typenn-keras-"));
  const cached = join(tempDir, name);
  if (existsSync(cached)) return new Uint8Array(readFileSync(cached));
  let lastError;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const response = await fetch(kerasUrl(name));
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      await verifySource(name, bytes);
      writeFileSync(cached, bytes);
      return bytes;
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`${name}: download failed (${lastError?.message})`);
}

assert.deepEqual(
  fixture.sources,
  KERAS_SOURCES,
  "the fixture and keras.ts pin different sources",
);

const failures = [];
const rows = [];

for (const id of KERAS_IDS) {
  const label = id.padEnd(19);
  try {
    const expected = fixture.datasets[id];
    assert.ok(expected, "no fixture entry");
    const spec = kerasCardManifest(id);
    const files = new Map();
    for (const source of spec.sources) {
      const bytes = await sourceBytes(source.file);
      await verifySource(source.file, bytes);
      files.set(source.file, bytes);
    }
    const decoded = await decodeKeras(id, files);

    const checks = [];
    for (const split of ["train", "test"]) {
      const ours = decoded[split];
      const want = expected.splits[split];
      assert.equal(ours.count, want.count, `${split} count`);
      assert.equal(spec.splits[split].count, want.count, `${split} card count`);
      for (const field of ["x", "y", "offsets"]) {
        if (!want[field]) {
          assert.equal(ours[field], undefined, `${split}.${field} unexpected`);
          continue;
        }
        const array = ours[field];
        assert.ok(array, `${split}.${field} missing`);
        assert.equal(
          DTYPES[array.constructor.name],
          want[field].dtype,
          `${split}.${field} dtype`,
        );
        assert.deepEqual(
          decoded.manifest.splits[split][field].shape,
          want[field].shape,
          `${split}.${field} shape`,
        );
        const mine = bytesOf(array);
        assert.equal(
          mine.byteLength,
          want[field].bytes,
          `${split}.${field} byte length`,
        );
        assert.equal(sha(mine), want[field].sha256, `${split}.${field} sha256`);
        checks.push(`${split}.${field}`);
      }
    }
    if (expected.vocabulary) {
      assert.ok(decoded.vocabulary, "vocabulary missing");
      assert.equal(
        decoded.vocabulary.length,
        expected.vocabulary.length,
        "vocabulary length",
      );
      assert.equal(
        sha(JSON.stringify(decoded.vocabulary)),
        expected.vocabulary.sha256,
        "vocabulary sha256",
      );
      checks.push("vocabulary");
    } else {
      assert.equal(decoded.vocabulary, undefined, "vocabulary unexpected");
    }
    let normDiff = 0;
    if (expected.normalization) {
      const { mean, std } = decoded.manifest.normalization;
      assert.equal(
        mean.length,
        expected.normalization.mean.length,
        "normalization length",
      );
      for (let i = 0; i < mean.length; i++) {
        normDiff = Math.max(
          normDiff,
          Math.abs(mean[i] - expected.normalization.mean[i]) /
            Math.max(1, Math.abs(expected.normalization.mean[i])),
          Math.abs(std[i] - expected.normalization.std[i]) /
            Math.max(1e-300, Math.abs(expected.normalization.std[i])),
        );
      }
      assert.ok(normDiff < 1e-12, `normalization differs by ${normDiff}`);
    }

    let mirrorNote = "";
    if (mirrorAvailable) {
      compareWithMirror(id, decoded);
      mirrorNote = "; mirror equal";
    }
    rows.push(
      `${label} ok   train ${decoded.train.count} / test ${decoded.test.count}; ${checks.length} arrays` +
        (expected.normalization
          ? `; normalisation max relative diff ${normDiff.toExponential(1)}`
          : "") +
        mirrorNote,
    );
  } catch (error) {
    failures.push(id);
    rows.push(`${label} FAIL ${error.message.split("\n")[0]}`);
  }
}

if (tempDir) rmSync(tempDir, { recursive: true, force: true });
console.log(rows.join("\n"));
console.log(
  `${KERAS_IDS.length - failures.length}/${KERAS_IDS.length} datasets match ${
    mirrorAvailable
      ? "the fixture and the mirror"
      : "the fixture (no mirror found)"
  } exactly`,
);
if (failures.length > 0) process.exit(1);

/** Mirror mode: the arrays, the vocabulary and the metadata, straight from the mirror files. */
function compareWithMirror(id, decoded) {
  const manifest = mirrorManifest(id);
  for (const split of ["train", "test"]) {
    for (const field of ["x", "y", "offsets"]) {
      const theirs = manifest.splits[split][field];
      if (!theirs) continue;
      const mine = bytesOf(decoded[split][field]);
      assert.ok(
        mine.equals(mirrorField(id, theirs)),
        `${id} ${split}.${field} differs from the mirror`,
      );
    }
  }
  if (manifest.vocabularyFile) {
    const words = JSON.parse(
      readFileSync(join(mirrorDir, id, manifest.vocabularyFile.file), "utf8"),
    );
    assert.deepEqual(decoded.vocabulary, words, "vocabulary vs mirror");
  }
  for (const key of [
    "id",
    "title",
    "task",
    "license",
    "citation",
    "classes",
    "features",
    "target",
    "vocabulary",
    "tokenEncoding",
  ]) {
    assert.deepEqual(
      decoded.manifest[key] ?? null,
      manifest[key] ?? null,
      `manifest.${key}`,
    );
  }
}
