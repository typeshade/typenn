// Parity test for src/lab/keras.ts: decode the five Keras source files that the browser downloads,
// and compare every array byte-for-byte with the mirror that scripts/fetch_datasets.py built.
// The source files are read from the cache (no network). Run with:
//   npm run test:keras
// The cache and the mirror default to .dataset-cache and public/data. Override them with
//   TYPENN_DATASET_CACHE=/path/to/.dataset-cache TYPENN_DATASET_MIRROR=/path/to/public/data
// Each dataset must match, or the script exits with status 1. It names each mismatch.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  KERAS_IDS,
  KERAS_SOURCES,
  decodeKeras,
  kerasCardManifest,
  verifySource,
} from "../src/lab/keras.ts";

const here = dirname(fileURLToPath(import.meta.url));
const website = join(here, "..");
const cacheDir =
  process.env.TYPENN_DATASET_CACHE ?? join(website, ".dataset-cache");
const mirrorDir =
  process.env.TYPENN_DATASET_MIRROR ?? join(website, "public", "data");

for (const dir of [cacheDir, mirrorDir]) {
  if (!existsSync(dir)) {
    console.error(`missing directory: ${dir}`);
    process.exit(1);
  }
}

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const bytesOf = (typed) =>
  Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength);

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

const failures = [];
const rows = [];

for (const id of KERAS_IDS) {
  const label = id.padEnd(19);
  try {
    const manifest = JSON.parse(
      readFileSync(join(mirrorDir, id, "manifest.json"), "utf8"),
    );
    const files = new Map();
    for (const source of manifest.sources) {
      const bytes = new Uint8Array(readFileSync(join(cacheDir, source.file)));
      await verifySource(source.file, bytes);
      assert.equal(
        KERAS_SOURCES[source.file].sha256,
        source.sha256,
        `${source.file}: table vs mirror`,
      );
      files.set(source.file, bytes);
    }
    const decoded = await decodeKeras(id, files);

    // The card manifest, before any load, must give the same counts.
    const card = kerasCardManifest(id);
    assert.equal(
      card.splits.train.count,
      manifest.splits.train.count,
      "card train count",
    );
    assert.equal(
      card.splits.test.count,
      manifest.splits.test.count,
      "card test count",
    );

    const checks = [];
    for (const split of ["train", "test"]) {
      const ours = decoded[split];
      const theirs = manifest.splits[split];
      assert.equal(ours.count, theirs.count, `${split} count`);
      for (const field of ["x", "y", "offsets"]) {
        if (!theirs[field]) {
          assert.equal(
            ours[field],
            undefined,
            `${split}.${field} should be absent`,
          );
          continue;
        }
        const array = ours[field];
        const dtype = {
          Uint8Array: "uint8",
          Uint16Array: "uint16",
          Uint32Array: "uint32",
          Float32Array: "float32",
        }[array.constructor.name];
        assert.equal(dtype, theirs[field].dtype, `${split}.${field} dtype`);
        const got = decoded.manifest.splits[split][field].shape;
        assert.deepEqual(got, theirs[field].shape, `${split}.${field} shape`);
        const want = mirrorField(id, theirs[field]);
        const mine = bytesOf(array);
        assert.equal(
          mine.byteLength,
          want.byteLength,
          `${split}.${field} byte length`,
        );
        assert.equal(sha(mine), sha(want), `${split}.${field} sha256`);
        checks.push(`${split}.${field}`);
      }
    }

    // Vocabulary and metadata.
    if (theirs_vocab(manifest)) {
      const words = JSON.parse(
        readFileSync(join(mirrorDir, id, manifest.vocabularyFile.file), "utf8"),
      );
      assert.deepEqual(decoded.vocabulary, words, "vocabulary");
      checks.push("vocabulary");
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
    let normDiff = 0;
    if (manifest.normalization) {
      const { mean, std } = decoded.manifest.normalization;
      for (let i = 0; i < mean.length; i++) {
        normDiff = Math.max(
          normDiff,
          Math.abs(mean[i] - manifest.normalization.mean[i]) /
            Math.max(1, Math.abs(manifest.normalization.mean[i])),
          Math.abs(std[i] - manifest.normalization.std[i]) /
            Math.max(1e-300, Math.abs(manifest.normalization.std[i])),
        );
      }
      assert.ok(normDiff < 1e-12, `normalization differs by ${normDiff}`);
    }
    rows.push(
      `${label} ok   train ${decoded.train.count} / test ${decoded.test.count}; ${checks.length} arrays` +
        (manifest.normalization
          ? `; normalisation max relative diff ${normDiff.toExponential(1)}`
          : ""),
    );
  } catch (error) {
    failures.push(id);
    rows.push(`${label} FAIL ${error.message}`);
  }
}

console.log(rows.join("\n"));
const passed = KERAS_IDS.length - failures.length;
console.log(`${passed}/${KERAS_IDS.length} datasets match the mirror exactly`);
if (failures.length > 0) process.exit(1);

function theirs_vocab(manifest) {
  return Boolean(manifest.vocabularyFile);
}
