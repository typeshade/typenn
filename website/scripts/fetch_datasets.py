#!/usr/bin/env python3
"""Mirror the Keras datasets that the TypeNN lab trains on, as browser-ready binary parts.

The browser cannot fetch the Keras sources (they send no CORS headers), so the deploy builds
this mirror under website/public/data/ and Cloudflare serves it at /data/. Every output file is
at most 20 MiB, because Cloudflare limits one static asset to 25 MiB.

Usage (from website/):
    python3 scripts/fetch_datasets.py [--only mnist,imdb] [--cache DIR] [--out DIR]

Sources and conventions:
- mnist, imdb, reuters, california_housing: the .npz files and word indexes that
  keras/src/datasets/*.py downloads. The split and vocabulary logic is a copy of those loaders:
  mnist takes the arrays as they are; imdb and reuters use seed 113, start_char=1, oov_char=2,
  index_from=3 and num_words=10000; california_housing uses version "large", seed 113 and
  test_split=0.2.
- fashion_mnist: the four IDX .gz files of the same Keras base URL. The parser follows
  keras/src/datasets/fashion_mnist.py.
- cifar10, cifar100: the BINARY editions of the Krizhevsky archives (not the Python pickles
  that Keras downloads). The records are read from the tar in memory, never extracted to disk.

Hashes: a SHA-256 of each downloaded file is checked against the table below. Where Keras
publishes a file_hash for the same file, the value is Keras's (origin "keras"). The other values
are pinned from the bytes observed on 2026-10-13 (origin "pinned 2026-10-13"). A mismatch stops
the run.

The standard library and numpy are the only dependencies.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import pickle
import struct
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field

import numpy as np

MAX_FILE_BYTES = 20 * 1024 * 1024  # Cloudflare: 25 MiB per asset; keep a margin.
KERAS_BASE = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
PINNED = "pinned 2026-10-13"
KERAS = "keras"
SEED = 113
VOCABULARY = 10000
INDEX_FROM = 3
START_CHAR = 1
OOV_CHAR = 2

# name -> (url, sha256, origin). The name is also the cache file name.
SOURCES: dict[str, tuple[str, str, str]] = {
    "mnist.npz": (
        KERAS_BASE + "mnist.npz",
        "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
        KERAS,
    ),
    "train-labels-idx1-ubyte.gz": (
        KERAS_BASE + "train-labels-idx1-ubyte.gz",
        "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
        PINNED,
    ),
    "train-images-idx3-ubyte.gz": (
        KERAS_BASE + "train-images-idx3-ubyte.gz",
        "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
        PINNED,
    ),
    "t10k-labels-idx1-ubyte.gz": (
        KERAS_BASE + "t10k-labels-idx1-ubyte.gz",
        "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
        PINNED,
    ),
    "t10k-images-idx3-ubyte.gz": (
        KERAS_BASE + "t10k-images-idx3-ubyte.gz",
        "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
        PINNED,
    ),
    "cifar-10-binary.tar.gz": (
        "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd",
        PINNED,
    ),
    "cifar-100-binary.tar.gz": (
        "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
        "58a81ae192c23a4be8b1804d68e518ed807d710a4eb253b1f2a199162a40d8ec",
        PINNED,
    ),
    "imdb.npz": (
        KERAS_BASE + "imdb.npz",
        "69664113be75683a8fe16e3ed0ab59fda8886cb3cd7ada244f7d9544e4676b9f",
        KERAS,
    ),
    "imdb_word_index.json": (
        KERAS_BASE + "imdb_word_index.json",
        "027313fe9a320280f3bf0fac5baf58c4a683cf39a7ed9c3d3d3912bde4284cf6",
        "keras file_hash (md5) bfafd718b763782e994055a2d397834f; sha256 pinned 2026-10-13",
    ),
    "reuters.npz": (
        KERAS_BASE + "reuters.npz",
        "d6586e694ee56d7a4e65172e12b3e987c03096cb01eab99753921ef915959916",
        KERAS,
    ),
    "reuters_word_index.json": (
        KERAS_BASE + "reuters_word_index.json",
        "9b56000edc4b34a32e89fd5a23b2605594682c74efcf7d8f4e2a5514b771e633",
        "keras file_hash (md5) 4d44cc38712099c9e383dc6e5f11a921; sha256 pinned 2026-10-13",
    ),
    "california_housing.npz": (
        KERAS_BASE + "california_housing.npz",
        "1a2e3a52e0398de6463aebe6f4a8da34fb21fbb6b934cf88c3425e766f2a1a6f",
        KERAS,
    ),
}

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]  # The order of batches.meta.txt in cifar-10-binary.tar.gz.
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]  # The order of keras/src/datasets/fashion_mnist.py's documented labels.
CALIFORNIA_FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]  # The column order of the Keras california_housing.npz x (checked against the train means).

DATASET_IDS = [
    "mnist", "fashion_mnist", "cifar10", "cifar100",
    "imdb", "reuters", "california_housing",
]


@dataclass
class Part:
    file: str
    bytes: int
    sha256: str


@dataclass
class Field:
    dtype: str
    shape: list
    parts: list = field(default_factory=list)


@dataclass
class Split:
    count: int
    x: Field
    y: Field
    offsets: Field | None = None


# --------------------------------------------------------------------------- download and hash


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, cache: str) -> str:
    """Return the cached path of a source file, downloading it when absent."""
    url, expected, origin = SOURCES[name]
    path = os.path.join(cache, name)
    if not os.path.exists(path):
        print(f"  download {name}", flush=True)
        partial = path + ".part"
        request = urllib.request.Request(url, headers={"User-Agent": "typenn-site-datasets/1"})
        with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
        os.replace(partial, path)
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(
            f"SHA-256 mismatch for {name} ({origin}): expected {expected}, got {actual}. "
            f"Delete {path} if the source changed on purpose, then update the table."
        )
    return path


def read_bytes(name: str, cache: str) -> bytes:
    with open(fetch(name, cache), "rb") as handle:
        return handle.read()


# --------------------------------------------------------------------------- output


class Writer:
    """Writes one dataset directory and records each file's size and SHA-256."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.written: list[Part] = []
        os.makedirs(directory, exist_ok=True)

    def write(self, name: str, data: bytes) -> Part:
        assert len(data) <= MAX_FILE_BYTES, f"{name} is {len(data)} bytes, over 20 MiB"
        with open(os.path.join(self.directory, name), "wb") as handle:
            handle.write(data)
        part = Part(name, len(data), hashlib.sha256(data).hexdigest())
        self.written.append(part)
        return part

    def array_parts(self, base: str, array: np.ndarray, row_bytes: int) -> Field:
        """Split an array into parts of whole rows, each at most 20 MiB. Names are base.partN.bin."""
        rows = array.shape[0]
        per_part = max(1, MAX_FILE_BYTES // row_bytes)
        parts = []
        for number, start in enumerate(range(0, rows, per_part), start=1):
            chunk = np.ascontiguousarray(array[start : start + per_part])
            parts.append(self.write(f"{base}.part{number}.bin", chunk.tobytes()))
        return Field(_dtype_name(array), list(array.shape[1:]), parts)

    def token_parts(self, base: str, tokens: np.ndarray) -> Field:
        """Split the uint16 token stream into parts of whole tokens, each at most 20 MiB."""
        per_part = MAX_FILE_BYTES // 2
        parts = []
        for number, start in enumerate(range(0, tokens.size, per_part), start=1):
            chunk = tokens[start : start + per_part].astype("<u2")
            parts.append(self.write(f"{base}.part{number}.bin", chunk.tobytes()))
        return Field("uint16", [int(tokens.size)], parts)


def _dtype_name(array: np.ndarray) -> str:
    return {"uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "float32": "float32"}[str(array.dtype)]


def le(array: np.ndarray, dtype: str) -> np.ndarray:
    """Return the array in little-endian byte order of dtype."""
    return array.astype(np.dtype(dtype).newbyteorder("<"))


def write_labels(writer: Writer, name: str, labels: np.ndarray) -> Field:
    values = np.asarray(labels).reshape(-1)
    assert values.min() >= 0 and values.max() <= 255, f"{name}: label outside uint8"
    return Field("uint8", [], [writer.write(name, values.astype(np.uint8).tobytes())])


def write_tabular_y(writer: Writer, name: str, targets: np.ndarray) -> Field:
    values = le(np.asarray(targets, dtype=np.float64).reshape(-1), "float32")
    return Field("float32", [], [writer.write(name, values.tobytes())])


def dataset_total(directory: str) -> int:
    total = 0
    for root, _, files in os.walk(directory):
        total += sum(os.path.getsize(os.path.join(root, name)) for name in files)
    return total


def write_manifest(directory: str, manifest: dict) -> None:
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def field_json(value: Field | None) -> dict | None:
    if value is None:
        return None
    return {
        "dtype": value.dtype,
        "shape": value.shape,
        "parts": [asdict(p) for p in value.parts],
    }


def split_json(split: Split) -> dict:
    out = {"count": split.count, "x": field_json(split.x), "y": field_json(split.y)}
    if split.offsets is not None:
        out["offsets"] = field_json(split.offsets)
    return out


def sources_json(names: list[str]) -> list[dict]:
    return [{"file": n, "url": SOURCES[n][0], "sha256": SOURCES[n][1], "origin": SOURCES[n][2]} for n in names]


# --------------------------------------------------------------------------- Keras logic


_ALLOWED_PICKLE = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    """Allows only the numpy globals that Keras allows (keras/src/datasets/npz_utils.py)."""

    def find_class(self, module, name):
        if (module, name) in _ALLOWED_PICKLE:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def load_npz(data: bytes) -> dict[str, np.ndarray]:
    """Read an .npz; object arrays go through the restricted unpickler, as Keras does."""
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            raw = io.BytesIO(archive.read(member))
            try:
                arrays[member[:-4]] = np.lib.format.read_array(raw, allow_pickle=False)
            except ValueError:
                raw.seek(0)
                version = np.lib.format.read_magic(raw)
                if version[0] == 1:
                    np.lib.format.read_array_header_1_0(raw)
                else:
                    np.lib.format.read_array_header_2_0(raw)
                arrays[member[:-4]] = _RestrictedUnpickler(raw).load()
    return arrays


def keras_shuffle(length: int, seed: int) -> np.ndarray:
    indices = np.arange(length)
    np.random.RandomState(seed).shuffle(indices)
    return indices


def ragged_tokens(sequences) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ragged sequences into (tokens uint16, offsets uint32 of length n+1)."""
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    assert offsets[-1] < 2**32, "token stream too long for uint32 offsets"
    tokens = np.fromiter((t for s in sequences for t in s), dtype=np.int64, count=int(offsets[-1]))
    assert tokens.size == 0 or (tokens.min() >= 0 and tokens.max() < 2**16), "token id outside uint16"
    return tokens.astype(np.uint16), offsets.astype(np.uint32)


def keras_text(xs: list, num_words: int) -> list:
    """Apply the start_char, index_from and oov_char steps of keras imdb and reuters load_data."""
    shifted = [[START_CHAR] + [w + INDEX_FROM for w in x] for x in xs]
    return [[w if w < num_words else OOV_CHAR for w in x] for x in shifted]


def load_imdb(data: bytes) -> tuple[tuple, tuple]:
    """Return ((tokens, offsets, labels) train, same for test), as keras imdb.load_data orders them."""
    f = load_npz(data)
    x_train, labels_train = f["x_train"], f["y_train"]
    x_test, labels_test = f["x_test"], f["y_test"]
    rng = np.random.RandomState(SEED)
    indices = np.arange(len(x_train))
    rng.shuffle(indices)
    x_train, labels_train = x_train[indices], labels_train[indices]
    indices = np.arange(len(x_test))
    rng.shuffle(indices)
    x_test, labels_test = x_test[indices], labels_test[indices]
    xs = keras_text(list(x_train) + list(x_test), VOCABULARY)
    n_train = len(x_train)
    labels = np.concatenate([labels_train, labels_test])
    assert set(np.unique(labels).tolist()) <= {0, 1}, "imdb labels must be 0 or 1"
    tr = ragged_tokens(xs[:n_train]) + (labels_train,)
    te = ragged_tokens(xs[n_train:]) + (labels_test,)
    return tr, te


def load_reuters(data: bytes) -> tuple[tuple, tuple]:
    """Return ((tokens, offsets, labels) train, same for test), as keras reuters.load_data does."""
    f = load_npz(data)
    xs, labels = f["x"], f["y"]
    indices = keras_shuffle(len(xs), SEED)
    xs = list(xs[indices])
    labels = labels[indices]
    xs = keras_text(xs, VOCABULARY)
    idx = int(len(xs) * (1 - 0.2))
    tr = ragged_tokens(xs[:idx]) + (labels[:idx],)
    te = ragged_tokens(xs[idx:]) + (labels[idx:],)
    return tr, te


def word_list(index: dict[str, int]) -> list[str]:
    """Word for each token id: id = rank + INDEX_FROM; ids 0..2 are the reserved tokens."""
    words = ["<pad>", "<start>", "<oov>"] + [""] * (VOCABULARY - 3)
    for word, rank in index.items():
        token = rank + INDEX_FROM
        if 3 <= token < VOCABULARY:
            words[token] = word
    return words


# --------------------------------------------------------------------------- datasets


def build_mnist(cache: str, out: str) -> dict:
    f = load_npz(read_bytes("mnist.npz", cache))
    writer = Writer(os.path.join(out, "mnist"))
    splits = {}
    for name, (x_key, y_key) in {"train": ("x_train", "y_train"), "test": ("x_test", "y_test")}.items():
        images = f[x_key].astype(np.uint8).reshape(-1, 28, 28, 1)
        labels = f[y_key]
        assert images.shape[0] == labels.shape[0]
        x = writer.array_parts(f"{name}-x", images, 28 * 28)
        y = write_labels(writer, f"{name}-y.bin", labels)
        splits[name] = Split(int(images.shape[0]), x, y)
    manifest = {
        "id": "mnist",
        "title": "MNIST handwritten digits",
        "task": "image-classification",
        "sources": sources_json(["mnist.npz"]),
        "license": "Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0), per the Keras documentation.",
        "citation": "LeCun, Cortes and Burges, The MNIST database of handwritten digits.",
        "classes": [str(i) for i in range(10)],
        "splits": {k: split_json(v) for k, v in splits.items()},
    }
    return finish("mnist", manifest, writer, out)


def build_fashion(cache: str, out: str) -> dict:
    def idx(name: str, magic: int) -> np.ndarray:
        """Read an IDX file: a big-endian header, then the data (keras fashion_mnist.py's layout)."""
        with gzip.open(io.BytesIO(read_bytes(name, cache)), "rb") as handle:
            raw = handle.read()
        assert struct.unpack(">I", raw[:4])[0] == magic, f"{name}: unexpected IDX magic"
        if magic == 2049:
            return np.frombuffer(raw, np.uint8, offset=8)
        count, rows, cols = struct.unpack(">III", raw[4:16])
        assert (rows, cols) == (28, 28), f"{name}: expected 28x28 images"
        return np.frombuffer(raw, np.uint8, offset=16).reshape(count, 28, 28, 1)

    writer = Writer(os.path.join(out, "fashion_mnist"))
    splits = {}
    for name, (img, lab) in {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
    }.items():
        images = idx(img, 2051)
        labels = idx(lab, 2049)
        assert images.shape[0] == labels.shape[0]
        x = writer.array_parts(f"{name}-x", images, 28 * 28)
        y = write_labels(writer, f"{name}-y.bin", labels)
        splits[name] = Split(int(images.shape[0]), x, y)
    manifest = {
        "id": "fashion_mnist",
        "title": "Fashion-MNIST clothing",
        "task": "image-classification",
        "sources": sources_json([n for n in SOURCES if "ubyte" in n]),
        "license": "MIT License, Zalando SE (github.com/zalandoresearch/fashion-mnist).",
        "citation": "Xiao, Rasul and Vollgraf, Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, 2017.",
        "classes": FASHION_CLASSES,
        "splits": {k: split_json(v) for k, v in splits.items()},
    }
    return finish("fashion_mnist", manifest, writer, out)


def read_cifar_members(data: bytes, wanted: list[str]) -> dict[str, bytes]:
    """Read the wanted members of a binary CIFAR tar.gz in one pass, in memory."""
    found = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            base = os.path.basename(member.name)
            if member.isfile() and base in wanted:
                found[base] = archive.extractfile(member).read()
    missing = set(wanted) - set(found)
    assert not missing, f"CIFAR archive is missing {sorted(missing)}"
    return found


def cifar_images(raw: bytes, label_bytes: int) -> tuple[np.ndarray, np.ndarray]:
    """Records: label bytes, then 3072 bytes CHW (R, G, B planes). Returns NHWC uint8 and labels."""
    record = label_bytes + 3072
    assert len(raw) % record == 0, "CIFAR binary batch has a partial record"
    rows = np.frombuffer(raw, np.uint8).reshape(-1, record)
    labels = rows[:, label_bytes - 1]  # The fine label is the last label byte.
    pixels = rows[:, label_bytes:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(pixels), labels


def build_cifar(cache: str, out: str, dataset: str) -> dict:
    license_text = "Released by Krizhevsky (2009) for research use; see cs.toronto.edu/~kriz/cifar.html."
    citation = "Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009."
    if dataset == "cifar10":
        source, label_bytes, title = "cifar-10-binary.tar.gz", 1, "CIFAR-10 small images"
        batches = {"train": [f"data_batch_{i}.bin" for i in range(1, 6)], "test": ["test_batch.bin"]}
        wanted = [name for names in batches.values() for name in names] + ["batches.meta.txt"]
    else:
        source, label_bytes, title = "cifar-100-binary.tar.gz", 2, "CIFAR-100 small images (fine labels)"
        batches = {"train": ["train.bin"], "test": ["test.bin"]}
        wanted = ["train.bin", "test.bin", "fine_label_names.txt"]

    members = read_cifar_members(read_bytes(source, cache), wanted)
    if dataset == "cifar10":
        classes = [line for line in members["batches.meta.txt"].decode("utf-8").split("\n") if line.strip()]
        assert classes == CIFAR10_CLASSES, "batches.meta.txt does not match the CIFAR-10 class table"
    else:
        classes = members["fine_label_names.txt"].decode("utf-8").split()
        assert len(classes) == 100, "fine_label_names.txt must list 100 classes"

    writer = Writer(os.path.join(out, dataset))
    splits = {}
    for split_name, names in batches.items():
        parts = [cifar_images(members[n], label_bytes) for n in names]
        images = np.concatenate([p[0] for p in parts])
        labels = np.concatenate([p[1] for p in parts])
        x = writer.array_parts(f"{split_name}-x", images, 32 * 32 * 3)
        y = write_labels(writer, f"{split_name}-y.bin", labels)
        splits[split_name] = Split(int(images.shape[0]), x, y)
    manifest = {
        "id": dataset,
        "title": title,
        "task": "image-classification",
        "sources": sources_json([source]),
        "license": license_text,
        "citation": citation,
        "classes": classes,
        "splits": {k: split_json(v) for k, v in splits.items()},
    }
    return finish(dataset, manifest, writer, out)


def build_imdb(cache: str, out: str) -> dict:
    train, test = load_imdb(read_bytes("imdb.npz", cache))
    writer = Writer(os.path.join(out, "imdb"))
    splits = write_text_splits(writer, {"train": train, "test": test})
    index = json.loads(read_bytes("imdb_word_index.json", cache).decode("utf-8"))
    vocabulary = write_vocabulary(writer, word_list(index))
    manifest = {
        "id": "imdb",
        "title": "IMDB movie reviews (sentiment)",
        "task": "text-classification",
        "sources": sources_json(["imdb.npz", "imdb_word_index.json"]),
        "license": "The dataset is distributed for research use. Source: Maas et al. (2011), ai.stanford.edu/~amaas/data/sentiment.",
        "citation": "Maas, Daly, Pham, Huang, Ng and Potts, Learning Word Vectors for Sentiment Analysis, ACL 2011.",
        "classes": ["negative", "positive"],
        "vocabulary": VOCABULARY,
        "vocabularyFile": asdict(vocabulary),
        "tokenEncoding": {"start_char": START_CHAR, "oov_char": OOV_CHAR, "index_from": INDEX_FROM, "num_words": VOCABULARY},
        "splits": {k: split_json(v) for k, v in splits.items()},
    }
    return finish("imdb", manifest, writer, out)


def build_reuters(cache: str, out: str) -> dict:
    train, test = load_reuters(read_bytes("reuters.npz", cache))
    writer = Writer(os.path.join(out, "reuters"))
    splits = write_text_splits(writer, {"train": train, "test": test})
    index = json.loads(read_bytes("reuters_word_index.json", cache).decode("utf-8"))
    vocabulary = write_vocabulary(writer, word_list(index))
    manifest = {
        "id": "reuters",
        "title": "Reuters newswire topics",
        "task": "text-classification",
        "sources": sources_json(["reuters.npz", "reuters_word_index.json"]),
        "license": "Reuters-21578 (distributed by Keras). Reuters-21578 is public domain under the terms of the Reuters distribution.",
        "citation": "Reuters-21578, Text Categorization Test Collection, distribution 1.0 (David D. Lewis).",
        "classes": [str(i) for i in range(46)],
        "vocabulary": VOCABULARY,
        "vocabularyFile": asdict(vocabulary),
        "tokenEncoding": {"start_char": START_CHAR, "oov_char": OOV_CHAR, "index_from": INDEX_FROM, "num_words": VOCABULARY},
        "splits": {k: split_json(v) for k, v in splits.items()},
    }
    return finish("reuters", manifest, writer, out)


def write_text_splits(writer: Writer, splits: dict[str, tuple]) -> dict[str, Split]:
    """Write token parts, offsets and labels for each split of a text dataset."""
    out = {}
    for name, (tokens, offsets, labels) in splits.items():
        x = writer.token_parts(f"{name}-tokens", tokens)
        off_bytes = le(offsets, "uint32").tobytes()
        off = Field("uint32", [int(offsets.size)], [writer.write(f"{name}-offsets.bin", off_bytes)])
        y = write_labels(writer, f"{name}-y.bin", labels)
        out[name] = Split(int(labels.size), x, y, off)
    return out


def write_vocabulary(writer: Writer, words: list[str]) -> Part:
    data = json.dumps(words, ensure_ascii=False).encode("utf-8")
    return writer.write("vocabulary.json", data)


def build_california(cache: str, out: str) -> dict:
    f = load_npz(read_bytes("california_housing.npz", cache))
    x_all = f["x"].astype(np.float64)
    y_all = f["y"].astype(np.float64)
    indices = keras_shuffle(len(x_all), SEED)
    x_all, y_all = x_all[indices], y_all[indices]
    cut = int(len(x_all) * (1 - 0.2))
    x_train, x_test = x_all[:cut], x_all[cut:]
    y_train, y_test = y_all[:cut], y_all[cut:]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    writer = Writer(os.path.join(out, "california_housing"))
    splits = {}
    for name, x, y in (("train", x_train, y_train), ("test", x_test, y_test)):
        xf = le(x, "float32")
        x_field = Field("float32", [int(x.shape[1])], [writer.write(f"{name}-x.bin", xf.tobytes())])
        y_field = write_tabular_y(writer, f"{name}-y.bin", y)
        splits[name] = Split(int(x.shape[0]), x_field, y_field)
    manifest = {
        "id": "california_housing",
        "title": "California housing prices (regression)",
        "task": "regression",
        "sources": sources_json(["california_housing.npz"]),
        "license": "StatLib dataset, Pace and Barry (1997); distributed by Keras. Use it for research with attribution.",
        "citation": "Pace and Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters, 1997 (StatLib).",
        "classes": None,
        "features": CALIFORNIA_FEATURES,
        "target": "median_house_value, in US dollars (checked: values such as 252300.0)",
        "normalization": {
            "basis": "train",
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
        },
        "splits": {k: split_json(v) for k, v in splits.items()},
    }
    return finish("california_housing", manifest, writer, out)


def finish(dataset: str, manifest: dict, writer: Writer, out: str) -> dict:
    """Check the writer's counts, write the manifest and return the index entry."""
    for split in manifest["splits"].values():
        assert split["count"] > 0
    write_manifest(writer.directory, manifest)
    total = dataset_total(writer.directory)
    print(f"  {dataset}: {len(writer.written)} data files, {total} bytes", flush=True)
    return {
        "id": dataset,
        "title": manifest["title"],
        "task": manifest["task"],
        "totalBytes": total,
        "manifest": f"{dataset}/manifest.json",
    }


BUILDERS = {
    "mnist": build_mnist,
    "fashion_mnist": build_fashion,
    "cifar10": lambda c, o: build_cifar(c, o, "cifar10"),
    "cifar100": lambda c, o: build_cifar(c, o, "cifar100"),
    "imdb": build_imdb,
    "reuters": build_reuters,
    "california_housing": build_california,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--only", help="comma-separated dataset ids, e.g. mnist,imdb")
    here = os.path.dirname(os.path.abspath(__file__))
    website = os.path.dirname(here)
    parser.add_argument("--cache", default=os.path.join(website, ".dataset-cache"))
    parser.add_argument("--out", default=os.path.join(website, "public", "data"))
    args = parser.parse_args(argv)

    ids = DATASET_IDS if not args.only else [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = [i for i in ids if i not in BUILDERS]
    if unknown:
        parser.error(f"unknown dataset ids: {', '.join(unknown)}")
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    entries = []
    for dataset in ids:
        print(f"{dataset}", flush=True)
        entries.append(BUILDERS[dataset](args.cache, args.out))

    index_path = os.path.join(args.out, "index.json")
    existing = []
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as handle:
            existing = [e for e in json.load(handle) if e["id"] not in {x["id"] for x in entries}]
    merged = sorted(existing + entries, key=lambda e: DATASET_IDS.index(e["id"]))
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
        handle.write("\n")
    print(f"wrote {index_path} ({len(merged)} datasets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
