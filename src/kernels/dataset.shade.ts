"use typeshade";

// Device-resident dataset gathers. A source is stored in u32 words: `width` bits per element and
// `perWord` elements per word (8 and 4 for u8, 16 and 2 for u16, 32 and 1 for u32 and f32). A
// float source keeps its f32 bits (`isFloat` is 1). Element `e` of a shard is word `e / perWord`,
// lane `e % perWord`.

/** The unsigned value of element `index` of a word that holds `perWord` elements of `width` bits. */
function laneValue(word: u32, index: u32, perWord: u32, width: u32): u32 {
  const shift = (index % perWord) * width;
  const one: u32 = 1;
  const zero: u32 = 0;
  const allBits: u32 = ~zero;
  const mask: u32 = select((one << width) - one, allBits, width === 32);
  return (word >> shift) & mask;
}

/**
 * The value of one sample feature, normalised as `(value - mean) / std`. `word` is the word that
 * holds element `index`; a float source (`isFloat` 1) reads the bits of that word as f32.
 */
function featureValue(
  word: u32,
  index: u32,
  perWord: u32,
  width: u32,
  isFloat: u32,
  mean: f32,
  std: f32,
): f32 {
  const element = laneValue(word, index, perWord, width);
  const raw = select(f32(element), bitcast<f32>(word), isFloat === 1);
  return (raw - mean) / std;
}

/**
 * `out` holds the batch of the first shard: row `r` is sample `perm[start + r]`, and a sample not
 * in this shard (`first` to `first + rows`) writes 0. Later shards read the output of the call
 * before them in `prev`, and change only the rows of their own shard.
 */
export function gatherFirst(
  out: array<f32>,
  perm: array<u32>,
  words: array<u32>,
  mean: array<f32>,
  std: array<f32>,
  start: u32,
  first: u32,
  rows: u32,
  features: u32,
  perWord: u32,
  width: u32,
  isFloat: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const row = i / features;
    const feature = i % features;
    const sample = perm[start + row];
    const inShard = sample >= first && sample - first < rows;
    const local = select(0, sample - first, inShard);
    const index = local * features + feature;
    const value = featureValue(
      words[index / perWord],
      index,
      perWord,
      width,
      isFloat,
      mean[feature],
      std[feature],
    );
    out[i] = select(0.0, value, inShard);
  }
}

/** As `gatherFirst`, and the rows outside this shard keep the value of `prev`. */
export function gatherNext(
  out: array<f32>,
  prev: array<f32>,
  perm: array<u32>,
  words: array<u32>,
  mean: array<f32>,
  std: array<f32>,
  start: u32,
  first: u32,
  rows: u32,
  features: u32,
  perWord: u32,
  width: u32,
  isFloat: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const row = i / features;
    const feature = i % features;
    const sample = perm[start + row];
    const inShard = sample >= first && sample - first < rows;
    const local = select(0, sample - first, inShard);
    const index = local * features + feature;
    const value = featureValue(
      words[index / perWord],
      index,
      perWord,
      width,
      isFloat,
      mean[feature],
      std[feature],
    );
    out[i] = select(prev[i], value, inShard);
  }
}

/**
 * Multi-hot bag of words: `out[r * vocab + t]` is 1 when the sample of row `r` holds token `t`,
 * and 0 otherwise. Tokens are packed like a source; `offsets[s]` to `offsets[s + 1]` bound sample
 * `s`. Tokens at or above `vocab` match no output.
 */
export function bagOfWords(
  out: array<f32>,
  perm: array<u32>,
  tokens: array<u32>,
  offsets: array<u32>,
  start: u32,
  vocab: u32,
  perWord: u32,
  width: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const row = i / vocab;
    const token = i % vocab;
    const sample = perm[start + row];
    const low = offsets[sample];
    const high = offsets[sample + 1];
    let hit: f32 = 0.0;
    for (let j: u32 = low; j < high; j++) {
      const id = laneValue(tokens[j / perWord], j, perWord, width);
      hit = select(hit, 1.0, id === token);
    }
    out[i] = hit;
  }
}
