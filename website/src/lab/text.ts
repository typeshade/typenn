// Keras-style tokenisation for the text datasets, against the dataset's own vocabulary.json.
// The vocabulary lists each word at its token id: `vocabulary[id]` is the word with that id, as
// the Keras loader numbers them (index_from 3; 0 pad, 1 start, 2 out-of-vocabulary).

export interface TokenEncoding {
  readonly start_char: number;
  readonly oov_char: number;
  readonly index_from: number;
  readonly num_words: number;
}

// The characters Keras's text_to_word_sequence replaces with a space (the apostrophe is kept).
const FILTERS = /[!"#$%&()*+,\-./:;<=>?@[\\\]^_`{|}~\t\n]/g;

/** Lowercase, strip punctuation, split on spaces: Keras's text_to_word_sequence. */
export function words(text: string): string[] {
  return text
    .toLowerCase()
    .replace(FILTERS, " ")
    .split(" ")
    .filter((word) => word.length > 0);
}

/** Map each word to its token id, where the word is in the vocabulary below `numWords`. */
export function wordIndex(vocabulary: readonly string[]): Map<string, number> {
  const index = new Map<string, number>();
  for (let id = 0; id < vocabulary.length; id++) {
    const word = vocabulary[id]!;
    if (word !== "" && !index.has(word)) index.set(word, id);
  }
  return index;
}

/**
 * Encode a text as the dataset encodes its reviews: the start token, then one id per word. A word
 * outside the vocabulary, or an id at or above `num_words`, becomes the out-of-vocabulary id.
 */
export function encode(
  text: string,
  index: Map<string, number>,
  encoding: TokenEncoding,
): number[] {
  const ids = [encoding.start_char];
  for (const word of words(text)) {
    const id = index.get(word);
    ids.push(
      id === undefined || id >= encoding.num_words ? encoding.oov_char : id,
    );
  }
  return ids;
}

/** The binary bag of words of token ids: 1 for each id below `size`. */
export function bagOf(ids: readonly number[], size: number): Float32Array {
  const bag = new Float32Array(size);
  for (const id of ids) if (id < size) bag[id] = 1;
  return bag;
}
