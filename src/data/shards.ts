/**
 * The default largest shard, in bytes, of a device dataset source. It is the WebGPU default of
 * `maxStorageBufferBindingSize`. A dataset reads the limit of its device when TypeShade exposes it
 * (see `Device.storageBindingLimit`), and uses this value otherwise. Pass `shardBytes` to override.
 * 134217728 is 128 MiB: WebGPU's guaranteed `maxStorageBufferBindingSize`.
 */
export const DEFAULT_SHARD_BYTES = 134217728;

/** How many elements of `width` bits one u32 word holds: 4 for u8, 2 for u16, 1 for u32 and f32. */
export function elementsPerWord(width: 8 | 16 | 32): number {
  return 32 / width;
}

/**
 * The most samples one shard holds: a shard of `shardBytes` holds `floor(shardBytes / 4)` words, and
 * each sample takes `features` elements of `perWord` per word. Returns 0 when one sample does not fit.
 */
export function rowsPerShard(
  features: number,
  perWord: number,
  shardBytes: number,
): number {
  const words = Math.floor(shardBytes / 4);
  return Math.floor((words * perWord) / features);
}
