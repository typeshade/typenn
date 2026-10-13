// A recording stand-in for a WebGPU device, after the `fakeGpu` of TypeShade's own fuse tests
// (src/fuse.test.ts at 96dbaa9). It makes nothing, records the commands it is given in order, and
// reads every buffer back as zeros: it proves the plan (the dispatches, the submissions and the
// groups), never a value. Hand it to TypeNN with `nn.device({ device: "webgpu", gpuDevice })`.

/** `GPUBufferUsage.STORAGE` and `GPUBufferUsage.MAP_READ`, which the specification fixes. */
const STORAGE = 128;
const MAP_READ = 1;

export function fakeGpu(options = {}) {
  const commands = [];
  let pops = 0;
  /**
   * Cumulative counters of the calls the device is given: buffers made, uploads (`writeBuffer`)
   * with their bytes and the bytes of the uploads that are all zeros, readbacks (a copy into a
   * `MAP_READ` buffer) with their bytes, and the submissions and dispatches.
   */
  const stats = {
    buffers: 0,
    uploads: 0,
    uploadBytes: 0,
    zeroUploadBytes: 0,
    readbacks: 0,
    readBytes: 0,
    submissions: 0,
    dispatches: 0,
  };
  /** The WGSL of every shader module the device was asked to make. */
  const shaders = [];
  /** Every buffer the device was asked to make, in order. */
  const made = [];
  const buffer = (size, usage = 0) => {
    const record = {
      size,
      usage,
      written: false,
      destroyed: false,
      destroys: 0,
    };
    made.push(record);
    return {
      record,
      mapAsync: async () => {},
      getMappedRange: () => new ArrayBuffer(size),
      unmap: () => {},
      destroy: () => {
        record.destroyed = true;
        record.destroys++;
      },
    };
  };
  const pass = {
    setPipeline: () => {},
    setBindGroup: () => {},
    dispatchWorkgroups: (x, y, z) => {
      stats.dispatches++;
      commands.push(`dispatch ${x} ${y} ${z}`);
    },
    end: () => {},
  };
  const encoder = {
    beginComputePass: () => pass,
    copyBufferToBuffer: (
      _source,
      _sourceOffset,
      target,
      _targetOffset,
      size,
    ) => {
      if (
        target.record !== undefined &&
        (target.record.usage & MAP_READ) !== 0
      ) {
        stats.readbacks++;
        stats.readBytes += size;
      }
      commands.push(`copy ${size}`);
    },
    finish: () => ({}),
  };
  const device = {
    createShaderModule: (descriptor) => {
      shaders.push(descriptor.code);
      return { getCompilationInfo: async () => ({ messages: [] }) };
    },
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createBuffer: (descriptor) => {
      stats.buffers++;
      return buffer(descriptor.size, descriptor.usage);
    },
    createBindGroup: () => ({}),
    createCommandEncoder: () => encoder,
    pushErrorScope: () => {},
    // `validation` makes the error scopes report a validation error: every one, or only the
    // `validationAt`-th (counting from 1), which fails one plan and lets the next run.
    popErrorScope: async () => {
      pops++;
      const fails =
        options.validation !== undefined &&
        (options.validationAt === undefined || options.validationAt === pops);
      return fails ? { message: options.validation } : null;
    },
    queue: {
      writeBuffer: (target, _offset, data) => {
        if (target.record !== undefined) target.record.written = true;
        const bytes =
          data instanceof ArrayBuffer
            ? new Uint8Array(data)
            : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        stats.uploads++;
        stats.uploadBytes += bytes.byteLength;
        if (bytes.every((byte) => byte === 0))
          stats.zeroUploadBytes += bytes.byteLength;
      },
      submit: () => {
        stats.submissions++;
        commands.push("submit");
      },
    },
    lost: new Promise(() => {}),
    ...(options.storageBuffers !== undefined
      ? { limits: { maxStorageBuffersPerShaderStage: options.storageBuffers } }
      : {}),
  };
  return {
    device,
    shaders,
    made,
    /** A copy of the cumulative counters. Take two and subtract them for one step. */
    get stats() {
      return { ...stats };
    },
    /** The storage buffers the device made and was given nothing to put in. */
    get zeroed() {
      return made.filter((b) => (b.usage & STORAGE) !== 0 && !b.written);
    },
    /** The dispatches and the submissions alone, in order. */
    get flow() {
      return commands.filter((c) => c.startsWith("dispatch") || c === "submit");
    },
    /** Forget the commands so far: what comes next is what one `data()` does. */
    mark() {
      commands.length = 0;
    },
  };
}
