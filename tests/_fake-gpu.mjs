// A recording stand-in for a WebGPU device, after the `fakeGpu` of TypeShade's own fuse tests
// (src/fuse.test.ts at 5fb79c7). It makes nothing, records the commands it is given in order, and
// reads every buffer back as zeros: it proves the plan (the dispatches, the submissions and the
// groups), never a value. Hand it to TypeNN with `nn.device({ device: "webgpu", gpuDevice })`.

/** `GPUBufferUsage.STORAGE`, which the specification fixes. */
const STORAGE = 128;

export function fakeGpu(options = {}) {
  const commands = [];
  /** The WGSL of every shader module the device was asked to make. */
  const shaders = [];
  /** Every buffer the device was asked to make, in order. */
  const made = [];
  const buffer = (size, usage = 0) => {
    const record = { size, usage, written: false, destroyed: false };
    made.push(record);
    return {
      record,
      mapAsync: async () => {},
      getMappedRange: () => new ArrayBuffer(size),
      unmap: () => {},
      destroy: () => {
        record.destroyed = true;
      },
    };
  };
  const pass = {
    setPipeline: () => {},
    setBindGroup: () => {},
    dispatchWorkgroups: (x, y, z) => commands.push(`dispatch ${x} ${y} ${z}`),
    end: () => {},
  };
  const encoder = {
    beginComputePass: () => pass,
    copyBufferToBuffer: (
      _source,
      _sourceOffset,
      _target,
      _targetOffset,
      size,
    ) => commands.push(`copy ${size}`),
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
    createBuffer: (descriptor) => buffer(descriptor.size, descriptor.usage),
    createBindGroup: () => ({}),
    createCommandEncoder: () => encoder,
    pushErrorScope: () => {},
    popErrorScope: async () => null,
    queue: {
      writeBuffer: (target) => {
        if (target.record !== undefined) target.record.written = true;
      },
      submit: () => commands.push("submit"),
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
