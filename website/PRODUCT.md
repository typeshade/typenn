# TypeNN product

## Platform

web

Astro static site with React islands, TailwindCSS v4, and Ant Design v5. English is served at
`/`; Korean is served at `/ko/`. The canonical host is `typenn.typeshade.dev`; the `.com` host
redirects to it.

## Users

Primary users are TypeScript developers exploring neural-network code and GPU execution. They
need to see how a Tensor API reaches compiled kernels and how far the current training path goes.

The site should let a visitor understand the framework in one screen, inspect a real TypeNN
operation and TypeShade kernel, open the MNIST example, and reach the current implementation
status.

## Product purpose

TypeNN is a TypeScript neural-network framework built on TypeShade. TypeNN owns Tensor and Shape
semantics, its deferred Tensor graph, neural-network modules, Tensor-graph autograd, optimizers,
and graph scheduling. TypeShade owns the shader language, compiler, generated backend programs,
and execution runtime.

Tensor operations add deferred graph nodes. Materializing a result executes the reachable work.
TypeNN's internal `.shade.ts` kernels are compiled during the package build and run through
TypeShade for supported CPU and GPU tiers.

## Positioning

TypeNN makes the boundary between neural-network semantics and GPU compilation visible. Users
write Tensor and Module code in TypeScript. TypeNN lowers operations to TypeShade programs and
dispatches them on the selected device.

The source repository is the implementation reference. The site points visitors to actual API
examples, kernel files, the MNIST example, and the TODO list. A Frame may submit multiple kernel
dispatches together. Operation fusion remains a separate optimization goal.

GPU `Module → backward() → optimizer.step()` training schedules deferred gradient work and
in-place parameter updates through one TypeShade Frame. Adam moments are Device-owned Resident
buffers. The CPU optimizer path continues to update host arrays. A shared Frame can still contain
separate dispatches for separate Tensor graph nodes.

## Operating context

- TypeNN source: `https://github.com/typeshade/typenn`.
- TypeShade source: `https://github.com/typeshade/typeshade`.
- TypeNN is pre-release and its package metadata describes the npm package as a placeholder.
  The site directs readers to source and examples instead of promising an npm install.
- TypeShade's call-layer configuration is process-global, so TypeNN currently permits one
  active device per JavaScript context.

## Capabilities and constraints

- Deferred Tensor graphs execute through TypeShade kernels and the TypeShade program runtime.
- `device()` supports automatic WebGPU selection with CPU fallback, and explicit CPU, WebGPU,
  and WebGL2 choices where supported. WebGL2 uses a constrained lowered-compute path.
- The current operation set includes rank-2 MatMul, elementwise operations, activations,
  reductions, softmax, log-softmax, and logits-based softmax cross entropy. Shape and operation
  coverage is limited and documented in the TypeNN `TODO.md`.
- `Module`, `Sequential`, `Linear`, `Flatten`, `ReLU`, `Dropout`, and `CrossEntropyLoss` support
  the documented model path. The MNIST example demonstrates the user-defined Module training
  loop.
- Tensor-graph reverse-mode differentiation covers selected operation rules. TypeShade's
  `shadeGrad` differentiates pure TypeShade functions at a different layer.
- Graph-wide kernel fusion, model serialization, and WebNN are not implemented in the current
  README scope. Do not present them as available features.
- The normal Tensor graph executor still dispatches one kernel per node. MatMul uses a serial
  inner loop per output element; workgroup tiling and graph-wide buffer-liveness reuse are not
  implemented.
- The WebGPU and WebGL2 training paths have browser smoke coverage under Chromium with
  SwiftShader. Those checks are software-rendered, not physical-GPU performance benchmarks.
- Do not publish unmeasured GPU performance, comparison claims, usage counts, or testimonials.

## Brand commitments

- Name the framework TypeNN and identify TypeShade as its compiler and runtime foundation.
- Use the TypeShade site's IBM Plex and Ant Design token language.
- Use plain, factual copy. Keep headings short and sentence case.
- Show source code as evidence. Do not fabricate a compiler trace or simulate a GPU result.

## Evidence

- The Tensor API example is from the TypeNN README.
- The ReLU kernel is `src/kernels/relu.shade.ts` in TypeNN.
- The model training example is `examples/mnist` and the PyTorch-style README section.
- Feature boundaries and current limits are documented in TypeNN `README.md` and `TODO.md`.

## Product principles

1. Show real TypeNN code before expanding on the architecture.
2. Treat Tensor graphs and TypeShade Shader IR as separate layers with an explicit lowering
   boundary.
3. Link every capability claim to code, examples, or current project documentation.
4. Mark unfinished compiler optimization work as future work.
5. Keep the Tensor and Module API familiar while making TypeShade's compiler role clear.
