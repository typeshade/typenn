# TypeNN implementation TODO

## Pure tensor operations

Public function names are intended to match TensorFlow.js so users can map familiar
operations directly: `tf.add(a, b)` becomes `nn.add(a, b)`. These are tensor
operations, not composite layers such as `Dense` or `Conv2D`. A TypeNN operation is
one or more TypeShade kernel functions, chosen by broadcasting and shape.

| Priority | TensorFlow.js operation                               | TypeNN API                                         | Status                                                                                  |
| -------- | ----------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------- |
| P0       | `tf.add(a, b)`                                        | `nn.add(a, b)`                                     | Partial: TypeShade kernel supports NumPy-style broadcasting through rank 8              |
| P0       | `tf.sub(a, b)`                                        | `nn.sub(a, b)`                                     | Partial: TypeShade kernel supports NumPy-style broadcasting through rank 8              |
| P0       | `tf.mul(a, b)`                                        | `nn.mul(a, b)`                                     | Partial: TypeShade kernel supports NumPy-style broadcasting through rank 8              |
| P0       | `tf.div(a, b)`                                        | `nn.div(a, b)`                                     | Partial: TypeShade kernel supports NumPy-style broadcasting through rank 8              |
| P0       | `tf.matMul(a, b)`                                     | `nn.matMul(a, b)`                                  | Partial: rank-2 TypeShade kernel supports transposeA/transposeB; batched matmul is TODO |
| P0       | `tf.sum(x, axis)`                                     | `nn.sum(x, axis)`                                  | Implemented: axis or all-axis reduction; scalar output supported                        |
| P0       | `tf.mean(x, axis)`                                    | `nn.mean(x, axis)`                                 | Implemented: axis or all-axis reduction; scalar output supported                        |
| P0       | `tf.softmax(x, axis)`                                 | `nn.softmax(x, axis)`                              | Implemented: arbitrary rank and axis; stable normalized slices                          |
| P0       | `tf.losses.softmaxCrossEntropy(oneHotLabels, logits)` | `nn.softmaxCrossEntropyWithLogits(logits, labels)` | Forward and Tensor-graph gradients support per-row, sum, and mean reductions            |
| P0       | `tf.argMax(x, axis)`                                  | `nn.argMax(x, axis)`                               | Implemented: one axis; indices are represented as float32 tensors                       |
| P1       | `tf.maximum(a, b)`                                    | `nn.maximum(a, b)`                                 | Partial: TypeShade kernel supports NumPy-style broadcasting through rank 8              |
| P1       | `tf.minimum(a, b)`                                    | `nn.minimum(a, b)`                                 | Partial: TypeShade kernel supports NumPy-style broadcasting through rank 8              |
| P1       | `tf.pow(a, b)`                                        | `nn.pow(a, b)`                                     | Partial: TypeShade kernel supports NumPy-style broadcasting through rank 8              |
| P1       | `tf.neg(x)`                                           | `nn.neg(x)`                                        | Implemented                                                                             |
| P1       | `tf.abs(x)`                                           | `nn.abs(x)`                                        | Implemented                                                                             |
| P1       | `tf.square(x)`                                        | `nn.square(x)`                                     | Implemented                                                                             |
| P1       | `tf.sqrt(x)`                                          | `nn.sqrt(x)`                                       | Implemented                                                                             |
| P1       | `tf.exp(x)`                                           | `nn.exp(x)`                                        | Implemented                                                                             |
| P1       | `tf.log(x)`                                           | `nn.log(x)`                                        | Implemented                                                                             |
| P1       | `tf.clipByValue(x, min, max)`                         | `nn.clipByValue(x, min, max)`                      | Implemented                                                                             |
| P1       | `tf.max(x, axis)`                                     | `nn.max(x, axis)`                                  | Implemented: one axis or all axes                                                       |
| P1       | `tf.min(x, axis)`                                     | `nn.min(x, axis)`                                  | Implemented: one axis or all axes                                                       |
| P1       | `tf.sigmoid(x)`                                       | `nn.sigmoid(x)`                                    | Implemented                                                                             |
| P1       | `tf.tanh(x)`                                          | `nn.tanh(x)`                                       | Implemented                                                                             |
| P1       | `tf.logSoftmax(x, axis)`                              | `nn.logSoftmax(x, axis)`                           | Implemented: arbitrary rank and axis; stable normalized slices                          |
| P2       | `tf.transpose(x, perm)`                               | `nn.transpose(x, perm)`                            | TODO                                                                                    |
| P2       | `tf.reshape(x, shape)`                                | `nn.reshape(x, shape)`                             | Implemented: preserves element count; CPU/WebGPU and autograd                           |
| P2       | `tf.concat(tensors, axis)`                            | `nn.concat(tensors, axis)`                         | TODO                                                                                    |
| P2       | `tf.split(x, numOrSizeSplits, axis)`                  | `nn.split(x, numOrSizeSplits, axis)`               | TODO                                                                                    |
| P2       | `tf.gather(x, indices, axis)`                         | `nn.gather(x, indices, axis)`                      | TODO                                                                                    |
| P2       | `tf.pad(x, paddings)`                                 | `nn.pad(x, paddings)`                              | TODO                                                                                    |
| P2       | `tf.where(condition, a, b)`                           | `nn.where(condition, a, b)`                        | TODO                                                                                    |
| P2       | `tf.equal(a, b)`                                      | `nn.equal(a, b)`                                   | TODO                                                                                    |
| P2       | `tf.greater(a, b)`                                    | `nn.greater(a, b)`                                 | TODO                                                                                    |
| P2       | `tf.less(a, b)`                                       | `nn.less(a, b)`                                    | TODO                                                                                    |
| P2       | `tf.gelu(x)`                                          | `nn.gelu(x)`                                       | Implemented with tanh approximation                                                     |
| P2       | `tf.leakyRelu(x, alpha)`                              | `nn.leakyRelu(x, alpha)`                           | Implemented                                                                             |
| P2       | `tf.elu(x, alpha)`                                    | `nn.elu(x, alpha)`                                 | Implemented                                                                             |
| P2       | `tf.relu(x)`                                          | `nn.relu(x)`                                       | Implemented                                                                             |

## Additional TensorFlow.js WebGPU kernel backlog

The table above is the initial core-op subset, not TensorFlow.js WebGPU parity. The
[`register_all_kernels.ts`](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-webgpu/src/register_all_kernels.ts)
registry currently registers 162 kernel configs. The directory also contains helper
implementations; gradients, fused kernels, image/string operations, and backend-only
utilities must be distinguished from public `tf.*` operations before assigning API
parity. The entries below extend the public-operation backlog and are not implemented
unless already present in the table above.

| Priority | TensorFlow.js operation                                      | Planned TypeNN API                                           | Status                                                                                                      |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| P1       | `tf.conv2d(x, filter, ...)`                                  | `nn.conv2d(x, filter, ...)`                                  | Partial: NHWC/HWIO forward, explicit input/filter gradients and graph autodiff; no bias/dilation/groups yet |
| P1       | `tf.depthwiseConv2d(x, filter, ...)`                         | `nn.depthwiseConv2d(x, filter, ...)`                         | TODO                                                                                                        |
| P1       | `tf.maxPool(x, ...)`                                         | `nn.maxPool(x, ...)`                                         | Partial: NHWC 2D pooling, explicit input gradient and graph autodiff; first maximum wins ties               |
| P1       | `tf.avgPool(x, ...)`                                         | `nn.avgPool(x, ...)`                                         | Partial: NHWC 2D pooling, explicit input gradient and graph autodiff; excludes padded cells from average    |
| P1       | `tf.softplus(x)`                                             | `nn.softplus(x)`                                             | Implemented: numerically stable formula                                                                     |
| P1       | `tf.relu6(x)`                                                | `nn.relu6(x)`                                                | Implemented                                                                                                 |
| P1       | `tf.prelu(x, alpha)`                                         | `nn.prelu(x, alpha)`                                         | Implemented with broadcasting                                                                               |
| P1       | `tf.addN(tensors)`                                           | `nn.addN(tensors)`                                           | Implemented as a graph of pairwise adds                                                                     |
| P1       | `tf.squaredDifference(a, b)`                                 | `nn.squaredDifference(a, b)`                                 | Implemented as `square(sub(a, b))`                                                                          |
| P1       | `tf.prod(x, axis)`                                           | `nn.prod(x, axis)`                                           | Implemented                                                                                                 |
| P1       | `tf.all(x, axis)`                                            | `nn.all(x, axis)`                                            | TODO                                                                                                        |
| P1       | `tf.any(x, axis)`                                            | `nn.any(x, axis)`                                            | TODO                                                                                                        |
| P2       | `tf.acos(x)`                                                 | `nn.acos(x)`                                                 | Implemented with TypeShade builtin; CPU/WebGPU; Tensor-graph gradient                                       |
| P2       | `tf.acosh(x)`                                                | `nn.acosh(x)`                                                | Implemented with TypeShade builtin; CPU/WebGPU; Tensor-graph gradient                                       |
| P2       | `tf.asin(x)`                                                 | `nn.asin(x)`                                                 | Implemented with TypeShade builtin; CPU/WebGPU; Tensor-graph gradient                                       |
| P2       | `tf.asinh(x)`                                                | `nn.asinh(x)`                                                | Implemented with TypeShade builtin; CPU/WebGPU; Tensor-graph gradient                                       |
| P2       | `tf.atan(x)`                                                 | `nn.atan(x)`                                                 | Implemented with TypeShade builtin; CPU/WebGPU; Tensor-graph gradient                                       |
| P2       | `tf.atan2(a, b)`                                             | `nn.atan2(a, b)`                                             | TODO                                                                                                        |
| P2       | `tf.atanh(x)`                                                | `nn.atanh(x)`                                                | Implemented with TypeShade builtin; CPU/WebGPU; Tensor-graph gradient                                       |
| P2       | `tf.ceil(x)`                                                 | `nn.ceil(x)`                                                 | Implemented                                                                                                 |
| P2       | `tf.cos(x)`                                                  | `nn.cos(x)`                                                  | Implemented                                                                                                 |
| P2       | `tf.cosh(x)`                                                 | `nn.cosh(x)`                                                 | Implemented                                                                                                 |
| P2       | `tf.erf(x)`                                                  | `nn.erf(x)`                                                  | TODO                                                                                                        |
| P2       | `tf.expm1(x)`                                                | `nn.expm1(x)`                                                | Implemented with a small-input series                                                                       |
| P2       | `tf.floor(x)`                                                | `nn.floor(x)`                                                | Implemented                                                                                                 |
| P2       | `tf.floorDiv(a, b)`                                          | `nn.floorDiv(a, b)`                                          | TODO                                                                                                        |
| P2       | `tf.log1p(x)`                                                | `nn.log1p(x)`                                                | Implemented with a small-input series                                                                       |
| P2       | `tf.mod(a, b)`                                               | `nn.mod(a, b)`                                               | TODO                                                                                                        |
| P2       | `tf.notEqual(a, b)`                                          | `nn.notEqual(a, b)`                                          | TODO                                                                                                        |
| P2       | `tf.greaterEqual(a, b)`                                      | `nn.greaterEqual(a, b)`                                      | TODO                                                                                                        |
| P2       | `tf.lessEqual(a, b)`                                         | `nn.lessEqual(a, b)`                                         | TODO                                                                                                        |
| P2       | `tf.reciprocal(x)`                                           | `nn.reciprocal(x)`                                           | Implemented                                                                                                 |
| P2       | `tf.round(x)`                                                | `nn.round(x)`                                                | Implemented                                                                                                 |
| P2       | `tf.rsqrt(x)`                                                | `nn.rsqrt(x)`                                                | Implemented                                                                                                 |
| P2       | `tf.sign(x)`                                                 | `nn.sign(x)`                                                 | Implemented                                                                                                 |
| P2       | `tf.sin(x)`                                                  | `nn.sin(x)`                                                  | Implemented                                                                                                 |
| P2       | `tf.sinh(x)`                                                 | `nn.sinh(x)`                                                 | Implemented                                                                                                 |
| P2       | `tf.step(x)`                                                 | `nn.step(x)`                                                 | Implemented as 0 for negative inputs, 1 otherwise; zero Tensor-graph gradient                               |
| P2       | `tf.tan(x)`                                                  | `nn.tan(x)`                                                  | Implemented with sin/cos                                                                                    |
| P2       | `tf.expandDims(x, axis)`                                     | `nn.expandDims(x, axis)`                                     | TODO                                                                                                        |
| P2       | `tf.reverse(x, axis)`                                        | `nn.reverse(x, axis)`                                        | TODO                                                                                                        |
| P2       | `tf.slice(x, begin, size)`                                   | `nn.slice(x, begin, size)`                                   | TODO                                                                                                        |
| P2       | `tf.stridedSlice(x, ...)`                                    | `nn.stridedSlice(x, ...)`                                    | TODO                                                                                                        |
| P2       | `tf.tile(x, reps)`                                           | `nn.tile(x, reps)`                                           | TODO                                                                                                        |
| P2       | `tf.gatherND(x, indices)`                                    | `nn.gatherND(x, indices)`                                    | TODO                                                                                                        |
| P2       | `tf.pack(tensors, axis)`                                     | `nn.pack(tensors, axis)`                                     | TODO                                                                                                        |
| P2       | `tf.unstack(x, axis)`                                        | `nn.unstack(x, axis)`                                        | TODO                                                                                                        |
| P2       | `tf.diag(x)`                                                 | `nn.diag(x)`                                                 | TODO                                                                                                        |
| P2       | `tf.mirrorPad(x, paddings, mode)`                            | `nn.mirrorPad(x, paddings, mode)`                            | TODO                                                                                                        |
| P2       | `tf.spaceToBatchND(x, blockShape, paddings)`                 | `nn.spaceToBatchND(x, blockShape, paddings)`                 | TODO                                                                                                        |
| P2       | `tf.batchToSpaceND(x, blockShape, crops)`                    | `nn.batchToSpaceND(x, blockShape, crops)`                    | TODO                                                                                                        |
| P2       | `tf.cast(x, dtype)`                                          | `nn.cast(x, dtype)`                                          | TODO                                                                                                        |
| P2       | `tf.fill(shape, value)`                                      | `nn.fill(shape, value)`                                      | TODO                                                                                                        |
| P2       | `tf.onesLike(x)`                                             | `nn.onesLike(x)`                                             | TODO                                                                                                        |
| P2       | `tf.zerosLike(x)`                                            | `nn.zerosLike(x)`                                            | TODO                                                                                                        |
| P2       | `tf.oneHot(indices, depth)`                                  | `nn.oneHot(indices, depth)`                                  | TODO                                                                                                        |
| P2       | `tf.range(start, stop, step)`                                | `nn.range(start, stop, step)`                                | TODO                                                                                                        |
| P2       | `tf.linspace(start, stop, num)`                              | `nn.linspace(start, stop, num)`                              | TODO                                                                                                        |
| P2       | `tf.scatterND(indices, updates, shape)`                      | `nn.scatterND(indices, updates, shape)`                      | TODO                                                                                                        |
| P2       | `tf.tensorScatterUpdate(x, indices, updates)`                | `nn.tensorScatterUpdate(x, indices, updates)`                | TODO                                                                                                        |
| P2       | `tf.topk(x, k)`                                              | `nn.topk(x, k)`                                              | TODO                                                                                                        |
| P2       | `tf.searchSorted(sorted, values)`                            | `nn.searchSorted(sorted, values)`                            | TODO                                                                                                        |
| P2       | `tf.sparseSegmentSum(data, indices, segmentIds)`             | `nn.sparseSegmentSum(data, indices, segmentIds)`             | TODO                                                                                                        |
| P2       | `tf.sparseSegmentMean(data, indices, segmentIds)`            | `nn.sparseSegmentMean(data, indices, segmentIds)`            | TODO                                                                                                        |
| P3       | `tf.conv3d(x, filter, ...)`                                  | `nn.conv3d(x, filter, ...)`                                  | TODO                                                                                                        |
| P3       | `tf.avgPool3d(x, ...)`                                       | `nn.avgPool3d(x, ...)`                                       | TODO                                                                                                        |
| P3       | `tf.maxPool3d(x, ...)`                                       | `nn.maxPool3d(x, ...)`                                       | TODO                                                                                                        |
| P3       | `tf.cropAndResize(image, boxes, boxIndex, cropSize)`         | `nn.cropAndResize(image, boxes, boxIndex, cropSize)`         | TODO                                                                                                        |
| P3       | `tf.image.resizeBilinear(x, size)`                           | `nn.image.resizeBilinear(x, size)`                           | TODO                                                                                                        |
| P3       | `tf.image.resizeNearestNeighbor(x, size)`                    | `nn.image.resizeNearestNeighbor(x, size)`                    | TODO                                                                                                        |
| P3       | `tf.image.flipLeftRight(x)`                                  | `nn.image.flipLeftRight(x)`                                  | TODO                                                                                                        |
| P3       | `tf.image.rotateWithOffset(x, radians, fillValue)`           | `nn.image.rotateWithOffset(x, radians, fillValue)`           | TODO                                                                                                        |
| P3       | `tf.complex(real, imag)`                                     | `nn.complex(real, imag)`                                     | TODO                                                                                                        |
| P3       | `tf.spectral.fft(x)`                                         | `nn.fft(x)`                                                  | TODO                                                                                                        |
| P3       | `tf.spectral.ifft(x)`                                        | `nn.ifft(x)`                                                 | TODO                                                                                                        |
| P3       | `tf.multinomial(logits, numSamples)`                         | `nn.multinomial(logits, numSamples)`                         | TODO                                                                                                        |
| P3       | `tf.image.nonMaxSuppression(boxes, scores, maxOutputSize)`   | `nn.image.nonMaxSuppression(boxes, scores, maxOutputSize)`   | TODO                                                                                                        |
| P3       | `tf.string.ngrams(...)`                                      | `nn.string.ngrams(...)`                                      | TODO                                                                                                        |
| P2       | `tf.argMin(x, axis)`                                         | `nn.argMin(x, axis)`                                         | Implemented; indices are represented as float32 tensors                                                     |
| P2       | `tf.cumsum(x, axis)`                                         | `nn.cumsum(x, axis)`                                         | TODO                                                                                                        |
| P2       | `tf.cumprod(x, axis)`                                        | `nn.cumprod(x, axis)`                                        | TODO                                                                                                        |
| P2       | `tf.logicalAnd(a, b)`                                        | `nn.logicalAnd(a, b)`                                        | TODO                                                                                                        |
| P2       | `tf.logicalOr(a, b)`                                         | `nn.logicalOr(a, b)`                                         | TODO                                                                                                        |
| P2       | `tf.logicalNot(x)`                                           | `nn.logicalNot(x)`                                           | TODO                                                                                                        |
| P2       | `tf.isFinite(x)`                                             | `nn.isFinite(x)`                                             | TODO                                                                                                        |
| P2       | `tf.isInf(x)`                                                | `nn.isInf(x)`                                                | TODO                                                                                                        |
| P2       | `tf.isNaN(x)`                                                | `nn.isNaN(x)`                                                | TODO                                                                                                        |
| P2       | `tf.depthToSpace(x, blockSize)`                              | `nn.depthToSpace(x, blockSize)`                              | TODO                                                                                                        |
| P2       | `tf.dilation2d(x, filter, ...)`                              | `nn.dilation2d(x, filter, ...)`                              | TODO                                                                                                        |
| P2       | `tf.einsum(equation, tensors)`                               | `nn.einsum(equation, tensors)`                               | TODO                                                                                                        |
| P2       | `tf.unsortedSegmentSum(x, segmentIds, numSegments)`          | `nn.unsortedSegmentSum(x, segmentIds, numSegments)`          | TODO                                                                                                        |
| P2       | `tf.sparseToDense(sparseIndices, outputShape, sparseValues)` | `nn.sparseToDense(sparseIndices, outputShape, sparseValues)` | TODO                                                                                                        |
| P2       | `tf.bincount(x, weights, size)`                              | `nn.bincount(x, weights, size)`                              | TODO                                                                                                        |
| P2       | `tf.denseBincount(x, weights, size)`                         | `nn.denseBincount(x, weights, size)`                         | TODO                                                                                                        |
| P2       | `tf.real(x)`                                                 | `nn.real(x)`                                                 | TODO                                                                                                        |
| P2       | `tf.imag(x)`                                                 | `nn.imag(x)`                                                 | TODO                                                                                                        |
| P2       | `tf.abs(x)` for complex tensors                              | `nn.abs(x)` for complex tensors                              | Dtype expansion TODO; f32 implemented above                                                                 |
| P3       | `tf.maxPoolWithArgmax(x, ...)`                               | `nn.maxPoolWithArgmax(x, ...)`                               | TODO                                                                                                        |
| P3       | `tf.browser.fromPixels(pixels)`                              | `nn.image.fromPixels(pixels)`                                | TODO                                                                                                        |

Autograd-only configs such as `Conv2DBackpropFilter`, `Conv2DBackpropInput`, pool
gradients, and `EluGrad` are implementation requirements for gradient rules rather
than separate user-facing forward APIs. Fused configs such as `_FusedMatMul`,
`FusedConv2D`, and `FusedBatchNorm` are planner/optimization work and should not force
matching public names. This table is a prioritized public-op backlog, not a claim that
all 162 backend registrations have a one-to-one user-facing TypeNN API.

## Upstream WebGPU kernel filename audit

Checked on 2026-10-10 against [`tfjs-backend-webgpu/src/kernels`](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-webgpu/src/kernels) on `master`. The directory has 172 direct files: 162 registered kernel configs, six shared implementation helpers, and four WebGPU tests. The operation tables above track public API parity; this inventory records every filename, including backend-only and non-public files.

### Registered kernel config files (162)

`Abs.ts` `Acos.ts` `Acosh.ts` `Add.ts` `AddN.ts` `All.ts`
`Any.ts` `ArgMax.ts` `ArgMin.ts` `Asin.ts` `Asinh.ts` `Atan.ts`
`Atan2.ts` `Atanh.ts` `AvgPool.ts` `AvgPool3D.ts` `AvgPool3DGrad.ts` `AvgPoolGrad.ts`
`BatchMatMul.ts` `BatchToSpaceND.ts` `Bincount.ts` `BroadcastArgs.ts` `Cast.ts` `Ceil.ts`
`ClipByValue.ts` `Complex.ts` `ComplexAbs.ts` `Concat.ts` `Conv2D.ts` `Conv2DBackpropFilter.ts`
`Conv2DBackpropInput.ts` `Conv3D.ts` `Conv3DBackpropFilterV2.ts` `Conv3DBackpropInputV2.ts` `Cos.ts` `Cosh.ts`
`CropAndResize.ts` `Cumprod.ts` `Cumsum.ts` `DenseBincount.ts` `DepthToSpace.ts` `DepthwiseConv2dNative.ts`
`DepthwiseConv2dNativeBackpropFilter.ts` `DepthwiseConv2dNativeBackpropInput.ts` `Diag.ts` `Dilation2D.ts` `Dilation2DBackpropFilter.ts` `Dilation2DBackpropInput.ts`
`Draw.ts` `Einsum.ts` `Elu.ts` `EluGrad.ts` `Equal.ts` `Erf.ts`
`Exp.ts` `ExpandDims.ts` `Expm1.ts` `FFT.ts` `Fill.ts` `FlipLeftRight.ts`
`Floor.ts` `FloorDiv.ts` `FromPixels.ts` `FusedBatchNorm.ts` `FusedConv2D.ts` `FusedDepthwiseConv2D.ts`
`GatherNd.ts` `GatherV2.ts` `Greater.ts` `GreaterEqual.ts` `IFFT.ts` `Identity.ts`
`Imag.ts` `IsFinite.ts` `IsInf.ts` `IsNaN.ts` `LRN.ts` `LRNGrad.ts`
`LeakyRelu.ts` `Less.ts` `LessEqual.ts` `LinSpace.ts` `Log.ts` `Log1p.ts`
`LogicalAnd.ts` `LogicalNot.ts` `LogicalOr.ts` `Max.ts` `MaxPool.ts` `MaxPool3D.ts`
`MaxPool3DGrad.ts` `MaxPoolGrad.ts` `MaxPoolWithArgmax.ts` `Maximum.ts` `Mean.ts` `Min.ts`
`Minimum.ts` `MirrorPad.ts` `Mod.ts` `Multinomial.ts` `Multiply.ts` `Neg.ts`
`NonMaxSuppressionV3.ts` `NonMaxSuppressionV5.ts` `NotEqual.ts` `OneHot.ts` `OnesLike.ts` `Pack.ts`
`PadV2.ts` `Pow.ts` `Prelu.ts` `Prod.ts` `Range.ts` `Real.ts`
`RealDiv.ts` `Reciprocal.ts` `Relu.ts` `Relu6.ts` `Reshape.ts` `ResizeBilinear.ts`
`ResizeBilinearGrad.ts` `ResizeNearestNeighbor.ts` `ResizeNearestNeighborGrad.ts` `Reverse.ts` `RotateWithOffset.ts` `Round.ts`
`Rsqrt.ts` `ScatterNd.ts` `SearchSorted.ts` `Select.ts` `Selu.ts` `Sigmoid.ts`
`Sign.ts` `Sin.ts` `Sinh.ts` `Slice.ts` `Softmax.ts` `Softplus.ts`
`SpaceToBatchND.ts` `SparseSegmentMean.ts` `SparseSegmentSum.ts` `SparseToDense.ts` `SplitV.ts` `Sqrt.ts`
`Square.ts` `SquaredDifference.ts` `Step.ts` `StridedSlice.ts` `StringNGrams.ts` `Sub.ts`
`Sum.ts` `Tan.ts` `Tanh.ts` `TensorScatterUpdate.ts` `Tile.ts` `TopK.ts`
`Transform.ts` `Transpose.ts` `Unpack.ts` `UnsortedSegmentSum.ts` `ZerosLike.ts` `_FusedMatMul.ts`

### Shared implementation helper files (6)

`BatchMatMul_impl.ts` `Concat_impl.ts` `Conv2D_impl.ts` `Cum_impl.ts` `FFT_impl.ts` `Pool_impl.ts`

### WebGPU test files (4)

`conv2d_webgpu_test.ts` `depthwise_conv2d_webgpu_test.ts` `pool_webgpu_test.ts` `precision_webgpu_test.ts`

## Training support required alongside forward operations

These are not replacements for the forward API above. Each differentiable operation
needs a gradient rule and corresponding execution path before TypeNN can train arbitrary models.

- [x] Train one `Linear` classifier with `createTrainer()`. A step is a Tensor graph (`matmul`, bias, mean softmax cross entropy), `backward()` and an `SGD` step. The dedicated `linear-*` kernels are removed.
- [x] Add `createTrainer()` with batched integer-label data and `onEpoch` metrics across TypeShade CPU/WebGPU/WebGL2 execution.
- [ ] Generalize `createTrainer().fit()` to `Sequential` models. The explicit PyTorch-style loop now supports a user-defined `Module` with multiple `Linear` layers, ReLU, Dropout, `backward()`, and SGD/Adam.

- [x] Re-export TypeShade `grad` and `gradCheck` as `shadeGrad`/`shadeGradCheck` for pure Shade functions and custom activations; this does not differentiate TypeNN Tensor graphs.
- [ ] Expand TypeNN Tensor-graph gradient propagation beyond the initial supported reverse-mode subset; TypeShade `grad` operates on pure IR functions, not TypeNN Tensor nodes or storage bindings.
- [x] Add `nn.gradients(output, wrt, upstream?)` reverse traversal for ReLU, broadcast binary ops, unary arithmetic/trigonometric ops, rank-2 MatMul with transpose options, Conv2D, 2D pooling, sum/mean, and softmax cross entropy.
- [x] Add softmax/log-softmax gradient kernels to Tensor-graph autograd.
- [ ] Add `prod`/`max`/`min` reduction gradients and special activation gradient rules.
- [x] Connect `nn.softmaxCrossEntropyWithLogits()` and `sum`/`mean` reductions to Tensor-graph autograd.
- [x] Add trainable graph parameters, accumulated `Tensor.backward()` gradients, `zeroGrad()`, and SGD/Adam optimizer objects. SGD and Adam compute the gradients and update the parameters, and the Adam moments, on the device in one plan (`src/kernels/optim.shade.ts`); `Module.sync()` refreshes the host copies in the layers.
- [ ] Use TypeShade `gradCheck` against its f64 CPU oracle for generated Shade derivatives.

## API and kernel implementation notes

- [x] Write every operation as a TypeShade kernel function in `src/kernels`, in eight modules: `map`, `binary`, `reduce`, `matmul`, `softmax`, `loss`, `conv` and `pool`. The host calls `kernel(out, ...arrays, ...values)`.
- [x] Run every `data()` as one `fuse` scope (`typeshade/fuse`) through one executor, `src/runtime/execute.ts`. The three tiers share it. No separate CPU executor remains.
- [x] Keep one graph node type, `OpNode`, in `src/tensor/node.ts`. It holds the kernel function, the input Tensors, the scalar values and the gradient closure.
- [x] Split Tensor graph builders into elementwise, linear-algebra, spatial, normalization, loss, shape, and reduction modules under `src/tensor/operations`.
- [x] Split public operation wrappers by operation family; keep `src/nn/ops.ts` as a re-export barrel.
- [x] Split each layer class into its own file; keep `src/nn/layers.ts` as a re-export barrel and `Sequential` in its own file.
- [x] Add shape-checked `reshape()` with CPU/WebGPU execution and a matching autograd rule. It runs the `copy` kernel function.
- [x] Add `Flatten`, `Dropout`, `Dense`, `Module.to()/parameters()/train()/eval()`, sparse-label `CrossEntropyLoss`, `Tensor.backward()/zeroGrad()`, and explicit SGD/Adam updates for the MNIST model loop.
- [x] Reduce `src/runtime` to `device.ts` (tier selection, resident ownership, reports), `device-types.ts` and `execute.ts`.
- [x] Report the plan of each read as `device.lastReport` and `device.reports`.
- [x] Add `npm run lint`, `npm test` and `npm run test:gpu`. CI runs them as the `cpu` and `gpu` jobs.
- [x] Test every operation on the CPU tier against a plain JavaScript reference, and on WebGPU and WebGL2 against the CPU tier (`tests/_cases.mjs`).
- [ ] Export remaining listed operations as `nn.*` functions from `src/index.ts`; tensor instance methods may remain as conveniences. Newly completed inverse-trigonometric functions, `step`, and `reshape` are exported.
- [ ] Keep shape inference and validation in TypeNN's graph layer; keep device execution in TypeShade kernel functions.
- [ ] Record rank, dtype, broadcasting, and axis limits when an operation's support is partial.

## Open items

Each row gives the reason and the next action. A next action marked "proposal" is not decided.

| Item                                           | Reason                                                                                                                                                                                                                                    | Next action                                                                                                                                       |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| A long chain splits into several dispatches    | Every operation output is a `Resident` array, and one dispatch binds at most 8 storage buffers (F5). `a.add(b).add(c).add(d).add(e).add(f).relu()` is two dispatches. A later `data()` or a gradient graph may read any output.           | Proposal: an inference mode in which only the roots of a `data()` keep arrays, and the other outputs are scratch arrays.                          |
| `reshape` costs a dispatch                     | `reshape` runs the `copy` kernel function. The flatten is one of the five dispatches of the MNIST forward.                                                                                                                                | Proposal: share the array of the input with the reshaped Tensor, and count the owners before `dispose()` frees it.                                |
| Reductions over all axes run in one invocation | `sum()`, `mean()` and the other reductions without an axis have one output element, so one invocation reads the whole input (Rule 8.22).                                                                                                  | Proposal: a two-stage reduction as two kernel functions with a scratch array between them. Measure it first.                                      |
| Softmax and the losses repeat work             | Each output element finds the maximum and the sum of its slice again. The work of a slice grows with the square of the axis size.                                                                                                         | Proposal: one kernel function that writes the maximum and the sum of each slice once, and a second that reads them. Measure it on hardware first. |
| `matmul` and `conv2d` have no tiling           | Each output element is one invocation with a serial loop. A call that reads an array through an index stands apart from the call that wrote it (F3).                                                                                      | Benchmark tiled variants on hardware before replacing them.                                                                                       |
| `maxPoolGrad` repeats work                     | Each input element recomputes the maximum of every output window that contains it. This is correct for the tie rule.                                                                                                                      | Measure on hardware, then decide.                                                                                                                 |
| SwiftShader trigonometry                       | On SwiftShader, `sin`, `cos`, `tan`, `acos` and the gradients through them differ from the f32 reference by more than 1e-5 relative. The worst absolute error of `sin` and `cos` is 1.9e-4. Seven cases are listed in `SWIFTSHADER_TRIG`. | Run `npm run test:gpu` on a hardware GPU. Remove the list if the cases agree.                                                                     |
| The optimizer update runs on the host          | SGD and Adam read the gradients back and write the parameters from the host.                                                                                                                                                              | Proposal: write the update as kernel functions in the plan of the backward pass.                                                                  |
| No hardware GPU run                            | The GPU tests ran on SwiftShader in Chromium 141 only. No time on SwiftShader says how fast a hardware GPU trains.                                                                                                                        | Run `node scripts/bench-step.mjs` and `npm run test:gpu` on a machine with a GPU.                                                                 |
| `publish.yml` is stale                         | Its check expects an empty namespace (`Object.keys(nn).length !== 0`), and the package exports names. It was so before this change.                                                                                                       | The owner decides what the pre-publish check must hold.                                                                                           |

### Gaps in TypeShade at commit 96dbaa9

TypeNN works around each gap. None is filed as an issue yet. Each gap was re-checked at 96dbaa9
(TypeNN's pin, after the bump from 5fb79c7). The check and its result are in each item. The
re-checks ran the compiler on probe sources (`compile()` of `typeshade`, Node 22) and read the
source diff between the two pins. They did not re-run the GPU measurements.

- Closed at 96dbaa9 (typeshade #607, "A failed batch destroys every buffer it made"). The
  batch now destroys its owned and staged buffers in a `finally` (`src/core/host-entry.ts`).
  Re-checked by reading the source. Not re-run on the recording device.
- Still open: a kernel function cannot share the name of a builtin that it calls (TS8099 and
  TS8031). This is as specified. Re-checked: a kernel `clamp` that calls `clamp` gives TS8031
  (recursive call) at 96dbaa9. TypeNN names the maps `<operation>Map`.
- Partly changed, still open: an array parameter named like a reserved WGSL word compiles for the
  CPU and WebGL2 tiers. WebGPU refuses it at pipeline creation: `'filter' is a reserved keyword`.
  At 96dbaa9 the compiler refuses a reserved name on an entry binding with TS8068. A kernel
  function's array parameter named `filter` still gets no diagnostic (a `hostFace` probe, with
  `fuse: true`). The WebGPU refusal of that parameter was not re-run.
- Still open, not re-run: one `Resident` passed as two parameters of one call throws a
  `TypeError`. TypeNN passes a copy. Reading the diff of `src/core/resident.ts` and `src/core/host-kernel.ts`
  found no change to this check (a reading, not a run).
- Still open, not re-measured: on the CPU tier, `toCpu` (`src/core/host-entry.ts`, called by `onCpu`
  in `src/core/host-kernel.ts`) copies each array argument to a JavaScript array on each call, and
  `fromCpu` copies it back. The byte copy of 96dbaa9 covers `readInto` and `pack`, not the array
  path of `toCpu`. Three calls over 2^24 + 1 elements reached 3.3 GB of resident memory at
  5fb79c7. That figure was not measured again.

## Kernel function modules

`src/kernels` holds 85 kernel functions in eight modules. The numerics cases of `tests/_cases.mjs`
hold every operation: on the CPU tier against a plain JavaScript reference, and on WebGPU and WebGL2
against the CPU tier. A case is a Tensor graph built with the public API. Each case carries its
tolerance: exact for the operations that are exact on a GPU, 1e-5 relative for the rest.

| Source file        | Kernel functions | Cases that hold them                                                                                   |
| ------------------ | ---------------: | ------------------------------------------------------------------------------------------------------ |
| `map.shade.ts`     |               37 | `map <name>`, `gradient of <name>`, `graph <name>`, `chain relu((a + b) * c)`                          |
| `binary.shade.ts`  |               25 | `binary <op>, <kind>`, `gradient of binary <op> with respect to <target>, <kind>`                      |
| `reduce.shade.ts`  |                8 | `reduction <op>, shape, axis`, `reduction <op> with ties`, `gradient of <op>, shape, axis`             |
| `matmul.shade.ts`  |                1 | `matmul <label>`, `gradient of matmul <label> with respect to <target>`, `matmul, add a bias, relu`    |
| `softmax.shade.ts` |                4 | `softmax, shape, axis`, `logSoftmax, shape, axis`, their gradients, `softmax of large logits`          |
| `loss.shade.ts`    |                3 | `softmax cross entropy, <reduction>`, `gradient of softmax cross entropy with respect to the <target>` |
| `conv.shade.ts`    |                3 | `conv2d <label>`, `gradient of conv2d with respect to the <target>`, `conv2dGradients`                 |
| `pool.shade.ts`    |                4 | `maxPool` and `avgPool`, their gradients in the graph, `maxPoolGrad` and `avgPoolGrad`                 |

The review of 65 source files on 2026-10-11 no longer applies, because every source file was
rewritten as a kernel function. The cases above check values, and not the source text. A review of
the source for bounds and index calculations is open.

- [ ] Review the 85 kernel functions for bounds, index calculations and naming, as the earlier review did for its files.
- [ ] Run the numerics cases and benchmark representative shapes on a hardware GPU.

## Device-resident datasets

Status: in progress (branch `claude/typenn-device-dataset`, based on `main` 11bdce1, TypeShade pin 96dbaa9).

Goal: upload a dataset to the device once, and build every training and evaluation batch on the
device. A training step uploads no sample bytes and does no per-sample CPU work.

- [ ] Public API under the `nn` namespace: a dataset object for images, tabular, text
      bag-of-words and labels (u8, u16, u32 and f32 regression targets).
- [ ] Storage keeps the source dtype on the device: u8 packed 4 per u32 word, u16 2 per word.
- [ ] Batches from a device-resident permutation, one N-element u32 upload per epoch.
- [ ] Evaluation iterator over the test split in order.
- [ ] CrossEntropyLoss and MSELoss accept a device label or target Tensor.
- [ ] Sharding for sources above the storage binding limit (128 MiB default).
- [ ] Residency report (`residency`) with a reason when a tier cannot hold a source.
- [ ] `dispose()` and a `bytes` property, with a clear error after dispose.
- [ ] Parity tests against the host batch helpers (CPU tier and fake GPU), and a GPU test.
- [ ] README section and JSDoc on every export.
