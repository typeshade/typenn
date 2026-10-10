# TypeNN implementation TODO

## Pure tensor operations

Public function names are intended to match TensorFlow.js so users can map familiar
operations directly: `tf.add(a, b)` becomes `nn.add(a, b)`. These are tensor
operations, not composite layers such as `Dense` or `Conv2D`. A TypeNN operation may
be implemented by one or more TypeShade kernels depending on broadcasting, shape,
and backend constraints.

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
| P2       | `tf.reshape(x, shape)`                                | `nn.reshape(x, shape)`                             | TODO                                                                                    |
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

| Priority | TensorFlow.js operation                                      | Planned TypeNN API                                           | Status                                                                                                       |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| P1       | `tf.conv2d(x, filter, ...)`                                  | `nn.conv2d(x, filter, ...)`                                  | Partial: NHWC/HWIO forward plus explicit input/filter gradients; no graph autodiff, bias/dilation/groups yet |
| P1       | `tf.depthwiseConv2d(x, filter, ...)`                         | `nn.depthwiseConv2d(x, filter, ...)`                         | TODO                                                                                                         |
| P1       | `tf.maxPool(x, ...)`                                         | `nn.maxPool(x, ...)`                                         | Partial: NHWC 2D pooling and explicit input gradient; first maximum wins ties; no graph autodiff             |
| P1       | `tf.avgPool(x, ...)`                                         | `nn.avgPool(x, ...)`                                         | Partial: NHWC 2D pooling and explicit input gradient; excludes padded cells from average; no graph autodiff  |
| P1       | `tf.softplus(x)`                                             | `nn.softplus(x)`                                             | Implemented: numerically stable formula                                                                      |
| P1       | `tf.relu6(x)`                                                | `nn.relu6(x)`                                                | Implemented                                                                                                  |
| P1       | `tf.prelu(x, alpha)`                                         | `nn.prelu(x, alpha)`                                         | Implemented with broadcasting                                                                                |
| P1       | `tf.addN(tensors)`                                           | `nn.addN(tensors)`                                           | Implemented as a graph of pairwise adds                                                                      |
| P1       | `tf.squaredDifference(a, b)`                                 | `nn.squaredDifference(a, b)`                                 | Implemented as `square(sub(a, b))`                                                                           |
| P1       | `tf.prod(x, axis)`                                           | `nn.prod(x, axis)`                                           | Implemented                                                                                                  |
| P1       | `tf.all(x, axis)`                                            | `nn.all(x, axis)`                                            | TODO                                                                                                         |
| P1       | `tf.any(x, axis)`                                            | `nn.any(x, axis)`                                            | TODO                                                                                                         |
| P2       | `tf.acos(x)`                                                 | `nn.acos(x)`                                                 | TODO                                                                                                         |
| P2       | `tf.acosh(x)`                                                | `nn.acosh(x)`                                                | TODO                                                                                                         |
| P2       | `tf.asin(x)`                                                 | `nn.asin(x)`                                                 | TODO                                                                                                         |
| P2       | `tf.asinh(x)`                                                | `nn.asinh(x)`                                                | TODO                                                                                                         |
| P2       | `tf.atan(x)`                                                 | `nn.atan(x)`                                                 | TODO                                                                                                         |
| P2       | `tf.atan2(a, b)`                                             | `nn.atan2(a, b)`                                             | TODO                                                                                                         |
| P2       | `tf.atanh(x)`                                                | `nn.atanh(x)`                                                | TODO                                                                                                         |
| P2       | `tf.ceil(x)`                                                 | `nn.ceil(x)`                                                 | Implemented                                                                                                  |
| P2       | `tf.cos(x)`                                                  | `nn.cos(x)`                                                  | Implemented                                                                                                  |
| P2       | `tf.cosh(x)`                                                 | `nn.cosh(x)`                                                 | Implemented                                                                                                  |
| P2       | `tf.erf(x)`                                                  | `nn.erf(x)`                                                  | TODO                                                                                                         |
| P2       | `tf.expm1(x)`                                                | `nn.expm1(x)`                                                | Implemented with a small-input series                                                                        |
| P2       | `tf.floor(x)`                                                | `nn.floor(x)`                                                | Implemented                                                                                                  |
| P2       | `tf.floorDiv(a, b)`                                          | `nn.floorDiv(a, b)`                                          | TODO                                                                                                         |
| P2       | `tf.log1p(x)`                                                | `nn.log1p(x)`                                                | Implemented with a small-input series                                                                        |
| P2       | `tf.mod(a, b)`                                               | `nn.mod(a, b)`                                               | TODO                                                                                                         |
| P2       | `tf.notEqual(a, b)`                                          | `nn.notEqual(a, b)`                                          | TODO                                                                                                         |
| P2       | `tf.greaterEqual(a, b)`                                      | `nn.greaterEqual(a, b)`                                      | TODO                                                                                                         |
| P2       | `tf.lessEqual(a, b)`                                         | `nn.lessEqual(a, b)`                                         | TODO                                                                                                         |
| P2       | `tf.reciprocal(x)`                                           | `nn.reciprocal(x)`                                           | Implemented                                                                                                  |
| P2       | `tf.round(x)`                                                | `nn.round(x)`                                                | Implemented                                                                                                  |
| P2       | `tf.rsqrt(x)`                                                | `nn.rsqrt(x)`                                                | Implemented                                                                                                  |
| P2       | `tf.sign(x)`                                                 | `nn.sign(x)`                                                 | Implemented                                                                                                  |
| P2       | `tf.sin(x)`                                                  | `nn.sin(x)`                                                  | Implemented                                                                                                  |
| P2       | `tf.sinh(x)`                                                 | `nn.sinh(x)`                                                 | Implemented                                                                                                  |
| P2       | `tf.step(x)`                                                 | `nn.step(x)`                                                 | TODO                                                                                                         |
| P2       | `tf.tan(x)`                                                  | `nn.tan(x)`                                                  | Implemented with sin/cos                                                                                     |
| P2       | `tf.expandDims(x, axis)`                                     | `nn.expandDims(x, axis)`                                     | TODO                                                                                                         |
| P2       | `tf.reverse(x, axis)`                                        | `nn.reverse(x, axis)`                                        | TODO                                                                                                         |
| P2       | `tf.slice(x, begin, size)`                                   | `nn.slice(x, begin, size)`                                   | TODO                                                                                                         |
| P2       | `tf.stridedSlice(x, ...)`                                    | `nn.stridedSlice(x, ...)`                                    | TODO                                                                                                         |
| P2       | `tf.tile(x, reps)`                                           | `nn.tile(x, reps)`                                           | TODO                                                                                                         |
| P2       | `tf.gatherND(x, indices)`                                    | `nn.gatherND(x, indices)`                                    | TODO                                                                                                         |
| P2       | `tf.pack(tensors, axis)`                                     | `nn.pack(tensors, axis)`                                     | TODO                                                                                                         |
| P2       | `tf.unstack(x, axis)`                                        | `nn.unstack(x, axis)`                                        | TODO                                                                                                         |
| P2       | `tf.diag(x)`                                                 | `nn.diag(x)`                                                 | TODO                                                                                                         |
| P2       | `tf.mirrorPad(x, paddings, mode)`                            | `nn.mirrorPad(x, paddings, mode)`                            | TODO                                                                                                         |
| P2       | `tf.spaceToBatchND(x, blockShape, paddings)`                 | `nn.spaceToBatchND(x, blockShape, paddings)`                 | TODO                                                                                                         |
| P2       | `tf.batchToSpaceND(x, blockShape, crops)`                    | `nn.batchToSpaceND(x, blockShape, crops)`                    | TODO                                                                                                         |
| P2       | `tf.cast(x, dtype)`                                          | `nn.cast(x, dtype)`                                          | TODO                                                                                                         |
| P2       | `tf.fill(shape, value)`                                      | `nn.fill(shape, value)`                                      | TODO                                                                                                         |
| P2       | `tf.onesLike(x)`                                             | `nn.onesLike(x)`                                             | TODO                                                                                                         |
| P2       | `tf.zerosLike(x)`                                            | `nn.zerosLike(x)`                                            | TODO                                                                                                         |
| P2       | `tf.oneHot(indices, depth)`                                  | `nn.oneHot(indices, depth)`                                  | TODO                                                                                                         |
| P2       | `tf.range(start, stop, step)`                                | `nn.range(start, stop, step)`                                | TODO                                                                                                         |
| P2       | `tf.linspace(start, stop, num)`                              | `nn.linspace(start, stop, num)`                              | TODO                                                                                                         |
| P2       | `tf.scatterND(indices, updates, shape)`                      | `nn.scatterND(indices, updates, shape)`                      | TODO                                                                                                         |
| P2       | `tf.tensorScatterUpdate(x, indices, updates)`                | `nn.tensorScatterUpdate(x, indices, updates)`                | TODO                                                                                                         |
| P2       | `tf.topk(x, k)`                                              | `nn.topk(x, k)`                                              | TODO                                                                                                         |
| P2       | `tf.searchSorted(sorted, values)`                            | `nn.searchSorted(sorted, values)`                            | TODO                                                                                                         |
| P2       | `tf.sparseSegmentSum(data, indices, segmentIds)`             | `nn.sparseSegmentSum(data, indices, segmentIds)`             | TODO                                                                                                         |
| P2       | `tf.sparseSegmentMean(data, indices, segmentIds)`            | `nn.sparseSegmentMean(data, indices, segmentIds)`            | TODO                                                                                                         |
| P3       | `tf.conv3d(x, filter, ...)`                                  | `nn.conv3d(x, filter, ...)`                                  | TODO                                                                                                         |
| P3       | `tf.avgPool3d(x, ...)`                                       | `nn.avgPool3d(x, ...)`                                       | TODO                                                                                                         |
| P3       | `tf.maxPool3d(x, ...)`                                       | `nn.maxPool3d(x, ...)`                                       | TODO                                                                                                         |
| P3       | `tf.cropAndResize(image, boxes, boxIndex, cropSize)`         | `nn.cropAndResize(image, boxes, boxIndex, cropSize)`         | TODO                                                                                                         |
| P3       | `tf.image.resizeBilinear(x, size)`                           | `nn.image.resizeBilinear(x, size)`                           | TODO                                                                                                         |
| P3       | `tf.image.resizeNearestNeighbor(x, size)`                    | `nn.image.resizeNearestNeighbor(x, size)`                    | TODO                                                                                                         |
| P3       | `tf.image.flipLeftRight(x)`                                  | `nn.image.flipLeftRight(x)`                                  | TODO                                                                                                         |
| P3       | `tf.image.rotateWithOffset(x, radians, fillValue)`           | `nn.image.rotateWithOffset(x, radians, fillValue)`           | TODO                                                                                                         |
| P3       | `tf.complex(real, imag)`                                     | `nn.complex(real, imag)`                                     | TODO                                                                                                         |
| P3       | `tf.spectral.fft(x)`                                         | `nn.fft(x)`                                                  | TODO                                                                                                         |
| P3       | `tf.spectral.ifft(x)`                                        | `nn.ifft(x)`                                                 | TODO                                                                                                         |
| P3       | `tf.multinomial(logits, numSamples)`                         | `nn.multinomial(logits, numSamples)`                         | TODO                                                                                                         |
| P3       | `tf.image.nonMaxSuppression(boxes, scores, maxOutputSize)`   | `nn.image.nonMaxSuppression(boxes, scores, maxOutputSize)`   | TODO                                                                                                         |
| P3       | `tf.string.ngrams(...)`                                      | `nn.string.ngrams(...)`                                      | TODO                                                                                                         |
| P2       | `tf.argMin(x, axis)`                                         | `nn.argMin(x, axis)`                                         | Implemented; indices are represented as float32 tensors                                                      |
| P2       | `tf.cumsum(x, axis)`                                         | `nn.cumsum(x, axis)`                                         | TODO                                                                                                         |
| P2       | `tf.cumprod(x, axis)`                                        | `nn.cumprod(x, axis)`                                        | TODO                                                                                                         |
| P2       | `tf.logicalAnd(a, b)`                                        | `nn.logicalAnd(a, b)`                                        | TODO                                                                                                         |
| P2       | `tf.logicalOr(a, b)`                                         | `nn.logicalOr(a, b)`                                         | TODO                                                                                                         |
| P2       | `tf.logicalNot(x)`                                           | `nn.logicalNot(x)`                                           | TODO                                                                                                         |
| P2       | `tf.isFinite(x)`                                             | `nn.isFinite(x)`                                             | TODO                                                                                                         |
| P2       | `tf.isInf(x)`                                                | `nn.isInf(x)`                                                | TODO                                                                                                         |
| P2       | `tf.isNaN(x)`                                                | `nn.isNaN(x)`                                                | TODO                                                                                                         |
| P2       | `tf.depthToSpace(x, blockSize)`                              | `nn.depthToSpace(x, blockSize)`                              | TODO                                                                                                         |
| P2       | `tf.dilation2d(x, filter, ...)`                              | `nn.dilation2d(x, filter, ...)`                              | TODO                                                                                                         |
| P2       | `tf.einsum(equation, tensors)`                               | `nn.einsum(equation, tensors)`                               | TODO                                                                                                         |
| P2       | `tf.unsortedSegmentSum(x, segmentIds, numSegments)`          | `nn.unsortedSegmentSum(x, segmentIds, numSegments)`          | TODO                                                                                                         |
| P2       | `tf.sparseToDense(sparseIndices, outputShape, sparseValues)` | `nn.sparseToDense(sparseIndices, outputShape, sparseValues)` | TODO                                                                                                         |
| P2       | `tf.bincount(x, weights, size)`                              | `nn.bincount(x, weights, size)`                              | TODO                                                                                                         |
| P2       | `tf.denseBincount(x, weights, size)`                         | `nn.denseBincount(x, weights, size)`                         | TODO                                                                                                         |
| P2       | `tf.real(x)`                                                 | `nn.real(x)`                                                 | TODO                                                                                                         |
| P2       | `tf.imag(x)`                                                 | `nn.imag(x)`                                                 | TODO                                                                                                         |
| P2       | `tf.abs(x)` for complex tensors                              | `nn.abs(x)` for complex tensors                              | Dtype expansion TODO; f32 implemented above                                                                  |
| P3       | `tf.maxPoolWithArgmax(x, ...)`                               | `nn.maxPoolWithArgmax(x, ...)`                               | TODO                                                                                                         |
| P3       | `tf.browser.fromPixels(pixels)`                              | `nn.image.fromPixels(pixels)`                                | TODO                                                                                                         |

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

- [x] Implement explicit TypeShade forward, stable objective, backward, reduction, and SGD update kernels for one `Linear` classifier.
- [x] Add `createTrainer()` with batched integer-label data and `onEpoch` metrics across TypeShade CPU/WebGPU/WebGL2 execution.
- [ ] Generalize training to `Sequential` models with multiple trainable layers and activation backpropagation.

- [x] Re-export TypeShade `grad` and `gradCheck` as `shadeGrad`/`shadeGradCheck` for pure Shade functions and custom activations; this does not differentiate TypeNN Tensor graphs.
- [ ] Expand TypeNN Tensor-graph gradient propagation beyond the initial supported reverse-mode subset; TypeShade `grad` operates on pure IR functions, not TypeNN Tensor nodes or storage bindings.
- [x] Add `nn.gradients(output, wrt, upstream?)` reverse traversal for ReLU, broadcast binary ops, unary arithmetic/trigonometric ops, rank-2 MatMul with transpose options, Conv2D, 2D pooling, sum/mean, and softmax cross entropy.
- [x] Add softmax/log-softmax gradient kernels to Tensor-graph autograd.
- [ ] Add `prod`/`max`/`min` reduction gradients and special activation gradient rules.
- [x] Connect `nn.softmaxCrossEntropyWithLogits()` and `sum`/`mean` reductions to Tensor-graph autograd.
- [ ] Generalize optimizers to graph parameters; add gradient accumulation and Adam.
- [ ] Use TypeShade `gradCheck` against its f64 CPU oracle for generated Shade derivatives.

## API and kernel implementation notes

- [x] Keep one TypeShade compute entrypoint per `src/kernels/<operation>.shade.ts` file, with filenames following the operation name (for example, `add.shade.ts` and `relu.shade.ts`). The current split has 65 files and the runtime registry imports each file separately.
- [x] Split TypeShade CPU graph evaluation into node-kind dispatch and elementwise, reduction, normalization, loss, spatial, and linear-algebra executor modules.
- [x] Split TypeNN graph-node types into individual files under `src/tensor/nodes/` and keep `src/tensor/node.ts` as a compatibility re-export.
- [x] Split Tensor graph builders into elementwise, linear-algebra, spatial, normalization, loss, and reduction modules.
- [x] Split GPU graph execution, resident buffer ownership, backend selection, pipeline preparation, and linear-training dispatch out of `src/runtime/device.ts`.
- [x] Add `npm run lint` to check formatting across TypeScript, TypeShade, docs, and configuration files.
- [ ] Export remaining listed operations as `nn.*` functions from `src/index.ts`; tensor instance methods may remain as conveniences.
- [ ] Keep shape inference and validation in TypeNN's graph layer; keep device execution in TypeShade-backed kernels.
- [ ] Support CPU and WebGPU paths for each operation before marking it implemented. Record rank, dtype, broadcasting, and axis limits when support is partial.
- [ ] Add each operation to the graph node union, runtime dispatch, CPU reference path, and TypeShade program manifest as applicable.

## TypeShade kernel source review

Re-reviewed all 65 source files individually on 2026-10-11, including `add.shade.ts`.
The checklist records the single exported compute entrypoint in each file. Review scope
was bounds checks, shape/index calculations, operation semantics, and naming. The audit found and fixed
broadcast-gradient branching, temporary generated-style names, max-pool handling for
negative infinity, cross-entropy gradients for labels whose row sums are not one, and
manual `tan`/hyperbolic/reciprocal-square-root formulas where TypeShade built-ins exist.
`npm run build` compiles the TypeShade sources and generated CPU views; it does not
replace numerical runtime tests.

| Source file                                  | Entry point                           | Source review |
| -------------------------------------------- | ------------------------------------- | ------------- |
| `abs.shade.ts`                               | `absKernel`                           | Reviewed      |
| `add.shade.ts`                               | `addKernel`                           | Reviewed      |
| `arg-max.shade.ts`                           | `argMaxKernel`                        | Reviewed      |
| `arg-min.shade.ts`                           | `argMinKernel`                        | Reviewed      |
| `avg-pool-grad.shade.ts`                     | `avgPoolGradKernel`                   | Reviewed      |
| `avg-pool.shade.ts`                          | `avgPoolKernel`                       | Reviewed      |
| `binary-grad.shade.ts`                       | `binaryGradKernel`                    | Reviewed      |
| `ceil.shade.ts`                              | `ceilKernel`                          | Reviewed      |
| `clip-by-value.shade.ts`                     | `clipByValueKernel`                   | Reviewed      |
| `conv2d-filter-grad.shade.ts`                | `conv2dFilterGradKernel`              | Reviewed      |
| `conv2d-input-grad.shade.ts`                 | `conv2dInputGradKernel`               | Reviewed      |
| `conv2d.shade.ts`                            | `conv2dKernel`                        | Reviewed      |
| `cos.shade.ts`                               | `cosKernel`                           | Reviewed      |
| `cosh.shade.ts`                              | `coshKernel`                          | Reviewed      |
| `cross-entropy-labels-grad.shade.ts`         | `crossEntropyLabelsGradKernel`        | Reviewed      |
| `cross-entropy-logits-grad.shade.ts`         | `crossEntropyLogitsGradKernel`        | Reviewed      |
| `div.shade.ts`                               | `divKernel`                           | Reviewed      |
| `elu.shade.ts`                               | `eluKernel`                           | Reviewed      |
| `exp.shade.ts`                               | `expKernel`                           | Reviewed      |
| `expm1.shade.ts`                             | `expm1Kernel`                         | Reviewed      |
| `floor.shade.ts`                             | `floorKernel`                         | Reviewed      |
| `gelu.shade.ts`                              | `geluKernel`                          | Reviewed      |
| `leaky-relu.shade.ts`                        | `leakyReluKernel`                     | Reviewed      |
| `linear-backward.shade.ts`                   | `linearBackwardKernel`                | Reviewed      |
| `linear-forward.shade.ts`                    | `linearForwardKernel`                 | Reviewed      |
| `linear-objective.shade.ts`                  | `linearObjectiveKernel`               | Reviewed      |
| `linear-reduce.shade.ts`                     | `linearReduceKernel`                  | Reviewed      |
| `linear-update.shade.ts`                     | `linearUpdateKernel`                  | Reviewed      |
| `log-softmax.shade.ts`                       | `logSoftmaxKernel`                    | Reviewed      |
| `log.shade.ts`                               | `logKernel`                           | Reviewed      |
| `log1p.shade.ts`                             | `log1pKernel`                         | Reviewed      |
| `matmul.shade.ts`                            | `matmulKernel`                        | Reviewed      |
| `max-pool-grad.shade.ts`                     | `maxPoolGradKernel`                   | Reviewed      |
| `max-pool.shade.ts`                          | `maxPoolKernel`                       | Reviewed      |
| `max.shade.ts`                               | `maxKernel`                           | Reviewed      |
| `maximum.shade.ts`                           | `maximumKernel`                       | Reviewed      |
| `mean.shade.ts`                              | `meanKernel`                          | Reviewed      |
| `min.shade.ts`                               | `minKernel`                           | Reviewed      |
| `minimum.shade.ts`                           | `minimumKernel`                       | Reviewed      |
| `mul.shade.ts`                               | `mulKernel`                           | Reviewed      |
| `neg.shade.ts`                               | `negKernel`                           | Reviewed      |
| `pow.shade.ts`                               | `powKernel`                           | Reviewed      |
| `prelu.shade.ts`                             | `preluKernel`                         | Reviewed      |
| `prod.shade.ts`                              | `prodKernel`                          | Reviewed      |
| `reciprocal.shade.ts`                        | `reciprocalKernel`                    | Reviewed      |
| `reduce-grad.shade.ts`                       | `reduceGradKernel`                    | Reviewed      |
| `relu-grad.shade.ts`                         | `reluGradKernel`                      | Reviewed      |
| `relu.shade.ts`                              | `reluKernel`                          | Reviewed      |
| `relu6.shade.ts`                             | `relu6Kernel`                         | Reviewed      |
| `round.shade.ts`                             | `roundKernel`                         | Reviewed      |
| `rsqrt.shade.ts`                             | `rsqrtKernel`                         | Reviewed      |
| `sigmoid.shade.ts`                           | `sigmoidKernel`                       | Reviewed      |
| `sign.shade.ts`                              | `signKernel`                          | Reviewed      |
| `sin.shade.ts`                               | `sinKernel`                           | Reviewed      |
| `sinh.shade.ts`                              | `sinhKernel`                          | Reviewed      |
| `softmax-cross-entropy-with-logits.shade.ts` | `softmaxCrossEntropyWithLogitsKernel` | Reviewed      |
| `softmax-grad.shade.ts`                      | `softmaxGradKernel`                   | Reviewed      |
| `softmax.shade.ts`                           | `softmaxKernel`                       | Reviewed      |
| `softplus.shade.ts`                          | `softplusKernel`                      | Reviewed      |
| `sqrt.shade.ts`                              | `sqrtKernel`                          | Reviewed      |
| `square.shade.ts`                            | `squareKernel`                        | Reviewed      |
| `sub.shade.ts`                               | `subKernel`                           | Reviewed      |
| `sum.shade.ts`                               | `sumKernel`                           | Reviewed      |
| `tan.shade.ts`                               | `tanKernel`                           | Reviewed      |
| `tanh.shade.ts`                              | `tanhKernel`                          | Reviewed      |

### Findings and limits

- The source review did not identify another clear bounds or index-calculation defect.
  This is a source-level result; it does not establish CPU/WebGPU numerical parity.
- Reduction, matmul, and convolution kernels use per-output accumulation. Tiled or
  multi-pass variants need benchmark evidence before replacing these implementations.
- `maxPoolGrad` recomputes each intersecting output window's maximum for every input
  element. This is correct for the current tie rule but can repeat substantial work.
- Softmax and cross-entropy kernels reduce each row or axis slice within one invocation.
  Long axes may need a parallel reduction path.
- [ ] Add numerical CPU/WebGPU parity coverage and benchmark representative shapes.
