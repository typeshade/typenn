import { sameShape, sizeOf } from "../../tensor/shape.js";
import { Tensor } from "../../tensor/tensor.js";

/** Mean squared error: the mean of `(prediction - target)^2` over every element. */
export class MSELoss {
  forward(prediction: Tensor, target: Tensor | Float32Array): Tensor {
    prediction.assertUsable();
    let targetTensor: Tensor;
    if (target instanceof Tensor) {
      if (!sameShape(target.shape, prediction.shape))
        throw new RangeError("MSELoss target shape must match the prediction.");
      targetTensor = target;
    } else {
      if (target.length !== sizeOf(prediction.shape))
        throw new RangeError(
          "MSELoss target length must match the prediction size.",
        );
      targetTensor = prediction.device.tensor(target, {
        shape: prediction.shape,
      });
    }
    return prediction.sub(targetTensor).square().mean();
  }
}

export function mseLoss(): MSELoss {
  return new MSELoss();
}
