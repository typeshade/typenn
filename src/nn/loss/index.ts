export {
  CrossEntropyLoss,
  crossEntropyLoss,
  type CrossEntropyOptions,
} from "./cross-entropy.js";
export { MSELoss, mseLoss } from "./mse.js";
import { crossEntropyLoss } from "./cross-entropy.js";
import { mseLoss } from "./mse.js";

export const loss = {
  crossEntropy: crossEntropyLoss,
  mse: mseLoss,
};
