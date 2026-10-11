export {
  CrossEntropyLoss,
  crossEntropyLoss,
  type CrossEntropyOptions,
} from "./cross-entropy.js";
import { crossEntropyLoss } from "./cross-entropy.js";

export const loss = {
  crossEntropy: crossEntropyLoss,
};
