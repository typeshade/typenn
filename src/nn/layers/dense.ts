import { Linear, type LinearOptions } from "./linear.js";

/** Keras-style name for a fully connected linear layer. */
export class Dense extends Linear {
  constructor(
    inFeatures: number,
    outFeatures: number,
    options: LinearOptions = {},
  ) {
    super(inFeatures, outFeatures, options);
  }
}
