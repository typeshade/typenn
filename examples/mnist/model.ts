import * as nn from "typenn";

export class SimpleMLP extends nn.Module {
  private readonly flatten: nn.Flatten;
  private readonly fc1: nn.Linear;
  private readonly relu: nn.ReLU;
  private readonly dropout: nn.Dropout;
  private readonly fc2: nn.Linear;

  constructor() {
    super();
    this.flatten = this.register("flatten", new nn.Flatten());
    this.fc1 = this.register(
      "fc1",
      new nn.Linear(28 * 28, 128, {
        seed: 101,
        weightScale: 1 / Math.sqrt(28 * 28),
      }),
    );
    this.relu = this.register("relu", new nn.ReLU());
    this.dropout = this.register("dropout", new nn.Dropout(0.2));
    this.fc2 = this.register(
      "fc2",
      new nn.Linear(128, 10, { seed: 202, weightScale: 1 / Math.sqrt(128) }),
    );
  }

  forward(input: nn.Tensor): nn.Tensor {
    const flat = this.flatten.forward(input);
    const hidden = this.fc1.forward(flat);
    const activated = this.relu.forward(hidden);
    const regularized = this.dropout.forward(activated);
    return this.fc2.forward(regularized);
  }
}
