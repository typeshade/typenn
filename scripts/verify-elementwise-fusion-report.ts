export interface FusionTestResult {
  readonly id: string;
  readonly status: "passed" | "failed";
  readonly assertions: readonly string[];
  readonly error?: string;
}

export interface FusionVerificationReport {
  readonly commit: string;
  readonly worktreeFingerprint: string;
  readonly helperFingerprint: string;
  readonly gpu: "not-run";
  readonly tests: readonly FusionTestResult[];
}

export class FusionVerificationRecorder {
  private readonly results: FusionTestResult[] = [];

  async run(
    id: string,
    body: (
      check: (
        name: string,
        assertion: () => void | Promise<void>,
      ) => Promise<void>,
    ) => Promise<void>,
  ): Promise<void> {
    const assertions: string[] = [];
    try {
      await body(async (name, assertion) => {
        await assertion();
        assertions.push(name);
      });
      if (assertions.length === 0)
        throw new Error(`${id} completed without any recorded assertions.`);
      this.results.push({ id, status: "passed", assertions });
    } catch (error) {
      this.results.push({
        id,
        status: "failed",
        assertions,
        error:
          error instanceof Error
            ? (error.stack ?? error.message)
            : String(error),
      });
    }
  }

  resultFields(): {
    readonly tests: readonly FusionTestResult[];
    readonly passed: boolean;
  } {
    const ids = this.results.map((result) => result.id);
    if (new Set(ids).size !== ids.length)
      throw new Error("Fusion acceptance test IDs must be unique.");
    return {
      tests: this.results,
      passed: this.results.every((result) => result.status === "passed"),
    };
  }
}
