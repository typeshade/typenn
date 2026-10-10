# TypeNN

`typenn` is a placeholder package. Version `0.0.0` provides an empty ES module
and TypeScript declarations. It exports no APIs and has no runtime dependencies.

Install it with:

```sh
npm install typenn
```

Import the package namespace with:

```ts
import * as nn from 'typenn';
```

The `nn` namespace is currently empty. Neural network functionality is not yet
implemented.

## Publishing

Publishing a GitHub release runs the npm publishing workflow. A manual workflow
run defaults to a dry run; clear `dry_run` to publish. Actual publication requires
the `NPM_ACCESS_TOKEN` repository or organization secret with npm publish access.
The workflow validates the packed package before publishing it publicly with
provenance.

## License

Apache-2.0. See [LICENSE](LICENSE).
