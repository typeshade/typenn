# TypeNN repository guidance

Read [README.md](./README.md) for the package layout, supported checks, measurements, and publishing
steps. Read [TODO.md](./TODO.md) for open library work.

This repository owns the TypeNN library, tests, and npm releases. The TypeNN website is maintained
in the [TypeShade web repository](https://github.com/typeshade/web/tree/main/apps/typenn). Keep
website source, website-only dependencies, and website deployment configuration there. Do not
recreate a website app in this library repository.

Keep the library CI and publishing workflows focused on the package. Use `npm ci`, `npm run lint`,
`npm test`, and the GPU check when the required Chromium/WebGPU environment is available. Website
checks belong to the web repository's TypeNN app.
