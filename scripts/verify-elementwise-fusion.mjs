import { resolve } from "node:path";
import { createServer } from "vite";

const server = await createServer({
  configFile: resolve(".typenn-build/vite.config.mjs"),
  appType: "custom",
  server: { middlewareMode: true },
});

try {
  const verification = await server.ssrLoadModule(
    "/scripts/verify-elementwise-fusion.entry.ts",
  );
  await verification.runElementwiseFusionVerification();
} finally {
  await server.close();
}
