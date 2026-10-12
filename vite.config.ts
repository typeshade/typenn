import { defineConfig } from "vite";
import { typeshade } from "typeshade/vite";

export default defineConfig({
  plugins: [typeshade({ fuse: true })],
  build: {
    lib: {
      entry: "src/index.ts",
      formats: ["es"],
      fileName: "index",
    },
  },
});
