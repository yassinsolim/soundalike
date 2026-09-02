import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const stub = (name: string) =>
  fileURLToPath(new URL(`./test/stubs/${name}.ts`, import.meta.url));

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    environment: "node",
  },
  resolve: {
    alias: {
      "@react-native-async-storage/async-storage": stub("async-storage"),
      "expo-crypto": stub("expo-crypto"),
    },
  },
});
