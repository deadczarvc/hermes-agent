// Scoped Stryker config: one pure module + its focused Vitest file.
export default {
  testRunner: "vitest",
  mutator: "typescript",
  mutate: ["src/lib/chat-title.ts"],
  coverageAnalysis: "off",
  timeoutMS: 20000,
  concurrency: 2,
  reporters: ["clear-text", "progress"],
  vitest: { configFile: "vitest.config.ts" },
};
