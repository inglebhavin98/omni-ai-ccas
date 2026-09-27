// Dev-only browser harness for the workbench console (ADR-0022).
// Runs against a real uvicorn server with the LLM unconfigured, so what it asserts is
// the console's own behaviour -- rendering and wiring -- never a model's.
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  use: {
    baseURL: process.env.BROWSER_BASE_URL ?? "http://127.0.0.1:8765",
    trace: "off",
  },
  // The server is expected to be already running (make browser starts both);
  // Playwright only drives the browser.
  webServer: undefined,
  reporter: [["list"]],
});
