// The browser test 9.21 asked for: the two failure modes the contract tests cannot
// see. A "dead button looks like a slow one" -- so this asserts rendering (a spoken
// turn appears in the transcript) and wiring (the send button actually fires, and a
// terminal session disables the input). No LLM is configured on this server, so the
// session escalates honestly; that is the point -- the console must *show* it.
import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("h1")).toHaveText("omni-ai-ccas workbench");
});

test("the console boots and reports provider state", async ({ page }) => {
  // The make target strips the LLM keys, so the honest state is "routing disabled" --
  // and the chip must say it rather than claim a provider it does not have.
  await expect(page.locator("#ready")).not.toHaveText("checking…", { timeout: 10_000 });
  await expect(page.locator("#ready")).not.toHaveText("api unreachable");
  // The domain dropdown must be populated from /v1/domains -- wiring, not strings.
  await expect(page.locator("#domain option")).not.toHaveCount(0);
});

test("a redaction preview renders what the model would see", async ({ page }) => {
  await page.locator("#redactGo").click();
  await expect(page.locator("#redactOut pre")).toBeVisible();
  // The default preview input carries a PAN; the render must show the placeholder,
  // not the number -- Rule 2, watched through a real browser.
  await expect(page.locator("#redactOut pre")).not.toContainText("4111");
  await expect(page.locator("#redactOut")).toContainText("PAYMENT_CARD");
});

test("a spoken turn renders in the transcript and an escalation disables input", async ({
  page,
}) => {
  await page.locator("#new").click();
  await expect(page.locator("#sid")).not.toBeEmpty({ timeout: 10_000 });

  const say = page.locator("#say");
  await expect(say).toBeEnabled();
  await say.fill("no idea really");
  await page.locator("#send").click();

  // The caller's turn must actually appear -- rendering.
  const transcript = page.locator("#transcript");
  await expect(transcript).toContainText("no idea really", { timeout: 15_000 });
  // The no-LLM router cannot classify; the console must show the escalation honestly.
  await expect(page.locator("#routing")).toContainText("ESCALATED", { timeout: 15_000 });
  // A terminal session must disable the input -- event wiring and state, both live.
  await expect(say).toBeDisabled();
  await expect(page.locator("#send")).toBeDisabled();
  await expect(page.locator("#hint")).toContainText("terminal");
});
