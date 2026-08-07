import assert from "node:assert/strict";
import { readFile, access } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const siteDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryDirectory = resolve(siteDirectory, "..");

const [html, css, app, evidenceSource, sourceMetricsText, copiedMetricsText, sourceHistory, copiedHistory] =
  await Promise.all([
    readFile(join(siteDirectory, "index.html"), "utf8"),
    readFile(join(siteDirectory, "styles.css"), "utf8"),
    readFile(join(siteDirectory, "app.js"), "utf8"),
    readFile(join(siteDirectory, "data", "evidence.js"), "utf8"),
    readFile(join(repositoryDirectory, "results", "final_metrics.json"), "utf8"),
    readFile(join(siteDirectory, "data", "final_metrics.json"), "utf8"),
    readFile(join(repositoryDirectory, "results", "training_history.csv")),
    readFile(join(siteDirectory, "data", "training_history.csv")),
  ]);

assert.equal(copiedMetricsText, sourceMetricsText, "copied metrics must match the repository source");
assert.equal(Buffer.compare(copiedHistory, sourceHistory), 0, "copied history must match the repository source");

for (const filename of ["final_checkpoint.json", "generated_samples.md"]) {
  const [source, copy] = await Promise.all([
    readFile(join(repositoryDirectory, "results", filename)),
    readFile(join(siteDirectory, "data", filename)),
  ]);
  assert.equal(Buffer.compare(copy, source), 0, `${filename} must match the repository source`);
}

const context = { window: {} };
vm.runInNewContext(evidenceSource, context);
const embeddedMetrics = JSON.parse(JSON.stringify(context.window.MODEL_CARD_METRICS));
const embeddedHistory = JSON.parse(JSON.stringify(context.window.MODEL_CARD_HISTORY));
assert.deepEqual(embeddedMetrics, JSON.parse(sourceMetricsText), "embedded metrics must match source JSON");
assert.equal(embeddedHistory.train.length, 594);
assert.equal(embeddedHistory.validation.length, 25);
assert.equal(embeddedHistory.hellaswag.length, 2);
assert.equal(embeddedHistory.train[0].step, 13140);
assert.equal(embeddedHistory.validation.at(-1).step, 19073);
assert.equal(embeddedHistory.hellaswag.at(-1).accuracy, embeddedMetrics.hellaswag_results.accuracy);

const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]);
assert.equal(new Set(ids).size, ids.length, "HTML IDs must be unique");
for (const [, fragment] of html.matchAll(/href="#([^"]+)"/g)) {
  assert(ids.includes(fragment), `missing fragment target: #${fragment}`);
}

const localReferences = [
  ...html.matchAll(/(?:href|src)="(\.\/[^"#?]+)(?:[#?][^"]*)?"/g),
].map((match) => match[1]);
for (const reference of localReferences) await access(resolve(siteDirectory, reference));

for (const section of [
  "overview",
  "architecture",
  "data",
  "training",
  "results",
  "reliability",
  "samples",
  "responsible-use",
  "resources",
]) {
  assert(html.includes(`id="${section}"`), `required section is missing: ${section}`);
}

assert(css.includes("@media (max-width: 800px)"), "tablet/mobile breakpoint is required");
assert(css.includes("@media (max-width: 520px)"), "small-mobile breakpoint is required");
assert(css.includes("prefers-reduced-motion"), "reduced-motion support is required");
assert(app.includes("setupSamples()"), "sample interaction must initialize");
assert(app.includes("renderCharts(history)"), "evidence charts must initialize");

const socialCard = await readFile(join(siteDirectory, "social-card.png"));
assert.equal(socialCard.subarray(1, 4).toString("ascii"), "PNG");
assert.equal(socialCard.readUInt32BE(16), 1200);
assert.equal(socialCard.readUInt32BE(20), 630);

const repositoryReadme = await readFile(join(repositoryDirectory, "README.md"), "utf8");
assert(repositoryReadme.includes("https://ctxnn.github.io/gpt-2/"));
assert(html.includes("9,953,989,344 unique prepared tokens"), "exact prepared-token count is required");
assert(html.includes("45,755,680-token loader wrap"), "loader-wrap disclosure is required");
assert(/resource\s+estimate—not an invoice/.test(html), "cost estimate caveat is required");
assert(!html.includes("100M uint16 tokens each"), "partial final shard must not be described as full");

console.log(
  `Model-card checks passed: ${ids.length} IDs, ${localReferences.length} local assets, ${embeddedHistory.train.length + embeddedHistory.validation.length + embeddedHistory.hellaswag.length} evidence rows.`,
);
