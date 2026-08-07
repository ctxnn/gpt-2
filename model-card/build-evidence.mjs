import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const siteDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryDirectory = resolve(siteDirectory, "..");

const metrics = JSON.parse(
  await readFile(join(repositoryDirectory, "results", "final_metrics.json"), "utf8"),
);
const csv = await readFile(join(repositoryDirectory, "results", "training_history.csv"), "utf8");

const history = { train: [], validation: [], hellaswag: [] };
for (const line of csv.split(/\r?\n/).slice(1)) {
  const columns = line.split(",");
  const event = columns[0];
  const step = Number(columns[1]);
  if (!Number.isFinite(step)) continue;

  if (event === "train") {
    const loss = Number(columns[2]);
    const learningRate = Number(columns[3]);
    const throughput = Number(columns[6]);
    if (Number.isFinite(loss) && Number.isFinite(learningRate) && Number.isFinite(throughput)) {
      history.train.push({ step, loss, learningRate, throughput });
    }
  } else if (event === "validation") {
    const loss = Number(columns[10]);
    if (Number.isFinite(loss)) history.validation.push({ step, loss });
  } else if (event === "hellaswag") {
    const accuracy = Number(columns[11]);
    if (Number.isFinite(accuracy)) history.hellaswag.push({ step, accuracy });
  }
}

const output = `/* Generated from ../results by build-evidence.mjs. */\nwindow.MODEL_CARD_METRICS=${JSON.stringify(metrics)};\nwindow.MODEL_CARD_HISTORY=${JSON.stringify(history)};\n`;
await writeFile(join(siteDirectory, "data", "evidence.js"), output, "utf8");

console.log(
  `Wrote evidence.js: ${history.train.length} train rows, ${history.validation.length} validation rows, ${history.hellaswag.length} HellaSwag rows.`,
);
