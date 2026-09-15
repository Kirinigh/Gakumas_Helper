import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { readCalibration, saveCalibration, MAX_CALIBRATION_VARIANTS } from "./calibration-cache.mjs";

function identity(letter = "a") {
  return {
    schema_version: 8, engine_commit: letter.repeat(40), node_version: "24.19.0",
    platform: "win32", architecture: "x64", cpu_model: "fixture CPU", available_parallelism: 32,
  };
}

async function cache(t) {
  const directory = await mkdtemp(join(tmpdir(), "arena-calibration-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return join(directory, "worker-calibration-v8.json");
}

test("A to B to A reuses legacy persisted calibration without rewriting", async (t) => {
  const path = await cache(t);
  const original = { ...identity(), selected_workers: 16, benchmarks: [{ workers: 16 }] };
  await writeFile(path, JSON.stringify(original));
  await saveCalibration(path, { ...identity("b"), selected_workers: 8 });
  const before = await readFile(path, "utf8");
  assert.deepEqual(await readCalibration(path, identity()), original);
  assert.equal((await readCalibration(path, identity("b"))).selected_workers, 8);
  assert.equal(await readFile(path, "utf8"), before);
});

test("all existing engine, runtime, and hardware identity gates remain required", async (t) => {
  const path = await cache(t);
  await saveCalibration(path, { ...identity(), selected_workers: 16 });
  await saveCalibration(path, { ...identity("b"), selected_workers: 8 });
  for (const [key, value] of Object.entries(identity())) {
    const changed = { ...identity(), [key]: typeof value === "number" ? value + 1 : `${value}-changed` };
    assert.equal(await readCalibration(path, changed), null, key);
  }
});

test("retain only four unique identities with no recursive histories", async (t) => {
  const path = await cache(t);
  for (const letter of "abcde") await saveCalibration(path, { ...identity(letter), selected_workers: 16 });
  await saveCalibration(path, { ...identity("e"), selected_workers: 8 });
  const record = JSON.parse(await readFile(path, "utf8"));
  assert.equal(record.calibration_variants.length, MAX_CALIBRATION_VARIANTS - 1);
  assert.deepEqual(record.calibration_variants.map((entry) => entry.engine_commit),
    [..."dcb"].map((letter) => letter.repeat(40)));
  assert.ok(record.calibration_variants.every((entry) => !("calibration_variants" in entry)));
  assert.equal(await readCalibration(path, identity()), null);
  assert.equal((await readCalibration(path, identity("e"))).selected_workers, 8);
});

test("malformed data and out-of-range worker counts never supply a cache hit", async (t) => {
  const path = await cache(t);
  for (const invalid of ["{broken", "null", JSON.stringify({ ...identity(), selected_workers: 33 }),
    JSON.stringify({ ...identity("b"), selected_workers: 8, calibration_variants: [
      { ...identity(), selected_workers: true },
    ] })]) {
    await writeFile(path, invalid);
    assert.equal(await readCalibration(path, identity()), null);
  }
  await saveCalibration(path, { ...identity(), selected_workers: 16 });
  const before = await readFile(path, "utf8");
  await assert.rejects(saveCalibration(path, { ...identity("c"), selected_workers: 0 }));
  assert.equal(await readFile(path, "utf8"), before);
  assert.equal((await readCalibration(path, identity())).selected_workers, 16);
});
