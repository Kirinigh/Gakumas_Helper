import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

export const MAX_CALIBRATION_VARIANTS = 4;
const IDENTITY_FIELDS = [
  "schema_version", "engine_commit", "node_version", "platform",
  "architecture", "cpu_model", "available_parallelism",
];

function validCalibration(value) {
  return value && typeof value === "object" &&
    Number.isSafeInteger(value.schema_version) && value.schema_version > 0 &&
    typeof value.engine_commit === "string" && /^[0-9a-f]{40}$/.test(value.engine_commit) &&
    ["node_version", "platform", "architecture", "cpu_model"].every(
      (key) => typeof value[key] === "string" && value[key].length > 0,
    ) &&
    Number.isSafeInteger(value.available_parallelism) && value.available_parallelism >= 1 &&
    Number.isSafeInteger(value.selected_workers) && value.selected_workers >= 1 &&
    value.selected_workers <= value.available_parallelism;
}

function sameIdentity(value, identity) {
  return IDENTITY_FIELDS.every((key) => value[key] === identity[key]);
}

async function readRecord(cachePath) {
  try {
    return JSON.parse(await readFile(cachePath, "utf8"));
  } catch {
    return null;
  }
}

function entries(record) {
  const history = Array.isArray(record?.calibration_variants)
    ? record.calibration_variants.slice(0, MAX_CALIBRATION_VARIANTS - 1) : [];
  return [record, ...history].filter(validCalibration);
}

export async function readCalibration(cachePath, identity) {
  return entries(await readRecord(cachePath)).find((value) => sameIdentity(value, identity)) ?? null;
}

export async function saveCalibration(cachePath, value) {
  if (!validCalibration(value)) throw new Error("invalid worker calibration");
  const selected = [value];
  for (const entry of entries(await readRecord(cachePath))) {
    if (!selected.some((item) => sameIdentity(item, entry))) selected.push(entry);
    if (selected.length === MAX_CALIBRATION_VARIANTS) break;
  }
  // Strip nested histories so the cache stays bounded after repeated saves.
  const flat = selected.map(({ calibration_variants: _history, ...entry }) => entry);
  const record = { ...flat[0], calibration_variants: flat.slice(1) };
  await mkdir(dirname(cachePath), { recursive: true });
  const temporary = `${cachePath}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporary, `${JSON.stringify(record, null, 2)}\n`, "utf8");
    await rename(temporary, cachePath);
  } finally {
    await rm(temporary, { force: true }).catch(() => {});
  }
}
