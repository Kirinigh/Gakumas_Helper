import { availableParallelism, cpus } from "node:os";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve as resolvePath } from "node:path";
import { performance } from "node:perf_hooks";
import {
  Worker,
  isMainThread,
  parentPort,
  workerData,
} from "node:worker_threads";

import {
  IdolConfig,
  IdolStageConfig,
  StageConfig,
  StageEngine,
  StagePlayer,
  STRATEGIES,
  resetRand,
} from "gakumas-engine";
import { Stages } from "gakumas-data";
import {
  classifyBestOfThree,
  compareStageScores,
  countOutcomes,
  sumMemberScores,
} from "./scoring.mjs";

const SCHEMA_VERSION = "2.0";
const UPSTREAM_COMMIT = "3e9e8ebdc929dedd32cdd8d8911e5d8a905f76b6";
const SCORE_AGGREGATION = "raw_sum_plus_stage_wide_first_place_20_percent";
const MATCH_RULE = "best_of_three_strict_wins";
const OWN_SCORE_METHOD = "independent_empirical_own_three_stage_scores";
const OWN_SCORE_AGGREGATION = "raw_team_sum_without_cross_side_first_place_bonus";
const CALIBRATION_SCHEMA_VERSION = 8;
const CALIBRATION_RUNS = 4000;
const CALIBRATION_REPETITIONS = 3;
const CALIBRATION_FINALISTS = 4;
const CALIBRATION_WARMUP_RUNS = 128;
const WORKER_COST_WEIGHT = 0.0025;
const SELECTION_NOISE_TOLERANCE = 0.01;

function assertRequest(request) {
  if (!request || request.schema_version !== SCHEMA_VERSION) {
    throw new Error("protocol version mismatch");
  }
  if (!["arena_match_win_rate", "arena_own_score"].includes(request.operation)) {
    throw new Error("unsupported operation");
  }
  if (
    !Number.isSafeInteger(request.simulations) ||
    request.simulations < 1000
  ) {
    throw new Error("simulations must be a safe integer of at least 1000");
  }
  if (!Number.isSafeInteger(request.seed) || request.seed < 0 || request.seed > 0xffffffff) {
    throw new Error("seed must be an unsigned 32-bit integer");
  }
  if (!Number.isSafeInteger(request.season) || request.season < 1) {
    throw new Error("season must be a positive integer");
  }
  if (
    !Array.isArray(request.stageIds) ||
    request.stageIds.length !== 3 ||
    new Set(request.stageIds).size !== 3 ||
    request.stageIds.some((stageId) => !Number.isSafeInteger(stageId) || stageId < 1)
  ) {
    throw new Error("stageIds must contain three unique positive integers");
  }
  const expectedAggregation =
    request.operation === "arena_own_score" ? OWN_SCORE_AGGREGATION : SCORE_AGGREGATION;
  if (request.score_aggregation !== expectedAggregation) {
    throw new Error("unsupported score aggregation");
  }
  if (request.operation === "arena_match_win_rate" && request.match_rule !== MATCH_RULE) {
    throw new Error("unsupported match rule");
  }
  if (!request.own_team) {
    throw new Error("the player lineup is required");
  }
  if (
    request.operation === "arena_match_win_rate" &&
    (!Array.isArray(request.opponents) || request.opponents.length !== 3)
  ) {
    throw new Error("the player and exactly three visible opponents are required");
  }
  if (request.operation === "arena_own_score" && request.opponents !== undefined) {
    throw new Error("arena_own_score must not include opponents");
  }
  assertSide(request.own_team, "own_team", request.stageIds);
  (request.opponents || []).forEach((opponent, position) =>
    assertSide(opponent, `opponents[${position}]`, request.stageIds),
  );
  const teamIds = [request.own_team, ...(request.opponents || [])].map((team) => team.team_id);
  if (new Set(teamIds).size !== teamIds.length) {
    throw new Error("team IDs must be unique");
  }
  if (request.own_score_cache !== undefined) {
    if (request.operation !== "arena_match_win_rate") {
      throw new Error("own score cache is only valid for arena_match_win_rate");
    }
    assertOwnScoreCache(request.own_score_cache, request);
  }
}

function assertOwnScoreCache(cache, request) {
  if (
    !cache ||
    cache.upstream_commit !== UPSTREAM_COMMIT ||
    cache.season !== request.season ||
    cache.simulations !== request.simulations ||
    cache.seed !== request.seed ||
    !Array.isArray(cache.stageIds) ||
    cache.stageIds.length !== 3 ||
    cache.stageIds.some((stageId, index) => stageId !== request.stageIds[index]) ||
    !Array.isArray(cache.stages) ||
    cache.stages.length !== 3
  ) {
    throw new Error("own score cache metadata does not match the current request");
  }
  cache.stages.forEach((stage, index) => {
    const expectedMembers = request.own_team.stages[index].members.length;
    if (
      !stage ||
      stage.stage_number !== index + 1 ||
      stage.stage_id !== request.stageIds[index] ||
      !Array.isArray(stage.member_scores) ||
      stage.member_scores.length !== expectedMembers ||
      stage.member_scores.some(
        (scores) =>
          !Array.isArray(scores) ||
          scores.length !== request.simulations ||
          scores.some((score) => !Number.isSafeInteger(score) || score < 0),
      )
    ) {
      throw new Error(`own score cache stage ${index + 1} is invalid`);
    }
  });
}

function assertSide(side, path, stageIds) {
  if (!side || typeof side !== "object" || typeof side.team_id !== "string" || !side.team_id.trim()) {
    throw new Error(`${path} is invalid`);
  }
  if (typeof side.supportBonus !== "number" || side.supportBonus < 0 || side.supportBonus > 1) {
    throw new Error(`${path}.supportBonus is invalid`);
  }
  if (!Array.isArray(side.stages) || side.stages.length !== 3) {
    throw new Error(`${path}.stages must contain exactly three entries`);
  }
  side.stages.forEach((stage, index) => {
    if (
      !stage ||
      stage.stage_number !== index + 1 ||
      stage.stageId !== stageIds[index] ||
      !Array.isArray(stage.members) ||
      stage.members.length < 1 ||
      stage.members.length > 3 ||
      stage.members.some((member) => !member || typeof member.loadout !== "object")
    ) {
      throw new Error(`${path}.stages[${index}] is invalid`);
    }
  });
}

function mixSeed(baseSeed, stream, trial) {
  let value = (baseSeed ^ Math.imul(stream + 1, 0x9e3779b1) ^ Math.imul(trial + 1, 0x85ebca6b)) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  return (value ^ (value >>> 16)) >>> 0;
}

async function executeJob(job) {
  const stage = Stages.getById(job.loadout.stageId);
  if (!stage || stage.type !== "contest") {
    throw new Error(`unsupported contest stageId: ${job.loadout.stageId}`);
  }
  resetRand(mixSeed(job.base_seed, job.seed_stream, job.start_index || 0));
  const idolConfig = new IdolConfig(job.loadout);
  const stageConfig = new StageConfig(stage);
  const config = new IdolStageConfig(idolConfig, stageConfig, false);
  const engine = new StageEngine(config, []);
  const strategy = new STRATEGIES.HeuristicStrategy(engine);
  engine.strategy = strategy;
  const scores = [];
  for (let trial = 0; trial < job.simulations; trial++) {
    resetRand(mixSeed(job.base_seed, job.seed_stream, (job.start_index || 0) + trial));
    scores.push((await new StagePlayer(engine, strategy).play()).score);
  }
  return {
    job_id: job.job_id,
    logical_job_id: job.logical_job_id,
    chunk_index: job.chunk_index,
    scores,
  };
}

async function executeWorkerJobs(jobs) {
  const results = [];
  for (const job of jobs) results.push(await executeJob(job));
  return results;
}

function runWorker(jobs) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL(import.meta.url), { workerData: jobs });
    worker.once("message", resolve);
    worker.once("error", reject);
    worker.once("exit", (code) => {
      if (code !== 0) reject(new Error(`simulation worker exited with code ${code}`));
    });
  });
}

function splitJobs(jobs, workerCount) {
  const chunks = [];
  for (const job of jobs) {
    const baseSize = Math.floor(job.simulations / workerCount);
    const remainder = job.simulations % workerCount;
    let start = 0;
    for (let chunkIndex = 0; chunkIndex < workerCount; chunkIndex++) {
      const chunkSize = baseSize + (chunkIndex < remainder ? 1 : 0);
      if (chunkSize === 0) continue;
      chunks.push({
        ...job,
        job_id: `${job.job_id}:chunk:${chunkIndex}`,
        logical_job_id: job.job_id,
        chunk_index: chunkIndex,
        start_index: start,
        simulations: chunkSize,
      });
      start += chunkSize;
    }
  }
  return chunks;
}

async function executeParallel(jobs, workerCount) {
  const chunks = splitJobs(jobs, workerCount);
  const partitions = Array.from({ length: workerCount }, () => []);
  chunks.forEach((job, index) => partitions[index % workerCount].push(job));
  const batches = await Promise.all(partitions.map(runWorker));
  const grouped = new Map();
  for (const result of batches.flat()) {
    if (!grouped.has(result.logical_job_id)) grouped.set(result.logical_job_id, []);
    grouped.get(result.logical_job_id).push(result);
  }
  const results = new Map();
  for (const [jobId, jobChunks] of grouped) {
    jobChunks.sort((left, right) => left.chunk_index - right.chunk_index);
    results.set(jobId, jobChunks.flatMap((chunk) => chunk.scores));
  }
  return {
    workerCount,
    results,
  };
}

function calibrationCandidates(available, simulations) {
  let baseline;
  if (available >= 16) baseline = [8, 12, 16, 20, 24, 28, 32];
  else if (available >= 8) baseline = [4, 8, 12];
  else baseline = [1, 2, 4];
  const candidates = new Set([...baseline, Math.ceil(available * 0.75), available]);
  return [...candidates]
    .filter((count) => count >= 1 && count <= available && count <= simulations)
    .sort((left, right) => left - right);
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2) return sorted[middle];
  return (sorted[middle - 1] + sorted[middle]) / 2;
}

function calibrationIdentity(available) {
  return {
    schema_version: CALIBRATION_SCHEMA_VERSION,
    engine_commit: UPSTREAM_COMMIT,
    node_version: process.versions.node,
    platform: process.platform,
    architecture: process.arch,
    cpu_model: cpus()[0]?.model?.trim() || "unknown",
    available_parallelism: available,
  };
}

function isReusableCalibration(value, identity) {
  if (!value || typeof value !== "object") return false;
  for (const [key, expected] of Object.entries(identity)) {
    if (value[key] !== expected) return false;
  }
  return (
    Number.isSafeInteger(value.selected_workers) &&
    value.selected_workers >= 1 &&
    value.selected_workers <= identity.available_parallelism
  );
}

async function readCalibration(cachePath, identity) {
  try {
    const value = JSON.parse(await readFile(cachePath, "utf8"));
    return isReusableCalibration(value, identity) ? value : null;
  } catch {
    return null;
  }
}

async function saveCalibration(cachePath, value) {
  await mkdir(dirname(cachePath), { recursive: true });
  await writeFile(cachePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function chooseCalibratedWorkerCount(benchmarks) {
  const finalists = benchmarks.filter((item) => item.finalist);
  const bestScore = Math.min(...finalists.map((item) => item.selection_score));
  return finalists
    .filter((item) => item.selection_score <= bestScore * (1 + SELECTION_NOISE_TOLERANCE))
    .sort(
      (left, right) =>
        left.workers - right.workers ||
        left.selection_score - right.selection_score ||
        left.elapsed_ms - right.elapsed_ms,
    )[0].workers;
}

function workerCostScore(elapsedMs, workers, baselineWorkers) {
  return elapsedMs * (1 + WORKER_COST_WEIGHT) ** (workers - baselineWorkers);
}

function addWorkerCostScores(benchmarks, candidateBaselineWorkers) {
  const baselineWorkers = Math.min(
    ...benchmarks.filter((item) => item.finalist).map((item) => item.workers),
  );
  return {
    baselineWorkers,
    benchmarks: benchmarks.map((item) => ({
      ...item,
      finalist_selection_score: Number(
        workerCostScore(item.samples_ms[0], item.workers, candidateBaselineWorkers).toFixed(3),
      ),
      selection_score: item.finalist
        ? Number(workerCostScore(item.elapsed_ms, item.workers, baselineWorkers).toFixed(3))
        : null,
    })),
  };
}

async function calibrateParallelism(request, representativeJob, available, cachePath) {
  const calibrationRuns = Math.min(request.simulations, CALIBRATION_RUNS);
  const candidates = calibrationCandidates(available, calibrationRuns);
  const warmupJob = {
    ...representativeJob,
    job_id: "calibration:warmup",
    simulations: Math.min(calibrationRuns, CALIBRATION_WARMUP_RUNS),
    base_seed: request.seed + 900000000,
  };
  await executeParallel([warmupJob], Math.min(4, available));

  const samplesByWorker = new Map(candidates.map((workerCount) => [workerCount, []]));
  for (const workerCount of candidates) {
    const benchmarkJob = {
      ...representativeJob,
      job_id: `calibration:${workerCount}`,
      simulations: calibrationRuns,
      base_seed: request.seed + 1000000000,
    };
    const started = performance.now();
    await executeParallel([benchmarkJob], workerCount);
    samplesByWorker.get(workerCount).push(performance.now() - started);
  }

  const finalistWorkers = new Set(
    [...samplesByWorker.entries()]
      .sort((left, right) => {
        const baselineWorkers = candidates[0];
        return (
          workerCostScore(left[1][0], left[0], baselineWorkers) -
            workerCostScore(right[1][0], right[0], baselineWorkers) ||
          left[0] - right[0]
        );
      })
      .slice(0, Math.min(CALIBRATION_FINALISTS, candidates.length))
      .map(([workerCount]) => workerCount),
  );
  for (const workerCount of finalistWorkers) {
    const benchmarkJob = {
      ...representativeJob,
      job_id: `calibration:${workerCount}`,
      simulations: calibrationRuns,
      base_seed: request.seed + 1000000000,
    };
    for (let repetition = 1; repetition < CALIBRATION_REPETITIONS; repetition++) {
      const started = performance.now();
      await executeParallel([benchmarkJob], workerCount);
      samplesByWorker.get(workerCount).push(performance.now() - started);
    }
  }

  const rawBenchmarks = candidates.map((workerCount) => {
    const samplesMs = samplesByWorker.get(workerCount);
    const elapsedMs = median(samplesMs);
    return {
      workers: workerCount,
      finalist: finalistWorkers.has(workerCount),
      elapsed_ms: Number(elapsedMs.toFixed(3)),
      samples_ms: samplesMs.map((value) => Number(value.toFixed(3))),
      simulations: calibrationRuns,
      simulations_per_second: Number(((calibrationRuns * 1000) / elapsedMs).toFixed(3)),
    };
  });
  const candidateBaselineWorkers = candidates[0];
  const { baselineWorkers, benchmarks } = addWorkerCostScores(
    rawBenchmarks,
    candidateBaselineWorkers,
  );

  const identity = calibrationIdentity(available);
  const value = {
    ...identity,
    selected_workers: chooseCalibratedWorkerCount(benchmarks),
    calibration_runs: calibrationRuns,
    calibration_repetitions: CALIBRATION_REPETITIONS,
    calibration_finalists: Math.min(CALIBRATION_FINALISTS, candidates.length),
    worker_cost_weight_per_additional_worker: WORKER_COST_WEIGHT,
    selection_noise_tolerance: SELECTION_NOISE_TOLERANCE,
    worker_cost_candidate_baseline_workers: candidateBaselineWorkers,
    worker_cost_baseline_workers: baselineWorkers,
    benchmarks,
    calibrated_at: new Date().toISOString(),
  };
  let cacheStatus = "saved";
  try {
    await saveCalibration(cachePath, value);
  } catch (error) {
    cacheStatus = `write_failed: ${error?.code || error?.name || "unknown"}`;
  }
  return {
    mode: "auto",
    source: "calibrated",
    cache_status: cacheStatus,
    cache_path: cachePath,
    available_parallelism: available,
    selected_workers: value.selected_workers,
    calibration_runs: calibrationRuns,
    calibration_repetitions: CALIBRATION_REPETITIONS,
    calibration_finalists: Math.min(CALIBRATION_FINALISTS, candidates.length),
    worker_cost_weight_per_additional_worker: WORKER_COST_WEIGHT,
    selection_noise_tolerance: SELECTION_NOISE_TOLERANCE,
    worker_cost_candidate_baseline_workers: candidateBaselineWorkers,
    worker_cost_baseline_workers: baselineWorkers,
    benchmarks,
  };
}

async function resolveParallelism(request, representativeJob) {
  const available = Math.max(1, availableParallelism());
  const config = request.parallelism;
  if (config == null) {
    throw new Error("parallelism configuration is required");
  }
  if (!config || typeof config !== "object") {
    throw new Error("parallelism configuration must be an object");
  }
  if (config.mode === "fixed") {
    if (!Number.isSafeInteger(config.workers) || config.workers < 1 || config.workers > available) {
      throw new Error(`fixed workers must be an integer from 1 to ${available}`);
    }
    return {
      mode: "fixed",
      source: "request",
      available_parallelism: available,
      selected_workers: config.workers,
    };
  }
  if (config.mode !== "auto" || typeof config.cache_path !== "string" || !config.cache_path.trim()) {
    throw new Error("auto parallelism requires a non-empty cache_path");
  }

  const cachePath = resolvePath(config.cache_path);
  const identity = calibrationIdentity(available);
  const cached = await readCalibration(cachePath, identity);
  if (cached) {
    return {
      mode: "auto",
      source: "cache",
      cache_status: "reused",
      cache_path: cachePath,
      available_parallelism: available,
      selected_workers: cached.selected_workers,
      calibration_runs: cached.calibration_runs,
      calibration_repetitions: cached.calibration_repetitions,
      calibration_finalists: cached.calibration_finalists,
      worker_cost_weight_per_additional_worker: cached.worker_cost_weight_per_additional_worker,
      selection_noise_tolerance: cached.selection_noise_tolerance,
      worker_cost_candidate_baseline_workers: cached.worker_cost_candidate_baseline_workers,
      worker_cost_baseline_workers: cached.worker_cost_baseline_workers,
      benchmarks: cached.benchmarks,
    };
  }
  return calibrateParallelism(request, representativeJob, available, cachePath);
}

function summarize(scores) {
  const sorted = [...scores].sort((a, b) => a - b);
  const quantile = (ratio) => {
    const index = (sorted.length - 1) * ratio;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    if (lower === upper) return sorted[lower];
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
  };
  return {
    count: sorted.length,
    min: sorted[0],
    q1: quantile(0.25),
    median: quantile(0.5),
    mean: sorted.reduce((total, score) => total + score, 0) / sorted.length,
    q3: quantile(0.75),
    max: sorted[sorted.length - 1],
  };
}

function makeJobs(request) {
  const jobs = [];
  let seedStream = 0;
  if (request.own_score_cache === undefined) {
    request.own_team.stages.forEach((stage) => {
      stage.members.forEach((member, memberIndex) => {
        jobs.push({
          job_id: `own:stage:${stage.stage_number}:member:${memberIndex}`,
          loadout: {
            ...member.loadout,
            stageId: stage.stageId,
            supportBonus: request.own_team.supportBonus,
          },
          simulations: request.simulations,
          base_seed: request.seed,
          seed_stream: seedStream++,
        });
      });
    });
  } else {
    seedStream = request.own_team.stages.reduce(
      (total, stage) => total + stage.members.length,
      0,
    );
  }
  (request.opponents || []).forEach((opponent, position) => {
    opponent.stages.forEach((stage) => {
      stage.members.forEach((member, memberIndex) => {
        jobs.push({
          job_id: `opponent:${position}:stage:${stage.stage_number}:member:${memberIndex}`,
          loadout: {
            ...member.loadout,
            stageId: stage.stageId,
            supportBonus: opponent.supportBonus,
          },
          simulations: request.simulations,
          base_seed: request.seed,
          seed_stream: seedStream++,
        });
      });
    });
  });
  return jobs;
}

async function run(request) {
  assertRequest(request);
  const jobs = makeJobs(request);
  const parallelism = await resolveParallelism(request, jobs[0]);
  const execution = await executeParallel(jobs, parallelism.selected_workers);
  const ownStageScores = request.own_score_cache
    ? request.own_score_cache.stages.map((stage) => stage.member_scores)
    : request.own_team.stages.map((stage) =>
        stage.members.map((_, memberIndex) =>
          execution.results.get(`own:stage:${stage.stage_number}:member:${memberIndex}`),
        ),
      );
  const stages = request.own_team.stages.map((stage, stageIndex) => ({
    stage_number: stage.stage_number,
    stage_id: stage.stageId,
    own_member_distributions: ownStageScores[stageIndex].map(summarize),
    own_raw_team_distribution: summarize(sumMemberScores(ownStageScores[stageIndex])),
    ...(request.operation === "arena_own_score"
      ? { own_member_scores: ownStageScores[stageIndex] }
      : {}),
  }));

  if (request.operation === "arena_own_score") {
    return {
      schema_version: SCHEMA_VERSION,
      request_id: request.request_id,
      engine: { project: "gakumas-tools", commit: UPSTREAM_COMMIT },
      method: OWN_SCORE_METHOD,
      score_aggregation: OWN_SCORE_AGGREGATION,
      workers: execution.workerCount,
      parallelism,
      stages,
    };
  }

  const candidates = [];
  for (let position = 0; position < request.opponents.length; position++) {
    const opponent = request.opponents[position];
    const candidateStages = [];
    const stageOutcomes = [];
    for (let stageIndex = 0; stageIndex < opponent.stages.length; stageIndex++) {
      const opponentStage = opponent.stages[stageIndex];
      const opponentMemberScores = opponentStage.members.map((_, memberIndex) =>
        execution.results.get(
          `opponent:${position}:stage:${opponentStage.stage_number}:member:${memberIndex}`,
        ),
      );
      const comparison = compareStageScores(ownStageScores[stageIndex], opponentMemberScores);
      stageOutcomes.push(comparison.outcomes);
      candidateStages.push({
        stage_number: opponentStage.stage_number,
        stage_id: opponentStage.stageId,
        ...countOutcomes(comparison.outcomes),
        first_place_ties: comparison.firstPlaceTies,
        own_effective_team_distribution: summarize(comparison.ownEffectiveTotals),
        opponent_member_distributions: opponentMemberScores.map(summarize),
        opponent_raw_team_distribution: summarize(comparison.opponentRawTotals),
        opponent_effective_team_distribution: summarize(comparison.opponentEffectiveTotals),
      });
    }

    const matchOutcomes = classifyBestOfThree(stageOutcomes);
    candidates.push({
      opponent_id: opponent.team_id,
      position,
      ...countOutcomes(matchOutcomes),
      stages: candidateStages,
    });
  }
  return {
    schema_version: SCHEMA_VERSION,
    request_id: request.request_id,
    engine: { project: "gakumas-tools", commit: UPSTREAM_COMMIT },
    method: "independent_empirical_three_stage_match",
    score_aggregation: SCORE_AGGREGATION,
    match_rule: MATCH_RULE,
    workers: execution.workerCount,
    parallelism,
    stages,
    candidates,
  };
}

if (!isMainThread) {
  try {
    parentPort.postMessage(await executeWorkerJobs(workerData));
  } catch (error) {
    throw error;
  }
} else {
  let input = "";
  process.stdin.setEncoding("utf8");
  for await (const chunk of process.stdin) input += chunk;

  try {
    const result = await run(JSON.parse(input));
    process.stdout.write(JSON.stringify(result));
  } catch (error) {
    process.stderr.write(`${error?.stack || error}\n`);
    process.exitCode = 2;
  }
}
