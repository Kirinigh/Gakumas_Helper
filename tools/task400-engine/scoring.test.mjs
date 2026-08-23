import assert from "node:assert/strict";

import {
  classifyBestOfThree,
  compareStageScores,
  countOutcomes,
} from "./scoring.mjs";

const firstPlaceBonus = compareStageScores([[100]], [[99]]);
assert.deepEqual(firstPlaceBonus.ownEffectiveTotals, [120]);
assert.deepEqual(firstPlaceBonus.opponentEffectiveTotals, [99]);
assert.deepEqual(firstPlaceBonus.outcomes, [1]);

const firstAndFourthCanLose = compareStageScores([[100], [0]], [[99], [99]]);
assert.deepEqual(firstAndFourthCanLose.ownEffectiveTotals, [120]);
assert.deepEqual(firstAndFourthCanLose.opponentEffectiveTotals, [198]);
assert.deepEqual(firstAndFourthCanLose.outcomes, [-1]);

const invariantCrossSideTie = compareStageScores([[100], [100]], [[100], [0]]);
assert.deepEqual(invariantCrossSideTie.outcomes, [1]);
assert.equal(invariantCrossSideTie.firstPlaceTies, 1);

const ambiguousCrossSideTie = compareStageScores([[100]], [[100]]);
assert.deepEqual(ambiguousCrossSideTie.outcomes, [0]);
assert.equal(ambiguousCrossSideTie.firstPlaceTies, 1);
assert.deepEqual(countOutcomes(ambiguousCrossSideTie.outcomes), {
  wins: 0,
  losses: 0,
  ties: 1,
  trials: 1,
});

assert.deepEqual(
  classifyBestOfThree([
    [1, 1, 1],
    [1, 0, -1],
    [-1, -1, 0],
  ]),
  [1, 0, 0],
);

process.stdout.write("task400 scoring tests passed\n");
