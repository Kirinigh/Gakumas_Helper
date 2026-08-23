function assertScoreMatrix(memberScores, label) {
  if (!Array.isArray(memberScores) || memberScores.length < 1 || memberScores.length > 3) {
    throw new Error(`${label} must contain one to three members`);
  }
  const trials = memberScores[0]?.length;
  if (!Number.isSafeInteger(trials) || trials < 1) {
    throw new Error(`${label} must contain at least one trial`);
  }
  if (
    memberScores.some(
      (scores) =>
        !Array.isArray(scores) ||
        scores.length !== trials ||
        scores.some((score) => !Number.isSafeInteger(score) || score < 0),
    )
  ) {
    throw new Error(`${label} contains invalid or misaligned scores`);
  }
  return trials;
}

export function sumMemberScores(memberScores) {
  assertScoreMatrix(memberScores, "member scores");
  return memberScores[0].map((_, trial) =>
    memberScores.reduce((total, scores) => total + scores[trial], 0),
  );
}

export function compareStageScores(ownMemberScores, opponentMemberScores) {
  const ownTrials = assertScoreMatrix(ownMemberScores, "own member scores");
  const opponentTrials = assertScoreMatrix(opponentMemberScores, "opponent member scores");
  if (ownTrials !== opponentTrials) {
    throw new Error("own and opponent score trials must align");
  }

  const ownRawTotals = sumMemberScores(ownMemberScores);
  const opponentRawTotals = sumMemberScores(opponentMemberScores);
  const ownEffectiveScaledTotals = [];
  const opponentEffectiveScaledTotals = [];
  const outcomes = [];
  let firstPlaceTies = 0;

  for (let trial = 0; trial < ownRawTotals.length; trial++) {
    const ownTop = Math.max(...ownMemberScores.map((scores) => scores[trial]));
    const opponentTop = Math.max(...opponentMemberScores.map((scores) => scores[trial]));
    const ownRawScaled = ownRawTotals[trial] * 5;
    const opponentRawScaled = opponentRawTotals[trial] * 5;

    let ownEffectiveScaled;
    let opponentEffectiveScaled;
    let outcome;
    if (ownTop > opponentTop) {
      ownEffectiveScaled = ownRawScaled + ownTop;
      opponentEffectiveScaled = opponentRawScaled;
      outcome = Math.sign(ownEffectiveScaled - opponentEffectiveScaled);
    } else if (ownTop < opponentTop) {
      ownEffectiveScaled = ownRawScaled;
      opponentEffectiveScaled = opponentRawScaled + opponentTop;
      outcome = Math.sign(ownEffectiveScaled - opponentEffectiveScaled);
    } else {
      // The visible rules do not define which side receives first place when
      // the maximum raw score is tied across sides. Only call the result when
      // it is invariant; otherwise preserve it as an explicit non-win tie.
      firstPlaceTies++;
      const ownWorst = ownRawScaled;
      const ownBest = ownRawScaled + ownTop;
      const opponentWorst = opponentRawScaled;
      const opponentBest = opponentRawScaled + opponentTop;
      ownEffectiveScaled = ownWorst;
      opponentEffectiveScaled = opponentBest;
      if (ownWorst > opponentBest) outcome = 1;
      else if (ownBest < opponentWorst) outcome = -1;
      else outcome = 0;
    }
    ownEffectiveScaledTotals.push(ownEffectiveScaled);
    opponentEffectiveScaledTotals.push(opponentEffectiveScaled);
    outcomes.push(outcome);
  }

  return {
    ownRawTotals,
    opponentRawTotals,
    ownEffectiveTotals: ownEffectiveScaledTotals.map((score) => score / 5),
    opponentEffectiveTotals: opponentEffectiveScaledTotals.map((score) => score / 5),
    outcomes,
    firstPlaceTies,
  };
}

export function countOutcomes(outcomes) {
  if (!Array.isArray(outcomes) || outcomes.some((outcome) => ![-1, 0, 1].includes(outcome))) {
    throw new Error("outcomes must contain only -1, 0, or 1");
  }
  return {
    wins: outcomes.filter((outcome) => outcome > 0).length,
    losses: outcomes.filter((outcome) => outcome < 0).length,
    ties: outcomes.filter((outcome) => outcome === 0).length,
    trials: outcomes.length,
  };
}

export function classifyBestOfThree(stageOutcomes) {
  if (
    !Array.isArray(stageOutcomes) ||
    stageOutcomes.length !== 3 ||
    stageOutcomes.some((outcomes) => !Array.isArray(outcomes))
  ) {
    throw new Error("best-of-three requires exactly three stage outcome arrays");
  }
  const trials = stageOutcomes[0].length;
  if (
    stageOutcomes.some(
      (outcomes) =>
        outcomes.length !== trials || outcomes.some((outcome) => ![-1, 0, 1].includes(outcome)),
    )
  ) {
    throw new Error("stage outcomes must align and contain only -1, 0, or 1");
  }
  return Array.from({ length: trials }, (_, trial) => {
    const stageWins = stageOutcomes.filter((outcomes) => outcomes[trial] > 0).length;
    const stageLosses = stageOutcomes.filter((outcomes) => outcomes[trial] < 0).length;
    if (stageWins >= 2) return 1;
    if (stageLosses >= 2) return -1;
    return 0;
  });
}
