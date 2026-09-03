"""Fail-closed state shared by arena selection and post-battle recording."""

from __future__ import annotations

import re
import json
import uuid
from typing import Any, Mapping, Sequence
from pathlib import Path
from datetime import datetime, timezone, timedelta

CHALLENGE_RECORD_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHALLENGE_RECORD_ROOT = PROJECT_ROOT / ".local" / "arena-win-rate"
CHALLENGE_LIFECYCLE_INTENT = "CHALLENGE_INTENT"
CHALLENGE_LIFECYCLE_RESULT_RECORDED = "RESULT_RECORDED_RETURN_PENDING"
CHALLENGE_LIFECYCLE_RETURN_UNVERIFIED = "RESULT_RECORDED_RETURN_UNVERIFIED"
CHALLENGE_LIFECYCLE_FINISHED = "FINISHED"
CHALLENGE_LIFECYCLE_RECOVERED_FINISHED = "RECOVERED_FINISHED"


def _score_tokens(text: str) -> tuple[int, ...]:
    """Extract every integer score token from one OCR row."""

    return tuple(
        int(token.replace(",", "").replace(".", ""))
        for token in re.findall(r"(?<![0-9])[0-9][0-9,.]*", text)
        if token.rstrip(".,")
    )


def _reconcile_displayed_score(
    displayed: int,
    members: Sequence[int],
    bonus: int,
) -> int | None:
    """Cross-check a displayed team total against its three members and bonus.

    The result page sometimes crops exactly one leading digit from either the
    large displayed total or one small member score. Accept only those two
    explainable cases; every other disagreement remains unresolved so the
    challenge flow fails closed instead of recording a silently wrong score.
    """

    if len(members) != 3 or any(value < 0 for value in (*members, bonus)):
        return None
    component_total = sum(members) + bonus
    if component_total == displayed:
        return displayed

    displayed_text = str(displayed)
    component_text = str(component_total)
    if (
        component_total > displayed
        and len(component_text) == len(displayed_text) + 1
        and component_text.endswith(displayed_text)
    ):
        return component_total

    if displayed > component_total:
        difference = displayed - component_total
        values = (*members, *((bonus,) if bonus else ()))
        if any(
            difference == prefix * 10 ** len(str(value))
            for value in values
            for prefix in range(1, 10)
        ):
            return displayed
    return None


def _stage_component_scores_by_side(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage_y: float,
    frame_width: int,
    frame_height: int,
) -> tuple[
    tuple[tuple[int, ...], int] | None,
    tuple[tuple[int, ...], int] | None,
]:
    """Return independently complete member/bonus evidence for each side."""

    lower_y = stage_y + frame_height * 0.015
    upper_y = stage_y + frame_height * 0.075
    members: list[list[int]] = [[], []]
    bonuses: list[list[int]] = [[], []]
    midpoint = frame_width / 2
    for row in rows:
        box = row["box"]
        center_y = box[1] + box[3] / 2
        if not lower_y <= center_y <= upper_y:
            continue
        values = _score_tokens(str(row["text"]))
        if not values:
            continue
        side = 0 if box[0] + box[2] / 2 < midpoint else 1
        if str(row["text"]).lstrip().startswith("+"):
            bonuses[side].extend(values)
        else:
            members[side].extend(values)

    return tuple(
        (tuple(members[side]), bonuses[side][0] if bonuses[side] else 0)
        if len(members[side]) == 3 and len(bonuses[side]) <= 1
        else None
        for side in range(2)
    )


def _stage_component_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage_y: float,
    frame_width: int,
    frame_height: int,
) -> tuple[tuple[tuple[int, ...], int], tuple[tuple[int, ...], int]] | None:
    """Return three member scores and the bonus for both result sides."""

    by_side = _stage_component_scores_by_side(
        rows,
        stage_y=stage_y,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if by_side[0] is None or by_side[1] is None:
        return None
    return by_side[0], by_side[1]


class ArenaChallengeFlowError(RuntimeError):
    """Raised before a new challenge whenever prior state is unresolved."""


def new_challenge_id(position: int, now: datetime | None = None) -> str:
    """Create a collision-resistant ID for one ticket-consuming challenge."""

    if position not in (0, 1, 2):
        raise ValueError("challenge position must be 0, 1, or 2")
    current = datetime.now().astimezone() if now is None else now
    if current.tzinfo is None:
        raise ValueError("challenge-id time must be timezone-aware")
    stamp = current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"arena-{stamp}-p{position}-{uuid.uuid4().hex[:12]}"


def contest_day_key(now: datetime | None = None) -> str:
    """Return the local contest day, whose rollover is 04:00."""

    current = datetime.now().astimezone() if now is None else now
    if current.tzinfo is None:
        raise ValueError("contest-day time must be timezone-aware")
    return (current - timedelta(hours=4)).date().isoformat()


def serialise_result_observations(
    observations: Sequence[Any],
) -> dict[str, Any]:
    """Keep OCR text/geometry only; never persist the source screenshot."""

    rows: list[dict[str, Any]] = []
    numbers: list[dict[str, Any]] = []
    outcome_rows: list[tuple[int, str]] = []
    for item in observations:
        if isinstance(item, Mapping):
            text = str(item.get("text", "")).strip()
            box_value = item.get("box")
            score = item.get("score")
        else:
            text = str(getattr(item, "text", "")).strip()
            box_value = getattr(item, "box", None)
            score = getattr(item, "score", None)
        if not text or box_value is None:
            continue
        try:
            box_components = list(box_value)
        except TypeError:
            continue
        if len(box_components) != 4:
            continue
        box = [int(round(value)) for value in box_components]
        row = {"text": text, "box": box}
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            row["score"] = round(float(score), 6)
        rows.append(row)

        normalised = text.upper().replace(" ", "")
        if normalised in {"WIN", "VICTORY", "勝利", "勝ち"}:
            outcome_rows.append((box[1], "WIN"))
        if normalised in {"LOSE", "LOSS", "DEFEAT", "敗北", "負け"}:
            outcome_rows.append((box[1], "LOSS"))
        numeric = re.fullmatch(
            r"([0-9][0-9,.]*)\s*(?:PT|Pt|pt|P|p)?\s*\+?",
            text,
        )
        if numeric:
            numbers.append(
                {
                    "value": int(
                        numeric.group(1).replace(",", "").replace(".", "")
                    ),
                    "text": text,
                    "box": box,
                }
            )

    score_rows = [
        row
        for row in numbers
        if row["value"] >= 1000 and re.search(r"[Pp]", row["text"])
    ]
    stage_results: list[dict[str, Any]] = []
    score_repairs: list[dict[str, Any]] = []
    score_clusters: list[list[dict[str, Any]]] = []
    recovered_score_row_ids: set[int] = set()
    dual_score_recovery = False
    dual_stage_win_rows: list[dict[str, Any]] = []
    frame_height = 0
    if len(score_rows) >= 4 and rows:
        frame_width = max(row["box"][0] + row["box"][2] for row in rows)
        frame_height = max(row["box"][1] + row["box"][3] for row in rows)
        midpoint = frame_width / 2

        def cluster_scores(tolerance: float) -> list[list[dict[str, Any]]]:
            clusters: list[list[dict[str, Any]]] = []
            for row in sorted(score_rows, key=lambda item: item["box"][1]):
                center_y = row["box"][1] + row["box"][3] / 2
                if not clusters:
                    clusters.append([row])
                    continue
                previous_y = sum(
                    item["box"][1] + item["box"][3] / 2
                    for item in clusters[-1]
                ) / len(clusters[-1])
                if abs(center_y - previous_y) <= tolerance:
                    clusters[-1].append(row)
                else:
                    clusters.append([row])
            return clusters

        if len(score_rows) >= 6:
            score_clusters = [
                cluster
                for cluster in cluster_scores(frame_height * 0.08)
                if len(cluster) == 2
            ]
        elif len(score_rows) == 5:
            pair_y_tolerance = max(8.0, frame_height * 0.025)
            clusters = cluster_scores(pair_y_tolerance)
            recovered_clusters: list[list[dict[str, Any]]] = []
            if len(clusters) == 3 and sorted(map(len, clusters)) == [1, 2, 2]:
                recovery_failed = False
                for cluster in clusters:
                    completed = list(cluster)
                    if len(completed) == 1:
                        seed = completed[0]
                        seed_box = seed["box"]
                        seed_center_y = seed_box[1] + seed_box[3] / 2
                        seed_is_left = seed_box[0] + seed_box[2] / 2 < midpoint
                        candidates = [
                            row
                            for row in numbers
                            if row["value"] >= 1000
                            and not re.search(r"[Pp]", row["text"])
                            and (
                                row["box"][0] + row["box"][2] / 2 < midpoint
                            )
                            != seed_is_left
                            and abs(
                                row["box"][1]
                                + row["box"][3] / 2
                                - seed_center_y
                            )
                            <= pair_y_tolerance
                            and seed_box[3] * 0.65
                            <= row["box"][3]
                            <= seed_box[3] * 1.5
                        ]
                        if len(candidates) != 1:
                            recovery_failed = True
                            break
                        completed.append(candidates[0])
                        recovered_score_row_ids.add(id(candidates[0]))
                    left_sides = {
                        row["box"][0] + row["box"][2] / 2 < midpoint
                        for row in completed
                    }
                    if len(completed) != 2 or left_sides != {False, True}:
                        recovery_failed = True
                        break
                    recovered_clusters.append(completed)
                if not recovery_failed:
                    score_clusters = recovered_clusters
        else:
            pair_y_tolerance = max(8.0, frame_height * 0.025)
            clusters = cluster_scores(pair_y_tolerance)
            recovery_failed = not (
                len(clusters) == 2 and all(len(cluster) == 2 for cluster in clusters)
            )
            explicit_clusters: list[list[dict[str, Any]]] = []
            matched_win_rows: list[dict[str, Any]] = []
            stage_win_rows: list[dict[str, Any]] = []
            score_offsets: list[float] = []

            if not recovery_failed:
                for cluster in clusters:
                    ordered = sorted(cluster, key=lambda item: item["box"][0])
                    sides = {
                        row["box"][0] + row["box"][2] / 2 < midpoint
                        for row in ordered
                    }
                    if sides != {False, True}:
                        recovery_failed = True
                        break
                    explicit_clusters.append(ordered)

            win_rows = [
                row for row in rows if row["text"].upper().strip() == "WIN"
            ]
            if not recovery_failed:
                for cluster in explicit_clusters:
                    stage_y = sum(
                        row["box"][1] + row["box"][3] / 2 for row in cluster
                    ) / 2
                    matches = [
                        row
                        for row in win_rows
                        if 0
                        < stage_y - (row["box"][1] + row["box"][3] / 2)
                        <= frame_height * 0.06
                    ]
                    if len(matches) != 1 or matches[0] in matched_win_rows:
                        recovery_failed = True
                        break
                    matched_win_rows.append(matches[0])
                    score_offsets.append(
                        stage_y
                        - (matches[0]["box"][1] + matches[0]["box"][3] / 2)
                    )

            if not recovery_failed:
                reference_win_width = sum(
                    row["box"][2] for row in matched_win_rows
                ) / len(matched_win_rows)
                reference_win_height = sum(
                    row["box"][3] for row in matched_win_rows
                ) / len(matched_win_rows)
                stage_win_rows = [
                    row
                    for row in win_rows
                    if reference_win_width * 0.65
                    <= row["box"][2]
                    <= reference_win_width * 1.5
                    and reference_win_height * 0.65
                    <= row["box"][3]
                    <= reference_win_height * 1.5
                ]
                stage_win_rows.sort(
                    key=lambda row: row["box"][1] + row["box"][3] / 2
                )
                if len(stage_win_rows) != 3 or not all(
                    row in stage_win_rows for row in matched_win_rows
                ):
                    recovery_failed = True

            if not recovery_failed:
                stage_win_centers = [
                    row["box"][1] + row["box"][3] / 2
                    for row in stage_win_rows
                ]
                stage_pitches = [
                    stage_win_centers[index + 1] - stage_win_centers[index]
                    for index in range(2)
                ]
                pitch_tolerance = max(12.0, frame_height * 0.02)
                if (
                    min(stage_pitches) <= 0
                    or abs(stage_pitches[0] - stage_pitches[1])
                    > pitch_tolerance
                    or max(score_offsets) - min(score_offsets)
                    > max(8.0, frame_height * 0.01)
                ):
                    recovery_failed = True

            missing_stage_index = -1
            if not recovery_failed:
                matched_indices = [
                    stage_win_rows.index(row) for row in matched_win_rows
                ]
                missing_indices = sorted({0, 1, 2}.difference(matched_indices))
                if len(set(matched_indices)) != 2 or len(missing_indices) != 1:
                    recovery_failed = True
                else:
                    missing_stage_index = missing_indices[0]

            side_references: list[list[dict[str, Any]]] = [[], []]
            if not recovery_failed:
                for cluster in explicit_clusters:
                    side_references[0].append(cluster[0])
                    side_references[1].append(cluster[1])
                column_tolerance = max(12.0, frame_width * 0.03)
                height_tolerance = max(8.0, frame_height * 0.0125)
                for references in side_references:
                    center_xs = [
                        row["box"][0] + row["box"][2] / 2
                        for row in references
                    ]
                    heights = [row["box"][3] for row in references]
                    widths = [row["box"][2] for row in references]
                    if (
                        max(center_xs) - min(center_xs) > column_tolerance
                        or max(heights) - min(heights) > height_tolerance
                        or min(widths) < max(widths) * 0.7
                    ):
                        recovery_failed = True
                        break

            recovered_cluster: list[dict[str, Any]] = []
            if not recovery_failed:
                missing_win_center = (
                    stage_win_rows[missing_stage_index]["box"][1]
                    + stage_win_rows[missing_stage_index]["box"][3] / 2
                )
                expected_score_y = missing_win_center + sum(score_offsets) / len(
                    score_offsets
                )
                score_y_tolerance = max(8.0, frame_height * 0.015)
                column_tolerance = max(12.0, frame_width * 0.03)
                for side, references in enumerate(side_references):
                    expected_center_x = sum(
                        row["box"][0] + row["box"][2] / 2
                        for row in references
                    ) / len(references)
                    expected_width = sum(
                        row["box"][2] for row in references
                    ) / len(references)
                    expected_height = sum(
                        row["box"][3] for row in references
                    ) / len(references)
                    candidates = [
                        row
                        for row in numbers
                        if row["value"] >= 1000
                        and re.fullmatch(r"[0-9][0-9,.]*", row["text"])
                        and (
                            row["box"][0] + row["box"][2] / 2 < midpoint
                        )
                        == (side == 0)
                        and abs(
                            row["box"][1]
                            + row["box"][3] / 2
                            - expected_score_y
                        )
                        <= score_y_tolerance
                        and abs(
                            row["box"][0]
                            + row["box"][2] / 2
                            - expected_center_x
                        )
                        <= column_tolerance
                        and expected_width * 0.7
                        <= row["box"][2]
                        <= expected_width * 1.35
                        and expected_height * 0.7
                        <= row["box"][3]
                        <= expected_height * 1.35
                    ]
                    if len(candidates) != 1:
                        recovery_failed = True
                        break
                    recovered_cluster.append(candidates[0])

            if not recovery_failed:
                recovered_centers = [
                    row["box"][1] + row["box"][3] / 2
                    for row in recovered_cluster
                ]
                if abs(recovered_centers[0] - recovered_centers[1]) > pair_y_tolerance:
                    recovery_failed = True
                else:
                    recovered_score_row_ids.update(map(id, recovered_cluster))
                    score_clusters = sorted(
                        [*explicit_clusters, recovered_cluster],
                        key=lambda cluster: sum(
                            row["box"][1] + row["box"][3] / 2
                            for row in cluster
                        )
                        / len(cluster),
                    )
                    dual_score_recovery = True
                    dual_stage_win_rows = stage_win_rows
        if len(score_clusters) == 3:
            win_rows = [row for row in rows if row["text"].upper().strip() == "WIN"]
            for stage_number, cluster in enumerate(score_clusters, start=1):
                ordered = sorted(cluster, key=lambda item: item["box"][0])
                recovered_suffix = any(
                    id(row) in recovered_score_row_ids for row in cluster
                )
                stage_y = sum(
                    item["box"][1] + item["box"][3] / 2 for item in cluster
                ) / 2
                nearby_wins = (
                    [dual_stage_win_rows[stage_number - 1]]
                    if dual_score_recovery
                    else [
                        row
                        for row in win_rows
                        if abs(
                            row["box"][1] + row["box"][3] / 2 - stage_y
                        )
                        <= frame_height * 0.08
                    ]
                )
                winner = "UNKNOWN"
                if len(nearby_wins) == 1:
                    winner = (
                        "OWN"
                        if nearby_wins[0]["box"][0]
                        + nearby_wins[0]["box"][2] / 2
                        < midpoint
                        else "OPPONENT"
                    )
                elif ordered[0]["value"] == ordered[1]["value"]:
                    winner = "TIE"
                own_score = ordered[0]["value"]
                opponent_score = ordered[1]["value"]
                observed_scores = (own_score, opponent_score)
                if dual_score_recovery:
                    component_scores_by_side = _stage_component_scores_by_side(
                        rows,
                        stage_y=stage_y,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                    displayed_scores = [own_score, opponent_score]
                    complete_side_count = 0
                    component_conflict = False
                    for side, components in enumerate(component_scores_by_side):
                        if components is None:
                            continue
                        reconciled = _reconcile_displayed_score(
                            displayed_scores[side],
                            components[0],
                            components[1],
                        )
                        if reconciled is None:
                            component_conflict = True
                            break
                        displayed_scores[side] = reconciled
                        complete_side_count += 1
                    if component_conflict or (
                        recovered_suffix and complete_side_count == 0
                    ):
                        stage_results = []
                        break
                    own_score, opponent_score = displayed_scores
                else:
                    component_scores = _stage_component_scores(
                        rows,
                        stage_y=stage_y,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                    components_certified = False
                    if component_scores is not None:
                        own_components, opponent_components = component_scores
                        reconciled_own = _reconcile_displayed_score(
                            own_score,
                            own_components[0],
                            own_components[1],
                        )
                        reconciled_opponent = _reconcile_displayed_score(
                            opponent_score,
                            opponent_components[0],
                            opponent_components[1],
                        )
                        if (
                            reconciled_own is not None
                            and reconciled_opponent is not None
                        ):
                            own_score = reconciled_own
                            opponent_score = reconciled_opponent
                            components_certified = True
                    if recovered_suffix and not components_certified:
                        stage_results = []
                        break
                expected_winner = (
                    "OWN"
                    if own_score > opponent_score
                    else "OPPONENT"
                    if opponent_score > own_score
                    else "TIE"
                )
                if winner != expected_winner:
                    if dual_score_recovery:
                        stage_results = []
                        break
                    winner = "UNKNOWN"
                for side, observed_score, resolved_score in zip(
                    ("own", "opponent"),
                    observed_scores,
                    (own_score, opponent_score),
                    strict=True,
                ):
                    if observed_score != resolved_score:
                        score_repairs.append(
                            {
                                "stage_number": stage_number,
                                "side": side,
                                "observed_score": observed_score,
                                "resolved_score": resolved_score,
                                "rule": (
                                    "displayed_total_missing_one_leading_digit"
                                ),
                                "evidence": (
                                    "three_member_scores_plus_unique_first_place_bonus"
                                ),
                            }
                        )
                stage_results.append(
                    {
                        "stage_number": stage_number,
                        "own_score": own_score,
                        "opponent_score": opponent_score,
                        "winner": winner,
                    }
                )
    outcome = "UNKNOWN"
    eligible_outcomes: list[tuple[int, str]] = []
    if outcome_rows:
        eligible_outcomes = outcome_rows
        if len(score_clusters) == 3:
            first_score_top = min(row["box"][1] for row in score_clusters[0])
            eligible_outcomes = [
                row
                for row in outcome_rows
                if row[0] <= first_score_top - frame_height * 0.03
            ]
        if eligible_outcomes:
            outcome = min(eligible_outcomes)[1]
    if dual_score_recovery:
        own_wins = sum(stage["winner"] == "OWN" for stage in stage_results)
        opponent_wins = sum(
            stage["winner"] == "OPPONENT" for stage in stage_results
        )
        outcome_matches_stages = (
            outcome == "WIN" and own_wins >= 2
        ) or (outcome == "LOSS" and opponent_wins >= 2)
        if len(eligible_outcomes) != 1 or not outcome_matches_stages:
            stage_results = []
    if len(stage_results) != 3:
        score_repairs = []
    return {
        "outcome": outcome,
        "stage_results": stage_results,
        "score_repairs": score_repairs,
        "ocr_observations": rows,
        "numeric_observations": numbers,
        "screenshots_persisted": False,
    }


def result_observations_complete(result: Mapping[str, Any]) -> bool:
    """Return whether a result can release the next challenge gate."""

    stages = result.get("stage_results")
    structurally_complete = bool(
        result.get("outcome") in {"WIN", "LOSS"}
        and isinstance(stages, list)
        and len(stages) == 3
        and all(
            isinstance(stage, Mapping)
            and stage.get("stage_number") == index
            and stage.get("winner") in {"OWN", "OPPONENT", "TIE"}
            and isinstance(stage.get("own_score"), int)
            and not isinstance(stage.get("own_score"), bool)
            and isinstance(stage.get("opponent_score"), int)
            and not isinstance(stage.get("opponent_score"), bool)
            and stage["own_score"] >= 0
            and stage["opponent_score"] >= 0
            for index, stage in enumerate(stages, start=1)
        )
    )
    if not structurally_complete:
        return False
    own_wins = sum(stage["winner"] == "OWN" for stage in stages)
    opponent_wins = sum(stage["winner"] == "OPPONENT" for stage in stages)
    return bool(
        (result["outcome"] == "WIN" and own_wins >= 2)
        or (result["outcome"] == "LOSS" and opponent_wins >= 2)
    )


def classify_post_challenge_page(
    *,
    opponent_count: int,
    close_count: int,
    ocr_texts: Sequence[str],
    reward_close_clicks: int,
    finish_count: int = 0,
    finish_resend_clicks: int = 0,
    finish_resend_authorized: bool = False,
    arena_main_exhausted: bool = False,
) -> str:
    """Choose the sole safe post-battle action from low-dimensional evidence."""

    normalised = {str(text).strip().upper() for text in ocr_texts}
    outcome_kinds: set[str] = set()
    if normalised.intersection({"WIN", "VICTORY", "勝利"}):
        outcome_kinds.add("WIN")
    if normalised.intersection({"LOSE", "LOSS", "DEFEAT", "敗北"}):
        outcome_kinds.add("LOSS")
    if finish_count not in (0, 1):
        return "AMBIGUOUS"
    if arena_main_exhausted:
        if (
            opponent_count == 0
            and close_count == 0
            and finish_count == 0
            and not outcome_kinds
            and "レート報酬" not in normalised
        ):
            return "RETURNED"
        return "AMBIGUOUS"
    if finish_count:
        if opponent_count or close_count:
            return "AMBIGUOUS"
        if finish_resend_authorized and finish_resend_clicks == 0:
            return "RESEND_FINISH"
        return "AMBIGUOUS"
    # Opponent cards are rendered progressively after the arena main page
    # returns.  One or two otherwise unobstructed anchors are not sufficient
    # to certify the page, but they are also not a contradictory state.  Keep
    # sampling without input until all three anchors appear or the existing
    # outer deadline expires.  Any overlay/result evidence remains ambiguous.
    if (
        opponent_count in (1, 2)
        and close_count == 0
        and not normalised
    ):
        return "WAIT"
    if opponent_count == 3 and close_count == 0:
        return "RETURNED"
    if opponent_count not in (0, 3) or close_count not in (0, 1):
        return "AMBIGUOUS"
    if opponent_count == 3:
        return "AMBIGUOUS"
    if close_count == 0:
        return "WAIT"
    if (
        reward_close_clicks < 3
        and len(outcome_kinds) == 1
        and ("GRADE" in normalised or "レート報酬" in normalised)
    ):
        return "CLOSE_REWARD"
    return "AMBIGUOUS"


class ArenaChallengeRecordStore:
    """Persist one pending intent and one atomic low-dimensional result per round."""

    def __init__(self, root: str | Path = DEFAULT_CHALLENGE_RECORD_ROOT) -> None:
        self.root = Path(root).resolve()
        self.pending_path = self.root / "pending-challenge-v1.json"
        self.results_root = self.root / "challenge-results-v1"

    def begin(self, intent: Mapping[str, Any]) -> dict[str, Any]:
        capture_id = str(intent.get("capture_id", "")).strip()
        if not capture_id or not re.fullmatch(r"[A-Za-z0-9._-]+", capture_id):
            raise ArenaChallengeFlowError("challenge capture_id is missing or unsafe")
        self._clear_completed_pending()
        if self.pending_path.exists():
            raise ArenaChallengeFlowError(
                "a prior arena challenge is not strictly finished; refusing another click"
            )
        record = {
            "schema_version": CHALLENGE_RECORD_SCHEMA_VERSION,
            **dict(intent),
            "lifecycle_state": CHALLENGE_LIFECYCLE_INTENT,
        }
        self._ensure_directory(self.root)
        self._write_atomic(self.pending_path, record)
        return record

    def abort(self, capture_id: str) -> None:
        pending = self.load_pending()
        if pending is None:
            return
        if pending.get("capture_id") != capture_id:
            raise ArenaChallengeFlowError("pending challenge capture_id changed")
        if pending.get("lifecycle_state") != CHALLENGE_LIFECYCLE_INTENT:
            raise ArenaChallengeFlowError(
                "challenge intent cannot be aborted after its result was recorded"
            )
        if (self.results_root / f"{capture_id}.json").is_file():
            raise ArenaChallengeFlowError(
                "challenge intent cannot be aborted after its result file exists"
            )
        self._unlink_pending()

    def load_pending(self) -> dict[str, Any] | None:
        if not self.pending_path.is_file():
            return None
        try:
            value = json.loads(self.pending_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArenaChallengeFlowError("pending challenge record is unreadable") from error
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != CHALLENGE_RECORD_SCHEMA_VERSION
        ):
            raise ArenaChallengeFlowError("pending challenge record has an unknown schema")
        return value

    def complete(self, result: Mapping[str, Any]) -> Path:
        pending = self.load_pending()
        if pending is None:
            raise ArenaChallengeFlowError("result page has no pending challenge intent")
        pending = dict(pending)
        if pending.get("lifecycle_state") != CHALLENGE_LIFECYCLE_INTENT:
            raise ArenaChallengeFlowError(
                "challenge result was already recorded; refusing to overwrite it"
            )
        incomplete_result = pending.pop("incomplete_result", None)
        capture_id = str(pending.get("capture_id", ""))
        destination = self.results_root / f"{capture_id}.json"
        if destination.is_file():
            existing = self._load_result(destination)
            self._require_result_capture_id(existing, capture_id)
            raise ArenaChallengeFlowError(
                "challenge result was already recorded; refusing to overwrite it"
            )
        lifecycle = {
            "state": CHALLENGE_LIFECYCLE_RESULT_RECORDED,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        record = {
            "schema_version": CHALLENGE_RECORD_SCHEMA_VERSION,
            "intent": pending,
            "result": dict(result),
            "lifecycle": lifecycle,
            "return_verification": {
                "status": "pending",
                "evidence_origin": "product",
                "updated_at": lifecycle["updated_at"],
            },
        }
        if isinstance(incomplete_result, Mapping):
            record["incomplete_result_attempt"] = dict(incomplete_result)
        self._ensure_directory(self.results_root)
        self._write_atomic(destination, record)
        pending["lifecycle_state"] = CHALLENGE_LIFECYCLE_RESULT_RECORDED
        pending["result_recorded_at"] = lifecycle["updated_at"]
        self._write_atomic(self.pending_path, pending)
        return destination

    def note_return_unverified(
        self,
        capture_id: str,
        *,
        reason_code: str,
        evidence: Mapping[str, Any],
    ) -> Path:
        """Persist a product-side return failure without releasing the next round."""

        pending = self._pending_for_capture(capture_id)
        now = datetime.now(timezone.utc).isoformat()
        result_path = self.results_root / f"{capture_id}.json"
        result_recorded = result_path.is_file()
        record: dict[str, Any] | None = None
        if result_recorded:
            record = self._load_result(result_path)
            self._require_result_capture_id(record, capture_id)
        verification = {
            "status": "unverified",
            "reason_code": str(reason_code),
            "evidence_origin": "product",
            "evidence": dict(evidence),
            "updated_at": now,
        }
        if result_recorded:
            pending["lifecycle_state"] = CHALLENGE_LIFECYCLE_RETURN_UNVERIFIED
        pending["return_verification"] = verification
        self._write_atomic(self.pending_path, pending)
        if result_recorded:
            assert record is not None
            record["lifecycle"] = {
                "state": CHALLENGE_LIFECYCLE_RETURN_UNVERIFIED,
                "updated_at": now,
            }
            record["return_verification"] = verification
            self._write_atomic(result_path, record)
        return result_path

    def complete_return(
        self,
        capture_id: str,
        *,
        evidence: Mapping[str, Any],
    ) -> Path:
        """Atomically finish a recorded round only from product return evidence."""

        pending = self._pending_for_capture(capture_id)
        result_path = self.results_root / f"{capture_id}.json"
        if not result_path.is_file():
            now = datetime.now(timezone.utc).isoformat()
            pending["return_verification"] = {
                "status": "verified",
                "evidence_origin": "product",
                "result_unresolved": True,
                "evidence": dict(evidence),
                "updated_at": now,
            }
            self._write_atomic(self.pending_path, pending)
            raise ArenaChallengeFlowError(
                "arena return was verified but the challenge result is still unresolved"
            )
        record = self._load_result(result_path)
        self._require_result_capture_id(record, capture_id)
        previous_state = str(pending.get("lifecycle_state", ""))
        record_lifecycle = record.get("lifecycle")
        record_state = (
            str(record_lifecycle.get("state", ""))
            if isinstance(record_lifecycle, Mapping)
            else ""
        )
        if (
            previous_state == CHALLENGE_LIFECYCLE_INTENT
            and record_state == CHALLENGE_LIFECYCLE_RESULT_RECORDED
        ):
            # The result file is written before the pending-state transition.
            # If that second atomic write failed, the immutable capture ID and
            # result lifecycle safely recover the interrupted transition.
            previous_state = CHALLENGE_LIFECYCLE_RESULT_RECORDED
        if previous_state not in {
            CHALLENGE_LIFECYCLE_RESULT_RECORDED,
            CHALLENGE_LIFECYCLE_RETURN_UNVERIFIED,
        }:
            raise ArenaChallengeFlowError(
                "arena return cannot finish before the challenge result is recorded"
            )
        recovered = previous_state == CHALLENGE_LIFECYCLE_RETURN_UNVERIFIED
        terminal_state = (
            CHALLENGE_LIFECYCLE_RECOVERED_FINISHED
            if recovered
            else CHALLENGE_LIFECYCLE_FINISHED
        )
        now = datetime.now(timezone.utc).isoformat()
        record["lifecycle"] = {"state": terminal_state, "updated_at": now}
        record["return_verification"] = {
            "status": "verified",
            "evidence_origin": "product",
            "recovered_after_unverified": recovered,
            "evidence": dict(evidence),
            "updated_at": now,
        }
        self._write_atomic(result_path, record)
        self._unlink_pending()
        return result_path

    def note_incomplete_result(self, result: Mapping[str, Any]) -> Path:
        """Attach low-dimensional OCR evidence to the unresolved intent."""

        pending = self.load_pending()
        if pending is None:
            raise ArenaChallengeFlowError("incomplete result has no pending challenge intent")
        if pending.get("lifecycle_state") != CHALLENGE_LIFECYCLE_INTENT:
            raise ArenaChallengeFlowError(
                "challenge result was already recorded; refusing incomplete overwrite"
            )
        capture_id = str(pending.get("capture_id", ""))
        if (self.results_root / f"{capture_id}.json").is_file():
            raise ArenaChallengeFlowError(
                "challenge result file already exists; refusing incomplete overwrite"
            )
        updated = {**pending, "incomplete_result": dict(result)}
        self._write_atomic(self.pending_path, updated)
        return self.pending_path

    def _clear_completed_pending(self) -> None:
        pending = self.load_pending()
        if pending is None:
            return
        capture_id = str(pending.get("capture_id", ""))
        result_path = self.results_root / f"{capture_id}.json"
        if not result_path.is_file():
            return
        record = self._load_result(result_path)
        self._require_result_capture_id(record, capture_id)
        lifecycle = record.get("lifecycle")
        state = lifecycle.get("state") if isinstance(lifecycle, Mapping) else None
        if state in {
            CHALLENGE_LIFECYCLE_FINISHED,
            CHALLENGE_LIFECYCLE_RECOVERED_FINISHED,
        }:
            self._unlink_pending()

    def _pending_for_capture(self, capture_id: str) -> dict[str, Any]:
        pending = self.load_pending()
        if pending is None:
            raise ArenaChallengeFlowError("arena return has no pending challenge")
        if pending.get("capture_id") != capture_id:
            raise ArenaChallengeFlowError("pending challenge capture_id changed")
        return dict(pending)

    @staticmethod
    def _load_result(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArenaChallengeFlowError("challenge result record is unreadable") from error
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != CHALLENGE_RECORD_SCHEMA_VERSION
        ):
            raise ArenaChallengeFlowError("challenge result record has an unknown schema")
        return value

    @staticmethod
    def _require_result_capture_id(
        record: Mapping[str, Any],
        capture_id: str,
    ) -> None:
        intent = record.get("intent")
        recorded_capture_id = (
            intent.get("capture_id") if isinstance(intent, Mapping) else None
        )
        if recorded_capture_id != capture_id:
            raise ArenaChallengeFlowError(
                "challenge result record capture_id does not match pending intent"
            )

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ArenaChallengeFlowError(
                "challenge record directory could not be created"
            ) from error

    def _unlink_pending(self) -> None:
        try:
            self.pending_path.unlink(missing_ok=True)
        except OSError as error:
            raise ArenaChallengeFlowError(
                "pending challenge record could not be removed"
            ) from error

    @staticmethod
    def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as error:
            raise ArenaChallengeFlowError(
                "challenge record could not be written atomically"
            ) from error
