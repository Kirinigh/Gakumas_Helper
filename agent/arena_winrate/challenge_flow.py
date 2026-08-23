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


def _stage_component_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage_y: float,
    frame_width: int,
    frame_height: int,
) -> tuple[tuple[tuple[int, ...], int], tuple[tuple[int, ...], int]] | None:
    """Return three member scores and the bonus for both result sides."""

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

    if any(len(values) != 3 for values in members):
        return None
    if any(len(values) > 1 for values in bonuses):
        return None
    return (
        (tuple(members[0]), bonuses[0][0] if bonuses[0] else 0),
        (tuple(members[1]), bonuses[1][0] if bonuses[1] else 0),
    )


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
    score_clusters: list[list[dict[str, Any]]] = []
    frame_height = 0
    if len(score_rows) >= 6 and rows:
        frame_width = max(row["box"][0] + row["box"][2] for row in rows)
        frame_height = max(row["box"][1] + row["box"][3] for row in rows)
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
            if abs(center_y - previous_y) <= frame_height * 0.08:
                clusters[-1].append(row)
            else:
                clusters.append([row])
        score_clusters = [cluster for cluster in clusters if len(cluster) == 2]
        if len(score_clusters) == 3:
            win_rows = [row for row in rows if row["text"].upper().strip() == "WIN"]
            midpoint = frame_width / 2
            for stage_number, cluster in enumerate(score_clusters, start=1):
                ordered = sorted(cluster, key=lambda item: item["box"][0])
                stage_y = sum(
                    item["box"][1] + item["box"][3] / 2 for item in cluster
                ) / 2
                nearby_wins = [
                    row
                    for row in win_rows
                    if abs(
                        row["box"][1] + row["box"][3] / 2 - stage_y
                    )
                    <= frame_height * 0.08
                ]
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
                component_scores = _stage_component_scores(
                    rows,
                    stage_y=stage_y,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
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
                    if reconciled_own is not None and reconciled_opponent is not None:
                        own_score = reconciled_own
                        opponent_score = reconciled_opponent
                expected_winner = (
                    "OWN"
                    if own_score > opponent_score
                    else "OPPONENT"
                    if opponent_score > own_score
                    else "TIE"
                )
                if winner != expected_winner:
                    winner = "UNKNOWN"
                stage_results.append(
                    {
                        "stage_number": stage_number,
                        "own_score": own_score,
                        "opponent_score": opponent_score,
                        "winner": winner,
                    }
                )
    outcome = "UNKNOWN"
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
    return {
        "outcome": outcome,
        "stage_results": stage_results,
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
) -> str:
    """Choose the sole safe post-battle action from low-dimensional evidence."""

    if opponent_count == 3 and close_count == 0:
        return "RETURNED"
    if opponent_count not in (0, 3) or close_count not in (0, 1):
        return "AMBIGUOUS"
    if opponent_count == 3:
        return "AMBIGUOUS"
    if close_count == 0:
        return "WAIT"
    normalised = {str(text).strip().upper() for text in ocr_texts}
    outcome_kinds: set[str] = set()
    if normalised.intersection({"WIN", "VICTORY", "勝利"}):
        outcome_kinds.add("WIN")
    if normalised.intersection({"LOSE", "LOSS", "DEFEAT", "敗北"}):
        outcome_kinds.add("LOSS")
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
                "a prior arena challenge has no recorded result; refusing another click"
            )
        record = {
            "schema_version": CHALLENGE_RECORD_SCHEMA_VERSION,
            **dict(intent),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_atomic(self.pending_path, record)
        return record

    def abort(self, capture_id: str) -> None:
        pending = self.load_pending()
        if pending is None:
            return
        if pending.get("capture_id") != capture_id:
            raise ArenaChallengeFlowError("pending challenge capture_id changed")
        self.pending_path.unlink(missing_ok=True)

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
        incomplete_result = pending.pop("incomplete_result", None)
        capture_id = str(pending.get("capture_id", ""))
        record = {
            "schema_version": CHALLENGE_RECORD_SCHEMA_VERSION,
            "intent": pending,
            "result": dict(result),
        }
        if isinstance(incomplete_result, Mapping):
            record["incomplete_result_attempt"] = dict(incomplete_result)
        self.results_root.mkdir(parents=True, exist_ok=True)
        destination = self.results_root / f"{capture_id}.json"
        self._write_atomic(destination, record)
        self.pending_path.unlink(missing_ok=True)
        return destination

    def note_incomplete_result(self, result: Mapping[str, Any]) -> Path:
        """Attach low-dimensional OCR evidence to the unresolved intent."""

        pending = self.load_pending()
        if pending is None:
            raise ArenaChallengeFlowError("incomplete result has no pending challenge intent")
        updated = {**pending, "incomplete_result": dict(result)}
        self._write_atomic(self.pending_path, updated)
        return self.pending_path

    def _clear_completed_pending(self) -> None:
        pending = self.load_pending()
        if pending is None:
            return
        capture_id = str(pending.get("capture_id", ""))
        if (self.results_root / f"{capture_id}.json").is_file():
            self.pending_path.unlink(missing_ok=True)

    @staticmethod
    def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
