"""The explicit own-only task must have no route into real challenges."""

import ast
import json
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock
from collections.abc import Mapping

import pytest

from agent.arena_winrate import ArenaRuntimeConfig
from agent.arena_winrate.task_log import ArenaTaskLog

ROOT = Path(__file__).resolve().parents[1]


def _load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _action_namespace():
    """Execute the real action with inert dependencies, without Maa IPC."""
    path = ROOT / "agent/custom/action/challenge.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [
        node
        for node in tree.body
        if getattr(node, "name", None)
        in {"_resolved_own_score_params", "_failure_location", "ArenaOwnScoreRecalculate"}
    ]
    for node in body:
        if isinstance(node, ast.ClassDef):
            node.decorator_list = []
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    method.decorator_list = []
    namespace = {
        "Context": object,
        "CustomAction": type("CustomAction", (), {"RunArg": object}),
        "Mapping": Mapping,
        "json": json,
        "ArenaRuntimeConfig": ArenaRuntimeConfig,
        "logger": Mock(),
        "user_logger": Mock(),
        "arena_task_log": ArenaTaskLog(),
        "DEFAULT_OWN_SCORE_CACHE": "unused-cache",
    }
    for name in (
        "ArenaComponentError", "ArenaReaderError", "AdapterError", "OwnScoreCacheError"
    ):
        namespace[name] = type(name, (Exception,), {})
    for name in (
        "resolve_arena_component", "reset_failed_arena_component_checks", "SubprocessArenaAdapter", "OwnScoreCacheStore",
        "reset_prepared_own_score_cache", "MaaArenaReaderBackend", "ArenaLineupReader",
        "ArenaOwnScoreService", "_log_cost_customization_fallbacks",
        "_log_own_score_evaluation", "own_score_user_status", "_publish_status",
        "_stop_with_error",
        "cancellation_for",
    ):
        namespace[name] = Mock()
    namespace["_stop_with_error"].return_value = False
    namespace["resolve_arena_component"].return_value = SimpleNamespace(
        bundle_dir="fixed-test-bundle", status="active", commit="a" * 40,
        season=SimpleNamespace(season=51, preview=False), warnings=(),
    )
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class OwnContext:
    def __init__(self, nodes):
        self.nodes = nodes
        self.requested = []

    def get_node_data(self, name):
        self.requested.append(name)
        assert name in {"ArenaOwnScoreRun", "ChallengeSeasonConfig"}
        return self.nodes.get(name)


def _context():
    nodes = _load("assets/resource/base/pipeline/ArenaOwnScore.json")
    nodes["ChallengeSeasonConfig"] = {"attach": {"season": 51}}
    return OwnContext(nodes)


def test_own_task_routes_are_statically_disconnected_from_challenge():
    nodes = {}
    for path in (ROOT / "assets/resource/base/pipeline").glob("*.json"):
        nodes.update(json.loads(path.read_text(encoding="utf-8")))
    entry = next(
        task["entry"] for task in _load("assets/interface.json")["task"]
        if task["name"] == "arena_recalculate_own_score"
    )
    reached, pending = set(), [entry]
    while pending:
        name = pending.pop().removeprefix("[JumpBack]")
        if name in reached:
            continue
        reached.add(name)
        node = nodes[name]
        for key in ("next", "on_error"):
            pending.extend(node.get(key, []))
    assert not reached & {
        "Challenge", "ChallengeChoose", "ChallengeSelector", "ChallengeAuto",
        "ChallengeStart", "ChallengeIndex", "ChallengePrepareOwnScore",
    }
    assert {"ArenaOwnScoreRun", "ArenaOwnScoreStop"} <= reached
    assert nodes["ArenaOwnScoreRun"]["next"] == []
    assert nodes["ArenaOwnScoreStop"]["next"] == []
    assert nodes[entry]["action"]["param"]["custom_action_param"]["capture_mode"] == "off"


def test_home_navigation_preserves_existing_recognition_click_and_wait_parameters():
    own = _load("assets/resource/base/pipeline/ArenaOwnScore.json")
    challenge = _load("assets/resource/base/pipeline/Challenge.json")
    for suffix in ("HomeFlag", "Click", "Enter"):
        left = deepcopy(own[f"ArenaOwnScore{suffix}"])
        right = deepcopy(challenge[f"Challenge{suffix}"])
        left.pop("next")
        right.pop("next")
        assert left == right
    assert own["ArenaOwnScoreReady"]["recognition"] == {
        "type": "Custom", "param": {
            "custom_recognition": "ArenaPipelineText",
            "custom_recognition_param": {"expected": ["リハーサル"]},
        },
    }
    assert own["ArenaOwnScoreReady"]["next"] == ["ArenaOwnScoreRun"]
    assert "challenge_battle_total.png" not in json.dumps(own)
    assert own["ArenaOwnScoreRun"]["action"]["param"]["custom_action"] == "ArenaOwnScoreRecalculate"


@pytest.mark.parametrize("status", ["calculated", "incomplete_input", "adapter_failure", "cache_failure"])
def test_own_action_calls_own_service_and_terminates_for_every_business_result(status):
    namespace = _action_namespace()
    context = _context()
    evaluation = SimpleNamespace(
        status=status, error="detail" if status != "calculated" else None,
        error_detail=None, summary_error_detail=None,
    )
    service = namespace["ArenaOwnScoreService"].return_value
    service.calculate.return_value = evaluation
    reader = namespace["ArenaLineupReader"].return_value
    read_metrics = {
        "member_count": 9 if status == "calculated" else 0,
        "duration_percentiles_seconds": {
            "skill_card_detail_transaction": {"count": 36, "p50": 2.1, "p95": 2.9}
        },
    }
    reader.last_read_metrics_summary.return_value = read_metrics
    result = namespace["ArenaOwnScoreRecalculate"]().run(context, object())
    assert result is (status == "calculated")
    assert context.requested == ["ArenaOwnScoreRun", "ChallengeSeasonConfig"]
    backend = namespace["MaaArenaReaderBackend"]
    backend.assert_called_once_with(
        context, namespace["resolve_arena_component"].return_value.season, "fixed-test-bundle"
    )
    # No known_grade is injected: read_own retains its fresh Grade-first contract.
    service.calculate.assert_called_once_with(
        reader, before_cache_save=namespace["cancellation_for"].return_value.check,
    )
    reader.last_read_metrics_summary.assert_called_once_with()
    records = [
        json.loads(call.args[0])
        for call in namespace["logger"].info.call_args_list
        if isinstance(call.args[0], str) and call.args[0].startswith("{")
    ]
    assert records == [{"event": "arena_own_read_metrics", "status": status, **read_metrics}]
    namespace["_publish_status"].assert_called_once()
    namespace["_log_own_score_evaluation"].assert_called_once()
    namespace["reset_prepared_own_score_cache"].assert_called_once()
    namespace["reset_failed_arena_component_checks"].assert_called_once_with()


@pytest.mark.parametrize("injected", [{"mode": "fixed"}, {"index": 1}, {"grade_override": 7}])
def test_own_action_rejects_mode_and_grade_injection_before_component_or_read(injected):
    namespace = _action_namespace()
    context = _context()
    context.nodes["ArenaOwnScoreRun"]["attach"].update(injected)
    assert namespace["ArenaOwnScoreRecalculate"]().run(context, object()) is False
    namespace["resolve_arena_component"].assert_not_called()
    namespace["ArenaOwnScoreService"].assert_not_called()


def test_own_action_component_failure_stops_before_reader():
    namespace = _action_namespace()
    namespace["resolve_arena_component"].side_effect = namespace["ArenaComponentError"]("offline")
    assert namespace["ArenaOwnScoreRecalculate"]().run(_context(), object()) is False
    namespace["ArenaOwnScoreService"].assert_not_called()
    namespace["_stop_with_error"].assert_called_once()
    namespace["user_logger"].info.assert_not_called()
    namespace["user_logger"].warning.assert_not_called()


def test_own_recalculation_shows_actual_preview_season_before_the_first_read():
    namespace = _action_namespace()
    context = _context()
    context.nodes["ChallengeSeasonConfig"] = {"attach": {"season": "latest"}}
    namespace["resolve_arena_component"].return_value.season = SimpleNamespace(season=52, preview=True)

    def first_read(*_args):
        namespace["user_logger"].warning.assert_called_once()
        assert "第52期（RIS预览，latest）" in namespace["user_logger"].warning.call_args.args[0]
        raise namespace["ArenaReaderError"]("first member read failed")

    namespace["MaaArenaReaderBackend"].side_effect = first_read
    assert not namespace["ArenaOwnScoreRecalculate"]().run(context, None)
    namespace["MaaArenaReaderBackend"].assert_called_once()
    namespace["ArenaOwnScoreService"].assert_not_called()
    namespace["user_logger"].warning.assert_called_once()
