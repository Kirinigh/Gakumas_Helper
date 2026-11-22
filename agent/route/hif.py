"""HIF large-deck route slice, translated from audited hiro behaviour.

This owns scene routing only. Card/deck changes require the common identity and
deck-ledger producers; Live strategy remains the battle adapter's responsibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .engine import Outcome, Decision, Observation, propose_unique, validate_observation

PRESET_ID = "hif-main-current-v1"
SUPPORTED_IDOL = "篠澤広(ガラクタロード)"
ATTRIBUTES = frozenset({"Vo", "Da", "Vi"})
BATTLE_SCENES = frozenset({"battle_entry", "live", "battle_result", "retry"})
MANUAL_SCENES = frozenset({"card_choice", "card_exchange", "card_delete", "card_customize", "event_choice", "supply"})
HOME_SCENES = ("lesson", "training", "outing", "consult_entry", "rest")


@dataclass(frozen=True)
class HifConfig:
    schema_version: int = 1
    mode: str = "off"
    preset: str = PRESET_ID
    battle_policy: str | None = None
    customization_enabled: bool | None = None
    main_attribute: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict) -> HifConfig:
        if not isinstance(raw, dict) or set(raw) - cls.__dataclass_fields__.keys():
            raise ValueError("unsupported HIF configuration fields")
        result = cls(**raw)
        if type(result.schema_version) is not int or result.schema_version != 1:
            raise ValueError("unsupported HIF schema version")
        if result.mode not in {"off", "observe", "advance"} or result.preset != PRESET_ID:
            raise ValueError("unsupported HIF mode or preset")
        from .battle_policy import selected_profile
        selected_profile(result.battle_policy)
        if result.customization_enabled is not None and type(result.customization_enabled) is not bool:
            raise ValueError('HIF customization switch must be explicit or inherited')
        if result.main_attribute is not None and (
                not isinstance(result.main_attribute, str) or result.main_attribute not in ATTRIBUTES):
            raise ValueError('unsupported HIF main attribute')
        return result


def training_main_attribute(context):
    read = getattr(context, "get_node_data", None)
    settings = read("ProduceHifMainAttributeSettings") if callable(read) else None
    attribute = ((settings or {}).get("attach") or {}).get("attribute")
    if attribute is not None and (not isinstance(attribute, str) or attribute not in ATTRIBUTES):
        raise ValueError("unsupported HIF main attribute")
    return attribute


def requested_qualifier_memory(context):
    read = getattr(context, "get_node_data", None)
    settings = read("ProduceHifQualifierMemorySettings") if callable(read) else None
    stats = ((settings or {}).get("attach") or {}).get("stats", "")
    if stats == "":
        return None
    if not isinstance(stats, str) or not re.fullmatch(r"[1-9][0-9]{2,3}(?:,[1-9][0-9]{2,3}){2}", stats):
        raise ValueError("HIF qualifier memory requires Vo,Da,Vi")
    return tuple(int(value) for value in stats.split(","))


def day_from_countdown(value: int | None) -> int | None:
    """Unknown countdown never means the second lesson or the final day."""
    return 7 - value if type(value) is int and 0 <= value <= 6 else None


def known_hif_role(idol, plan):
    from live_decision.hif_profiles import supported_hif_identity

    try:
        supported_hif_identity(idol, plan)
    except ValueError:
        return False
    return True


def active_hif_role(host):
    """Use the role bound to the same cultivation's creation or notebook."""
    from .new_run import start_record

    run_id = host['run_id'] if isinstance(host, dict) else host.run_id
    handoff = (host.get('handoff') if isinstance(host, dict) else getattr(host, 'handoff', None)) or {}
    origin = start_record(handoff) or {}
    selected = tuple((origin.get('request') or {}).get('identity') or ())
    while isinstance(handoff, dict):
        for source in (handoff.get('recovery_identity'), handoff.get('live_context'), handoff):
            if not isinstance(source, dict) or source.get('run_id', run_id) != run_id:
                continue
            role = source.get('idol'), source.get('plan')
            if known_hif_role(*role):
                if selected and selected != role:
                    raise ValueError('HIF notebook role differs from the confirmed start')
                return role
        handoff = handoff.get('nonbattle_context')
    if len(selected) == 2 and known_hif_role(*selected):
        return selected
    # Preliminary route navigation has no confirmed identity yet. The start
    # transaction reads the selected role independently before spending AP.
    return SUPPORTED_IDOL, 'sense'


def decide_hif(observation: Observation, config: HifConfig = HifConfig()) -> Decision:
    config = HifConfig.from_mapping(vars(config))
    if config.mode == "off":
        return Decision(Outcome.DISABLED, "hif_disabled")
    error = validate_observation(observation)
    if error:
        return Decision(Outcome.STOP, error)
    if observation.scenario != "HIF" or not known_hif_role(observation.idol, observation.plan):
        return Decision(Outcome.STOP, "unsupported_or_unknown_combination")
    if observation.scene in BATTLE_SCENES:
        return Decision(Outcome.PAUSE, "battle_requires_task275")
    if observation.scene in MANUAL_SCENES:
        return Decision(Outcome.PAUSE, f"{observation.scene}_requires_verified_selection")
    if observation.scene == "complete":
        return Decision(Outcome.PAUSE, "hif_complete_no_repeat")
    if observation.day is None:
        return Decision(Outcome.STOP, "unknown_day")

    def select(kind, expected, *, attribute=None, special=None):
        candidates = tuple(
            target for target in observation.targets
            if target.kind == kind and (attribute is None or target.attribute == attribute)
            and (special is None or target.special is special)
        )
        return propose_unique(observation, candidates, f"{PRESET_ID}:{observation.day}:{kind}", expected)

    if observation.scene == "lesson" and observation.day in {1, 4}:
        destinations = ("lesson", "lesson_selected", "lesson_choice", "event_choice", "card_choice", "training", "drink_offer")
        if observation.day == 1:
            destinations += ("support_reward",)
        return select("lesson", destinations, attribute="Vi")
    if observation.scene in {"lesson_choice", "lesson_choice_preview"} and observation.day in {1, 4}:
        # User-approved temporary Sense policy. Ordinal keys must come from
        # a verified lesson-choice producer, never from generic event OCR.
        options = tuple(target for target in observation.targets if target.kind == "lesson_option")
        allowed = {"lesson_option_1", "lesson_option_2", "lesson_option_3"}
        keys = {target.key for target in options}
        if (len(options) != len(keys) or 'lesson_option_1' not in keys or not keys <= allowed
                or any(type(target.available) is not bool for target in options if target.key == 'lesson_option_1')):
            return Decision(Outcome.STOP, "incomplete_lesson_options")
        first = tuple(target for target in options if target.key == "lesson_option_1")
        expected = ("lesson_choice_preview", "card_choice", "card_exchange", "next", "training")
        return propose_unique(observation, first, "sense-lesson-first-temporary-v1", expected)
    if observation.scene == "training" and observation.day in {2, 5}:
        training = tuple(target for target in observation.targets if target.kind == "training")
        # SP absence must be observed independently for all three attributes.
        if len(training) != 3 or {target.attribute for target in training} != ATTRIBUTES:
            return Decision(Outcome.STOP, "incomplete_training_options")
        if any(type(target.special) is not bool or type(target.available) is not bool for target in training):
            return Decision(Outcome.STOP, "unknown_training_availability_or_sp")
        preferred = next((target for target in training
            if target.attribute == config.main_attribute and target.available), None)
        if preferred is not None:
            return select("training", ("training_selected", "battle_entry", "live"),
                attribute=preferred.attribute, special=preferred.special)
        for attribute in ("Da", "Vo", "Vi"):
            if any(target.attribute == attribute and target.special and target.available for target in training):
                return select("training", ("training_selected", "battle_entry", "live"), attribute=attribute, special=True)
        return select("training", ("training_selected", "battle_entry", "live"), attribute="Vo", special=False)
    if observation.scene == "training_selected" and observation.day in {2, 5}:
        # The producer must preserve the selected attribute/SP identity and
        # verify the preview before exposing this confirmation target.
        return select("confirm_training", ("training_result", "battle_entry", "live"))
    if observation.scene == "training_result" and observation.day in {2, 5}:
        # Public lessons have a non-battle parameter receipt. Its confirmed
        # enclosing scene is required; generic text matching cannot enable it.
        return select("finish_training_result", ("next", "card_choice", "drink_offer", *HOME_SCENES))
    if observation.scene == "outing" and observation.day == 3:
        return select("outing", ("event_choice", "card_choice", "drink_offer", "lesson"))
    if observation.scene == "consult_entry" and observation.day == 6:
        return select("consult", ("consult",))
    if observation.scene == "consult" and observation.day == 6:
        return select("finish_consult", ("battle_entry", "live", "next", "card_choice"))
    if observation.scene == "drink_offer":
        return select("decline_drink", ("drink_decline_confirm", "next", *HOME_SCENES))
    if observation.scene == "drink_decline_confirm":
        return select("confirm_decline_drink", ("next", *HOME_SCENES))
    if observation.scene == "next":
        # The producer must identify the enclosing result page; a generic Next
        # OCR hit alone is not a scene identity or permission to continue.
        return select("next", (*HOME_SCENES, "next", "battle_entry", "live", "card_choice", "drink_offer"))
    return Decision(Outcome.STOP, "unsupported_scene_or_day")
