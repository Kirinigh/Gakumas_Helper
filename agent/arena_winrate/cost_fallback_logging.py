"""Pure log payload formatting for bounded arena cost assumptions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def cost_customization_fallback_log_payloads(
    records: Sequence[Mapping[str, object]],
    *,
    side: str,
) -> tuple[dict[str, object], ...]:
    """Format per-own-card or one opponent-summary warning payload."""

    if side == "own":
        return tuple(
            {
                "event": "arena_cost_customization_fallback",
                "message": (
                    f"己方 舞台{record['stage_number']}/成员{record['member_slot']}/"
                    f"第{int(record['group_index']) + 1}组第{record['card_slot']}槽/"
                    f"卡ID {record['card_id']} 的费用强化无法唯一读取，按未强化计算"
                ),
                "fallback": dict(record),
            }
            for record in records
        )
    if side != "opponent":
        raise ValueError("cost fallback log side must be own or opponent")
    if not records:
        return ()
    locations = [
        (
            f"{int(record['opponent_position']) + 1}号对手 "
            f"舞台{record['stage_number']}-{record['member_slot']} "
            f"第{int(record['group_index']) + 1}组第{record['card_slot']}槽"
            f"(ID {record['card_id']})"
        )
        for record in records
    ]
    return (
        {
            "event": "arena_cost_customization_fallback_summary",
            "message": (
                f"本轮对手有 {len(records)} 张卡的费用强化无法唯一读取，均按已强化计算："
                + "；".join(locations)
            ),
            "count": len(records),
            "locations": locations,
            "fallbacks": [dict(record) for record in records],
        },
    )
