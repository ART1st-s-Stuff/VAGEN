import math
from collections import Counter
from typing import Dict, Iterable, List


VALID_NAVIGATION_ACTIONS = (
    "moveahead",
    "moveback",
    "moveright",
    "moveleft",
    "rotateright",
    "rotateleft",
    "lookup",
    "lookdown",
)

FORBIDDEN_NAVIGATION_ACTIONS = (
    "stay",
    "stop",
    "end",
    "done",
    "terminate",
    "wait",
    "noop",
)


def summarize_action_distribution(records: Iterable[Dict]) -> Dict[str, float]:
    actions: List[str] = []
    too_many_actions = []
    format_correct = []
    format_error_types = []
    action_validity_errors = []

    for record in records:
        info = record.get("info", {})
        actions.extend(str(action).strip().lower() for action in info.get("actions", []) if str(action).strip())
        too_many_actions.append(float(bool(info.get("too_many_actions", False))))
        if "format_correct" in info:
            format_correct.append(float(bool(info["format_correct"])))
        if "format_error_type" in info:
            format_error_types.append(str(info["format_error_type"]))
        elif info.get("format_correct") is True:
            format_error_types.append("ok")
        elif info.get("format_correct") is False:
            format_error_types.append("unknown")
        metrics = info.get("metrics", {})
        turn_metrics = metrics.get("turn_metrics", {}) if isinstance(metrics, dict) else {}
        if "action_validity_error" in turn_metrics:
            action_validity_errors.append(str(turn_metrics["action_validity_error"]))

    counts = Counter(actions)
    format_error_counts = Counter(format_error_types)
    action_validity_error_counts = Counter(action_validity_errors)
    total = sum(counts.values())
    valid_count = sum(counts[action] for action in VALID_NAVIGATION_ACTIONS)
    forbidden_count = sum(counts[action] for action in FORBIDDEN_NAVIGATION_ACTIONS)
    invalid_count = max(total - valid_count, 0)
    invalid_typo_count = max(invalid_count - forbidden_count, 0)
    metrics = {
        "action/count": float(total),
        "action/top_share": 0.0,
        "action/entropy": 0.0,
        "action/all_same_traj": 0.0,
        "action/valid_vocab_rate": valid_count / total if total else 0.0,
        "action/forbidden_stay_stop_end_rate": forbidden_count / total if total else 0.0,
        "action/invalid_typo_rate": invalid_typo_count / total if total else 0.0,
        "format/too_many_actions": sum(too_many_actions) / len(too_many_actions) if too_many_actions else 0.0,
        "format/correct": sum(format_correct) / len(format_correct) if format_correct else 0.0,
    }

    if total:
        probs = [count / total for count in counts.values()]
        metrics["action/top_share"] = max(probs)
        metrics["action/entropy"] = -sum(prob * math.log(prob) for prob in probs if prob > 0)
        metrics["action/all_same_traj"] = float(total >= 2 and len(counts) == 1)

    for action in VALID_NAVIGATION_ACTIONS:
        metrics[f"action/share/{action}"] = counts[action] / total if total else 0.0

    format_error_total = len(format_error_types)
    for error_type in (
        "ok",
        "missing_or_malformed_tags",
        "empty_answer",
        "too_many_actions",
        "invalid_action_name",
        "unknown",
    ):
        metrics[f"format/error/{error_type}"] = (
            format_error_counts[error_type] / format_error_total if format_error_total else 0.0
        )

    action_validity_error_total = len(action_validity_errors)
    for error_type in ("ok", "no_action", "invalid_action_name"):
        metrics[f"action/error/{error_type}"] = (
            action_validity_error_counts[error_type] / action_validity_error_total
            if action_validity_error_total
            else 0.0
        )

    return metrics
