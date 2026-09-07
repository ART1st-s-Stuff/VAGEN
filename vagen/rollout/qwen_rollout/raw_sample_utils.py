from typing import Dict, Iterable, List


def truncate_text(value, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def collect_raw_response_samples(
    log_rst: Iterable[Dict],
    limit: int = 8,
    max_chars: int = 2000,
) -> List[Dict[str, str]]:
    if limit <= 0:
        return []

    candidates = []
    for item in log_rst:
        history = item.get("history", [])
        for turn_idx, record in enumerate(history):
            info = record.get("info", {})
            raw_response = info.get("llm_raw_response")
            if raw_response is None:
                continue
            metrics = info.get("metrics", {})
            turn_metrics = metrics.get("turn_metrics", {}) if isinstance(metrics, dict) else {}
            format_correct = bool(info.get("format_correct", turn_metrics.get("format_correct", False)))
            action_is_valid = bool(turn_metrics.get("action_is_valid", bool(info.get("actions"))))
            too_many_actions = bool(info.get("too_many_actions", turn_metrics.get("too_many_actions", False)))
            priority = 0 if (not format_correct or not action_is_valid or too_many_actions) else 1
            candidates.append(
                (
                    priority,
                    {
                        "env_id": str(item.get("env_id", "")),
                        "config_id": str(item.get("config_id", "")),
                        "turn": str(turn_idx),
                        "raw_response": truncate_text(raw_response, max_chars),
                        "llm_response": truncate_text(info.get("llm_response", ""), max_chars),
                        "think_content": truncate_text(info.get("think_content", ""), max_chars),
                        "action_content": truncate_text(info.get("action_content", ""), max_chars),
                        "actions": ", ".join(str(action) for action in info.get("actions", [])),
                        "format_correct": str(format_correct),
                        "format_error_type": str(info.get("format_error_type", "unknown")),
                        "too_many_actions": str(too_many_actions),
                        "action_is_valid": str(action_is_valid),
                        "action_validity_error": str(turn_metrics.get("action_validity_error", "unknown")),
                        "reward": str(record.get("reward", "")),
                        "done": str(record.get("done", "")),
                        "instruction": truncate_text(info.get("instruction", ""), max_chars),
                    },
                )
            )

    candidates.sort(key=lambda pair: pair[0])
    return [sample for _, sample in candidates[:limit]]
