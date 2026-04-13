"""Build Phase1 navigation SFT data from EB-Nav trajectories.

This script converts successful EB-Nav episodes into step-level multimodal
conversation samples for SFT. It supports both the curated single-step file and
the original multi-step file.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import Any

import pandas as pd

from verl.workers.roles.utils.action_schema import (
    ACTION_NAME_TO_TOKEN,
    ACTION_START_TOKEN,
    ACTION_END_TOKEN,
    LATENT_TOKEN,
    normalize_action_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EB-Nav step-level SFT data")
    parser.add_argument("--output", type=str, required=True, help="Output parquet/json/jsonl path")
    parser.add_argument("--history_mode", choices=("no_history", "with_history"), default="no_history")
    parser.add_argument("--include_sources", choices=("single_step", "multi_step", "both"), default="both")
    parser.add_argument("--single_step_file", type=str, default=None, help="Path to eb-nav_dataset_single_step.json")
    parser.add_argument("--multi_step_file", type=str, default=None, help="Path to eb-nav_dataset_multi_step.json")
    parser.add_argument("--image_root", type=str, default="", help="Optional root for resolving relative image paths")
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional cap on written samples")
    parser.add_argument(
        "--only_successful_actions",
        action="store_true",
        help="Keep only steps whose executable action is marked successful",
    )
    return parser.parse_args()


_VERBOSE_ACTION_NAME_MAP: dict[str, str] = {
    "move forward by 0.25": "move_forward",
    "move backward by 0.25": "move_backward",
    "move rightward by 0.25": "move_right",
    "move leftward by 0.25": "move_left",
    "rotate to the right by 90 degrees.": "turn_right",
    "rotate to the left by 90 degrees.": "turn_left",
    "tilt the camera upward by 30 degrees.": "look_up",
    "tilt the camera downward by 30 degrees.": "look_down",
}


def _normalize_eb_action_name(raw_name: str) -> str:
    normalized = normalize_action_name(raw_name)
    if normalized in ACTION_NAME_TO_TOKEN:
        return normalized

    lowered = str(raw_name).strip().lower()
    lowered = lowered.rstrip(".")
    mapped = _VERBOSE_ACTION_NAME_MAP.get(lowered) or _VERBOSE_ACTION_NAME_MAP.get(f"{lowered}.")
    if mapped is not None:
        return mapped

    if lowered.startswith("move forward"):
        return "move_forward"
    if lowered.startswith("move backward"):
        return "move_backward"
    if lowered.startswith("move rightward"):
        return "move_right"
    if lowered.startswith("move leftward"):
        return "move_left"
    if lowered.startswith("rotate to the right"):
        return "turn_right"
    if lowered.startswith("rotate to the left"):
        return "turn_left"
    if lowered.startswith("tilt the camera upward"):
        return "look_up"
    if lowered.startswith("tilt the camera downward"):
        return "look_down"

    raise ValueError(f"Unsupported EB-Nav action name: {raw_name!r}")


def _resolve_image_path(image_path: str | None, image_root: str) -> str | None:
    if not image_path:
        return None
    if os.path.isabs(image_path):
        return image_path
    if image_root:
        return os.path.abspath(os.path.join(image_root, image_path))
    return image_path


def _build_user_message(task_prompt: str, image_path: str | None) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": task_prompt}]
    if image_path:
        content.append({"type": "image", "image": image_path})
    return {"role": "user", "content": content, "loss_mask": 0}


def _build_assistant_text(cot: str, action_token: str, close_action: bool) -> str:
    cot_text = (cot or "").strip()
    response = f"<think>{cot_text}</think>{LATENT_TOKEN}{ACTION_START_TOKEN}{action_token}"
    if close_action:
        response += ACTION_END_TOKEN
    return response


def _build_assistant_message(cot: str, action_token: str, close_action: bool, loss_mask: int) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": _build_assistant_text(cot=cot, action_token=action_token, close_action=close_action),
        "loss_mask": loss_mask,
    }


def _load_json(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _expand_episode_steps(
    episode: dict[str, Any],
    source: str,
    image_root: str,
    only_successful_actions: bool,
) -> list[dict[str, Any]]:
    task_prompt = episode["input"]
    expanded_steps: list[dict[str, Any]] = []

    for traj_index, traj_step in enumerate(episode.get("trajectory", [])):
        executable_plan = traj_step.get("executable_plan") or []
        if not executable_plan:
            continue

        current_image_path = _resolve_image_path(traj_step.get("input_image_path"), image_root=image_root)
        cot = (traj_step.get("reasoning_and_reflection") or "").strip()

        for action_index, exec_step in enumerate(executable_plan):
            action_success = bool(exec_step.get("action_success", True))
            if only_successful_actions and not action_success:
                current_image_path = _resolve_image_path(exec_step.get("img_path"), image_root=image_root)
                continue

            raw_action = exec_step.get("action")
            if not isinstance(raw_action, list | tuple) or len(raw_action) < 2:
                raise ValueError(f"Unexpected executable action format: {raw_action!r}")

            gt_action_name = _normalize_eb_action_name(raw_action[1])
            gt_action_token = ACTION_NAME_TO_TOKEN[gt_action_name]
            expanded_steps.append(
                {
                    "source": source,
                    "model_name": episode.get("model_name"),
                    "eval_set": episode.get("eval_set"),
                    "episode_id": str(episode.get("episode_id")),
                    "trajectory_index": traj_index,
                    "plan_action_index": action_index,
                    "step_id": int(exec_step.get("step_id", len(expanded_steps))),
                    "success": bool(episode.get("success", False)),
                    "action_success": action_success,
                    "task_prompt": task_prompt,
                    "cot": cot,
                    "current_image_path": current_image_path,
                    "next_image_path": _resolve_image_path(exec_step.get("img_path"), image_root=image_root),
                    "gt_action_name": gt_action_name,
                    "gt_action_token": gt_action_token,
                    "env_feedback": exec_step.get("env_feedback", ""),
                }
            )
            current_image_path = _resolve_image_path(exec_step.get("img_path"), image_root=image_root)

    return expanded_steps


def _build_messages(step_records: list[dict[str, Any]], current_index: int, history_mode: str) -> list[dict[str, Any]]:
    current_step = step_records[current_index]
    if history_mode == "no_history":
        return [
            _build_user_message(current_step["task_prompt"], current_step["current_image_path"]),
            _build_assistant_message(
                cot=current_step["cot"],
                action_token=current_step["gt_action_token"],
                close_action=False,
                loss_mask=1,
            ),
        ]

    if history_mode != "with_history":
        raise ValueError(f"Unsupported history mode: {history_mode}")

    messages: list[dict[str, Any]] = []
    for idx, step in enumerate(step_records[: current_index + 1]):
        is_current = idx == current_index
        messages.append(_build_user_message(step["task_prompt"], step["current_image_path"]))
        messages.append(
            _build_assistant_message(
                cot=step["cot"],
                action_token=step["gt_action_token"],
                close_action=not is_current,
                loss_mask=1 if is_current else 0,
            )
        )
    return messages


def build_rows(
    episodes: list[dict[str, Any]],
    source: str,
    history_mode: str,
    image_root: str,
    only_successful_actions: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        if not bool(episode.get("success", False)):
            continue

        step_records = _expand_episode_steps(
            episode=episode,
            source=source,
            image_root=image_root,
            only_successful_actions=only_successful_actions,
        )
        for idx, step in enumerate(step_records):
            rows.append(
                {
                    "messages": deepcopy(_build_messages(step_records=step_records, current_index=idx, history_mode=history_mode)),
                    "enable_thinking": False,
                    "source": step["source"],
                    "model_name": step["model_name"],
                    "eval_set": step["eval_set"],
                    "episode_id": step["episode_id"],
                    "trajectory_index": step["trajectory_index"],
                    "plan_action_index": step["plan_action_index"],
                    "step_id": step["step_id"],
                    "success": step["success"],
                    "action_success": step["action_success"],
                    "history_mode": history_mode,
                    "gt_action_name": step["gt_action_name"],
                    "gt_action_token": step["gt_action_token"],
                    "input_image_path": step["current_image_path"],
                    "next_image_path": step["next_image_path"],
                    "task_prompt": step["task_prompt"],
                    "cot": step["cot"],
                    "assistant_target": _build_assistant_text(
                        cot=step["cot"],
                        action_token=step["gt_action_token"],
                        close_action=False,
                    ),
                    "env_feedback": step["env_feedback"],
                }
            )
    return rows


def _write_rows(rows: list[dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if output_path.endswith(".parquet"):
        pd.DataFrame(rows).to_parquet(output_path, index=False)
        return
    if output_path.endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return
    if output_path.endswith(".jsonl"):
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    raise ValueError(f"Unsupported output format for {output_path!r}")


def main() -> None:
    args = parse_args()

    source_to_path = {
        "single_step": args.single_step_file,
        "multi_step": args.multi_step_file,
    }
    if args.include_sources == "both":
        sources = ("single_step", "multi_step")
    else:
        sources = (args.include_sources,)

    rows: list[dict[str, Any]] = []
    for source in sources:
        dataset_path = source_to_path[source]
        if dataset_path is None:
            raise ValueError(f"Missing input path for source {source!r}")
        rows.extend(
            build_rows(
                episodes=_load_json(dataset_path),
                source=source,
                history_mode=args.history_mode,
                image_root=args.image_root,
                only_successful_actions=args.only_successful_actions,
            )
        )

    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    _write_rows(rows=rows, output_path=args.output)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
