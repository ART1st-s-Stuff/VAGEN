"""Evidence-backed source-behavior reconstruction for the hligb step60 run.

This module does not claim to be the unavailable source commit. Its prompt and
reward behavior are bound to the committed reconstruction evidence artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

STEP60_RECONSTRUCTION_MODE = "step60_source_reconstruction"
RECONSTRUCTION_BASE_COMMIT = "3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a"
EVIDENCE_FORMAT = "vagen_step60_reconstruction_evidence_v1"
SOURCE_ACTION_NAMES = (
    "moveahead",
    "moveback",
    "moveright",
    "moveleft",
    "rotateright",
    "rotateleft",
    "lookup",
    "lookdown",
)
_EVIDENCE_PATH = (
    Path(__file__).with_name("evidence")
    / "vagen_step60_reconstruction_evidence_v1.json"
)
_STRICT_RE = re.compile(
    r"^<think><observation>(.*?)</observation>"
    r"<reasoning>(.*?)</reasoning>"
    r"<prediction>(.*?)</prediction></think>"
    r"<answer>(.*?)</answer>$",
    re.DOTALL,
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_evidence() -> dict[str, Any]:
    evidence = json.loads(_EVIDENCE_PATH.read_text(encoding="utf-8"))
    if evidence.get("format") != EVIDENCE_FORMAT:
        raise ValueError("step60 reconstruction evidence format mismatch")
    claimed = evidence.get("manifest_sha256")
    payload = {key: value for key, value in evidence.items() if key != "manifest_sha256"}
    if claimed != _canonical_sha256(payload):
        raise ValueError("step60 reconstruction evidence manifest hash mismatch")
    if evidence.get("reward_counts") != {
        "-0.2": 596,
        "0": 35,
        "0.02": 6361,
        "10.02": 359,
    }:
        raise ValueError("step60 reconstruction reward evidence mismatch")
    return evidence


_EVIDENCE = _load_evidence()
_TEMPLATES = _EVIDENCE["prompt_templates"]
_EVIDENCE_FILE_SHA256 = hashlib.sha256(_EVIDENCE_PATH.read_bytes()).hexdigest()
_EXPECTED_ENVIRONMENT_CONFIG = {
    "render_mode": "vision",
    "prompt_format": STEP60_RECONSTRUCTION_MODE,
    "use_state_reward": False,
    "max_actions_per_step": 1,
    "format_reward": 0.02,
    "invalid_action_penalty": -0.2,
    "success_threshold": 1.5,
    "step_length": 0.5,
    "success_reward": 10.0,
}
_ENVIRONMENT_ASSETS = {
    "base": {
        "rows": 60,
        "sha256": "6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a",
    },
    "common_sense": {
        "rows": 60,
        "sha256": "3e7d2cb4246b6e2edaeaabd318dba93e4dbbff114c8368ed0c862e64f417afcf",
    },
}


def validate_environment_config(config: dict[str, Any]) -> None:
    expected_fields = {*_EXPECTED_ENVIRONMENT_CONFIG, "eval_set", "gpu_device"}
    if set(config) != expected_fields:
        raise ValueError("step60 reconstruction environment fields drift")
    if config.get("eval_set") not in {"base", "common_sense"}:
        raise ValueError("step60 reconstruction eval_set is unsupported")
    gpu_device = config.get("gpu_device")
    if isinstance(gpu_device, bool) or not isinstance(gpu_device, int) or gpu_device < 0:
        raise ValueError("step60 reconstruction gpu_device is invalid")
    for key, expected in _EXPECTED_ENVIRONMENT_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"step60 reconstruction environment drift: {key}")


def _computed_environment_assets(root: Path) -> dict[str, Any]:
    assets = {}
    dataset_root = root / "vagen/env/navigation/datasets"
    for eval_set, expected in _ENVIRONMENT_ASSETS.items():
        path = dataset_root / f"{eval_set}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        tasks = payload.get("tasks")
        actual = {
            "rows": len(tasks) if isinstance(tasks, list) else -1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if actual != expected:
            raise ValueError(f"reconstruction environment asset drift: {eval_set}")
        assets[eval_set] = actual
    return assets


def service_runtime_identity(*, service_routes: list[str]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    if status:
        raise ValueError("reconstruction service worktree is dirty")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    parent = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD^"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    diff = subprocess.check_output(
        [
            "git", "-C", str(root), "--no-pager", "diff", "--binary",
            "--full-index", "--no-ext-diff", f"{RECONSTRUCTION_BASE_COMMIT}..HEAD", "--",
        ]
    )
    return {
        "reconstruction_identity": {
            "base_commit": RECONSTRUCTION_BASE_COMMIT,
            "runtime_head": head,
            "runtime_parent": parent,
            "runtime_tree": tree,
            "commit_count": int(
                subprocess.check_output(
                    ["git", "-C", str(root), "rev-list", "--count", f"{RECONSTRUCTION_BASE_COMMIT}..HEAD"],
                    text=True,
                ).strip()
            ),
            "parent_count": len(
                subprocess.check_output(
                    ["git", "-C", str(root), "rev-list", "--parents", "-n", "1", "HEAD"],
                    text=True,
                ).split()
            ) - 1,
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
            "git_version": subprocess.check_output(
                ["git", "--version"], text=True
            ).strip(),
        },
        "evidence_artifact": {
            "sha256": _EVIDENCE_FILE_SHA256,
            "manifest_sha256": _EVIDENCE["manifest_sha256"],
        },
        "environment_assets": _computed_environment_assets(root),
        "environment_config": {
            **_EXPECTED_ENVIRONMENT_CONFIG,
            "source_prompt_format": "grounding_worldmodeling",
            "action_names": list(SOURCE_ACTION_NAMES),
        },
        "service_api_contract": "legacy_batch_environment_v1",
        "service_routes": sorted(service_routes),
    }


def render_system_prompt() -> str:
    return str(_TEMPLATES["system"])


def render_initial_prompt(*, observation: str, instruction: str) -> str:
    template = str(_TEMPLATES["initial"])
    if template.count("<image>") != 1 or template.count("<INSTRUCTION>") != 1:
        raise ValueError("invalid reconstructed initial prompt template")
    return template.replace("<image>", observation).replace(
        "<INSTRUCTION>", instruction
    )


def render_step_prompt(
    *,
    extracted_actions: Any,
    env_feedback: str,
    reward: Any,
    done: Any,
    observation: str,
    instruction: str,
) -> str:
    template = str(_TEMPLATES["post_step"])
    replacements = {
        "<ACTIONS>": str(extracted_actions),
        "<FEEDBACK>": str(env_feedback),
        "<REWARD>": str(reward),
        "<DONE>": str(done),
        "<image>": observation,
        "<INSTRUCTION>": instruction,
    }
    for placeholder, value in replacements.items():
        if template.count(placeholder) != 1:
            raise ValueError(f"invalid reconstructed step template: {placeholder}")
        template = template.replace(placeholder, value)
    return template


def classify_response(response: str, *, max_actions: int = 1) -> dict[str, Any]:
    """Classify strict source responses without truncating extra actions."""

    if not isinstance(response, str):
        raise TypeError("step60 response must be text")
    match = _STRICT_RE.fullmatch(response)
    if match is None:
        return {
            "actions": [],
            "format_correct": False,
            "error_kind": "missing_or_malformed_tags",
        }
    if not all(match.group(index).strip() for index in (1, 2, 3)):
        return {
            "actions": [],
            "format_correct": False,
            "error_kind": "missing_or_malformed_tags",
        }
    answer = match.group(4)
    if "," not in answer and answer != answer.strip():
        return {
            "actions": [],
            "format_correct": False,
            "error_kind": "missing_or_malformed_tags",
        }
    raw_actions = answer.split(",")
    if any(not item.strip() for item in raw_actions):
        return {
            "actions": [],
            "format_correct": False,
            "error_kind": "missing_or_malformed_tags",
        }
    actions = [item.strip() for item in raw_actions]
    if len(actions) > int(max_actions):
        if actions and all(action in SOURCE_ACTION_NAMES for action in actions):
            error_kind = "too_many_actions"
        else:
            error_kind = "invalid_action_name"
        return {
            "actions": [],
            "format_correct": False,
            "error_kind": error_kind,
        }
    if len(actions) != 1 or actions[0] not in SOURCE_ACTION_NAMES:
        return {
            "actions": [],
            "format_correct": False,
            "error_kind": "invalid_action_name",
        }
    return {
        "actions": actions,
        "format_correct": True,
        "error_kind": "ok",
    }


def parse_response(
    response: str,
    special_token_list=None,
    action_sep=",",
    max_actions=1,
) -> dict[str, Any]:
    # Legacy BaseEnvConfig supplies tag names here. The strict parser does not
    # rewrite response bytes or extracted contents with that list.
    del special_token_list
    if action_sep != ",":
        raise ValueError("step60 reconstruction action separator must be ','")
    parsed = classify_response(response, max_actions=max_actions)
    return {
        "llm_raw_response": response,
        "llm_response": response if parsed["format_correct"] else "",
        "action_content": parsed["actions"][0] if parsed["actions"] else "",
        **parsed,
    }


def turn_reward(parsed: dict[str, Any], *, success: bool) -> float:
    if parsed.get("format_correct") and parsed.get("actions"):
        return 10.02 if success else 0.02
    if parsed.get("error_kind") == "too_many_actions":
        return 0.0
    return -0.2
