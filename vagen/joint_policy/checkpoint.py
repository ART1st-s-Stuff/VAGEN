"""Atomic sidecar and completion marker for exact joint-update resume."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

JOINT_CHECKPOINT_SCHEMA = "vagen_joint_training_checkpoint_v1"
JOINT_CHECKPOINT_FILENAME = "joint_training.pt"
JOINT_COMPLETION_SCHEMA = "vagen_joint_checkpoint_complete_v1"
JOINT_COMPLETION_FILENAME = "joint_checkpoint_complete.json"


def assemble_joint_checkpoint(
    *,
    global_step: int,
    run_seed: int,
    rank_exports: Sequence[Mapping[str, Any]],
    owner_checkpoint_state: Mapping[str, Any],
    expected_world_size: int,
    dataloader_sha256: str,
    training_contract_id: str,
) -> dict[str, Any]:
    step = _positive_int(global_step, "global_step")
    seed = _nonnegative_int(run_seed, "run_seed")
    world_size = _positive_int(expected_world_size, "expected_world_size")
    if (
        not isinstance(training_contract_id, str)
        or not training_contract_id.startswith("sha256:")
        or len(training_contract_id) != 71
    ):
        raise ValueError("joint checkpoint training_contract_id is invalid")
    if (
        not isinstance(dataloader_sha256, str)
        or not dataloader_sha256.startswith("sha256:")
        or len(dataloader_sha256) != 71
    ):
        raise ValueError("joint checkpoint dataloader_sha256 is invalid")
    exports = list(rank_exports)
    if len(exports) != world_size:
        raise ValueError("joint checkpoint rank export count mismatch")
    fields = {
        "rank",
        "world_size",
        "completed_updates",
        "source_step",
        "snapshot_id",
        "contract_id",
        "score_dtype",
        "optimizer_fingerprint",
        "checkpoint_payload",
    }
    records = []
    for raw in exports:
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ValueError("joint checkpoint rank export fields are invalid")
        records.append(dict(raw))
    if sorted(record["rank"] for record in records) != list(range(world_size)):
        raise ValueError("joint checkpoint rank identities are incomplete")
    if any(record["world_size"] != world_size for record in records):
        raise ValueError("joint checkpoint worker world size mismatch")
    reference = records[0]
    for field in (
        "completed_updates",
        "source_step",
        "snapshot_id",
        "contract_id",
        "score_dtype",
        "optimizer_fingerprint",
    ):
        if any(record[field] != reference[field] for record in records[1:]):
            raise ValueError(f"joint checkpoint {field} diverged across ranks")
    payloads = [
        record for record in records if record["checkpoint_payload"] is not None
    ]
    if len(payloads) != 1 or payloads[0]["rank"] != 0:
        raise ValueError("only joint checkpoint rank zero may return payload")
    actor_payload = payloads[0]["checkpoint_payload"]
    if not isinstance(actor_payload, Mapping):
        raise ValueError("joint checkpoint actor payload must be a mapping")
    for field in (
        "completed_updates",
        "source_step",
        "snapshot_id",
        "contract_id",
        "score_dtype",
    ):
        if actor_payload.get(field) != reference[field]:
            raise ValueError(f"joint checkpoint actor payload {field} mismatch")
    if (
        actor_payload.get("critic_optimizer_fingerprint")
        != reference["optimizer_fingerprint"]
    ):
        raise ValueError("joint checkpoint actor optimizer fingerprint mismatch")
    if not isinstance(owner_checkpoint_state, Mapping):
        raise ValueError("joint checkpoint frozen Q owner state must be a mapping")
    owner = dict(owner_checkpoint_state)
    if set(owner) != {
        "schema",
        "activation_version",
        "active_snapshot_state",
    }:
        raise ValueError("joint checkpoint frozen Q owner fields are invalid")
    active = owner["active_snapshot_state"]
    if not isinstance(active, Mapping):
        raise ValueError("joint checkpoint active snapshot state must be a mapping")
    for field in ("source_step", "snapshot_id", "contract_id", "score_dtype"):
        if active.get(field) != reference[field]:
            raise ValueError(f"joint checkpoint active snapshot {field} mismatch")
    if owner["activation_version"] != reference["completed_updates"]:
        raise ValueError("joint checkpoint activation version mismatch")
    return {
        "schema": JOINT_CHECKPOINT_SCHEMA,
        "global_step": step,
        "run_seed": seed,
        "world_size": world_size,
        "dataloader_sha256": dataloader_sha256,
        "training_contract_id": training_contract_id,
        "actor_critic": dict(actor_payload),
        "frozen_q_owner": owner,
    }


def save_atomic_joint_checkpoint(
    folder: str | os.PathLike[str],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != JOINT_CHECKPOINT_SCHEMA:
        raise ValueError("joint checkpoint payload schema is invalid")
    sidecar = root / JOINT_CHECKPOINT_FILENAME
    marker = root / JOINT_COMPLETION_FILENAME
    if marker.exists():
        raise FileExistsError(f"joint checkpoint completion marker exists: {marker}")
    temp_sidecar = root / f".{JOINT_CHECKPOINT_FILENAME}.tmp.{os.getpid()}"
    temp_marker = root / f".{JOINT_COMPLETION_FILENAME}.tmp.{os.getpid()}"
    try:
        with temp_sidecar.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_sidecar, sidecar)
        digest = _sha256_file(sidecar)
        completion = {
            "schema": JOINT_COMPLETION_SCHEMA,
            "global_step": payload["global_step"],
            "sidecar": JOINT_CHECKPOINT_FILENAME,
            "sidecar_sha256": digest,
            "snapshot_id": payload["actor_critic"]["snapshot_id"],
            "source_step": payload["actor_critic"]["source_step"],
            "dataloader_sha256": payload["dataloader_sha256"],
        }
        with temp_marker.open("w", encoding="utf-8") as handle:
            json.dump(completion, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_marker, marker)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return completion
    finally:
        temp_sidecar.unlink(missing_ok=True)
        temp_marker.unlink(missing_ok=True)


def load_complete_joint_checkpoint(
    folder: str | os.PathLike[str],
) -> dict[str, Any]:
    import torch

    root = Path(folder)
    marker_path = root / JOINT_COMPLETION_FILENAME
    sidecar_path = root / JOINT_CHECKPOINT_FILENAME
    with marker_path.open("r", encoding="utf-8") as handle:
        marker = json.load(handle)
    if set(marker) != {
        "schema",
        "global_step",
        "sidecar",
        "sidecar_sha256",
        "snapshot_id",
        "source_step",
        "dataloader_sha256",
    } or marker["schema"] != JOINT_COMPLETION_SCHEMA:
        raise ValueError("joint checkpoint completion marker is invalid")
    if marker["sidecar"] != JOINT_CHECKPOINT_FILENAME:
        raise ValueError("joint checkpoint marker sidecar name is invalid")
    if _sha256_file(sidecar_path) != marker["sidecar_sha256"]:
        raise ValueError("joint checkpoint sidecar digest mismatch")
    if _sha256_file(root / "data.pt") != marker["dataloader_sha256"]:
        raise ValueError("joint checkpoint dataloader digest mismatch")
    payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != JOINT_CHECKPOINT_SCHEMA:
        raise ValueError("joint checkpoint sidecar payload is invalid")
    if (
        payload.get("global_step") != marker["global_step"]
        or payload.get("actor_critic", {}).get("snapshot_id") != marker["snapshot_id"]
        or payload.get("actor_critic", {}).get("source_step") != marker["source_step"]
        or payload.get("dataloader_sha256") != marker["dataloader_sha256"]
    ):
        raise ValueError("joint checkpoint marker and sidecar identities mismatch")
    return dict(payload)


def find_latest_complete_joint_checkpoint(
    root: str | os.PathLike[str],
) -> str | None:
    base = Path(root)
    candidates = []
    for path in base.glob("global_step_*"):
        try:
            step = int(path.name.removeprefix("global_step_"))
        except ValueError:
            continue
        if (path / JOINT_COMPLETION_FILENAME).is_file():
            candidates.append((step, path))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])


def sha256_file(path: str | os.PathLike[str]) -> str:
    return _sha256_file(Path(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"joint checkpoint {field} must be non-negative int")
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result < 1:
        raise ValueError(f"joint checkpoint {field} must be positive")
    return result


__all__ = [
    "JOINT_CHECKPOINT_FILENAME",
    "JOINT_CHECKPOINT_SCHEMA",
    "JOINT_COMPLETION_FILENAME",
    "assemble_joint_checkpoint",
    "find_latest_complete_joint_checkpoint",
    "load_complete_joint_checkpoint",
    "save_atomic_joint_checkpoint",
    "sha256_file",
]
