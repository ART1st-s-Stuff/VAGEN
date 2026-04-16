#!/usr/bin/env python3
"""Build Phase1 rollout testset and evaluate format/action-prior accuracy.

Evaluation protocol:
1) Select the last N successful rollouts from downloaded EB-Nav datasets.
2) Expand them into step-level samples (same action schema as Phase1).
3) Run model generation for each step sample.
4) Truncate at the first emitted action token, then:
   - check whether the prefix format matches:
     <think>...</think><|latent|><|action_start|>
   - check whether the first action token matches GT action token.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList

from verl.workers.roles.utils.action_schema import (
    ACTION_NAME_TO_TOKEN,
    ACTION_TOKEN_TO_NAME,
    ACTION_START_TOKEN,
    LATENT_TOKEN,
    normalize_action_name,
)


FORMAT_PREFIX_PATTERN = re.compile(
    rf"^\s*<think>.*?</think>\s*{re.escape(LATENT_TOKEN)}\s*{re.escape(ACTION_START_TOKEN)}\s*$",
    re.DOTALL,
)

VERBOSE_ACTION_NAME_MAP: dict[str, str] = {
    "move forward by 0.25": "move_forward",
    "move backward by 0.25": "move_backward",
    "move rightward by 0.25": "move_right",
    "move leftward by 0.25": "move_left",
    "rotate to the right by 90 degrees": "turn_right",
    "rotate to the left by 90 degrees": "turn_left",
    "tilt the camera upward by 30 degrees": "look_up",
    "tilt the camera downward by 30 degrees": "look_down",
}


@dataclass
class EvalSample:
    source: str
    episode_id: str
    trajectory_index: int
    plan_action_index: int
    step_id: int
    task_prompt: str
    cot: str
    image_path: str
    gt_action_name: str
    gt_action_token: str


@dataclass
class EvalResult:
    source: str
    episode_id: str
    trajectory_index: int
    plan_action_index: int
    step_id: int
    gt_action_name: str
    gt_action_token: str
    pred_action_name: str | None
    pred_action_token: str | None
    action_prior_correct: bool
    format_correct: bool
    prefix_before_action_token: str
    visible_model_output: str
    model_output: str
    model_output_token_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase1 rollout testset + eval")
    parser.add_argument(
        "--single-step-file",
        type=str,
        default="datasets/navigation/eb-nav_dataset_single_step.json",
    )
    parser.add_argument(
        "--multi-step-file",
        type=str,
        default="datasets/navigation/eb-nav_dataset_multi_step.json",
    )
    parser.add_argument(
        "--include-sources",
        choices=("single_step", "multi_step", "both"),
        default="single_step",
    )
    parser.add_argument("--image-root", type=str, default="datasets/navigation/images")
    parser.add_argument("--num-rollouts", type=int, default=100)
    parser.add_argument("--model-path", type=str, default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help="Base model path when --model-path is a LoRA adapter directory.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size for evaluation.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs/phase1_rollout_eval")
    parser.add_argument("--save-testset", action="store_true")
    return parser.parse_args()


def _read_json(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_image_path(image_path: str | None, image_root: str) -> str | None:
    if not image_path:
        return None
    if os.path.isabs(image_path):
        return image_path
    primary = os.path.abspath(os.path.join(image_root, image_path))
    if os.path.exists(primary):
        return primary
    fallback = os.path.abspath(os.path.join(image_root, "images", image_path))
    if os.path.exists(fallback):
        return fallback
    return primary


def _normalize_eb_action_name(raw_name: str) -> str:
    normalized = normalize_action_name(raw_name)
    if normalized in ACTION_NAME_TO_TOKEN:
        return normalized
    lowered = str(raw_name).strip().lower().rstrip(".")
    mapped = VERBOSE_ACTION_NAME_MAP.get(lowered)
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


def _collect_success_rollouts(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_to_path = {
        "single_step": args.single_step_file,
        "multi_step": args.multi_step_file,
    }
    if args.include_sources == "both":
        sources = ("single_step", "multi_step")
    else:
        sources = (args.include_sources,)

    merged: list[dict[str, Any]] = []
    for source in sources:
        episodes = _read_json(source_to_path[source])
        for ep in episodes:
            payload = dict(ep)
            payload["_source"] = source
            merged.append(payload)

    success_rollouts = [ep for ep in merged if bool(ep.get("success", False))]
    if len(success_rollouts) < args.num_rollouts:
        raise ValueError(
            f"Successful rollouts ({len(success_rollouts)}) < required num_rollouts ({args.num_rollouts})."
        )
    return success_rollouts[-args.num_rollouts :]


def _build_eval_samples(rollouts: list[dict[str, Any]], image_root: str) -> list[EvalSample]:
    samples: list[EvalSample] = []
    for episode in rollouts:
        source = str(episode.get("_source", "unknown"))
        episode_id = str(episode.get("episode_id"))
        task_prompt = str(episode.get("input", ""))

        for traj_idx, traj in enumerate(episode.get("trajectory", [])):
            executable_plan = traj.get("executable_plan") or []
            if not executable_plan:
                continue
            current_image_path = _resolve_image_path(traj.get("input_image_path"), image_root=image_root)
            cot = str(traj.get("reasoning_and_reflection") or "").strip()

            for plan_idx, exec_step in enumerate(executable_plan):
                raw_action = exec_step.get("action")
                if not isinstance(raw_action, (list, tuple)) or len(raw_action) < 2:
                    raise ValueError(f"Unexpected executable action format: {raw_action!r}")
                action_name = _normalize_eb_action_name(raw_action[1])
                action_token = ACTION_NAME_TO_TOKEN[action_name]
                if current_image_path is None:
                    raise ValueError(
                        f"Missing image path for episode={episode_id}, traj={traj_idx}, action_idx={plan_idx}"
                    )
                samples.append(
                    EvalSample(
                        source=source,
                        episode_id=episode_id,
                        trajectory_index=traj_idx,
                        plan_action_index=plan_idx,
                        step_id=int(exec_step.get("step_id", len(samples))),
                        task_prompt=task_prompt,
                        cot=cot,
                        image_path=current_image_path,
                        gt_action_name=action_name,
                        gt_action_token=action_token,
                    )
                )
                current_image_path = _resolve_image_path(exec_step.get("img_path"), image_root=image_root)
    return samples


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _pick_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "fp32":
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _has_hf_weights(model_dir: str) -> bool:
    p = Path(model_dir)
    if (p / "model.safetensors").is_file():
        return True
    if (p / "model.safetensors.index.json").is_file():
        return True
    if (p / "pytorch_model.bin").is_file():
        return True
    if (p / "pytorch_model.bin.index.json").is_file():
        return True
    return any(p.glob("model-*.safetensors"))


def _has_lora_adapter(model_dir: str) -> bool:
    p = Path(model_dir)
    if not (p / "adapter_config.json").is_file():
        return False
    if (p / "adapter_model.safetensors").is_file() or (p / "adapter_model.bin").is_file():
        return True
    return any(p.glob("adapter_model-*.safetensors"))


def _has_processor_files(model_dir: str) -> bool:
    p = Path(model_dir)
    required_any = (
        p / "tokenizer.json",
        p / "tokenizer_config.json",
        p / "preprocessor_config.json",
    )
    return any(x.is_file() for x in required_any)


def _looks_like_local_path(path: str) -> bool:
    return path.startswith("/") or path.startswith("./") or path.startswith("../")


def _resolve_existing_local_path(path: str) -> str | None:
    candidates: list[str] = []
    candidates.append(path)
    candidates.append(str(Path(path).expanduser()))
    candidates.append(str(Path(path).expanduser().resolve(strict=False)))

    # Common HPC/container mount remap candidates.
    remap_rules = (
        ("/project/peilab/atst/VAGEN", "/project/VAGEN"),
        ("/project/peilab/atst", "/project"),
    )
    for src, dst in remap_rules:
        if path.startswith(src):
            remapped = path.replace(src, dst, 1)
            candidates.append(remapped)
            candidates.append(str(Path(remapped).expanduser()))

    for cand in candidates:
        if Path(cand).exists():
            return str(Path(cand))
    return None


def _normalize_model_path(path: str, arg_name: str) -> str:
    if _looks_like_local_path(path):
        resolved = _resolve_existing_local_path(path)
        if resolved is None:
            raise ValueError(
                f"{arg_name} points to a local path but it does not exist in current runtime: {path}. "
                "If running inside container, make sure host path is mounted correctly (e.g. /project/peilab/atst -> /project)."
            )
        return resolved
    return path


def _prepare_lora_adapter_for_peft(adapter_dir: str) -> str:
    adapter_path = Path(adapter_dir)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        return adapter_dir

    with open(config_path, encoding="utf-8") as f:
        adapter_cfg = json.load(f)

    target_modules = adapter_cfg.get("target_modules")
    if isinstance(target_modules, list):
        # Remove numeric path fragments (e.g. "0", "2"), which can make PEFT
        # match container modules like VisionBlock instead of Linear layers.
        filtered: list[str] = []
        seen: set[str] = set()
        for item in target_modules:
            token = str(item)
            if token.isdigit():
                continue
            if token not in seen:
                seen.add(token)
                filtered.append(token)
        adapter_cfg["target_modules"] = filtered

    if int(adapter_cfg.get("lora_alpha", 0)) <= 0:
        rank = int(adapter_cfg.get("r", 16))
        adapter_cfg["lora_alpha"] = rank
        warnings.warn(
            f"adapter_config has non-positive lora_alpha, fallback to r={rank} for inference.",
            stacklevel=1,
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="phase1_lora_adapter_"))
    shutil.copy2(adapter_path / "adapter_model.safetensors", tmp_dir / "adapter_model.safetensors")
    with open(tmp_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_cfg, f, ensure_ascii=False, indent=2)
    return str(tmp_dir)


def _load_eval_model(
    model_path: str,
    base_model_path: str | None,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Qwen2_5_VLForConditionalGeneration, str]:
    model_path = _normalize_model_path(model_path, "--model-path")
    if base_model_path is not None:
        base_model_path = _normalize_model_path(base_model_path, "--base-model-path")

    if _has_hf_weights(model_path):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
            trust_remote_code=True,
        )
        return model.to(device), model_path

    if _has_lora_adapter(model_path):
        if not base_model_path:
            raise ValueError(
                "Detected LoRA adapter files in --model-path, but --base-model-path is missing."
            )
        from peft import PeftModel

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            local_files_only=True,
            trust_remote_code=True,
        )
        adapter_path_for_loading = _prepare_lora_adapter_for_peft(model_path)
        model = PeftModel.from_pretrained(base_model, adapter_path_for_loading, local_files_only=True)
        # Prefer tokenizer/processor from exported checkpoint dir (parent of lora_adapter),
        # because it may include project-specific special tokens.
        adapter_parent = str(Path(model_path).parent)
        processor_source = adapter_parent if _has_processor_files(adapter_parent) else base_model_path
        return model.to(device), processor_source

    raise ValueError(
        f"Cannot load model from {model_path}. "
        "Expected full HF weights (model.safetensors/model-*.safetensors) "
        "or LoRA adapter files (adapter_config.json + adapter_model.safetensors)."
    )


def _build_processor_inputs(processor: Any, sample: EvalSample, device: torch.device) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample.task_prompt},
                {"type": "image", "image": sample.image_path},
            ],
        }
    ]
    try:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    image = Image.open(sample.image_path).convert("RGB")
    model_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    return {k: v.to(device) for k, v in model_inputs.items()}


def _build_batched_processor_inputs(
    processor: Any, samples: list[EvalSample], device: torch.device
) -> tuple[dict[str, torch.Tensor], list[int]]:
    prompt_texts: list[str] = []
    images: list[Any] = []
    input_lens: list[int] = []

    for sample in samples:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample.task_prompt},
                    {"type": "image", "image": sample.image_path},
                ],
            }
        ]
        try:
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_texts.append(prompt_text)
        images.append(Image.open(sample.image_path).convert("RGB"))

    model_inputs = processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)
    for i in range(model_inputs["input_ids"].shape[0]):
        # attention_mask sum gives effective prompt length for each sample.
        input_lens.append(int(model_inputs["attention_mask"][i].sum().item()))
    return {k: v.to(device) for k, v in model_inputs.items()}, input_lens


def _extract_first_action_token(text: str) -> tuple[str | None, int]:
    first_token = None
    first_pos = -1
    for tok in ACTION_TOKEN_TO_NAME:
        pos = text.find(tok)
        if pos >= 0 and (first_pos < 0 or pos < first_pos):
            first_pos = pos
            first_token = tok
    return first_token, first_pos


class _StopOnActionToken(StoppingCriteria):
    """Stop generation once any action token is generated."""

    def __init__(self, action_token_ids: set[int]):
        super().__init__()
        self.action_token_ids = action_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] == 0 or input_ids.shape[0] == 0:
            return False
        # In batched generation, stop only when all rows have reached an action token.
        for row in range(input_ids.shape[0]):
            last_token_id = int(input_ids[row, -1].item())
            if last_token_id not in self.action_token_ids:
                return False
        return True


def _resolve_action_token_ids(tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    for token in ACTION_TOKEN_TO_NAME:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    return ids


def _evaluate_one_sample(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Any,
    sample: EvalSample,
    device: torch.device,
    max_new_tokens: int,
    action_token_ids: set[int],
) -> EvalResult:
    model_inputs = _build_processor_inputs(processor, sample, device=device)
    input_len = int(model_inputs["input_ids"].shape[1])
    with torch.no_grad():
        stopping_criteria = None
        if action_token_ids:
            stopping_criteria = StoppingCriteriaList([_StopOnActionToken(action_token_ids)])
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
    new_token_ids = generated[:, input_len:]
    generated_ids = new_token_ids[0].tolist()
    output_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
    visible_output_text = output_text.encode("unicode_escape").decode("utf-8")

    pred_action_token, action_pos = _extract_first_action_token(output_text)
    prefix = output_text[:action_pos] if action_pos >= 0 else output_text
    format_correct = bool(pred_action_token) and bool(FORMAT_PREFIX_PATTERN.match(prefix))
    pred_action_name = ACTION_TOKEN_TO_NAME.get(pred_action_token) if pred_action_token else None
    action_prior_correct = pred_action_token == sample.gt_action_token

    return EvalResult(
        source=sample.source,
        episode_id=sample.episode_id,
        trajectory_index=sample.trajectory_index,
        plan_action_index=sample.plan_action_index,
        step_id=sample.step_id,
        gt_action_name=sample.gt_action_name,
        gt_action_token=sample.gt_action_token,
        pred_action_name=pred_action_name,
        pred_action_token=pred_action_token,
        action_prior_correct=action_prior_correct,
        format_correct=format_correct,
        prefix_before_action_token=prefix,
        visible_model_output=visible_output_text,
        model_output=output_text,
        model_output_token_ids=generated_ids,
    )


def _evaluate_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Any,
    samples: list[EvalSample],
    device: torch.device,
    max_new_tokens: int,
    action_token_ids: set[int],
) -> list[EvalResult]:
    model_inputs, input_lens = _build_batched_processor_inputs(processor, samples, device=device)
    with torch.no_grad():
        stopping_criteria = None
        if action_token_ids:
            stopping_criteria = StoppingCriteriaList([_StopOnActionToken(action_token_ids)])
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

    results: list[EvalResult] = []
    for idx, sample in enumerate(samples):
        new_token_ids = generated[idx, input_lens[idx] :]
        generated_ids = new_token_ids.tolist()
        output_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
        visible_output_text = output_text.encode("unicode_escape").decode("utf-8")

        pred_action_token, action_pos = _extract_first_action_token(output_text)
        prefix = output_text[:action_pos] if action_pos >= 0 else output_text
        format_correct = bool(pred_action_token) and bool(FORMAT_PREFIX_PATTERN.match(prefix))
        pred_action_name = ACTION_TOKEN_TO_NAME.get(pred_action_token) if pred_action_token else None
        action_prior_correct = pred_action_token == sample.gt_action_token

        results.append(
            EvalResult(
                source=sample.source,
                episode_id=sample.episode_id,
                trajectory_index=sample.trajectory_index,
                plan_action_index=sample.plan_action_index,
                step_id=sample.step_id,
                gt_action_name=sample.gt_action_name,
                gt_action_token=sample.gt_action_token,
                pred_action_name=pred_action_name,
                pred_action_token=pred_action_token,
                action_prior_correct=action_prior_correct,
                format_correct=format_correct,
                prefix_before_action_token=prefix,
                visible_model_output=visible_output_text,
                model_output=output_text,
                model_output_token_ids=generated_ids,
            )
        )
    return results


def _safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    device = _pick_device(args.device)
    dtype = _pick_dtype(args.dtype, device=device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rollouts = _collect_success_rollouts(args)
    eval_samples = _build_eval_samples(selected_rollouts, image_root=args.image_root)

    if args.save_testset:
        _write_json(
            output_dir / "testset_rollouts.json",
            selected_rollouts,
        )
        _write_jsonl(
            output_dir / "testset_samples.jsonl",
            [asdict(sample) for sample in eval_samples],
        )

    model, processor_source = _load_eval_model(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        dtype=dtype,
        device=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    action_token_ids = _resolve_action_token_ids(processor.tokenizer)
    if not action_token_ids:
        warnings.warn(
            "No action token ids found in tokenizer. Generation will not early-stop on action token.",
            stacklevel=1,
        )

    results: list[EvalResult] = []
    batch_size = max(1, int(args.batch_size))
    if batch_size == 1:
        for sample in tqdm(eval_samples, desc="Evaluating"):
            results.append(
                _evaluate_one_sample(
                    model=model,
                    processor=processor,
                    sample=sample,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    action_token_ids=action_token_ids,
                )
            )
    else:
        for start in tqdm(range(0, len(eval_samples), batch_size), desc="Evaluating(batch)"):
            batch_samples = eval_samples[start : start + batch_size]
            results.extend(
                _evaluate_batch(
                    model=model,
                    processor=processor,
                    samples=batch_samples,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    action_token_ids=action_token_ids,
                )
            )

    total = len(results)
    format_correct_count = sum(int(r.format_correct) for r in results)
    action_prior_correct_count = sum(int(r.action_prior_correct) for r in results)
    parsed_action_count = sum(int(r.pred_action_token is not None) for r in results)

    summary = {
        "num_rollouts": len(selected_rollouts),
        "num_samples": total,
        "parsed_action_count": parsed_action_count,
        "format_correct_count": format_correct_count,
        "action_prior_correct_count": action_prior_correct_count,
        "format_accuracy": _safe_ratio(format_correct_count, total),
        "action_prior_accuracy": _safe_ratio(action_prior_correct_count, total),
        "action_prior_accuracy_on_parsed": _safe_ratio(action_prior_correct_count, parsed_action_count),
        "config": {
            "include_sources": args.include_sources,
            "single_step_file": args.single_step_file,
            "multi_step_file": args.multi_step_file,
            "image_root": args.image_root,
            "model_path": args.model_path,
            "base_model_path": args.base_model_path,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "device": str(device),
            "dtype": str(dtype),
        },
    }

    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(output_dir / "predictions.jsonl", [asdict(r) for r in results])

    print("===== Phase1 Rollout Eval Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {output_dir / 'summary.json'}")
    print(f"Saved predictions to: {output_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
