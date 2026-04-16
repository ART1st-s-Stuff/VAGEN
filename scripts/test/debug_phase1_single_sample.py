#!/usr/bin/env python3
"""Debug a single Phase1 sample generation in detail."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from verl.workers.roles.utils.action_schema import ACTION_TOKEN_TO_NAME


def _load_eval_module(repo_root: Path):
    module_path = repo_root / "scripts/test/phase1_rollout_testset_and_eval.py"
    module_name = "phase1_eval_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    # Python 3.12 dataclass resolution needs module to exist in sys.modules.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug one sample of Phase1 eval")
    parser.add_argument("--single-step-file", type=str, default="datasets/navigation/eb-nav_dataset_single_step.json")
    parser.add_argument("--multi-step-file", type=str, default="datasets/navigation/eb-nav_dataset_multi_step.json")
    parser.add_argument("--include-sources", choices=("single_step", "multi_step", "both"), default="single_step")
    parser.add_argument("--image-root", type=str, default="datasets/navigation")
    parser.add_argument("--num-rollouts", type=int, default=20)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    return parser.parse_args()


def _decode_topk(processor: Any, logits: torch.Tensor, k: int) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k=min(k, probs.shape[-1]))
    out: list[dict[str, Any]] = []
    for p, token_id in zip(vals.tolist(), idx.tolist(), strict=True):
        tok = processor.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        out.append({"token_id": int(token_id), "token": tok, "prob": float(p)})
    return out


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    mod = _load_eval_module(repo_root)

    device = mod._pick_device(args.device)
    dtype = mod._pick_dtype(args.dtype, device=device)

    helper_args = argparse.Namespace(
        single_step_file=args.single_step_file,
        multi_step_file=args.multi_step_file,
        include_sources=args.include_sources,
        num_rollouts=args.num_rollouts,
    )
    rollouts = mod._collect_success_rollouts(helper_args)
    samples = mod._build_eval_samples(rollouts, image_root=args.image_root)
    if not (0 <= args.sample_index < len(samples)):
        raise ValueError(f"sample-index out of range: {args.sample_index}, total={len(samples)}")
    sample = samples[args.sample_index]

    model, processor_source = mod._load_eval_model(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        dtype=dtype,
        device=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True, local_files_only=True)

    messages = [{"role": "user", "content": [{"type": "text", "text": sample.task_prompt}, {"type": "image", "image": sample.image_path}]}]
    try:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image = Image.open(sample.image_path).convert("RGB")
    model_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    input_len = int(model_inputs["input_ids"].shape[1])

    with torch.no_grad():
        outputs = model(**model_inputs)
        first_step_logits = outputs.logits[0, -1, :].float().cpu()
        generated = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    gen_ids = generated[:, input_len:][0].tolist()
    gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
    visible_text = gen_text.encode("unicode_escape").decode("utf-8")

    first_action_token = None
    for tok in ACTION_TOKEN_TO_NAME:
        if tok in gen_text:
            first_action_token = tok
            break

    report = {
        "sample_meta": {
            "sample_index": args.sample_index,
            "source": sample.source,
            "episode_id": sample.episode_id,
            "trajectory_index": sample.trajectory_index,
            "plan_action_index": sample.plan_action_index,
            "step_id": sample.step_id,
            "image_path": sample.image_path,
            "gt_action_name": sample.gt_action_name,
            "gt_action_token": sample.gt_action_token,
        },
        "prompt_preview": prompt_text[:1200],
        "generated": {
            "token_ids": gen_ids,
            "text_visible": visible_text,
            "first_action_token_found": first_action_token,
        },
        "first_step_topk": _decode_topk(processor, first_step_logits, args.topk),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
