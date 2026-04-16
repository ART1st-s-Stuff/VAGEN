#!/usr/bin/env python3
"""Inspect Phase1 SFT samples right before feeding into the model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

# Allow running from arbitrary cwd without pre-setting PYTHONPATH.
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vagen.datasets.navigation_multimodal_sft_dataset import (
    NavigationMultimodalSFTDataset,
    _normalize_message_content_schema,
)
from verl.workers.roles.utils.action_schema import ACTION_TOKEN_TO_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Phase1 training inputs")
    parser.add_argument(
        "--train-file",
        type=str,
        default="datasets/phase1_debug/ebnav_phase1_no_history.parquet",
        help="Parquet/json/jsonl used by phase1 SFT.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/Qwen2.5-VL-3B-Instruct",
        help="Model/processor path used in training.",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index to inspect.")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--truncation", choices=("error", "left", "right"), default="error")
    parser.add_argument("--history-mode", type=str, default="no_history")
    parser.add_argument(
        "--print-token-window",
        type=int,
        default=256,
        help="How many leading tokens to decode for preview.",
    )
    return parser.parse_args()


def _to_python(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def _escape_visible(text: str) -> str:
    return text.encode("unicode_escape").decode("utf-8")


def _keyword_hits(text: str) -> dict[str, bool]:
    action_tokens = tuple(ACTION_TOKEN_TO_NAME.keys())
    return {
        "contains_im_start_assistant": "<|im_start|>assistant" in text,
        "contains_im_end": "<|im_end|>" in text,
        "contains_think_open": "<think>" in text,
        "contains_think_close": "</think>" in text,
        "contains_latent": "<|latent|>" in text,
        "contains_action_start": "<|action_start|>" in text,
        "contains_action_end": "<|action_end|>" in text,
        "contains_any_action_token": any(tok in text for tok in action_tokens),
    }


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    train_file = str((root / args.train_file).resolve()) if not Path(args.train_file).is_absolute() else args.train_file
    model_path = str((root / args.model_path).resolve()) if not Path(args.model_path).is_absolute() else args.model_path

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer = processor.tokenizer

    dataset_config = {
        "pad_mode": "no_padding",
        "truncation": args.truncation,
        "max_length": args.max_length,
        "messages_key": "messages",
        "apply_chat_template_kwargs": {},
        "builder": {"history_mode": args.history_mode},
    }
    dataset = NavigationMultimodalSFTDataset(
        parquet_files=train_file,
        tokenizer=tokenizer,
        config=dataset_config,
        max_samples=-1,
        processor=processor,
    )
    if not (0 <= args.sample_index < len(dataset)):
        raise ValueError(f"sample-index out of range: {args.sample_index}, total={len(dataset)}")

    raw_messages = _to_python(dataset.messages[args.sample_index])
    norm_messages = _normalize_message_content_schema(raw_messages)
    enable_thinking = dataset.enable_thinking[args.sample_index]
    full_text = dataset._apply_chat_template(
        norm_messages,
        enable_thinking=enable_thinking,
        add_generation_prompt=False,
    )

    sample = dataset[args.sample_index]
    input_ids = sample["input_ids"].tolist()
    loss_mask = sample["loss_mask"].tolist()
    attention_mask = sample.get("attention_mask")
    attention_mask_list = attention_mask.tolist() if attention_mask is not None else None

    decode_len = min(len(input_ids), args.print_token_window)
    decoded_prefix = tokenizer.decode(input_ids[:decode_len], skip_special_tokens=False)
    supervised_ids = [tid for tid, m in zip(input_ids, loss_mask, strict=True) if int(m) == 1]
    supervised_text = tokenizer.decode(supervised_ids[: args.print_token_window], skip_special_tokens=False)
    supervised_text_full = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    supervised_token_counts: dict[int, int] = {}
    for token_id in supervised_ids:
        supervised_token_counts[token_id] = supervised_token_counts.get(token_id, 0) + 1
    top_supervised_token_ids = sorted(
        supervised_token_counts.items(), key=lambda kv: kv[1], reverse=True
    )[: min(20, len(supervised_token_counts))]
    top_supervised_tokens = [
        {
            "token_id": int(token_id),
            "count": int(count),
            "token_visible": _escape_visible(tokenizer.decode([token_id], skip_special_tokens=False)),
        }
        for token_id, count in top_supervised_token_ids
    ]
    supervised_preview_head = tokenizer.decode(
        supervised_ids[: args.print_token_window],
        skip_special_tokens=False,
    )
    supervised_preview_tail = tokenizer.decode(
        supervised_ids[-args.print_token_window :] if args.print_token_window > 0 else supervised_ids,
        skip_special_tokens=False,
    )

    report = {
        "meta": {
            "train_file": train_file,
            "model_path": model_path,
            "sample_index": args.sample_index,
            "dataset_len": len(dataset),
        },
        "raw_messages": raw_messages,
        "normalized_messages": norm_messages,
        "chat_template_text_visible": _escape_visible(full_text),
        "tensor_stats": {
            "input_len": len(input_ids),
            "num_supervised_tokens": int(sum(int(x) for x in loss_mask)),
            "num_ignored_tokens": int(len(loss_mask) - sum(int(x) for x in loss_mask)),
            "attention_len": len(attention_mask_list) if attention_mask_list is not None else None,
            "first_50_loss_mask": loss_mask[:50],
        },
        "token_preview": {
            "first_token_ids": input_ids[: args.print_token_window],
            "decoded_prefix_visible": _escape_visible(decoded_prefix),
            "supervised_prefix_visible": _escape_visible(supervised_text),
        },
        "supervised_stats": {
            "keyword_hits": _keyword_hits(supervised_text_full),
            "top_supervised_tokens": top_supervised_tokens,
            "head_visible": _escape_visible(supervised_preview_head),
            "tail_visible": _escape_visible(supervised_preview_tail),
        },
        "multi_modal_inputs_keys": sorted(list(sample["multi_modal_inputs"].keys())),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
