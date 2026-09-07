#!/usr/bin/env python3
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from vagen.inference.model_interface.vllm.model import VLLMModelInterface
from vagen.inference.model_interface.vllm.model_config import VLLMModelConfig
from vagen.rollout.inference_rollout.inference_rollout_service import InferenceRolloutService


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run chunked Navigation inference and write JSON results.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_files_path", required=True)
    parser.add_argument("--server_url", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--server_max_workers", type=int, default=1)
    parser.add_argument("--server_timeout", type=int, default=1200)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def env_configs_from_frame(df: pd.DataFrame) -> List[Dict[str, Any]]:
    env_configs = []
    for _, row in df.iterrows():
        extra_info = row.get("extra_info", {}) or {}
        env_configs.append(
            {
                "env_name": extra_info.get("env_name", "navigation"),
                "env_config": extra_info.get("env_config", {}),
                "seed": extra_info.get("seed", 42),
            }
        )
    return env_configs


def jsonable_result(result: Dict[str, Any], source_index: int) -> Dict[str, Any]:
    item = dict(result)
    image_data = item.pop("image_data", None) or []
    item["num_images"] = len(image_data)
    item["source_index"] = int(source_index)
    return item


def update_summary(summary: Dict[str, Any], result: Dict[str, Any]) -> None:
    metrics = result.get("metrics", {}) or {}
    config_id = result.get("config_id", "unknown")
    by_config = summary["by_config"].setdefault(
        config_id,
        {"total": 0, "success_count": 0, "done_count": 0, "score_sum": 0.0, "step_sum": 0.0},
    )

    success = float(metrics.get("success", 0) or 0)
    done = float(metrics.get("done", 0) or 0)
    score = float(metrics.get("score", 0) or 0)
    step = float(metrics.get("step", 0) or 0)

    summary["total"] += 1
    summary["success_count"] += int(success > 0)
    summary["done_count"] += int(done > 0)
    summary["score_sum"] += score
    summary["step_sum"] += step

    by_config["total"] += 1
    by_config["success_count"] += int(success > 0)
    by_config["done_count"] += int(done > 0)
    by_config["score_sum"] += score
    by_config["step_sum"] += step


def finalize_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    def add_rates(node):
        total = node.get("total", 0)
        if total:
            node["success_rate"] = node.get("success_count", 0) / total
            node["done_rate"] = node.get("done_count", 0) / total
            node["score_mean"] = node.get("score_sum", 0.0) / total
            node["step_mean"] = node.get("step_sum", 0.0) / total
        else:
            node["success_rate"] = 0.0
            node["done_rate"] = 0.0
            node["score_mean"] = 0.0
            node["step_mean"] = 0.0

    add_rates(summary)
    for node in summary["by_config"].values():
        add_rates(node)
    return summary


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    df = pd.read_parquet(args.val_files_path)
    if args.max_examples and args.max_examples > 0:
        df = df.head(args.max_examples)

    model_config = VLLMModelConfig(
        model_name=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        trust_remote_code=True,
    )
    model = VLLMModelInterface(model_config)

    config = {
        "max_steps": args.max_steps,
        "window_size": args.window_size,
        "show_progress": True,
        "use_wandb": False,
        "output_dir": str(output_dir),
    }

    summary = {
        "model_path": args.model_path,
        "val_files_path": args.val_files_path,
        "split": args.split,
        "max_steps": args.max_steps,
        "chunk_size": args.chunk_size,
        "total": 0,
        "success_count": 0,
        "done_count": 0,
        "score_sum": 0.0,
        "step_sum": 0.0,
        "by_config": {},
    }

    with results_path.open("w", encoding="utf-8") as out:
        for start in range(0, len(df), args.chunk_size):
            chunk = df.iloc[start : start + args.chunk_size]
            logger.info("Running chunk %s:%s", start, start + len(chunk))
            service = InferenceRolloutService(
                config=config,
                model_interface=model,
                base_url=args.server_url,
                timeout=args.server_timeout,
                max_workers=args.server_max_workers,
                split=args.split,
                debug=args.debug,
            )
            try:
                service.reset(env_configs_from_frame(chunk))
                service.run(max_steps=args.max_steps)
                for offset, result in enumerate(service.recording_to_log()):
                    row = jsonable_result(result, source_index=start + offset)
                    update_summary(summary, row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
            finally:
                service.close()

            finalized = finalize_summary(json.loads(json.dumps(summary)))
            summary_path.write_text(json.dumps(finalized, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(
                "Progress total=%s success=%s rate=%.4f",
                finalized["total"],
                finalized["success_count"],
                finalized["success_rate"],
            )

    summary_path.write_text(
        json.dumps(finalize_summary(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(finalize_summary(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
