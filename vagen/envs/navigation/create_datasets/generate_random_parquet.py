"""Generate navigation SFT parquet with random policy rollouts.

Each row corresponds to one env step and contains:
  - prompt: current observation text (obs_str)
  - response: random action string
  - image_path: current RGB observation image path
  - next_image_path: next observation image path (if available)
  - reward, done, success
  - clip_gt, mae_gt (optional; extracted from current image)
  - next_clip_gt, next_mae_gt (optional; extracted from next image)
  - eval_set, episode_id, step_id

Usage:
  python -m vagen.envs.navigation.create_datasets.generate_random_parquet \
    --output /tmp/navigation_random.parquet \
    --image_dir /tmp/navigation_images \
    --episodes 100 \
    --max_steps 10 \
    --eval_set base \
    --prompt_format latent_plan
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
from dataclasses import asdict
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, ViTImageProcessor, ViTMAEModel

from vagen.envs.navigation.navigation_env import NavigationEnv, NavigationEnvConfig
from verl.workers.roles.utils.action_schema import (
    ACTION_END_TOKEN,
    ACTION_NAME_TO_TOKEN,
    ACTION_NAMES,
    ACTION_START_TOKEN,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random-policy navigation parquet")
    parser.add_argument("--output", type=str, required=True, help="Output parquet path")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory to save observation images")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=10, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed")
    parser.add_argument("--eval_set", type=str, default="base", help="Navigation eval_set")
    parser.add_argument("--prompt_format", type=str, default="latent_plan", help="Env prompt format")
    parser.add_argument("--max_actions_per_step", type=int, default=3, help="Upper bound random action count")
    parser.add_argument("--action_sep", type=str, default="|", help="Action separator for textual formats")
    parser.add_argument("--gpu_device", type=int, default=0, help="AI2-THOR gpu device")
    parser.add_argument("--compute_features", action="store_true", help="Whether to write CLIP/MAE feature columns")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model id")
    parser.add_argument("--mae_model_name", type=str, default="facebook/vit-mae-base", help="MAE model id")
    return parser.parse_args()


def random_action_str(prompt_format: str, max_actions_per_step: int, action_sep: str) -> str:
    n = random.randint(1, max_actions_per_step)
    actions = [random.choice(ACTION_NAMES) for _ in range(n)]
    if prompt_format == "latent_plan":
        tokens = " ".join(ACTION_NAME_TO_TOKEN[a] for a in actions)
        return f"{ACTION_START_TOKEN}{tokens}{ACTION_END_TOKEN}"
    action_text = action_sep.join(actions)
    return f"<think>random explore</think><action>{action_text}</action>"


def save_obs_image(obs: dict[str, Any], path: str) -> str | None:
    images = obs.get("multi_modal_input", {}).get("<image>", []) or []
    if not images:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images[0].save(path)
    return path


def get_obs_image(obs: dict[str, Any]):
    images = obs.get("multi_modal_input", {}).get("<image>", []) or []
    return images[0] if images else None


def build_feature_extractors(args: argparse.Namespace):
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    clip_model = CLIPModel.from_pretrained(args.clip_model_name).eval()
    mae_processor = ViTImageProcessor.from_pretrained(args.mae_model_name)
    mae_model = ViTMAEModel.from_pretrained(args.mae_model_name).eval()
    return clip_processor, clip_model, mae_processor, mae_model


@torch.no_grad()
def extract_clip_mae_features(image, clip_processor, clip_model, mae_processor, mae_model):
    clip_inputs = clip_processor(images=image, return_tensors="pt")
    clip_feat = clip_model.get_image_features(**clip_inputs).squeeze(0).to(torch.float32)
    clip_feat = torch.nn.functional.normalize(clip_feat, dim=-1)

    mae_inputs = mae_processor(images=image, return_tensors="pt")
    mae_out = mae_model(**mae_inputs)
    mae_feat = mae_out.last_hidden_state.mean(dim=1).squeeze(0).to(torch.float32)
    return clip_feat.tolist(), mae_feat.tolist()


async def generate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    random.seed(args.seed)

    env_cfg = asdict(
        NavigationEnvConfig(
            eval_set=args.eval_set,
            prompt_format=args.prompt_format,
            max_steps=args.max_steps,
            action_sep=args.action_sep,
            max_actions_per_step=max(args.max_actions_per_step, 1),
            gpu_device=args.gpu_device,
        )
    )
    env = NavigationEnv(env_cfg)
    rows: list[dict[str, Any]] = []
    clip_processor = clip_model = mae_processor = mae_model = None
    if args.compute_features:
        clip_processor, clip_model, mae_processor, mae_model = build_feature_extractors(args)

    try:
        pbar = tqdm(range(args.episodes), desc="Generating episodes", unit="ep")
        for ep in pbar:
            obs, _ = await env.reset(seed=args.seed + ep)
            done = False
            step_id = 0
            while not done and step_id < args.max_steps:
                prompt = obs.get("obs_str", "")
                image_path = os.path.join(args.image_dir, f"ep{ep:06d}_step{step_id:04d}.png")
                image_path = save_obs_image(obs, image_path)
                image = get_obs_image(obs)
                clip_gt = mae_gt = None
                if args.compute_features and image is not None:
                    clip_gt, mae_gt = extract_clip_mae_features(
                        image=image,
                        clip_processor=clip_processor,
                        clip_model=clip_model,
                        mae_processor=mae_processor,
                        mae_model=mae_model,
                    )

                response = random_action_str(
                    prompt_format=args.prompt_format,
                    max_actions_per_step=args.max_actions_per_step,
                    action_sep=args.action_sep,
                )
                next_obs, reward, done, info = await env.step(response)
                next_image_path = os.path.join(args.image_dir, f"ep{ep:06d}_step{step_id:04d}_next.png")
                next_image_path = save_obs_image(next_obs, next_image_path)
                next_image = get_obs_image(next_obs)
                next_clip_gt = next_mae_gt = None
                if args.compute_features and next_image is not None:
                    next_clip_gt, next_mae_gt = extract_clip_mae_features(
                        image=next_image,
                        clip_processor=clip_processor,
                        clip_model=clip_model,
                        mae_processor=mae_processor,
                        mae_model=mae_model,
                    )

                rows.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "image_path": image_path,
                        "next_image_path": next_image_path,
                        "reward": float(reward),
                        "done": bool(done),
                        "success": bool(info.get("success", False)),
                        "clip_gt": clip_gt,
                        "mae_gt": mae_gt,
                        "next_clip_gt": next_clip_gt,
                        "next_mae_gt": next_mae_gt,
                        "eval_set": args.eval_set,
                        "episode_id": ep,
                        "step_id": step_id,
                    }
                )

                obs = next_obs
                step_id += 1
            pbar.set_postfix(samples=len(rows))
    finally:
        await env.close()

    return rows


async def _main() -> None:
    args = parse_args()
    rows = await generate_rows(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
