#!/usr/bin/env python
"""Vision prefix-invariance probe for VAGEN navigation.

Compare Path A (train: full trajectory, all images in one forward) vs
Path B (rollout: per-step prefix, only images up to current step).

Two measurements:
1. Vision-tower non-invariance: get_image_features([images_0..k]) vs
   get_image_features([img_k]) — compare the last N_k tokens (tail).
2. Downstream forward: Path A (full) vs Path B (prefix at step k)
   — compare last-hidden and logits at overlapping prefix positions.
"""
import argparse, json, os, sys
from types import SimpleNamespace

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from vagen.server.client import BatchEnvClient
from vagen.rollout.qwen_rollout.rollout_manager import QwenVLRolloutManager
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.dataset.rl_dataset import process_image


# -- helpers ------------------------------------------------------------------

def drive_trajectory(client, env_id, env_config, seed, max_turns, action_str):
    """Drive env for max_turns, return (system_prompt, recording list)."""
    client.create_environments_batch({env_id: {"env_name": "navigation",
                                               "env_config": env_config, "seed": seed}})
    obs, info = client.reset(env_id, seed=seed)
    recording = []
    def record(obs, reward, done, llm_raw_response):
        entry = {"env_id": env_id, "done": bool(done), "reward": float(reward),
                 "info": {"llm_raw_response": llm_raw_response}, "obs_str": obs["obs_str"]}
        mmd = obs.get("multi_modal_data", {})
        if "<image>" in mmd:
            entry["image_data"] = [process_image(im) for im in mmd["<image>"]]
        recording.append(entry)
    record(obs, 0.0, False, "")
    for k in range(max_turns - 1):
        obs, reward, done, info = client.step(env_id, action_str)
        record(obs, reward, done, action_str)
        if done:
            break
    sys_prompt = client.get_system_prompt(env_id)
    try:
        client.close(env_id)
    except Exception:
        pass
    return sys_prompt, recording


class NormHook:
    def __init__(self, model):
        self.captured = None
        self.h = model.model.norm.register_forward_hook(self._hook)
    def _hook(self, module, inp, out):
        self.captured = inp[0].detach()
    def remove(self):
        self.h.remove()


# -- input builders (matching VAGEN) ------------------------------------------

def build_full(mgr, recording, max_len):
    """Path A: _generate_input_for_uptate (all images, do_embedding=True)."""
    return mgr._generate_input_for_uptate(recording, step=len(recording) - 1,
                                          window_size=len(recording))

def build_prefix(mgr, recording, step, window_size, max_len):
    """Path B: _generate_input_for_rollout (prefix up to step)."""
    return mgr._generate_input_for_rollout(recording, step=step, window_size=window_size)


def _forward(model, mm, device, hook):
    """Run model on batched(1) input, return (last_hidden(pre-norm), logits)."""
    pv = grid = None
    if mm.get("pixel_values") is not None:
        pv = mm["pixel_values"].to(device, dtype=model.dtype)
    if mm.get("image_grid_thw") is not None:
        grid = mm["image_grid_thw"].to(device)
    ids = mm["input_ids"].to(device).unsqueeze(0)
    attn = mm["attention_mask"].to(device).unsqueeze(0)
    pos = mm["position_ids"].to(device)
    if pos.dim() == 3:
        pos = pos.unsqueeze(1)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, position_ids=pos,
                    pixel_values=pv, image_grid_thw=grid, use_cache=False)
    return hook.captured, out.logits


# -- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8400")
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-actions-per-step", type=int, default=5)
    ap.add_argument("--window-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-length", type=int, default=4096)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[probe] loading {args.model_path} attn={args.attn_implementation}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=args.attn_implementation).to(device).eval()

    env_config = {"render_mode": "vision", "prompt_format": "worldmodeling",
                  "use_state_reward": False, "eval_set": "base",
                  "max_actions_per_step": args.max_actions_per_step}
    action_str = ("<|box_start|><reasoning>probe: step forward.</reasoning>"
                  "<prediction>scene advances.</prediction>"
                  "<answer>moveahead,moveahead,rotateright</answer><|box_end|>")
    client = BatchEnvClient(args.base_url, timeout=300)
    assert client.wait_for_server(max_retries=60, retry_delay=2.0), "env server not ready"
    print("[probe] collecting real trajectory", flush=True)
    sys_prompt, recording = drive_trajectory(client, "probe0", env_config,
                                             args.seed, args.max_turns, action_str)
    imgs = [r["image_data"][0] for r in recording if r.get("image_data")]
    n_imgs = len(imgs)
    print(f"[probe] {len(recording)} turns, {n_imgs} images", flush=True)

    # -- vision-tower non-invariance: tail-based comparison --------------------
    print("[probe] vision prefix-vs-single diff", flush=True)
    per_image_diff = []
    for k in range(n_imgs):
        s_inp = processor.image_processor([imgs[k]], return_tensors="pt")
        s_pv = s_inp["pixel_values"].to(device, dtype=model.dtype)
        s_grid = s_inp["image_grid_thw"].to(device)
        with torch.no_grad():
            s_feats = model.model.get_image_features(pixel_values=s_pv, image_grid_thw=s_grid)[0]
        N_k = s_feats.shape[0]
        p_inp = processor.image_processor(imgs[:k + 1], return_tensors="pt")
        p_pv = p_inp["pixel_values"].to(device, dtype=model.dtype)
        p_grid = p_inp["image_grid_thw"].to(device)
        with torch.no_grad():
            p_feats = model.model.get_image_features(pixel_values=p_pv, image_grid_thw=p_grid)[0]
        diff = (p_feats[-N_k:].float() - s_feats.float()).abs().max().item()
        per_image_diff.append(diff)
        print(f"[probe] image {k}: N={N_k} tail_vs_single_max_diff={diff}", flush=True)

    # -- Path A vs Path B downstream -------------------------------------------
    print("[probe] building VAGEN inputs", flush=True)
    cfg = SimpleNamespace(special_token_for_loss_mask=["<|box_start|>", "<|box_end|>"],
                          max_trajectory_length=args.max_length,
                          truncation="left",
                          use_multi_turn_reward=True, use_loss_mask=True,
                          use_gae_mask=True, n_gpus_per_node=1)
    env_shim = SimpleNamespace(system_prompt=lambda: sys_prompt,
                               config={"image_placeholder": "<image>"})
    mgr = QwenVLRolloutManager(actor_rollout_wg=None, config=cfg,
                               tokenizer=tokenizer, processor=processor)
    mgr.envs = {"probe0": env_shim}
    mgr.recorder = None

    pathA = build_full(mgr, recording, args.max_length)
    hook = NormHook(model)
    prefix_results = []
    for k in range(len(recording)):
        print(f"[probe] step {k} prefix vs full overlay", flush=True)
        pathB = build_prefix(mgr, recording, k, args.window_size, args.max_length)
        try:
            hA, logitsA = _forward(model, pathA, device, hook)
            hB, logitsB = _forward(model, pathB, device, hook)
            L = min(hA.shape[1], hB.shape[1])
            hdiff = (hA[0, :L].float() - hB[0, :L].float()).abs().max().item()
            ldiff = (logitsA[0, :L].float() - logitsB[0, :L].float()).abs().max().item()
            prefix_results.append({"step": k, "overlap_len": L,
                                   "hidden_max_abs_diff": hdiff,
                                   "logits_max_abs_diff": ldiff})
            print(f"[probe] step={k} L={L} hidden={hdiff} logits={ldiff}", flush=True)
        except Exception as e:
            print(f"[probe] step={k} error: {e}", flush=True)
            prefix_results.append({"step": k, "error": str(e)})
    hook.remove()

    report = {
        "model": args.model_path,
        "attn_implementation": args.attn_implementation,
        "n_turns": len(recording), "n_images": n_imgs,
        "vision_features_per_image_max_diff": per_image_diff,
        "vision_features_batch_max_diff": max(per_image_diff) if per_image_diff else 0.0,
        "verdict_vision": "NOT PREFIX-INVARIANT" if (per_image_diff and max(per_image_diff) > 1e-4) else "within_noise",
        "prefix_hidden_logits_diffs": prefix_results,
    }
    out = os.path.join(args.out_dir, "report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[probe] report saved to {out}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
