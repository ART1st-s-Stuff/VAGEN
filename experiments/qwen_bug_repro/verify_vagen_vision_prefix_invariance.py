#!/usr/bin/env python
"""Vision prefix-invariance probe for VAGEN navigation (qwen-bug-repro).

Goal: on a real multi-image VAGEN navigation trajectory, measure whether the
Qwen2.5-VL visual encoder is prefix/batch invariant — i.e. whether the same
image produces the same vision features when encoded alone vs. batched with the
rest of the trajectory's images, and whether the full-trajectory forward agrees
with the per-prefix forward on the overlapping prefix tokens.

This reuses VAGEN's own input construction (QwenVLRolloutManager helpers) and
the legacy verl get_rope_index, so the inputs match what VAGEN's train/rollout
path actually feeds the model. Images are collected live from the running
VAGEN env server (BatchEnvClient) so they are real navigation frames.

Outputs a JSON report + stdout summary under --out-dir.
"""
import argparse
import json
import os
from types import SimpleNamespace

import torch

from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from vagen.server.client import BatchEnvClient
from vagen.rollout.qwen_rollout.rollout_manager import QwenVLRolloutManager
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.dataset.rl_dataset import process_image


def collect_trajectory(client, env_id, env_config, seed, max_turns, action_str):
    """Drive the env server for max_turns turns, return (system_prompt, recording)."""
    client.create_environments_batch({env_id: {"env_name": "navigation",
                                               "env_config": env_config,
                                               "seed": seed}})
    obs, info = client.reset(env_id, seed=seed)
    image_placeholder = "<image>"
    recording = []

    def rec(obs, reward, done, llm_raw_response):
        entry = {"env_id": env_id, "done": bool(done), "reward": float(reward),
                 "info": {"llm_raw_response": llm_raw_response},
                 "obs_str": obs["obs_str"]}
        mmd = obs.get("multi_modal_data", {})
        if image_placeholder in mmd:
            entry["image_data"] = [process_image(im) for im in mmd[image_placeholder]]
        recording.append(entry)

    rec(obs, 0.0, False, "")
    for k in range(max_turns - 1):
        obs, reward, done, info = client.step(env_id, action_str)
        rec(obs, reward, done, action_str)
        if done:
            break
    sys_prompt = client.get_system_prompt(env_id)
    try:
        client.close(env_id)
    except Exception:
        pass
    return sys_prompt, recording


def build_prefix_input(mgr, recording, step, window_size, max_length):
    """Mirror QwenVLRolloutManager._generate_final_input_for_rollout but with
    is_final=False (prefix up to `step`, add_generation_prompt=True), embedded."""
    rst = mgr._single_recording_to_prompt(recording, step, window_size,
                                          is_final=False, prep_for_loss_mask=False)
    prompt = rst["prompt"]
    image_data = rst["image_data"]
    row_dict = {}
    if image_data:
        prompt, row_dict, image_grid_thw, _ = mgr._handle_multi_modal_data(
            prompt, row_dict, image_data, do_embedding=True)
    input_ids, attention_mask = _tokenize(mgr.tokenizer, prompt, max_length, left_pad=False)
    position_ids = _rope_position(mgr, input_ids[0], attention_mask[0], image_grid_thw)
    return {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "position_ids": position_ids,
        "multi_modal_inputs": row_dict.get("multi_modal_inputs", {}),
        "prompt": prompt,
    }


def build_full_input(mgr, recording, max_length):
    """Path A: full trajectory, do_embedding=True (train-side representation)."""
    row = mgr._generate_final_input_for_rollout(recording, step=len(recording) - 1,
                                                window_size=len(recording))
    return row


def _tokenize(tokenizer, prompt, max_length, left_pad=False):
    from verl.utils.torch_functional import tokenize_and_postprocess_data
    return tokenize_and_postprocess_data(prompt=prompt, tokenizer=tokenizer,
                                         max_length=max_length,
                                         pad_token_id=tokenizer.pad_token_id,
                                         left_pad=left_pad, truncation="left")


def _rope_position(mgr, input_ids_1d, attention_mask_1d, image_grid_thw):
    if image_grid_thw is not None:
        return get_rope_index(mgr.processor, input_ids_1d, image_grid_thw,
                              attention_mask=attention_mask_1d)
    pos = torch.arange(input_ids_1d.numel(), device=input_ids_1d.device)
    return pos.view(1, -1).expand(3, -1)


def _to_batched(t):
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() == 2 and t.shape[0] == 3:  # (3, seq) mrope -> (3, 1, seq)
        return t.unsqueeze(1)
    return t.unsqueeze(0)


def forward_hidden(model, inputs, device, norm_hook):
    mm = inputs.get("multi_modal_inputs", {})
    input_ids = inputs["input_ids"].to(device)
    attn = inputs["attention_mask"].to(device)
    pos = inputs["position_ids"].to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids.unsqueeze(0),
                       attention_mask=attn.unsqueeze(0),
                       position_ids=_to_batched(pos),
                       pixel_values=mm.get("pixel_values").to(device) if mm.get("pixel_values") is not None else None,
                       image_grid_thw=mm.get("image_grid_thw").to(device) if mm.get("image_grid_thw") is not None else None,
                       use_cache=False).logits
    hidden = norm_hook.captured
    return hidden, logits


class NormHook:
    def __init__(self, model):
        self.captured = None
        self.h = model.model.norm.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.captured = inp[0].detach()  # pre-norm last hidden, (1, seq, H)

    def remove(self):
        self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8400")
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-actions-per-step", type=int, default=5)
    ap.add_argument("--window-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--attn-implementation", default="sdpa",
                    choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-length", type=int, default=4096)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- model + processor ---
    print(f"[probe] loading {args.model_path} attn={args.attn_implementation}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=args.attn_implementation).to(device).eval()

    # --- collect a real trajectory from the env server ---
    env_config = {"render_mode": "vision", "prompt_format": "worldmodeling",
                  "use_state_reward": False, "eval_set": "base",
                  "max_actions_per_step": args.max_actions_per_step}
    action_str = ("<|box_start|><reasoning>probe: step forward to collect frames."
                  "</reasoning><prediction>scene advances slightly.</prediction>"
                  "<answer>moveahead,moveahead,rotateright</answer><|box_end|>")
    client = BatchEnvClient(args.base_url, timeout=300)
    assert client.wait_for_server(max_retries=60, retry_delay=2.0), "env server not ready"
    print("[probe] collecting real trajectory from env server", flush=True)
    sys_prompt, recording = collect_trajectory(client, "probe0", env_config, args.seed,
                                               args.max_turns, action_str)
    n_imgs = sum(1 for r in recording if r.get("image_data"))
    print(f"[probe] collected {len(recording)} turns, {n_imgs} images", flush=True)

    # --- vision tower non-invariance: prefix-batched vs single get_image_features ---
    print("[probe] measuring vision tower prefix-batched-vs-single features", flush=True)
    all_imgs = [r["image_data"][0] for r in recording if r.get("image_data")]
    per_image_diff = []
    for k in range(len(all_imgs)):
        # single image k
        s_in = processor.image_processor([all_imgs[k]], return_tensors="pt")
        with torch.no_grad():
            sf = model.model.get_image_features(
                pixel_values=s_in["pixel_values"].to(device, dtype=model.dtype),
                image_grid_thw=s_in["image_grid_thw"].to(device))
        if isinstance(sf, tuple):
            sf = sf[0]  # newer HF: (features, vision_hidden)
        # prefix-batched images 0..k
        b_in = processor.image_processor(all_imgs[:k+1], return_tensors="pt")
        with torch.no_grad():
            bf = model.model.get_image_features(
                pixel_values=b_in["pixel_values"].to(device, dtype=model.dtype),
                image_grid_thw=b_in["image_grid_thw"].to(device))
        if isinstance(bf, tuple):
            bf = bf[0]
        # image k's features are the tail of the prefix-batched output
        prefix_feats_k = bf[-sf.shape[0]:]
        diff = (prefix_feats_k.float() - sf.float()).abs().max().item()
        per_image_diff.append(diff)
        print(f"[probe] image {k}: prefix_batched_vs_single_max_diff={diff}", flush=True)

    # --- build VAGEN inputs (Path A full, Path B prefix per step) ---
    cfg = SimpleNamespace(special_token_for_loss_mask=["<|box_start|>", "<|box_end|>"],
                          max_trajectory_length=args.max_length,
                          use_multi_turn_reward=True, use_loss_mask=True,
                          use_gae_mask=True, n_gpus_per_node=1)
    env_shim = SimpleNamespace(system_prompt=lambda: sys_prompt,
                               config={"image_placeholder": "<image>"})
    mgr = QwenVLRolloutManager(actor_rollout_wg=None, config=cfg,
                               tokenizer=tokenizer, processor=processor)
    mgr.envs = {"probe0": env_shim}
    mgr.recorder = None

    pathA = build_full_input(mgr, recording, args.max_length)
    pathB_list = []
    for k in range(len(recording)):
        pathB_list.append(build_prefix_input(mgr, recording, k, args.window_size, args.max_length))

    # --- alignment: does Path A's input_ids start with Path B (last step) tokens? ---
    lastB = pathB_list[-1]
    a_ids = pathA["input_ids"]
    b_ids = lastB["input_ids"]
    common = 0
    for i in range(min(a_ids.numel(), b_ids.numel())):
        if a_ids[i].item() == b_ids[i].item():
            common += 1
        else:
            break
    input_ids_prefix_match = (common == b_ids.numel())
    print(f"[probe] last-step prefix len={b_ids.numel()} common_prefix={common} "
          f"input_ids_prefix_match={input_ids_prefix_match}", flush=True)

    # --- downstream hidden/logits prefix diff: Path A vs Path B per step ---
    hook = NormHook(model)
    pathA_row = pathA  # from build_full_input
    per_step_hidden_diff = []
    per_step_logits_diff = []
    per_step_common_len = []
    try:
        hA, logitsA = forward_hidden(model, pathA_row, device, hook)
        for k, pathB_row in enumerate(pathB_list):
            # align by longest common input_ids prefix
            a_ids = pathA_row["input_ids"]
            b_ids = pathB_row["input_ids"]
            common = 0
            for i in range(min(a_ids.numel(), b_ids.numel())):
                if a_ids[i].item() == b_ids[i].item():
                    common += 1
                else:
                    break
            per_step_common_len.append(common)
            if common == 0:
                per_step_hidden_diff.append(None)
                per_step_logits_diff.append(None)
                print(f"[probe] step {k}: no common prefix, skipping", flush=True)
                continue
            hB, logitsB = forward_hidden(model, pathB_row, device, hook)
            hd = (hA[0, :common].float() - hB[0, :common].float()).abs().max().item()
            ld = (logitsA[0, :common].float() - logitsB[0, :common].float()).abs().max().item()
            per_step_hidden_diff.append(hd)
            per_step_logits_diff.append(ld)
            print(f"[probe] step {k}: common_prefix={common} "
                  f"hidden_max_abs_diff={hd} logits_max_abs_diff={ld}", flush=True)
    finally:
        hook.remove()

    report = {
        "model": args.model_path,
        "attn_implementation": args.attn_implementation,
        "n_turns": len(recording),
        "n_images": n_imgs,
        "vision_features_prefix_batched_vs_single_per_image_diff": per_image_diff,
        "vision_features_batch_max_diff": max(per_image_diff) if per_image_diff else 0.0,
        "per_step_common_prefix_len": per_step_common_len,
        "per_step_hidden_max_abs_diff": per_step_hidden_diff,
        "per_step_logits_max_abs_diff": per_step_logits_diff,
        "verdict_vision": "vision_encoder_not_prefix_invariant" if
                          (per_image_diff and max(per_image_diff) > 1e-3) else "vision_features_within_noise",
        "verdict_forward": "full_vs_prefix_not_equivalent" if
                           (per_step_hidden_diff and any(
                               d is not None and d > 1e-3 for d in per_step_hidden_diff))
                           else "full_vs_prefix_hidden_within_noise",
    }
    out = os.path.join(args.out_dir, "report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[probe] report written to {out}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
