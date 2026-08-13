# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

import asyncio
import logging
import math
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from PIL import Image
from .agent_loop_no_concat import AgentLoopBase, AgentLoopOutput, register
from .decision_ledger import (
    build_decision_ledger_from_env_info,
    parse_decision_ledger_enabled,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from ..envs.gym_image_env import GymImageEnv
from omegaconf import OmegaConf
import traceback
import importlib
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
from .gym_agent_loop import convert_obs_to_content, extract_success, _flatten_text_only_content, _normalize_images

def _nimloth_turn_generation_spec(
    tokenizer: Any,
    *,
    latent_token_count: int,
    max_response_tokens: int,
) -> Any:
    """Build Nimloth's shared real-CoT/K-slot/action generation contract."""

    from nimloth.backbone.qwen25vl.policy import reasoning_forbidden_token_ids
    from nimloth.backbone.qwen25vl.turn_generation import TurnGenerationSpec
    from nimloth.latent import LatentActionTokens, latent_state_tokens, special_token_ids

    tokens = LatentActionTokens()
    token_id_map = special_token_ids(
        tokenizer,
        latent_token_count=latent_token_count,
    )
    close_ids = tuple(
        int(value) for value in tokenizer.encode("</think>", add_special_tokens=False)
    )
    injected_ids = tuple(
        token_id_map[token]
        for token in (
            *latent_state_tokens(latent_token_count, tokens),
            tokens.action_start,
        )
    )
    action_token_ids = tuple(token_id_map[token] for token in tokens.action_tokens)
    protocol_overhead = len(close_ids) + len(injected_ids) + 2
    if max_response_tokens <= protocol_overhead:
        raise ValueError(
            "response limit is too small for Nimloth turn protocol: "
            f"{max_response_tokens} <= {protocol_overhead}"
        )
    return TurnGenerationSpec(
        close_text="</think>",
        close_token_ids=close_ids,
        injected_token_ids=injected_ids,
        action_token_ids=action_token_ids,
        action_end_token_id=token_id_map[tokens.action_end],
        forbidden_reasoning_token_ids=reasoning_forbidden_token_ids(
            tokenizer,
            token_id_map,
            close_token_ids=close_ids,
        ),
        max_reasoning_tokens=max_response_tokens - protocol_overhead,
    )


def _nimloth_response_mask(response_ids: List[int], spec: Any) -> List[int]:
    """Mark sampled CoT/action tokens while excluding forced protocol tokens."""

    from nimloth.backbone.qwen25vl.turn_generation import find_token_subsequence

    injected_start = find_token_subsequence(response_ids, spec.injected_token_ids)
    if injected_start is None:
        raise RuntimeError("Nimloth turn response did not inject latent queries")
    injected_end = injected_start + len(spec.injected_token_ids)
    if tuple(response_ids[injected_start:injected_end]) != spec.injected_token_ids:
        raise RuntimeError("Nimloth turn response has an invalid injected prefix")
    if len(response_ids) != injected_end + 2:
        raise RuntimeError("Nimloth turn response has an invalid action suffix length")
    if response_ids[injected_end] not in spec.action_token_ids:
        raise RuntimeError("Nimloth turn response has an invalid action token")
    if response_ids[injected_end + 1] != spec.action_end_token_id:
        raise RuntimeError("Nimloth turn response did not end at action_end")

    mask = [0] * len(response_ids)
    for index in range(min(injected_start, spec.max_reasoning_tokens)):
        mask[index] = 1
    mask[injected_end] = 1
    return mask


def _strip_trailing_generation_terminators(
    token_ids: List[int],
    tokenizer: Any,
) -> List[int]:
    """Strip only tokenizer-declared generation terminators for env text decoding."""

    terminator_ids: set[int] = set()
    for attribute in ("eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            terminator_ids.add(value)
        elif isinstance(value, (list, tuple, set)):
            terminator_ids.update(
                item
                for item in value
                if isinstance(item, int) and not isinstance(item, bool)
            )
    trimmed = list(token_ids)
    while trimmed and trimmed[-1] in terminator_ids:
        trimmed.pop()
    return trimmed


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    INTERACTING = "interacting"
    TERMINATED = "terminated"


class AgentData:
    """Container for all mutable trajectory state."""
    def __init__(
        self,
        metrics: Dict[str, Any],
        request_id: str,
        env: GymImageEnv,
        response_limit: int,
        env_name: str,
        sys_msg: Optional[Dict[str, Any]] = None,
        sys_images: Optional[List[Image.Image]] = None,
        cur_msg: Optional[Dict[str, Any]] = None,
        cur_images: Optional[List[Image.Image]] = None,
        group_idx: int = 0,
        traj_idx: int = 0,
    ):
        self.sys_msg: Optional[Dict[str, Any]] = sys_msg
        self.sys_images: Optional[List[Image.Image]] = sys_images
        
        self.cur_msg: Optional[Dict[str, Any]] = cur_msg
        self.cur_images: Optional[List[Image.Image]] = cur_images
        
        self.metrics = metrics
        self.request_id = request_id
        self.env = env
        self.response_limit = response_limit
        self.env_name = env_name
        self.group_idx = group_idx
        self.traj_idx = traj_idx
        # Token buffers
        self.turn_prompt_ids: Optional[List[int]] = None
        self.turn_response_ids: Optional[List[int]] = None
        self.turn_response_mask: Optional[List[int]] = None
        self.turn_response_logprobs: Optional[List[float]] = None
        self.turn_policy_state: Optional[Dict[str, Any]] = None
        self.turn_generation_id: Optional[str] = None

        # Env stats
        self.env_turns: int = 0


        # Cached assistant text to step env
        self.last_assistant_text: Optional[str] = None
        self.outputs: List[AgentLoopOutput] = []

# -------------------- Gym Agent Loop --------------------

class GymAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level GymAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
        
        # Store module paths for lazy loading; environments are imported on first use
        cls.env_registry_paths = dict(config.env_registry.items())
        cls.env_registry = {}
            
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.decision_ledger_enabled = parse_decision_ledger_enabled(
            config.get("decision_ledger")
        )
        

    @rollout_trace_op
    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics: Dict[str, Any] = {}
        request_id = uuid4().hex

        # Build env (lazy import on first use)
        env_name = kwargs["env_name"]
        if env_name not in self.env_registry:
            if env_name not in self.env_registry_paths:
                raise KeyError(f"Unknown env: {env_name}. Available: {list(self.env_registry_paths.keys())}")
            module_path, class_name = self.env_registry_paths[env_name].rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.env_registry[env_name] = getattr(module, class_name)
        env_cls = self.env_registry[env_name]
        env_config = kwargs["config"]
        seed = kwargs["seed"]
        self.env_max_turns = kwargs.get("max_turns", None)
        env: GymImageEnv = env_cls(env_config=env_config)
        try:
            # Bootstrap: reset -> system_prompt (message order: system, then initial user)
            init_obs, info = await env.reset(seed=seed)
            sys_obs = await env.system_prompt()
            prompt_format = env_config.get("prompt_format", "free_think")
            latent_token_count = env_config.get("latent_token_count")
            if prompt_format == "nimloth" and (
                isinstance(latent_token_count, bool)
                or not isinstance(latent_token_count, int)
                or latent_token_count < 1
            ):
                raise ValueError(
                    "prompt_format=nimloth requires explicit positive "
                    "latent_token_count"
                )

            sys_msg={"role": "system", "content": convert_obs_to_content(sys_obs, **kwargs)}
            sys_images=_normalize_images(sys_obs.get("multi_modal_input", {}).get("<image>", []) or [])

            cur_msg={"role": "user", "content": convert_obs_to_content(init_obs, **kwargs)}
            cur_images=_normalize_images(init_obs.get("multi_modal_input", {}).get("<image>", []) or [])

            per_turn_response_limit = int(kwargs.get("response_length_per_turn") or self.response_length)
            per_turn_response_limit = min(per_turn_response_limit, self.response_length)
            if per_turn_response_limit <= 0:
                per_turn_response_limit = 1

            agent_data = AgentData(
                sys_msg=sys_msg,
                sys_images=sys_images,
                cur_msg=cur_msg,
                cur_images=cur_images,
                metrics=metrics,
                request_id=request_id,
                env=env,
                response_limit=per_turn_response_limit,
                env_name=kwargs["env_name"],
                group_idx=kwargs["group_idx"],
                traj_idx=kwargs["traj_idx"],
            )

            if prompt_format == "nimloth":
                # The opening tag is a prompt prefix; generated IDs begin with the
                # model's real thought and close it before forcing K slots/action.
                agent_data.cur_msg = {
                    "role": "user",
                    "content": cur_msg["content"],
                }

            # State machine: always GENERATE -> INTERACT, and decide termination inside INTERACT
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.INTERACTING:
                    state = await self._handle_env_state(agent_data, **kwargs)
                else:
                    logger.error(f"Invalid state: {state}")
                    state = AgentState.TERMINATED
            return agent_data.outputs
        finally:
            await env.close()

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: Dict[str, Any]) -> AgentState:
        """Encode initial (system + first user) messages into prompt_ids."""
        image_data = agent_data.sys_images + agent_data.cur_images
        prompt_format = agent_data.env.config.get("prompt_format", "free_think")
        messages = [agent_data.sys_msg, agent_data.cur_msg]
        chat_template_args = {
            "add_generation_prompt": True,
            **self.apply_chat_template_kwargs,
        }
        if prompt_format == "nimloth":
            messages.append({"role": "assistant", "content": "<think>"})
            chat_template_args.update(
                {
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                }
            )
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    **chat_template_args,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            agent_data.turn_prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if image_data:
                raise ValueError("Environment returned images but `processor` is None.")
            flat_messages = [_flatten_text_only_content(m) for m in messages]
            agent_data.turn_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    flat_messages,
                    tokenize=True,
                    return_dict=False,
                    **chat_template_args,
                ),
            )
        
        if len(agent_data.turn_prompt_ids)>self.prompt_length:
            logger.warning(f"In env:{agent_data.env_name}, initial prompt length {len(agent_data.turn_prompt_ids)} exceeds prompt_length {self.prompt_length}")
        return AgentState.GENERATING

    
    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: Dict[str, Any]
    ) -> AgentState:
        """Generate assistant output and mark generated tokens with mask=1."""
        sampling_params_for_turn = sampling_params.copy()
        max_new_tokens=sampling_params_for_turn.get("max_new_tokens", None) or agent_data.response_limit
        max_new_tokens = min(max_new_tokens, agent_data.response_limit)
        sampling_params_for_turn["max_new_tokens"] = max_new_tokens
        image_data = agent_data.sys_images + agent_data.cur_images
        prompt_format = agent_data.env.config.get("prompt_format", "free_think")
        nimloth_spec = None
        if prompt_format == "nimloth":
            nimloth_spec = _nimloth_turn_generation_spec(
                self.tokenizer,
                latent_token_count=agent_data.env.config["latent_token_count"],
                max_response_tokens=max_new_tokens,
            )
            sampling_params_for_turn.update(
                {
                    "max_new_tokens": nimloth_spec.max_output_tokens,
                    "logprobs": len(nimloth_spec.action_token_ids),
                    "ignore_eos": True,
                    "stop_token_ids": [nimloth_spec.action_end_token_id],
                    "extra_args": nimloth_spec.to_extra_args(),
                }
            )

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                require_unique_generation=nimloth_spec is not None,
                prompt_ids=agent_data.turn_prompt_ids,
                sampling_params = sampling_params_for_turn,
                image_data = image_data,
            )


        agent_data.turn_response_ids = output.token_ids
        agent_data.turn_response_mask = (
            _nimloth_response_mask(output.token_ids, nimloth_spec)
            if nimloth_spec is not None
            else [1] * len(output.token_ids)
        )
        agent_data.turn_prompt_ids += agent_data.turn_response_ids
        if nimloth_spec is not None:
            if output.policy_state is None:
                raise RuntimeError("Nimloth turn requires same-generation policy state")
            if output.policy_state.get("request_id") != agent_data.request_id:
                raise RuntimeError("Nimloth policy state request identity mismatch")
            generation_id = output.policy_state.get("generation_id")
            if (
                not isinstance(generation_id, str)
                or not generation_id
                or generation_id == agent_data.request_id
            ):
                raise RuntimeError("Nimloth policy state generation identity mismatch")
            agent_data.turn_generation_id = generation_id
            latent_hidden = output.policy_state.get("latent_hidden")
            action_logits = output.policy_state.get("action_logits")
            if (
                output.policy_state.get("schema") != "nimloth_policy_state_v2"
                or output.policy_state.get("latent_token_ids")
                != list(nimloth_spec.injected_token_ids[:-1])
                or output.policy_state.get("action_start_token_id")
                != nimloth_spec.injected_token_ids[-1]
                or output.policy_state.get("action_token_ids")
                != list(nimloth_spec.action_token_ids)
                or not isinstance(latent_hidden, list)
                or len(latent_hidden) != len(nimloth_spec.injected_token_ids) - 1
                or any(not isinstance(row, list) or not row for row in latent_hidden)
                or not isinstance(action_logits, list)
                or len(action_logits) != len(nimloth_spec.action_token_ids)
                or any(
                    not math.isfinite(float(value))
                    for row in latent_hidden
                    for value in row
                )
                or any(not math.isfinite(float(value)) for value in action_logits)
            ):
                raise RuntimeError("Nimloth turn returned invalid policy state")
            agent_data.turn_policy_state = output.policy_state
            if output.log_probs is None or len(output.log_probs) != len(output.token_ids):
                raise RuntimeError(
                    "Nimloth turn requires one rollout log-prob per response token"
                )
            if any(
                not math.isfinite(float(value))
                for value, sampled in zip(
                    output.log_probs,
                    agent_data.turn_response_mask,
                    strict=True,
                )
                if sampled
            ):
                raise RuntimeError(
                    "Nimloth sampled response token has non-finite log-probability"
                )
        if output.log_probs:
            agent_data.turn_response_logprobs = output.log_probs

        # Keep raw generated IDs for PPO, but omit trailing model terminators
        # from environment parsing while preserving Nimloth control tokens.
        env_response_ids = _strip_trailing_generation_terminators(
            agent_data.turn_response_ids,
            self.tokenizer,
        )
        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(env_response_ids, skip_special_tokens=False)
        )
        agent_data.last_assistant_text = (
            f"<think>{assistant_message}"
            if nimloth_spec is not None
            else assistant_message
        )
        return AgentState.INTERACTING

    async def _handle_env_state(self, agent_data: AgentData, **kwargs) -> AgentState:
        """
        Step the environment with last assistant action; always collect reward first.
        If terminal (done/success/turn-limit/token-limit), stop WITHOUT appending user suffix,
        so the episode ends on an assistant turn.
        """
        action_str = agent_data.last_assistant_text or ""
        try:
            obs, reward, done, info = await agent_data.env.step(action_str)
            # traceback
        except Exception as exc:
            logger.error(
                "Environment step failed in '%s' with action %r: %s",
                agent_data.env_name,
                action_str,
                exc,
            )
            logger.error("Environment traceback:\n%s", traceback.format_exc())
            obs, reward, done, info = {"obs_str":"Environment Error"}, 0.0, True, {"traj_success": False}

        traj_success = extract_success(info)
        agent_data.env_turns += 1
        last_turn=False
        
        
        
        if done:
            last_turn = True

        if self.env_max_turns is not None and agent_data.env_turns >= int(self.env_max_turns):
            last_turn = True

        
        if len(agent_data.turn_response_mask) >= self.response_length:
            last_turn = True

        decision_ledger = None
        if self.decision_ledger_enabled:
            decision_ledger = build_decision_ledger_from_env_info(
                info,
                env_turn_reward=reward,
                env_terminated=done,
                rollout_truncated=last_turn and not done,
            )

        turn_images=agent_data.sys_images+agent_data.cur_images
        
        resp_len = len(agent_data.turn_response_mask)
        response_ids = agent_data.turn_prompt_ids[-resp_len:] if resp_len else []
        prompt_ids = agent_data.turn_prompt_ids[: len(agent_data.turn_prompt_ids) - resp_len]
        multi_modal_data = {"image": turn_images} if turn_images else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids[-self.prompt_length:],
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.turn_response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=(
                agent_data.turn_response_logprobs[: self.response_length] if agent_data.turn_response_logprobs else None
            ),
            reward_score=float(reward),
            num_turns=1,
            metrics=agent_data.metrics,
            extra_fields={"reward_extra_info": {
                "traj_success": float(traj_success)},
                "image_data": turn_images,
                "last_turn": last_turn,
                **(
                    {"decision_ledger": decision_ledger}
                    if decision_ledger is not None
                    else {}
                ),
                **(
                    {"policy_state": agent_data.turn_policy_state}
                    if agent_data.turn_policy_state is not None
                    else {}
                ),
                "group_idx": agent_data.group_idx,
                "traj_idx": agent_data.traj_idx,
                "turn_idx": agent_data.env_turns,
                          
            },
        )
        agent_data.outputs.append(output)
        
        # update cur msg and images
        cur_msg={"role": "user", "content": convert_obs_to_content(obs, **kwargs)}
        cur_images=_normalize_images(obs.get("multi_modal_input", {}).get("<image>", []) or [])
        agent_data.cur_msg = cur_msg
        agent_data.cur_images = cur_images
        if last_turn:
            return AgentState.TERMINATED

        return AgentState.PENDING
