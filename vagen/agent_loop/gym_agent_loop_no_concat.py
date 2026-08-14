# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

import asyncio
import logging
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from PIL import Image
from .agent_loop_no_concat import AgentLoopBase, AgentLoopOutput, register
from .decision_ledger import (
    build_decision_ledger_from_env_info,
    build_guided_decision_ledger,
    parse_decision_ledger_enabled,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from ..envs.gym_image_env import GymImageEnv
from omegaconf import OmegaConf
from vagen.joint_policy import (
    FrozenQGuidedPolicyConfig,
    GuidedActionDrawKey,
    parse_joint_policy_section,
    sample_frozen_q_guided_action,
    validate_guided_action_execution_result,
)
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


@dataclass(frozen=True)
class JointGuidedTurnArtifacts:
    """Complete immutable provenance for one authorized environment step."""

    batch_pin: Any
    scoring_record: Any
    response_trace: Any
    action_draw: Any
    execution: Any


async def _build_joint_guided_execution(
    *,
    frozen_q_owner: Any,
    batch_pin: Mapping[str, Any],
    expected_draw_key: Mapping[str, Any],
    policy_config: FrozenQGuidedPolicyConfig,
    policy_state: Mapping[str, Any],
    response_ids: Sequence[int],
    response_mask: Sequence[int | bool],
    response_logprobs: Sequence[float],
    raw_response: str,
    generation_spec: Any,
    tokenizer: Any,
    action_space: str,
    action_space_names: Sequence[str],
) -> JointGuidedTurnArtifacts:
    """Score capture, consume a coordinator key, and authorize one action."""

    from nimloth.training.rl.joint_behavior import (
        NimlothPolicyResponseTrace,
        build_guided_execution_from_scoring,
    )
    from nimloth.training.rl.joint_frozen_q_owner import (
        FROZEN_Q_OWNER_SCORE_REQUEST_SCHEMA,
        FrozenQBatchPin,
        FrozenQOwnerScoringResult,
    )

    if frozen_q_owner is None:
        raise RuntimeError("joint guided execution requires a frozen Q owner")
    if not isinstance(policy_config, FrozenQGuidedPolicyConfig):
        raise ValueError(
            "joint guided execution requires FrozenQGuidedPolicyConfig"
        )
    if not isinstance(policy_state, Mapping):
        raise ValueError("joint guided execution policy_state must be a mapping")
    pin = FrozenQBatchPin.from_mapping(batch_pin)
    draw_key = GuidedActionDrawKey.from_mapping(expected_draw_key)
    config = FrozenQGuidedPolicyConfig.from_mapping(
        {
            field: getattr(policy_config, field)
            for field in policy_config.__dataclass_fields__
        }
    )
    if (
        draw_key.policy_step != pin.policy_step
        or draw_key.snapshot_id != pin.snapshot_id
        or draw_key.contract_id != pin.contract_id
    ):
        raise ValueError(
            "guided action draw key does not match the pinned rollout batch"
        )
    expected_contract = config.contract_id(
        action_space,
        action_space_names,
        generation_spec.action_token_ids,
    )
    if expected_contract != pin.contract_id:
        raise ValueError(
            "joint policy config and action table do not match frozen Q contract"
        )
    request_id = policy_state.get("request_id")
    generation_id = policy_state.get("generation_id")
    if (
        not isinstance(request_id, str)
        or not request_id
        or not isinstance(generation_id, str)
        or not generation_id
        or request_id == generation_id
    ):
        raise ValueError(
            "joint guided execution requires distinct request and generation identities"
        )
    score_result = FrozenQOwnerScoringResult.from_mapping(
        await frozen_q_owner.score.remote(
            {
                "schema": FROZEN_Q_OWNER_SCORE_REQUEST_SCHEMA,
                "batch_pin": pin.to_mapping(),
                "policy_state": dict(policy_state),
                "expected_request_id": request_id,
                "expected_generation_id": generation_id,
                "expected_latent_token_ids": list(
                    generation_spec.injected_token_ids[:-1]
                ),
                "expected_action_start_token_id": (
                    generation_spec.injected_token_ids[-1]
                ),
                "expected_action_token_ids": list(
                    generation_spec.action_token_ids
                ),
                "expected_contract_id": pin.contract_id,
            }
        )
    )
    if score_result.batch_pin != pin:
        raise ValueError("frozen Q scoring result does not match rollout batch pin")
    score = score_result.scoring_record
    trace = NimlothPolicyResponseTrace.build(
        request_id=request_id,
        generation_id=generation_id,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        raw_response=raw_response,
        generation_spec=generation_spec,
        tokenizer=tokenizer,
    )
    draw = sample_frozen_q_guided_action(
        action_space=action_space,
        action_space_names=action_space_names,
        action_token_ids=score.action_token_ids,
        prior_logits=score.prior_logits,
        frozen_all_action_q=score.frozen_all_action_q,
        draw_key=draw_key,
        config=config,
    )
    execution = build_guided_execution_from_scoring(
        scoring_record=score,
        expected_draw_key=draw_key,
        action_draw=draw,
        response_trace=trace,
        generation_spec=generation_spec,
        tokenizer=tokenizer,
        expected_request_id=request_id,
        expected_generation_id=generation_id,
        expected_snapshot_id=pin.snapshot_id,
        expected_contract_id=pin.contract_id,
        expected_generation_spec_id=trace.generation_spec_id,
    )
    return JointGuidedTurnArtifacts(
        batch_pin=pin,
        scoring_record=score,
        response_trace=trace,
        action_draw=draw,
        execution=execution,
    )


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
        joint_policy_batch_pin: Optional[Mapping[str, Any]] = None,
        guided_action_draw_keys: Optional[Sequence[Mapping[str, Any]]] = None,
        rollout_sample_id: Optional[str] = None,
        rollout_repeat_index: Optional[int] = None,
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
        self.joint_policy_batch_pin = joint_policy_batch_pin
        self.guided_action_draw_keys = guided_action_draw_keys
        self.rollout_sample_id = rollout_sample_id
        self.rollout_repeat_index = rollout_repeat_index
        # Token buffers
        self.turn_prompt_ids: Optional[List[int]] = None
        self.turn_response_ids: Optional[List[int]] = None
        self.turn_response_mask: Optional[List[int]] = None
        self.turn_response_logprobs: Optional[List[float]] = None
        self.turn_policy_state: Optional[Dict[str, Any]] = None
        self.turn_generation_id: Optional[str] = None
        self.turn_generation_spec: Optional[Any] = None
        self.turn_guided_artifacts: Optional[JointGuidedTurnArtifacts] = None

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
        raw_joint_policy = config.get("joint_policy")
        if OmegaConf.is_config(raw_joint_policy):
            raw_joint_policy = OmegaConf.to_container(
                raw_joint_policy,
                resolve=True,
            )
        cls.joint_policy_config = (
            None
            if raw_joint_policy is None
            else parse_joint_policy_section(raw_joint_policy)
        )
        if cls.joint_policy_config is not None and not cls.decision_ledger_enabled:
            raise ValueError(
                "joint guided rollout requires decision_ledger.enabled=true"
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
            prompt_format = env_config.get("prompt_format", "free_think")
            latent_token_count = env_config.get("latent_token_count")
            joint_enabled = self.joint_policy_config is not None
            if joint_enabled:
                if self.frozen_q_owner is None:
                    raise RuntimeError(
                        "joint guided rollout requires the frozen Q owner"
                    )
                if prompt_format != "nimloth":
                    raise ValueError(
                        "joint guided rollout supports only prompt_format=nimloth"
                    )
                if not isinstance(kwargs.get("joint_policy_batch_pin"), Mapping):
                    raise ValueError(
                        "joint guided rollout requires a manager-issued batch pin"
                    )
                draw_keys = kwargs.get("guided_action_draw_keys")
                if (
                    isinstance(draw_keys, (str, bytes, Mapping))
                    or not isinstance(draw_keys, Sequence)
                    or len(draw_keys) != int(kwargs["max_turns"])
                ):
                    raise ValueError(
                        "joint guided rollout requires one manager-issued draw key per turn"
                    )
                if (
                    not isinstance(kwargs.get("rollout_sample_id"), str)
                    or not kwargs["rollout_sample_id"]
                    or kwargs.get("rollout_repeat_index") is None
                ):
                    raise ValueError(
                        "joint guided rollout requires stable sample and repeat identity"
                    )
            elif self.frozen_q_owner is not None:
                raise RuntimeError(
                    "frozen Q owner cannot be attached while joint policy is disabled"
                )

            # Bootstrap: reset -> system_prompt (message order: system, then initial user)
            init_obs, info = await env.reset(seed=seed)
            sys_obs = await env.system_prompt()
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
                joint_policy_batch_pin=kwargs.get("joint_policy_batch_pin"),
                guided_action_draw_keys=kwargs.get("guided_action_draw_keys"),
                rollout_sample_id=kwargs.get("rollout_sample_id"),
                rollout_repeat_index=kwargs.get("rollout_repeat_index"),
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
        agent_data.turn_guided_artifacts = None
        agent_data.turn_generation_spec = None
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
            agent_data.turn_generation_spec = nimloth_spec
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
        if self.joint_policy_config is not None:
            if nimloth_spec is None or agent_data.turn_policy_state is None:
                raise RuntimeError(
                    "joint guided rollout requires validated Nimloth capture"
                )
            if agent_data.turn_response_logprobs is None:
                raise RuntimeError(
                    "joint guided rollout requires rollout response log-probabilities"
                )
            if agent_data.env_turns >= len(agent_data.guided_action_draw_keys):
                raise RuntimeError("joint guided rollout exhausted manager draw keys")
            from vagen.envs.navigation.utils.nimloth_format import ACTION_NAMES

            agent_data.turn_guided_artifacts = await _build_joint_guided_execution(
                frozen_q_owner=self.frozen_q_owner,
                batch_pin=agent_data.joint_policy_batch_pin,
                expected_draw_key=(
                    agent_data.guided_action_draw_keys[agent_data.env_turns]
                ),
                policy_config=self.joint_policy_config,
                policy_state=agent_data.turn_policy_state,
                response_ids=agent_data.turn_response_ids,
                response_mask=agent_data.turn_response_mask,
                response_logprobs=agent_data.turn_response_logprobs,
                raw_response=agent_data.last_assistant_text,
                generation_spec=nimloth_spec,
                tokenizer=self.tokenizer,
                action_space="navigation_v1",
                action_space_names=ACTION_NAMES,
            )
        return AgentState.INTERACTING

    async def _handle_env_state(self, agent_data: AgentData, **kwargs) -> AgentState:
        """
        Step the environment with last assistant action; always collect reward first.
        If terminal (done/success/turn-limit/token-limit), stop WITHOUT appending user suffix,
        so the episode ends on an assistant turn.
        """
        action_str = agent_data.last_assistant_text or ""
        artifacts = agent_data.turn_guided_artifacts
        if artifacts is not None:
            guided_action_execution = artifacts.execution.to_mapping()
            # Guided contract errors must escape; fabricating a terminal fallback
            # would hide whether the authorized action was actually executed.
            obs, reward, done, info = await agent_data.env.guided_step(
                action_str,
                guided_action_execution=guided_action_execution,
            )
            validate_guided_action_execution_result(
                info,
                artifacts.execution,
            )
        else:
            try:
                obs, reward, done, info = await agent_data.env.step(action_str)
            except Exception as exc:
                logger.error(
                    "Environment step failed in '%s' with action %r: %s",
                    agent_data.env_name,
                    action_str,
                    exc,
                )
                logger.error("Environment traceback:\n%s", traceback.format_exc())
                obs, reward, done, info = (
                    {"obs_str": "Environment Error"},
                    0.0,
                    True,
                    {"traj_success": False},
                )

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
            if artifacts is None:
                decision_ledger = build_decision_ledger_from_env_info(
                    info,
                    env_turn_reward=reward,
                    env_terminated=done,
                    rollout_truncated=last_turn and not done,
                )
            else:
                format_valid = info.get("format_correct")
                if not isinstance(format_valid, bool):
                    raise ValueError(
                        "guided environment info format_correct must be bool"
                    )
                decision_ledger = build_guided_decision_ledger(
                    behavior=artifacts.execution.behavior_record,
                    env_turn_reward=reward,
                    env_terminated=done,
                    rollout_truncated=last_turn and not done,
                    format_valid=format_valid,
                )

        turn_images=agent_data.sys_images+agent_data.cur_images
        
        resp_len = len(agent_data.turn_response_mask)
        response_ids = agent_data.turn_prompt_ids[-resp_len:] if resp_len else []
        prompt_ids = agent_data.turn_prompt_ids[: len(agent_data.turn_prompt_ids) - resp_len]
        multi_modal_data = {"image": turn_images} if turn_images else {}
        extra_fields: Dict[str, Any] = {
            "reward_extra_info": {"traj_success": float(traj_success)},
            "image_data": turn_images,
            "last_turn": last_turn,
            "group_idx": agent_data.group_idx,
            "traj_idx": agent_data.traj_idx,
            "turn_idx": agent_data.env_turns,
        }
        if decision_ledger is not None:
            extra_fields["decision_ledger"] = decision_ledger
        if agent_data.turn_policy_state is not None:
            extra_fields["policy_state"] = agent_data.turn_policy_state
        if agent_data.rollout_sample_id is not None:
            extra_fields["rollout_sample_id"] = agent_data.rollout_sample_id
        if agent_data.rollout_repeat_index is not None:
            extra_fields["rollout_repeat_index"] = agent_data.rollout_repeat_index
        if artifacts is not None:
            guided_turn_index = artifacts.action_draw.draw_key.turn_index
            if guided_turn_index != agent_data.env_turns - 1:
                raise RuntimeError(
                    "guided draw turn index does not match executed environment turn"
                )
            extra_fields.update(
                {
                    # Historical no-concat turn_idx is one-based. This explicit
                    # field preserves the coordinator's zero-based draw identity.
                    "guided_turn_index": guided_turn_index,
                    "joint_policy_batch_pin": artifacts.batch_pin.to_mapping(),
                    "frozen_q_scoring": artifacts.scoring_record.to_mapping(),
                    "policy_response_trace": artifacts.response_trace.to_mapping(),
                    "guided_action_draw": artifacts.action_draw.to_mapping(),
                    "guided_action_execution": artifacts.execution.to_mapping(),
                }
            )
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
            extra_fields=extra_fields,
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
