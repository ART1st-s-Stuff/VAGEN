# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import hashlib
import heapq
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from numbers import Integral
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.experimental.reward import RewardManagerWorker
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
    rollout_trace_op,
)
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class

from .decision_ledger import last_policy_token_index

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)
        self._generation_namespace = uuid4().hex
        self._generation_counter = 0

    def _next_generation_id(self) -> str:
        self._generation_counter += 1
        return f"{self._generation_namespace}:{self._generation_counter}"

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        require_unique_generation: bool = False,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        if not isinstance(require_unique_generation, bool):
            raise ValueError("require_unique_generation must be bool")
        generation_id = (
            self._next_generation_id()
            if require_unique_generation
            else request_id
        )
        server = self._choose_server(request_id)
        generate_kwargs = {
            "request_id": generation_id,
            "prompt_ids": prompt_ids,
            "sampling_params": sampling_params,
            "image_data": image_data,
        }
        if require_unique_generation:
            generate_kwargs["session_request_id"] = request_id
        output = await server.generate.remote(**generate_kwargs)
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


# make hydra.utils.instantiate happy
class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        frozen_q_owner: Optional[ray.actor.ActorHandle] = None,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
        """
        self.init_class(config=trainer_config.config, tokenizer=tokenizer, processor=processor, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.frozen_q_owner = frozen_q_owner
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor: AutoProcessor, **kwargs):
        """This is used to do heavy initialization work that should shared across all instances. It's only called once.

        Args:
            config (DictConfig): trainer config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process multi_modal data.
            **kwargs: extra kwargs from config file passed in by `hydra.utils.instantiate`.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorkerBase:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
        frozen_q_owner: Optional[ray.actor.ActorHandle] = None,
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config

        # for recipe to change
        if not hasattr(self, "server_manager"):
            self.server_manager = AsyncLLMServerManager(config, server_handles)

        self.reward_router_address = reward_router_address
        self.frozen_q_owner = frozen_q_owner

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, self.reward_router_address)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "rollout_sample_id" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["rollout_sample_id"]
        elif "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)))
        outputs = await asyncio.gather(*tasks)

        # Flatten the nested list since _run_agent_loop now returns List[_InternalAgentLoopOutput]
        flattened_outputs = [item for sublist in outputs for item in sublist]

        output = self._postprocess(flattened_outputs)
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> list[_InternalAgentLoopOutput]:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            instantiate_kwargs = {
                "config": agent_loop_config,
                "trainer_config": _DummyConfig(config=self.config),
                "server_manager": self.server_manager,
                "tokenizer": self.tokenizer,
                "processor": self.processor,
            }
            # Preserve the historical custom AgentLoop constructor signature
            # when joint rollout is disabled.
            if self.frozen_q_owner is not None:
                instantiate_kwargs["frozen_q_owner"] = self.frozen_q_owner
            agent_loop = hydra.utils.instantiate(**instantiate_kwargs)
            outputs: list[AgentLoopOutput] = await agent_loop.run(sampling_params, **kwargs)

            # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

            # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
            # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
            # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
            # input_ids: concatenation of prompt + response
            # Mask:
            # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
            # - prompt_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [0,0,0,0,1,1,1,1]
            # - response_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
            #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
            # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
            #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
            # - position_ids: sequential positions for tokens, starting at 0
            #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

            internal_outputs = []
            for output in outputs:
                self.tokenizer.padding_side = "left"
                prompt_output = self.tokenizer.pad(
                    {"input_ids": output.prompt_ids},
                    padding="max_length",
                    max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                if prompt_output["input_ids"].dim() == 1:
                    prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                    prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

                self.tokenizer.padding_side = "right"
                response_output = self.tokenizer.pad(
                    {"input_ids": output.response_ids},
                    padding="max_length",
                    max_length=self.config.actor_rollout_ref.rollout.response_length,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                if response_output["input_ids"].dim() == 1:
                    response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                    response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

                response_mask_output = self.tokenizer.pad(
                    {"input_ids": output.response_mask},
                    padding="max_length",
                    max_length=self.config.actor_rollout_ref.rollout.response_length,
                    return_tensors="pt",
                    return_attention_mask=False,
                )
                if response_mask_output["input_ids"].dim() == 1:
                    response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

                response_logprobs = None
                if output.response_logprobs is not None:
                    pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                    response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

                response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
                attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
                input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

                # Handle multi-modal inputs and position_ids calculation
                # Only support Qwen2VLImageProcessor for multi-modal processing currently
                # TODO: support other multi-modal inputs
                multi_modal_inputs = None
                if (
                    self.processor is not None
                    and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
                ):
                    from verl.models.transformers.qwen2_vl import get_rope_index

                    images = getattr(output, "multi_modal_data", {}).get("image", None)
                    current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                    multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                    multi_modal_inputs.pop("input_ids", None)
                    multi_modal_inputs.pop("attention_mask", None)

                    # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
                    # because np.array() only keeps the keys for BatchFeature.
                    multi_modal_inputs = dict(multi_modal_inputs)

                    image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                    video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                    second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                    vision_position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids.squeeze(0),
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        attention_mask=attention_mask.squeeze(0),
                    ).unsqueeze(0)  # (1, 3, seq_len)

                    valid_mask = attention_mask[0].bool()
                    text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                    text_position_ids = text_position_ids.unsqueeze(0)
                    position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
                else:
                    position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)
                enable_async_reward = (
                    self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
                ) or not self.config.reward_model.enable
                if output.reward_score is None and enable_async_reward:
                    batch = TensorDict(
                        {
                            "prompts": prompt_output["input_ids"],  # [1, prompt_length]
                            "responses": response_output["input_ids"],  # [1, response_length]
                            "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                            "input_ids": input_ids,  # [1, prompt_length + response_length]
                            "position_ids": position_ids,
                        },
                        batch_size=1,
                    )
                    non_tensor_batch = {
                        **{k: np.array([v]) for k, v in kwargs.items()},
                        "__num_turns__": np.array([output.num_turns]),
                        "tool_extra_fields": np.array([output.extra_fields], dtype=object),
                    }

                    data = DataProto(
                        batch=batch,
                        non_tensor_batch=non_tensor_batch,
                    )
                    result = await self.reward_manager_worker.compute_score.remote(data)
                    output.reward_score = result["reward_score"]
                    output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

                internal_outputs.append(_InternalAgentLoopOutput(
                    prompt_ids=prompt_output["input_ids"],
                    response_ids=response_output["input_ids"],
                    input_ids=input_ids,
                    position_ids=position_ids,
                    response_mask=response_mask,
                    attention_mask=attention_mask,
                    response_logprobs=response_logprobs,
                    multi_modal_inputs=multi_modal_inputs,
                    multi_modal_data=output.multi_modal_data,
                    reward_score=output.reward_score,
                    num_turns=output.num_turns,
                    metrics=output.metrics,
                    extra_fields=output.extra_fields,
                ))

            return internal_outputs

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            reward_positions = torch.tensor(
                [last_policy_token_index(mask.tolist()) for mask in response_mask],
                dtype=torch.long,
                device=response_mask.device,
            )
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[
                torch.arange(response_mask.size(0), device=response_mask.device),
                reward_positions,
            ] = torch.tensor(scores, dtype=torch.float32, device=response_mask.device)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = {"policy_state"}
        all_keys.update(
            key for input_item in inputs for key in input_item.extra_fields
        )
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

    def create_transferqueue_client(self, controller_infos, storage_infos, role):
        """Create a client for data system(transfer queue)."""
        from verl.single_controller.ray.base import get_random_string
        from verl.utils.transferqueue_utils import create_transferqueue_client

        client_name = get_random_string(length=6)
        create_transferqueue_client(
            client_id=f"{role}_worker_{client_name}",
            controller_infos=controller_infos,
            storage_infos=storage_infos,
        )


@ray.remote
class AgentLoopWorker(AgentLoopWorkerBase):
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
        frozen_q_owner: Optional[ray.actor.ActorHandle] = None,
    ):
        """Initialize agent loop manager.
        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_router_address (str): reward router address.
        """
        super().__init__(
            config,
            server_handles,
            reward_router_address,
            frozen_q_owner,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


def _nonnegative_integral(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"guided rollout {field} must be a non-negative integer")
    return int(value)


def _positive_integral(value: object, field: str) -> int:
    result = _nonnegative_integral(value, field)
    if result == 0:
        raise ValueError(f"guided rollout {field} must be positive")
    return result


def _frozen_q_batch_id(
    *,
    run_seed: int,
    policy_step: int,
    is_validation: bool,
    snapshot_id: str,
    activation_version: int,
    contract_id: str,
    trajectories: list[dict[str, Any]],
) -> str:
    payload = json.dumps(
        {
            "schema": "vagen_frozen_q_rollout_batch_identity_v1",
            "run_seed": run_seed,
            "policy_step": policy_step,
            "is_validation": is_validation,
            "snapshot_id": snapshot_id,
            "activation_version": activation_version,
            "contract_id": contract_id,
            "trajectories": trajectories,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rm_wg: RayWorkerGroup = None,
        *,
        initial_frozen_q_snapshot_state: Optional[dict[str, Any]] = None,
        frozen_q_activation_version: int = 0,
        guided_draw_run_seed: Optional[int] = None,
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
        """
        self.config = config
        self.worker_group = worker_group
        self.joint_policy_config = self._parse_joint_policy_config()
        self.guided_draw_run_seed = self._validate_joint_runtime_inputs(
            initial_frozen_q_snapshot_state,
            guided_draw_run_seed,
        )
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward import RewardModelManager

            self.reward_model_manager = RewardModelManager(config.reward_model, rm_wg)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = AgentLoopWorker

        self._initialize_llm_servers()
        self._init_frozen_q_owner(
            initial_frozen_q_snapshot_state,
            frozen_q_activation_version,
        )
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _parse_joint_policy_config(self):
        raw = self.config.get("joint_policy")
        if raw is None:
            return None
        from vagen.joint_policy import parse_joint_policy_section

        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
        return parse_joint_policy_section(raw)

    def _validate_joint_runtime_inputs(
        self,
        initial_snapshot_state: Optional[dict[str, Any]],
        guided_draw_run_seed: Optional[int],
    ) -> Optional[int]:
        enabled = self.joint_policy_config is not None
        if not enabled:
            if initial_snapshot_state is not None or guided_draw_run_seed is not None:
                raise ValueError(
                    "frozen Q runtime inputs require joint_policy.enabled=true"
                )
            return None
        if initial_snapshot_state is None:
            raise ValueError(
                "joint_policy.enabled=true requires an initial frozen Q snapshot state"
            )
        if (
            isinstance(guided_draw_run_seed, bool)
            or not isinstance(guided_draw_run_seed, Integral)
            or int(guided_draw_run_seed) < 0
        ):
            raise ValueError(
                "joint_policy.enabled=true requires a non-negative guided draw run seed"
            )
        return int(guided_draw_run_seed)

    def _initialize_llm_servers(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.worker_group:
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(rollout_config.prometheus, self.server_addresses)

    def _init_frozen_q_owner(
        self,
        initial_frozen_q_snapshot_state: Optional[dict[str, Any]],
        activation_version: int,
    ) -> None:
        self.frozen_q_owner = None
        if initial_frozen_q_snapshot_state is None:
            return
        from vagen.joint_policy.frozen_q_actor import FrozenQScoringActor

        owner = FrozenQScoringActor.remote(
            initial_frozen_q_snapshot_state,
            activation_version=activation_version,
        )
        try:
            # Construction, CPU/thread resource validation, and snapshot restore
            # must finish before any worker can receive the handle.
            status = ray.get(owner.status.remote())
            if status["score_dtype"] != self.joint_policy_config.score_dtype:
                raise ValueError(
                    "frozen Q snapshot score_dtype does not match joint policy config"
                )
        except BaseException:
            ray.kill(owner, no_restart=True)
            raise
        self.frozen_q_owner = owner

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.server_handles,
                    self.reward_router_address,
                    *(
                        (self.frozen_q_owner,)
                        if self.frozen_q_owner is not None
                        else ()
                    ),
                )
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Pin one snapshot around the complete distributed rollout batch."""

        rollout_awake = False
        reward_awake = False
        joint_policy_batch_pin = None
        try:
            if self.config.actor_rollout_ref.rollout.free_cache_engine:
                self.wake_up()
                rollout_awake = True
            if (
                self.reward_model_manager
                and self.config.reward_model.rollout.free_cache_engine
            ):
                self.reward_model_manager.wake_up()
                reward_awake = True

            if self.frozen_q_owner is not None:
                joint_policy_batch_pin = self._pin_frozen_q_batch(prompts)

            chunks = prompts.chunk(len(self.agent_loop_workers))
            outputs = ray.get(
                [
                    worker.generate_sequences.remote(chunk)
                    for worker, chunk in zip(
                        self.agent_loop_workers,
                        chunks,
                        strict=True,
                    )
                ]
            )
            output = DataProto.concat(outputs)

            metrics = [
                item.meta_info.pop("metrics") for item in outputs
            ]
            timing = self._performance_metrics(metrics, output)
            output.meta_info = {"timing": timing, **outputs[0].meta_info}
            return output
        finally:
            try:
                if joint_policy_batch_pin is not None:
                    ray.get(
                        self.frozen_q_owner.unpin_batch.remote(
                            joint_policy_batch_pin
                        )
                    )
            finally:
                if reward_awake:
                    self.reward_model_manager.sleep()
                if rollout_awake:
                    self.sleep()

    def _pin_frozen_q_batch(self, prompts: DataProto) -> dict[str, Any]:
        """Allocate authoritative per-turn draw keys and pin the active Q."""

        if self.frozen_q_owner is None or self.guided_draw_run_seed is None:
            raise RuntimeError("frozen Q batch pin requested while runtime is disabled")
        if len(prompts) <= 0:
            raise ValueError("frozen Q rollout batch must be non-empty")
        policy_step = _nonnegative_integral(
            prompts.meta_info.get("global_steps"),
            "global_steps",
        )
        is_validation = prompts.meta_info.get("validate", False)
        if not isinstance(is_validation, bool):
            raise ValueError("guided rollout validate metadata must be bool")
        required = {
            "rollout_sample_id",
            "rollout_repeat_index",
            "max_turns",
        }
        missing = required - set(prompts.non_tensor_batch)
        if missing:
            raise ValueError(
                "guided rollout batch is missing stable identity fields: "
                f"{sorted(missing)}"
            )

        sample_ids: list[str] = []
        repeat_indices: list[int] = []
        max_turns: list[int] = []
        for row in range(len(prompts)):
            sample_id = prompts.non_tensor_batch["rollout_sample_id"][row]
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("guided rollout sample identity must be non-empty str")
            repeat_index = _nonnegative_integral(
                prompts.non_tensor_batch["rollout_repeat_index"][row],
                "rollout_repeat_index",
            )
            turn_count = _positive_integral(
                prompts.non_tensor_batch["max_turns"][row],
                "max_turns",
            )
            sample_ids.append(sample_id)
            repeat_indices.append(repeat_index)
            max_turns.append(turn_count)
        trajectory_ids = list(zip(sample_ids, repeat_indices, strict=True))
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError(
                "guided rollout batch contains duplicate sample/repeat identities"
            )

        status = ray.get(self.frozen_q_owner.status.remote())
        batch_id = _frozen_q_batch_id(
            run_seed=self.guided_draw_run_seed,
            policy_step=policy_step,
            is_validation=is_validation,
            snapshot_id=status["active_snapshot_id"],
            activation_version=status["activation_version"],
            contract_id=status["contract_id"],
            trajectories=[
                {
                    "rollout_sample_id": sample_id,
                    "rollout_repeat_index": repeat_index,
                    "max_turns": turn_count,
                }
                for sample_id, repeat_index, turn_count in zip(
                    sample_ids,
                    repeat_indices,
                    max_turns,
                    strict=True,
                )
            ],
        )

        from nimloth.training.rl.joint_frozen_q_owner import FrozenQBatchPin
        from vagen.joint_policy import GuidedActionDrawCoordinator

        expected_pin = FrozenQBatchPin(
            schema="nimloth_frozen_q_batch_pin_v1",
            batch_id=batch_id,
            policy_step=policy_step,
            snapshot_id=status["active_snapshot_id"],
            snapshot_source_step=status["active_source_step"],
            contract_id=status["contract_id"],
            activation_version=status["activation_version"],
        )
        coordinator = GuidedActionDrawCoordinator(self.guided_draw_run_seed)
        draw_key_rows: list[list[dict[str, Any]]] = []
        for sample_id, repeat_index, turn_count in zip(
            sample_ids,
            repeat_indices,
            max_turns,
            strict=True,
        ):
            draw_key_rows.append(
                [
                    coordinator.key_for(
                        policy_step=policy_step,
                        rollout_sample_id=sample_id,
                        rollout_repeat_index=repeat_index,
                        turn_index=turn_index,
                        is_validation=is_validation,
                        snapshot_id=expected_pin.snapshot_id,
                        contract_id=expected_pin.contract_id,
                    ).to_mapping()
                    for turn_index in range(turn_count)
                ]
            )

        pin_rows = np.empty(len(prompts), dtype=object)
        pin_rows[:] = [expected_pin.to_mapping() for _ in range(len(prompts))]
        key_rows = np.empty(len(prompts), dtype=object)
        key_rows[:] = draw_key_rows

        try:
            raw_pin = ray.get(
                self.frozen_q_owner.pin_batch.remote(
                    {
                        "batch_id": batch_id,
                        "policy_step": policy_step,
                        "expected_snapshot_id": status["active_snapshot_id"],
                        "expected_activation_version": status["activation_version"],
                    }
                )
            )
        except BaseException:
            # The actor may have applied the idempotent pin before an object-store
            # or transport failure reached the manager. Use the already-derived
            # authoritative pin to make best-effort cleanup without replacing the
            # original failure.
            try:
                ray.get(
                    self.frozen_q_owner.unpin_batch.remote(
                        expected_pin.to_mapping()
                    )
                )
            except BaseException:
                logger.exception(
                    "failed to clean up frozen Q batch after pin RPC failure"
                )
            raise
        try:
            actual_pin = FrozenQBatchPin.from_mapping(raw_pin)
            if actual_pin != expected_pin:
                raise RuntimeError(
                    "frozen Q owner returned an unexpected batch pin"
                )
            prompts.non_tensor_batch["joint_policy_batch_pin"] = pin_rows
            prompts.non_tensor_batch["guided_action_draw_keys"] = key_rows
        except BaseException:
            ray.get(self.frozen_q_owner.unpin_batch.remote(raw_pin))
            raise
        return actual_pin.to_mapping()

    def frozen_q_status(self) -> dict[str, Any]:
        if self.frozen_q_owner is None:
            raise RuntimeError("frozen Q owner is disabled")
        return ray.get(self.frozen_q_owner.status.remote())

    def stage_frozen_q_snapshot(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.frozen_q_owner is None:
            raise RuntimeError("frozen Q owner is disabled")
        return ray.get(self.frozen_q_owner.stage_snapshot.remote(request))

    def activate_staged_frozen_q_snapshot(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        if self.frozen_q_owner is None:
            raise RuntimeError("frozen Q owner is disabled")
        return ray.get(self.frozen_q_owner.activate_staged.remote(request))

    def frozen_q_checkpoint_state(self) -> dict[str, Any]:
        if self.frozen_q_owner is None:
            raise RuntimeError("frozen Q owner is disabled")
        return ray.get(self.frozen_q_owner.checkpoint_state.remote())

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
