from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.vision_utils import process_image
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def _convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: _convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    if isinstance(data_item, list):
        return [_convert_nested_value_to_list_recursive(elem) for elem in data_item]
    if isinstance(data_item, np.ndarray):
        return _convert_nested_value_to_list_recursive(data_item.tolist())
    return data_item


class NavigationMultimodalSFTDataset(Dataset):
    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer,
        config=None,
        max_samples: int = -1,
        processor=None,
    ):
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.max_samples = max_samples
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", config.get("messages_key", "messages"))
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", config.get("enable_thinking_key", "enable_thinking"))

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        self.parquet_files = list(parquet_files)

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor
        if self.processor is None:
            raise ValueError("NavigationMultimodalSFTDataset requires a processor for multimodal inputs")

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_single_file(self, file_path: str) -> pd.DataFrame:
        if file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        if file_path.endswith(".json"):
            return pd.read_json(file_path)
        if file_path.endswith(".jsonl"):
            return pd.read_json(file_path, lines=True)
        raise ValueError(f"Unsupported dataset format: {file_path}")

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = [self._read_single_file(parquet_file) for parquet_file in self.parquet_files]
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        total = len(self.dataframe)
        print(f"dataset len: {total}")
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"selected {self.max_samples} random samples out of {total}")

        self.messages = self.dataframe[self.messages_key].apply(series_to_item).apply(_convert_nested_value_to_list_recursive).tolist()
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = [None] * len(self.messages)

    def __len__(self):
        return len(self.messages)

    def _apply_chat_template(self, messages, enable_thinking: Optional[bool], add_generation_prompt: bool) -> str:
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            **self.apply_chat_template_kwargs,
        )

    def _process_message_tokens(
        self,
        messages: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        if start_idx > 0:
            prev_applied_text = self._apply_chat_template(
                messages[:start_idx], enable_thinking=enable_thinking, add_generation_prompt=False
            )
            if is_assistant:
                prev_applied_text_w_generation_prompt = self._apply_chat_template(
                    messages[:start_idx], enable_thinking=enable_thinking, add_generation_prompt=True
                )
        else:
            prev_applied_text = ""

        cur_applied_text = self._apply_chat_template(
            messages[:end_idx], enable_thinking=enable_thinking, add_generation_prompt=False
        )
        if is_assistant:
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text) :]
            generation_prompt_tokens = self.tokenizer.encode(generation_prompt_text, add_special_tokens=False)
            message_tokens_only = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text_w_generation_prompt) :], add_special_tokens=False
            )
            message_tokens = generation_prompt_tokens + message_tokens_only
            loss_mask = [0] * len(generation_prompt_tokens) + [1] * len(message_tokens_only)
        else:
            message_tokens = self.tokenizer.encode(cur_applied_text[len(prev_applied_text) :], add_special_tokens=False)
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)
        return message_tokens, loss_mask, attention_mask

    def _validate_and_convert_tokens(
        self,
        full_tokens: torch.Tensor,
        concat_tokens: list[int],
        concat_loss_mask: list[int],
        concat_attention_mask: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_tokens_list = full_tokens.tolist()
        if len(concat_tokens) != len(full_tokens_list) or not all(
            left == right for left, right in zip(concat_tokens, full_tokens_list, strict=True)
        ):
            logger.warning(
                "Token mismatch detected in NavigationMultimodalSFTDataset. Falling back to concatenated tokens."
            )
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.long),
                torch.tensor(concat_attention_mask, dtype=torch.long),
            )
        return (
            full_tokens,
            torch.tensor(concat_loss_mask, dtype=torch.long),
            torch.tensor(concat_attention_mask, dtype=torch.long),
        )

    def _collect_images(self, messages: list[dict[str, Any]]) -> list[Any]:
        images: list[Any] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                image_value = item.get("image")
                if image_value is None:
                    continue
                if isinstance(image_value, dict):
                    images.append(process_image(image_value))
                else:
                    images.append(process_image({"image": image_value}))
        return images

    def _compute_position_ids(self, input_ids, attention_mask, model_inputs):
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask,
            )
            valid_mask = attention_mask.bool()
            text_position_ids = torch.ones((1, len(input_ids)), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            return torch.cat((text_position_ids, vision_position_ids), dim=0)

        if self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask,
            )
            valid_mask = attention_mask.bool()
            text_position_ids = torch.ones((1, len(input_ids)), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            return torch.cat((text_position_ids, vision_position_ids), dim=0)

        return compute_position_id_with_mask(attention_mask)

    def _truncate_tensor(self, tensor: torch.Tensor, max_length: int, keep: str) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor[-max_length:] if keep == "right" else tensor[:max_length]
        return tensor[..., -max_length:] if keep == "right" else tensor[..., :max_length]

    def __getitem__(self, item):
        messages = self.messages[item]
        enable_thinking = self.enable_thinking[item]
        images = self._collect_images(messages)

        full_text = self._apply_chat_template(messages, enable_thinking=enable_thinking, add_generation_prompt=False)
        processor_outputs = self.processor(text=[full_text], images=images or None, return_tensors="pt")
        input_ids = processor_outputs.pop("input_ids")[0]
        attention_mask = processor_outputs.pop("attention_mask")[0]

        concat_tokens: list[int] = []
        concat_loss_mask: list[int] = []
        concat_attention_mask: list[int] = []

        idx = 0
        while idx < len(messages):
            current_message = messages[idx]
            role = current_message["role"]
            if role == "assistant":
                tokens, loss_mask, attn_mask = self._process_message_tokens(
                    messages, idx, idx + 1, is_assistant=True, enable_thinking=enable_thinking
                )
                idx += 1
            elif role in ("user", "system"):
                tokens, loss_mask, attn_mask = self._process_message_tokens(
                    messages, idx, idx + 1, enable_thinking=enable_thinking
                )
                idx += 1
            else:
                raise ValueError(f"Unknown role: {role}")

            override_loss_mask = current_message.get("loss_mask")
            if override_loss_mask is not None:
                if isinstance(override_loss_mask, np.ndarray):
                    override_loss_mask = override_loss_mask.item()
                assert int(override_loss_mask) in (0, 1), f"loss_mask should be 0 or 1, got {override_loss_mask}"
                loss_mask = [int(override_loss_mask)] * len(tokens)

            concat_tokens.extend(tokens)
            concat_loss_mask.extend(loss_mask)
            concat_attention_mask.extend(attn_mask)

        input_ids, loss_mask, attention_mask_from_tokens = self._validate_and_convert_tokens(
            input_ids,
            concat_tokens,
            concat_loss_mask,
            concat_attention_mask,
        )
        position_ids = self._compute_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            model_inputs=processor_outputs,
        )

        sequence_length = input_ids.shape[0]
        if sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
                attention_mask_from_tokens = attention_mask_from_tokens[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                position_ids = self._truncate_tensor(position_ids, self.max_length, keep="right")
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                attention_mask_from_tokens = attention_mask_from_tokens[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                position_ids = self._truncate_tensor(position_ids, self.max_length, keep="left")
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        multi_modal_inputs = dict(processor_outputs)
        multi_modal_inputs.pop("second_per_grid_ts", None)
        sample = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask_from_tokens,
            "multi_modal_inputs": multi_modal_inputs,
        }

        if self.pad_mode == DatasetPadMode.NO_PADDING:
            sample.pop("attention_mask")
            return sample

        if self.pad_mode != DatasetPadMode.RIGHT:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

        current_length = input_ids.shape[0]
        if current_length < self.max_length:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.full((self.max_length - current_length,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros((self.max_length - current_length,), dtype=attention_mask_from_tokens.dtype)
            padded_loss_mask = torch.zeros((self.max_length - current_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask_from_tokens = torch.cat((attention_mask_from_tokens, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))

            if position_ids.dim() == 1:
                padded_position_ids = torch.zeros((self.max_length - current_length,), dtype=position_ids.dtype)
                position_ids = torch.cat((position_ids, padded_position_ids))
            else:
                padded_position_ids = torch.zeros(
                    (position_ids.shape[0], self.max_length - current_length), dtype=position_ids.dtype
                )
                position_ids = torch.cat((position_ids, padded_position_ids), dim=-1)

        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask_from_tokens
        sample["position_ids"] = position_ids
        sample["loss_mask"] = loss_mask
        return sample
