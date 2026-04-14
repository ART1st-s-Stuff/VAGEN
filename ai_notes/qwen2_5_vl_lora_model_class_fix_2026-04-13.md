# Qwen2.5-VL LoRA 模型类选择修复（2026-04-13）

- 现象：SFT FSDP 路径在 `get_peft_model(..., task_type=TaskType.CAUSAL_LM)` 时报错，提示底座模型 `Qwen2_5_VLModel` 缺少 `prepare_inputs_for_generation`。
- 根因：`verl/verl/utils/model.py` 中 `get_hf_auto_model_class()` 只按架构名片段匹配 `ForVision2Seq` / `ForCausalLM` 等后缀；`Qwen2_5_VLForConditionalGeneration` 没命中，错误回退到 `AutoModel`，因此加载成基座类而不是生成类。
- 修复：优先按 HF config 类型判断对应 auto class；若仍无法确定，则把 `*ForConditionalGeneration` 兜底映射到 `AutoModelForVision2Seq`。
- 影响：`verl/verl/workers/engine/fsdp/transformer_impl.py` 这条 SFT 引擎路径会重新拿到带生成接口的 VLM 模型，LoRA 包装与后续 `output.logits` 使用保持一致。
