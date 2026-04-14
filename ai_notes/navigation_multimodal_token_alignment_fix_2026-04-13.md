# Navigation 多模态 SFT token 对齐修复（2026-04-13）

- 背景：`NavigationMultimodalSFTDataset` 之前用 `processor(text, images)` 生成整条样本的多模态 `input_ids`，但又用 `tokenizer.encode()` 按 message 分段重构 token 和 `loss_mask`。
- 问题：对带图像的 Qwen2.5-VL 样本，`processor` 的 token 序列和纯文本 `tokenizer.encode()` 不一致，导致 warning 后 fallback 到一份不同长度的 `input_ids`，进一步和原始 `attention_mask` 失配。
- 修复：分段 token 计算改为统一走 `processor` 的前缀增量 token 化；同时在 token 协调后，若 `attention_mask` 与最终 `input_ids` 长度不一致，则改用 token-derived mask 计算 `position_ids`。
- 结果：多模态样本不再依赖“纯文本可重构”假设，`loss_mask`、`input_ids`、`position_ids` 在数据集阶段保持同一 token 空间。
