# Navigation Phase1 dataloader debug 记录（2026-04-13）

- 现象：`StatefulDataLoader` 在 `num_workers > 0` 时把 worker 内异常包装成 `ExceptionWrapper`，主进程只显示 `TypeError: cannot unpack non-iterable ExceptionWrapper object`，不利于定位真实错误。
- 处理：为 `verl/verl/trainer/sft_trainer.py` 增加 `data.num_workers` 配置项，默认保留 `8`。
- 对 `vagen/configs/navigation_sft.yaml` 单独设置 `data.num_workers: 0`，让当前 Phase1 多模态 SFT debug 在主进程直接抛出原始异常。
- 说明：这是调试可观测性修复，不替代业务逻辑修复；若后续定位清楚并稳定后，可再把 worker 数调回更高值。
