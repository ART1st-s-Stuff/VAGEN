# Predictor Pipeline 首轮执行记录（2026-04-10）

## 1) 本轮完成项

- 对齐了 `ActorConfig -> ActorWorker` 的 Predictor/MCTS 接口启用条件。
- 修复了 actor 侧 log-prob 输出字段与 trainer 读取字段不一致问题：
  - 输出由 `entropy` 对齐为 `entropys`（与 `ray_trainer.py` 消费侧一致）。
- 增强了 Predictor 相关训练可观测性：
  - `actor/world_model_enabled`
  - `actor/world_model_supervision_missing`
  - `actor/world_model_loss`
  - `actor/reward_loss`
  - `actor/state_loss`
- 在训练主循环中增加了规划动作摘要指标：
  - `predictor/planned_action_mean`
  - `predictor/planned_action_min`
  - `predictor/planned_action_max`
- 在主配置中补齐了 Predictor/MCTS 相关字段，默认保持 baseline（关闭）。

## 2) 关键变更文件

- `verl/verl/workers/roles/actor.py`
- `verl/verl/workers/roles/utils/losses.py`
- `verl/verl/workers/actor/dp_actor.py`
- `vagen/ray_trainer.py`
- `vagen/configs/vagen_multiturn.yaml`

## 3) 配置默认策略（避免回归）

- `vagen/configs/vagen_multiturn.yaml` 默认：
  - `actor_rollout_ref.actor.num_actions = 0`
  - `actor_rollout_ref.actor.world_state_dim = 0`
  - `actor_rollout_ref.actor.enable_latent_mcts = false`
- 这保证默认行为仍是 baseline，不会因为本轮代码引入而自动开启 Predictor/MCTS 路径。

## 4) 最小验收建议

1. baseline 配置启动训练，确认无回归。
2. 打开 Predictor 最小配置（设置 `num_actions/world_state_dim`），确认日志出现：
   - `actor/world_model_enabled=1`
   - 若监督字段缺失，出现 `actor/world_model_supervision_missing`
3. 打开 `enable_latent_mcts=true` 后确认出现：
   - `predictor/planned_action_mean/min/max`

## 5) 当前已知限制

- Predictor 的监督仍依赖 batch 中是否包含 `action_labels`、`step_rewards`、`next_latent`。
- 本轮未新增数据生产逻辑；若这些字段在数据管线中不存在，会记录 `world_model_supervision_missing`，但不会中断训练。

