# Phase2 Predictor 多方案改造记录（2026-04-14）

## 1) 新增统一接口

- 在 `verl/verl/workers/roles/utils/world_model.py` 新增统一 predictor 抽象：
  - `BaseLatentPredictor.forward_step(latent, action_ids)`
  - `BaseLatentPredictor.forward_multi_step(latent_0, action_seq, horizon)`
  - `BaseLatentPredictor.compute_auxiliary_losses(...)`
- 新增统一构建入口：
  - `build_latent_predictor(predictor_mode, latent_dim, num_actions, world_state_dim, hidden_dim, multi_step_horizon)`
- 新增统一 loss 路由 helper：
  - `compute_world_model_aux_loss(...)`

## 2) 三种 predictor 模式

- `world_state_with_decoder`（方案2a）  
  `latent -> world_state -> pred_next_world_state -> decoder -> pred_next_latent`，同时可预测 reward。
- `direct_latent_mlp`（方案2b）  
  直接用 MLP 做 `(latent, action_id) -> pred_next_latent`，同时可预测 reward。
- `lewm_latent_dynamics`（方案1）  
  适配 `lewm.module.ARPredictor`，支持 latent + action 的序列建模与多步 rollout。

兼容保留：
- `world_state_mlp`（历史模式）  
  保留旧行为（state 空间监督），用于默认配置不回归。

## 3) 训练路径改动

- `verl/verl/workers/roles/actor.py`
  - 从 `state_encoder + transition_reward_net` 改为统一 `world_model_predictor`。
  - planner 仅在 predictor 具备 `transition_reward_net` 时构建（即 world_state 路径）。
- `verl/verl/workers/roles/utils/losses.py`
  - `ppo_loss` 改为接收 `world_model_predictor`。
  - world model loss 统一通过 `compute_world_model_aux_loss` 计算。
- `verl/verl/workers/actor/dp_actor.py`
  - 同步接入统一 predictor 与统一 loss helper，避免角色实现分叉。

## 4) 配置开关

- `verl/verl/workers/config/actor.py`
  - 新增：
    - `predictor_mode: str = "world_state_mlp"`
    - `predictor_multi_step_horizon: int = 1`
  - 新增 `predictor_mode` 合法值校验与 `predictor_multi_step_horizon > 0` 校验。
- `vagen/configs/vagen_multiturn.yaml`
  - 新增默认项：
    - `actor_rollout_ref.actor.predictor_mode: world_state_mlp`
    - `actor_rollout_ref.actor.predictor_multi_step_horizon: 1`

## 5) 启动脚本

- `predictor_debug.sh` 新增环境变量：
  - `PREDICTOR_MODE`
  - `PREDICTOR_HORIZON`

## 6) 验证记录

- 语法检查通过：`py_compile` 覆盖 core 改动文件。
- 轻量 smoke：
  - `world_state_with_decoder`：前向/反传/多步通过。
  - `direct_latent_mlp`：前向/反传/多步通过。
  - `lewm_latent_dynamics`：当前环境缺少 `einops`，初始化时报错；代码路径已接通，运行依赖需补齐。
