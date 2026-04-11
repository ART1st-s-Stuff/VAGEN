# ARTI5T 近期修改记录

本文根据 `VAGEN` 主仓库和 `VAGEN/verl` 子仓库的 git 历史整理，仅总结作者为 `ARTI5T` 的近期已提交修改。  
说明：
- 统计范围以当前本地 git 记录为准。
- 只总结已提交记录，不包含当前工作区里尚未提交的修改。
- 时间范围主要覆盖 `2026-03-27` 到 `2026-04-10`。

## 总览

ARTI5T 最近这批修改的主线比较清晰，可以概括为 5 个阶段：

1. 在 `verl` 里打通 latent 提取能力，让 actor 能拿到语言模型最后一步的 hidden state。
2. 在 `verl` 里从“latent action head”继续演进到 world model + lightweight planner。
3. 把这套能力迁移到 FSDP 路径，并补 FSDP 相关修复。
4. 在 `VAGEN` 的 `navigation` 任务中增加 `latent_plan` 协议、planner trigger、动作 token 解析和 agent loop 接线。
5. 在 `VAGEN` 侧补训练监督字段、调试指标、样本生成脚本和一批路线文档，准备推进下一阶段实验。

从结果上看，这批改动已经把“latent 提取 -> predictor / planner 雏形 -> navigation 接线 -> 训练观测指标”这条链条基本搭起来了，但仍然更接近 DreamAgent 原型，而不是完整实现。

## VAGEN 主仓库

### 2026-04-10 `7459677` Prepare next step

这一提交主要是“整理现状 + 固定下一步路线”，不是大规模算法实现。

主要内容：
- 新增 `AI_README.md`，为 AI 协作和项目目标提供说明。
- 新增 `progress.md`，把当前缺口、阶段划分、开发优先级和 TODO 明确写出来。
- 新增 `wm.md`，总结 world model / latent planner 相关代码入口和配置项。
- 新增两份 `ai_notes`：
  - `ai_notes/codebase_assessment_2026-04-10.md`
  - `ai_notes/predictor_pipeline_round1_2026-04-10.md`
- 更新 `vagen/configs/vagen_multiturn.yaml`，补 predictor / MCTS 相关配置，并保持默认关闭，避免影响 baseline。
- 更新 `vagen/ray_trainer.py`，增加规划动作摘要指标等可观测性。

影响：
- 这次提交没有把 DreamAgent 直接做完，但把项目从“零散改动”推进到了“有阶段路线、有指标入口、有文档索引”的状态。

### 2026-03-31 `6d6ce71` Generate step1 training samples

这一提交新增了：
- `vagen/envs/navigation/create_datasets/generate_random_parquet.py`

主要作用：
- 为 `navigation` 环境生成 step-level 训练样本。
- 为后续 predictor / latent alignment / SFT 数据准备提供最小工具。

影响：
- 这是把“训练 world model 需要的数据”往前推进了一步，说明开发方向已经从纯在线 RL 开始转向带监督字段的数据构造。

### 2026-03-31 `33c72d1` Progress bar for sample generation

这是对上一个样本生成脚本的易用性补充：
- 为 `generate_random_parquet.py` 增加进度条。

影响：
- 算法影响不大，但改善了大批量样本生成时的可见性和使用体验。

### 2026-03-27 `df15fdf` Navigation MCTS

这是 `VAGEN` 主仓里最关键的一次功能接线提交之一，核心是把 `navigation` 任务改造成支持 latent planner 协议。

主要内容：
- 更新 `vagen/envs/navigation/utils/prompt.py`
  - 新增 `latent_plan` prompt format。
  - 明确规划触发 token `<|action_start|>` 和边界 token `<|action_end|>`。
  - 在 prompt 中加入 planner action token 提示。
- 更新 `vagen/envs/navigation/utils/parse.py`
  - 支持 `latent_plan` 格式解析。
  - 支持把结构化 action token 映射回动作名。
  - 增加 `planner_triggered` 等 planner 元信息。
- 更新 `vagen/agent_loop/gym_agent_loop.py` 和 `vagen/agent_loop/gym_agent_loop_no_concat.py`
  - 当模型只输出 planner trigger 时，自动补齐 action boundary。
  - 若没有具体 action token，则注入一个固定 fallback action token。
- 更新 `vagen/envs/navigation/navigation_env.py`
  - 让环境 step 路径能消费 planner 相关解析结果和指标。
- 更新训练 / 验证 / 评测配置：
  - `examples/train/navigation/train_navigation.yaml`
  - `examples/train/navigation/val_navigation.yaml`
  - `examples/evaluate/navigation/config.yaml`
  - 将 `prompt_format` 切到 `latent_plan`。
- 更新 `verl` 子模块指针，依赖下游 planner / world model 支持。

影响：
- 这次提交把 `navigation` 任务改造成了“可以触发 latent planner”的形态。
- 但从实现细节看，当时的 agent loop 仍然主要依赖 fallback token，而不是 planner 真正决定执行动作，所以更准确地说是“协议接线完成，决策闭环尚未完成”。

## verl 子仓库

### 2026-04-10 `19c6aa8` Add actor

这是对前面 world model / planner 接线的补强，重点在于可观测性和 actor 路径的一致性。

主要内容：
- 更新 `verl/workers/roles/actor.py`
  - 增加 `_can_enable_world_model()` 和 `_can_enable_latent_mcts()` 这类守卫逻辑，避免配置不完整时误开启。
  - 让 `compute_log_prob()` 在启用时输出 `planned_action_ids` / `planned_action_tokens`。
  - 把熵输出字段从单数形式切到 `entropys`，对齐上层 trainer 的消费方式。
- 更新 `verl/workers/actor/dp_actor.py`
  - 在 DP actor 训练路径里记录 `actor/world_model_enabled`。
  - 增加 `actor/world_model_loss`、`actor/world_model_supervision_missing` 等指标。
  - 在 reward loss 和 state loss 之上汇总 world model loss。
- 更新 `verl/workers/roles/utils/losses.py`
  - 在 PPO loss 里显式暴露 world model 相关 metrics。
  - 当启用了 world model 但监督字段缺失时，打出 `world_model_supervision_missing`。

影响：
- 这次提交的核心价值是“让 world model 分支可观测”，方便判断 predictor 分支到底有没有真正训练起来。

### 2026-03-28 `b24850d` Fix fsdp

这是一次 FSDP 路径的修复提交，涉及：
- `verl/utils/fsdp_utils.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/sharding_manager/fsdp_sglang.py`

影响：
- 主要是补 FSDP / sharding 相关问题，为前一天接入的 latent / planner 逻辑提供更稳定的分布式执行基础。
- 从提交位置看，它更像是前序 world model / planner 改动落到 FSDP 路径后的一次兼容性修正。

### 2026-03-27 `967cf6f` FSDP migration

这是把前面在 actor 路径上的 latent / planner 能力继续迁移到 FSDP worker 路径。

涉及文件：
- `verl/workers/actor/dp_actor.py`
- `verl/workers/fsdp_workers.py`

影响：
- 说明 ARTI5T 并不是只在单一路径上做实验代码，而是开始把 world model / planner 逻辑同步到 FSDP 训练实现中。

### 2026-03-27 `2f291ea` MCTS

这是 `verl` 里最核心的功能性提交之一，标志着 world model + planner 原型正式出现。

主要内容：
- 新增 `verl/workers/roles/utils/world_model.py`
  - `LatentStateEncoder`：把 LLM latent 压到 world state 空间。
  - `TransitionRewardNet`：输入 `(world_state, action_id)`，输出 `(next_world_state, expected_reward)`。
- 新增 `verl/workers/roles/utils/mcts_planner.py`
  - 提供一个基于 `TransitionRewardNet` 的 lightweight lookahead planner。
  - 具备 `depth`、`branching`、`discount` 等超参。
- 更新 `verl/workers/config/actor.py`
  - 引入 `world_state_dim`、`transition_hidden_dim`、`state_loss_coef`、`reward_loss_coef`、`enable_latent_mcts`、`mcts.*` 等配置项。
- 更新 `verl/workers/roles/actor.py`
  - 在 actor 初始化时按配置创建 state encoder、transition net、planner 和优化器。
  - 在 `compute_log_prob()` 中可选输出 latent、world_state、planned_action_ids。
- 更新 `verl/workers/roles/utils/losses.py`
  - 在 PPO loss 上叠加 reward loss / state loss。

影响：
- 这是 DreamAgent 原型链路里最关键的一步：第一次把 latent state、状态转移预测、reward 预测和 lookahead planner 放进了训练框架。
- 但这里的 `MCTSPlanner` 实现本质上仍是 lightweight rollout，不是完整 UCT / PUCT 树搜索。

### 2026-03-27 `2304d27` Add latent FFNN

这是 world model 之前的中间阶段，先做了一层更简单的 latent 动作头。

主要内容：
- 在 `ActorConfig` 中新增：
  - `num_actions`
  - `action_head_hidden_size`
  - `action_head_loss_coef`
  - `action_head_detach_latent`
  - `action_head_lr`
- 在 `verl/workers/roles/actor.py` 中新增 `LatentActionHead`
  - 从 latent 直接输出 action scores。
- 在 `verl/workers/roles/utils/losses.py` 中给 PPO loss 增加 action head 的交叉熵监督。

影响：
- 这一步可以看作是“先验证 latent 里能不能直接读出动作信息”的过渡方案。
- 后续 `world_model.py` 和 `mcts_planner.py` 的设计，基本是在这个 latent action head 基础上继续往世界模型方向升级。

### 2026-03-27 `c20f185` Extract latent to outside

这次提交把 latent 从模型内部拉到了 actor 外部可消费的接口。

主要内容：
- `ActorWorker.compute_log_prob()` 支持根据 `meta_info` 控制是否提取 latent。
- 支持在需要时保留 autograd 图，而不总是使用 `torch.no_grad()` 推理路径。
- 把最后一个 token 的 latent 作为外部输出返回。

影响：
- 这是后续所有 latent action head、world model、planner 逻辑的基础，没有这一步，后面的方案都无法在 actor 外消费 hidden state。

### 2026-03-27 `424488c` Add latent state extraction

这是整条链路的起点：先让底层 Megatron engine 有能力把 pre-logits hidden state 暴露出来。

主要内容：
- 更新 `verl/workers/engine/megatron/transformer_impl.py`
  - 新增 `extract_latent` 控制开关。
  - 在 output layer 之前注册 hook，捕获 pre-logits latent。
  - 在 `prepare_model_outputs()` 中把 `latent` 一并返回。

影响：
- 这是 latent world model 路线的基础设施提交，相当于先把“能拿到 latent”这件事打通。

## 一条连续演进线

如果把这些提交按技术路线串起来，可以得到一条很清楚的演进线：

1. `424488c`：底层 engine 支持输出 latent。
2. `c20f185`：actor 路径能把 latent 暴露到外部。
3. `2304d27`：先尝试 latent action head，验证 latent 对动作监督是否有用。
4. `2f291ea`：升级到 latent state encoder + transition/reward net + lightweight planner。
5. `967cf6f` / `b24850d`：把上述逻辑迁移并修复到 FSDP 路径。
6. `df15fdf`：在 `navigation` 任务里接入 latent-plan prompt、parser、agent loop 和配置。
7. `6d6ce71` / `33c72d1`：补样本生成脚本，为监督训练准备数据。
8. `7459677` / `19c6aa8`：补文档、指标和调试入口，为下一阶段实验做准备。

## 当前状态判断

从这些提交看，ARTI5T 最近完成的是：
- latent 提取基础设施；
- latent action/world-model 原型；
- lightweight planner 原型；
- navigation 环境的 latent-plan 协议接线；
- predictor / planner 的部分监督字段和观测指标。

但还没有完全完成的是：
- planner 真正接管环境动作执行；
- 基于 `D + V + P(a|z)` 的完整 latent MCTS；
- 基于 imagined rollout 的 Dreaming RL 闭环；
- 大规模 baseline / ablation 论证。

所以更准确的说法是：

> ARTI5T 最近把 DreamAgent 方向的原型主链路搭起来了，并补到了可训练、可观察、可在 navigation 上验证的程度；但当前状态仍然是“原型 + 接线 + 指标”，还不是完整 DreamAgent。
