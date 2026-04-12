## 项目进度
目前仍处于原型阶段，但相比上一轮评估，world model / latent planner 的基础接线已经明显前进：动作 schema 已收敛到共享定义，planner 元信息已进入 rollout，`action_labels / step_rewards / next_latent` 已能在 trainer 中按 no-concat 轨迹自动回填，且已经补了最小 debug 指标与测试脚本。

当前使用 `navigation` 环境，并将 planner 限制为一步 lookahead，适合作为最小可行验证场景。

开发计划统一维护在本文件中；`AI_README.md` 仅保留项目目标、目标架构和协作规则。若两者表述发生偏差，以本文件为准。

## 当前状态判断
- 已有内容：
  - 已有 `latent -> world_state` 编码器。
  - 已有 `(world_state, action) -> (next_state, reward)` predictor 雏形。
  - 已有基于 predictor 的一步/有限深度 lookahead planner 雏形。
  - 已在 actor loss 中接入 predictor 的监督损失入口。
  - 已有共享 `action schema`，用于统一 planner boundary token、action token、action id 映射。
  - 已在 `agent_loop -> reward_extra_info -> trainer batch` 链路中回填：
    - `action_labels`
    - `step_rewards`
    - `planner_triggered / planner_fallback_used / planner_parse_failed`
  - 已在 no-concat 路径中基于 `group_idx / traj_idx / turn_idx` 自动构造 `next_latent` 与 `next_latent_mask`。
  - 已补训练可观测指标：
    - `predictor/action_label_coverage`
    - `predictor/supervision_valid_ratio`
    - `predictor/next_latent_coverage`
    - `predictor/planner_*_rate`
    - `actor/world_model_*`
- 当前缺口：
  - planner 还没有真正消费 `planned_action_ids` 去决定环境执行动作，当前 latent-plan 仍以 fallback 补全为主。
  - 当前 planner 还不是完整 MCTS，更接近基于 learned dynamics 的穷举/轻量 rollout。
  - `action prior`、价值函数、真实的 tree policy 还未形成完整设计与实现。
  - predictor 数据流虽然已基本接通，但还缺系统实验来确认 loss 是否稳定生效、监督覆盖率是否足够。
  - 还没有形成 baseline、ablation 和一步 planner 收益验证结果。

## 对当前 TODO 的修正
- `SFT微调VLM输出格式`
  - 这是合理的，但目标要更具体：不是泛泛做 SFT，而是先把 VLM 输出稳定约束为可解析、可映射到离散 action id 的格式。
- `VLM和Predictor联合训练`
  - 这一步放得过早。当前更需要先打通 predictor 的监督数据与单独预训练，再考虑 joint training。
- `在Latent space上训练价值函数`
  - 这一步本身可能需要，但它不应该早于 predictor 数据闭环和 planner 验证。
  - 另外当前代码里的 planner 还没有真正使用 value function，这部分需要先补设计定义。
- `VLM、Predictor、MCTS联合RL训练`
  - 方向没错，但属于后期阶段，不应该作为当前主 TODO。

## 推荐开发流程

按 DreamAgent 的设想，开发顺序应分阶段推进，而不是一开始就直接做端到端联合 RL。当前 `progress.md` 中的阶段划分与 TODO，是项目的唯一开发计划来源。

### Phase 0: 先对齐任务定义与接口
- [x] 明确 action space：
  - 固定离散动作集合。
  - 明确 `action token -> action id -> env action` 的唯一映射。
- [ ] 明确 latent 监督目标：
  - `latent_t` 取哪个位置。
  - `next_latent` 是 next step 的哪个表示。
- [x] 明确训练 batch 必须提供的字段：
  - `action_labels`
  - `step_rewards`
  - `next_latent`
  - `done / valid_mask`（如后续需要多步 rollout）
- [x] 明确三个模块的职责边界：
  - Predictor 负责转移与即时奖励预测。
  - Critic / value head 负责价值估计。
  - Planner 负责基于 predictor/value/prior 选择动作。
- [x] 输出一份最小接口约定，避免后续边写边改。

### Phase 1: 先把动作输出与监督数据打通
- [ ] 做最小化 SFT 或模板约束，让 VLM 输出稳定 action 格式。
- [x] 增加 action parser / validator：
  - 能从 VLM 文本输出稳定解析出 action id。
  - 出错时可回退、可统计解析失败率。
- [x] 在 rollout / dataset 中补齐 predictor 监督字段：
  - 记录执行动作作为 `action_labels`。
  - 记录环境即时奖励作为 `step_rewards`。
  - 记录 next step latent 作为 `next_latent`。
- [ ] 验证训练日志中不再频繁出现 `world_model_supervision_missing`。

## Phase 0 细化：接口先统一

### 0.1 Action 映射单点定义
- [x] 建立唯一的 action schema，至少包含：
  - `action_name`
  - `action_token`
  - `action_id`
  - `env_action`
- [x] 目标是把当前多处重复定义收敛到一个共享位置，避免后续映射漂移。
- [ ] 当前可重点检查/收敛的文件：
  - `vagen/envs/navigation/utils/parse.py`
  - `vagen/agent_loop/gym_agent_loop.py`
  - `verl/verl/workers/roles/actor.py`
  - `vagen/envs/navigation/create_datasets/generate_random_parquet.py`
- [x] 验收标准：
  - prompt、parser、planner、rollout、dataset 使用同一套动作顺序。
  - 不再出现“token 顺序一套、id 顺序另一套”的隐性错误。

### 0.2 明确 planner trigger 协议
- [x] 明确 `latent_plan` 模式下模型的最小输出协议：
  - 推荐 VLM 只负责输出 `"<|action_start|>"` 作为 planner trigger。
  - planner 负责补全 action token，并闭合 `"<|action_end|>"`。
- [x] 明确兼容策略：
  - 若模型已经输出 action token，则沿用。
  - 若只输出 `action_start`，系统决定是否 fallback。
  - 若格式错误，记录错误类型并拒绝静默吞错。
- [ ] 当前已有 fallback，但它更像临时兜底；当前已能观测和统计，但尚未替换成 planner 真正决策动作的正式闭环。
- [x] 验收标准：
  - 能统计 `planner_triggered`、`planner_fallback_used`、`planner_parse_failed`。
  - 可以区分“模型自己给出可执行动作”和“模型触发 planner / fallback”的两种轨迹。

### 0.3 明确 predictor 监督字段契约
- [x] 写清楚训练 batch 中每个字段的语义、shape、dtype、来源：
  - `action_labels`: 当前 step 实际执行的离散动作 id，建议 `int64`
  - `step_rewards`: 当前 step 的即时奖励，建议 `float32`
  - `next_latent`: next step observation 对应的 latent，建议 `[bs, hidden]` 或兼容 `[bs, T, hidden]`
- [x] 写清楚这些字段是在什么阶段产生的：
  - rollout 结束后回填
  - actor 二次 forward 计算
  - dataset 预生成
- [x] 验收标准：
  - 能明确回答每个字段“谁产生、何时产生、在哪一层写入 batch”。

### 0.4 模块职责冻结
- [ ] 在设计文档中固定以下边界：
  - VLM: 编码 observation、输出文本/trigger、提供 latent。
  - Predictor: 预测 `(next_state, reward)`。
  - Planner: 基于 predictor/value/prior 做动作选择。
  - Critic/value head: 负责值函数，不与 predictor 混用概念。
- [ ] 验收标准：
  - 后续不会把 reward predictor、state predictor、value head 混成一个模块讨论。

## Phase 1 细化：打通最小监督闭环

### 1.1 先做最小可解析 action 输出
- [ ] 目标不是先提升策略性能，而是先降低解析不稳定性。
- [ ] 两种可选方案：
  - 方案 A：轻量 SFT，让模型稳定输出 action token 或 planner trigger。
  - 方案 B：先不训模型，只强化 prompt/template 约束，观察是否足够稳定。
- [ ] 建议顺序：
  - 先做模板约束。
  - 如果解析失败率仍高，再补小规模 SFT。
- [ ] 验收标准：
  - `navigation` 上 action 解析成功率足够高。
  - 不再依赖过强的 fallback 才能跑通。

### 1.2 补齐 `action_labels`
- [x] 推荐把“真正执行到环境里的动作”作为监督目标，而不是模型原始文本里声明的动作。
- [ ] 若一个 step 允许多个动作，需要先定义 predictor 的训练粒度：
  - 只训练每个 turn 的第一个动作。
  - 或把一个 turn 展开成多个 `(state_t, action_t, state_t+1)` 子样本。
- [x] 当前更建议先采用最小方案：
  - 单步 planner 场景下，每 turn 只监督一个离散动作。
- [x] 验收标准：
  - `action_labels` 与 planner/action id 共用同一映射。

### 1.3 补齐 `step_rewards`
- [x] 明确 reward 是监督：
  - 执行动作后的即时环境奖励。
  - 不是 trajectory return，也不是 token-level reward。
- [ ] 若当前一个 turn 会执行多个动作，需要定义 reward 聚合方式：
  - first-action reward
  - sum reward
  - discounted multi-action reward
- [x] 当前建议先和“单动作监督”保持一致，先使用一步即时奖励。
- [x] 验收标准：
  - predictor 的 reward 目标与 planner 的一步 lookahead 目标一致。

### 1.4 补齐 `next_latent`
- [ ] 这是当前最容易含糊的部分，需要先定语义再实现。
- [x] 推荐定义：
  - `next_latent` 对应执行完当前监督动作后的 next observation，经同一 VLM 编码得到的 latent。
- [ ] 可选实现路径：
  - 路径 A：rollout 时缓存 next observation，训练前再做一次 forward 得到 `next_latent`。
  - 路径 B：直接复用下一 turn 已经产生的 latent，对齐后写回当前样本。
- [x] 当前更推荐先做路径 B：
  - 成本更低。
  - 更容易和现有 multi-turn 轨迹对齐。
- [x] 需要额外定义：
  - terminal step 没有 next latent 时如何 mask。
  - 解析失败 / 无效动作时是否写入监督。
- [x] 验收标准：
  - `next_latent` 只在定义清楚的有效样本上参与损失。

### 1.5 增加数据完整性检查
- [x] 在训练前或 batch 构造阶段增加检查项：
  - 缺 `action_labels`
  - 缺 `step_rewards`
  - 缺 `next_latent`
  - id 越界
  - reward shape 不匹配
- [x] 训练日志建议增加：
  - `predictor/supervision_valid_ratio`
  - `predictor/action_label_coverage`
  - `predictor/next_latent_coverage`
  - `predictor/planner_trigger_rate`
- [x] 验收标准：
  - 能在首轮实验就快速定位“损失分支没训上”的原因。

### 1.6 做一个最小 debug 跑通路径
- [x] 先不要直接上完整训练，先做一个小规模 debug 流程：
  - 采几条 `navigation` 轨迹。
  - 打印每条样本的 `action_labels / step_rewards / next_latent` 是否存在。
  - 跑一个很小 batch，确认 `reward_loss/state_loss` 非空且会下降。
- [ ] 验收标准：
  - 可以在小样本上稳定复现 predictor loss。
  - 日志中不再主要出现 `world_model_supervision_missing`。

### Phase 2: Predictor 单独预训练
- [ ] 先冻结或基本冻结 VLM 主体，只训练：
  - `LatentStateEncoder`
  - `TransitionRewardNet`
- [ ] 从一步预测开始：
  - 先保证 reward prediction 可学。
  - 再加入 next latent / next state 预测。
- [ ] 建立 predictor 的离线验证指标：
  - reward MSE
  - next state MSE / cosine similarity
  - 基于 predictor 选出的 action 与 oracle / env 最优一步动作的一致率
- [ ] 只有 predictor 有基本可用性后，才进入 planner 阶段。

### Phase 3: 先验证一步 planner，而不是直接上完整 MCTS
- [ ] 使用 predictor 做一步 action ranking。
- [ ] 与以下 baseline 对比：
  - 无 planner 的原始 VLM policy
  - 仅规则/穷举的一步搜索
- [ ] 先回答一个核心问题：
  - predictor 产生的动作排序，是否比原始 policy 更好？
- [ ] 如果一步 planner 没有收益，暂停后续深度搜索与 RL 联训，优先回头修 predictor 或 action interface。

### Phase 4: 再决定是否需要“真正的 MCTS”
- [ ] 若一步 planner 有收益，再考虑多步规划。
- [ ] 若要实现真正的 MCTS，需要补齐：
  - action prior
  - value function / value head
  - tree expansion / backup 规则
  - `c_puct` 的真实使用逻辑
- [ ] 否则可以先保持为 learned dynamics + lookahead planner，不必过早追求完整 MCTS 形式。

### Phase 5: Joint training
- [ ] 在 predictor 已稳定后，再尝试联合训练 VLM 和 predictor。
- [ ] 联训时优先采用保守策略：
  - 小学习率。
  - predictor loss warmup。
  - 必要时只解冻部分层。
- [ ] 重点监控：
  - policy 性能是否提升
  - predictor loss 是否恶化
  - latent 空间是否因为 RL 更新而漂移过快

### Phase 6: 最终 RL 闭环
- [ ] 在 planner 与 predictor 均验证有效后，再做联合 RL 训练。
- [ ] 实验至少包含以下对照：
  - baseline VLM RL
  - VLM + predictor
  - VLM + predictor + one-step planner
  - VLM + predictor + multi-step planner / MCTS
- [ ] 以 ablation 的方式确认性能收益究竟来自：
  - 监督数据
  - predictor
  - planner
  - RL 联训

## 当前阶段的核心 TODO
- [x] 收敛 action schema，消除多处重复定义。
- [x] 固定 `latent_plan` 的 trigger 协议与 fallback 策略（基础版已接通，正式 planner 执行闭环仍未完成）。
- [x] 打通 `action_labels / step_rewards / next_latent` 数据流（当前覆盖 no-concat 主路径）。
- [ ] 先做一个最小 debug batch，确认 predictor loss 真正生效。
- [ ] 建立 predictor 单独训练脚本与验证指标。
- [ ] 在 `navigation` 环境验证一步 planner 是否优于 baseline。
- [ ] 让 planner 真正接管环境执行动作，而不是只通过 fallback 补 action boundary。
- [ ] 明确后续是继续做 lightweight lookahead，还是升级成真正的 latent MCTS。

## 当前优先级
当前最优先的事情，不是直接做完整 MCTS，也不是直接做端到端 Dreaming RL，而是先打通最小闭环：

1. 统一 action schema 和 latent-plan trigger 协议。
2. 让 planner 真正接管环境执行动作。
3. 打通 `action_labels / step_rewards / next_latent` 数据流。
4. 建立 predictor 的最小可训练闭环和调试指标。
5. 在单一环境上验证 one-step planner 是否优于 baseline。

如果以上 5 点没有完成，就不应过早推进 full MCTS 或 Dreaming RL。

## 暂不建议优先做的事
- [ ] 暂不把“轻量 lookahead”直接宣传为完整 latent MCTS。
- [ ] 暂不在没有 value / prior 的情况下声称 planner 已符合 DreamAgent 设想。
- [ ] 暂不直接做 VLM 和 Predictor 的全面联合训练。
- [ ] 暂不直接做完整 MCTS。
- [ ] 暂不直接做端到端联合 RL。
- [ ] 暂不在没有 baseline 和指标的情况下扩展到更复杂环境。

## 接近 DreamAgent 目标的完成标准
只有同时满足以下条件，才可以认为接近 DreamAgent 目标：

- [ ] planner 不是只产生日志，而是真正决定环境执行动作。
- [ ] world model 不只是 PPO 上附带的辅助损失，而能支持多步 latent rollout。
- [ ] latent planning 显式使用 predictor、value 和 action prior。
- [ ] RL 不只依赖真实环境采样，而能从 imagined rollouts 中获益。
- [ ] 有清晰 baseline、ablation 和评测结论支持设计有效性。