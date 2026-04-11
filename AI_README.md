## 项目目标
基于 VAGEN 开发符合 DreamAgent 设想的 VLM Agent：让模型不依赖显式未来文本或未来图像生成，而是在 latent space 中完成世界建模、规划和强化学习。

核心目标不是只做一个辅助 predictor，而是逐步实现 DreamAgent 的三件事：
- Pure Latent Dynamics：直接在 latent space 中预测状态转移，不把 world modeling 退化成长文本 CoT 或像素级未来渲染。
- Latent Planning：在真实执行动作前，先在 latent space 中利用 predictor 快速展开多步搜索，并由 planner 返回实际动作。
- Dreaming RL：不仅在真实环境里采样，也要能在 learned world model 内展开 imagined rollouts，用于训练 actor / critic。

目标架构：
- [VLM] Text prompt + Image -> CoT / planner trigger -> latent state `z_t`
- [Action Prior] `P(a|z_t)`：根据当前 latent state 提供候选动作先验
- [Predictor] Dynamics `D(z_t, a_t) -> z_(t+1)`，并预测一步 reward；后续可扩展不确定性或终止信号
- [Value] `V(z_t)` 或 `Q(z_t, a_t)`：为 latent planning 和 Dreaming RL 提供价值评估
- [Planner] 基于 `D + V + P(a|z)` 进行 latent MCTS / lookahead，输出最终执行动作
- [RL Loop] 结合真实环境轨迹与 imagined trajectories 联合优化 policy、value 和 world model

## 开发计划
按 DreamAgent 的设想，开发顺序应分阶段推进，而不是一开始就直接做端到端联合 RL。

### Phase 0: Baseline 与接口对齐
- 先跑通 text-based baseline（VAGEN 当前路径），记录 bad cases，尤其是多步任务中的 spurious reasoning。
- 固定 action schema、planner trigger、latent 提取位置、predictor 监督字段，避免后续接口漂移。
- 明确模块边界：
  - VLM 负责观测编码、文本输出和 latent 提供。
  - Predictor 负责状态转移与一步 reward 预测。
  - Value / Critic 负责价值估计。
  - Planner 负责基于 predictor / value / prior 做动作选择。
- 当前阶段的可执行 TODO：
  - 统一 action schema，保证 prompt、parser、planner、rollout、dataset 使用同一套动作顺序和 action id。
  - 固定 latent-plan 协议，明确模型输出 planner trigger 时，谁来补动作、谁来闭合 action boundary、何时 fallback。
  - 固定 predictor 监督字段契约：`action_labels`、`step_rewards`、`next_latent`、可选 `done / valid_mask`。
  - 明确 latent 提取位置：当前 step 用哪个 hidden state，next step 用哪个 latent 作为监督目标。
  - 在 batch 构造阶段增加数据校验和调试日志，确保监督字段缺失时能快速定位。
- 验收标准：
  - 可以明确回答每个监督字段“谁产生、何时产生、在哪一层写入 batch”。
  - prompt / parser / env action / predictor action id 不再各自维护不同映射。
  - `planner_triggered`、`planner_fallback_used`、`planner_parse_failed` 可以稳定统计。

### Phase 1: SFT Phase 1 - 先学会稳定触发规划
- 让 VLM 稳定输出可解析的 action token 或 planner trigger，而不是把动作选择逻辑混在自由文本里。
- 优先让 latent-plan 模式变成稳定协议：
  - 模型输出 planner trigger；
  - planner 真正补全动作并返回执行；
  - 失败时有显式 fallback 和统计指标。
- 这一步的目标是“动作接口稳定”，不是“性能已经最优”。
- 当前阶段的可执行 TODO：
  - 先用 prompt / template 约束，观察 action parser 成功率和 planner trigger 触发率。
  - 如果模板约束不够，再补最小化 SFT，让模型稳定输出 action token 或 trigger。
  - 把 planner 输出真正接入 environment execution，而不是只记录 `planned_action_ids` 日志。
  - 增加 planner 相关日志：模型原始输出、planner 选中动作、fallback 类型、解析失败原因。
- 验收标准：
  - latent-plan 模式下 planner 确实决定环境执行动作。
  - 解析失败率足够低，系统不再主要依赖兜底 token 才能跑通。
  - 可以清楚区分“模型自己给动作”和“模型只触发 planner”两类轨迹。

### Phase 2: SFT Phase 2 / Latent Alignment
- 训练 latent state 与下一步真实观测特征对齐，让模型学会在隐藏空间里表示未来状态。
- 训练 predictor：输入 `(z_t, a_t)`，输出 `z_(t+1)` 与一步 reward。
- 在这一阶段优先验证：
  - predictor 是否能学会一步 reward prediction；
  - latent transition 是否与 next observation latent 对齐；
  - latent 表示是否比纯文本 CoT 更稳定、更可复用。
- 当前阶段的可执行 TODO：
  - 先冻结或基本冻结 VLM 主体，只训练 state encoder、transition net，以及必要的对齐头。
  - 从一步预测开始，先保证 reward prediction 可学，再加入 next latent / next state 对齐。
  - 建立 predictor 的离线验证指标：
    - reward MSE
    - next latent / next state 的 MSE 或 cosine similarity
    - predictor 排序出的动作与 oracle / env 一步最优动作的一致率
  - 先做最小 debug batch，确认 `reward_loss` / `state_loss` 非空且会下降。
- 验收标准：
  - predictor 分支不再频繁出现“启用了但无监督”的情况。
  - 小样本和小 batch 上可以稳定复现 predictor loss。
  - predictor 对一步 reward 和 next latent 的预测具备基本可用性。

### Phase 3: Latent Planning
- 先验证 one-step lookahead 是否优于原始 policy，再逐步升级到 multi-step latent planning。
- 真正的目标是实现基于 `D + V + P(a|z)` 的 latent MCTS，而不是长期停留在“随机分支 + max rollout”的轻量近似。
- planner 完成的标志是：
  - planner 输出真正进入环境执行链路；
  - 搜索显式使用 action prior 与 value；
  - 能在复杂局面下比 baseline 有稳定收益。
- 当前阶段的可执行 TODO：
  - 先做 one-step planner，对所有合法动作打分并选最优动作。
  - 用明确对照实验验证：原始 policy vs one-step planner。
  - 如果 one-step planner 有收益，再实现真正的多步搜索。
  - 补齐 latent MCTS 所需组件：
    - action prior
    - value function / value head
    - tree expansion / backup 规则
    - `c_puct` 的真实使用逻辑
- 验收标准：
  - one-step planner 在目标环境上优于无 planner baseline。
  - 多步 planner 不再只是随机分支 rollout，而是显式树搜索。
  - action prior 和 value 已进入 planner 决策主路径。

### Phase 4: Dreaming RL
- 先用真实环境做少量 burn-in，收集 `(obs, action, reward, next_obs)` 轨迹。
- 基于 learned world model 从 `z_t` 出发做 imagined rollout，构建 latent 轨迹。
- 联合优化：
  - World Model Loss：状态转移、reward、latent alignment、价值等效等约束；
  - Policy / Value Loss：使用 PPO / GRPO 等方法，在真实轨迹与 imagined 轨迹上提升长期回报。
- 这一阶段的目标是摆脱完全依赖真实环境采样，把 latent dreaming 变成训练增益来源。
- 当前阶段的可执行 TODO：
  - 明确 replay buffer / latent rollout 的数据结构和训练入口。
  - 明确 imagined rollout 的停止条件、奖励累计方式和 done 语义。
  - 先做短 horizon imagined rollout，避免 predictor 误差快速累积。
  - 先验证 imagined trajectories 是否能给 actor / critic 带来增益，再扩大规模。
- 验收标准：
  - 训练中可以同时消费真实轨迹和 imagined trajectories。
  - imagined rollout 的统计信息可观测，例如 rollout 长度、预测 reward 分布、终止率。
  - 与纯真实环境 RL 相比，Dreaming RL 至少在部分场景中带来训练效率或性能收益。

### Phase 5: 实验验证与论证
- 与以下对象做对照：
  - text-based baseline VLM agent
  - 无 planner 的 predictor 版本
  - one-step planner
  - full latent planning / MCTS
- 必须验证 DreamAgent 的核心收益是否真实存在：
  - 推理更快；
  - 长程一致性更强；
  - 对环境变化或视觉风格变化更鲁棒。
- 必须做 ablation，区分收益究竟来自：
  - action interface
  - predictor
  - value / prior
  - planner
  - Dreaming RL
- 当前阶段的可执行 TODO：
  - 固定统一评测环境、成功率指标、速度指标和日志格式。
  - 补齐最小实验矩阵，至少覆盖 baseline、predictor、one-step planner、multi-step planner。
  - 记录典型 bad cases 和修复后的对照样例，避免只看平均指标。
- 验收标准：
  - 每个阶段结束后都有明确可复现实验结果，而不是只靠训练日志主观判断。
  - 可以回答“性能提升来自哪里”，而不是只知道最终指标变好或变差。

## 当前优先级
当前最优先的事情，不是直接做完整 MCTS，也不是直接做端到端 Dreaming RL，而是先打通最小闭环：

1. 统一 action schema 和 latent-plan trigger 协议。
2. 让 planner 真正接管环境执行动作。
3. 打通 `action_labels / step_rewards / next_latent` 数据流。
4. 建立 predictor 的最小可训练闭环和调试指标。
5. 在单一环境上验证 one-step planner 是否优于 baseline。

如果以上 5 点没有完成，就不应过早推进 full MCTS 或 Dreaming RL。

## 暂不优先做的事
- 暂不把“轻量 lookahead”直接宣传为完整 latent MCTS。
- 暂不在没有 value / prior 的情况下声称 planner 已符合 DreamAgent 设想。
- 暂不在 predictor 还不稳定时做全面联合训练。
- 暂不在没有 baseline 和 ablation 的情况下扩展到更复杂环境。

## 完成标准
只有同时满足以下条件，才可以认为接近 DreamAgent 目标：
- planner 不是只产生日志，而是真正决定环境执行动作。
- world model 不只是 PPO 上附带的辅助损失，而能支持多步 latent rollout。
- latent planning 显式使用 predictor、value 和 action prior。
- RL 不只依赖真实环境采样，而能从 imagined rollouts 中获益。
- 有清晰 baseline、ablation 和评测结论支持设计有效性。

## 开发行为准则
prompt/代码中没有清楚描述/你不理解的部分，征询人类开发者意见。
遇到可选超过一种方案的情况，征询人类开发者意见。
如果要进行大范围破坏性改动，征询人类开发者意见。

代码要求：
- 尽量复用现有代码
- 创建新代码时需要具备可复用性和可配置性，保证后续可持续发展
- 禁止硬编码参数，训练参数使用hydra等工具管理
- 保持代码逻辑简洁且易于理解
- 函数、变量的命名也要易于理解
- 与核心逻辑无关的内容（如库兼容性），将其单独放在utils/中

文件要求：
- 遵循项目结构
- 模块化，按职责分离

注释要求：
- python源码中的注释遵循python规范
- 简洁易懂，面向人类开发者
- 复杂逻辑额外添加注释说明

AI笔记：
- 由于对话历史可能会被清除，所以你需要在ai_notes/内保存所有关键信息
- 匹配当前代码库内容，每次修改后按需更新
- 尽量节约Token。例如对于简单阅读代码已经完全足够的情况，你可以保存信息的索引，甚至无需记录。
- 分类、分条保存，便于你自己查找
- 禁止包含用户个人信息。
- 尽量减少和当前运行环境有关的信息。
- 你需要自己判断合适的时机添加笔记，不要等待人类的指令。

## 项目结构
- AI_README.md: 这个文件。由人类开发者编写。【不要修改】
- ai_notes/: 由你自己记录的开发笔记，可以多文件按需保存
- vagen/: 主要源码
- verl/: 训练用的库