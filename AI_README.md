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
开发计划、阶段路线、当前优先级、完成标准统一维护在 `progress.md`。

本文件只保留以下内容：
- 项目目标
- 目标架构
- 开发行为准则
- 代码/文件/注释要求

如果 `AI_README.md` 与 `progress.md` 对阶段判断或 TODO 表述不一致，以 `progress.md` 为准。

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