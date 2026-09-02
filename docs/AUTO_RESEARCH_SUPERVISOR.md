# Concept Flow Auto Research 接管与监督指南

本文档规定长期 supervisor 如何接管并推进 Concept Flow。它不是新的状态页、实验注册表或环境手册；可变事实只写入各自的权威文件，本文档只定义读取顺序、推进规则和监督协议。

## 1. 权威来源与读取顺序

每次新 supervisor session、automation 唤醒或 writer 交接后，按以下顺序读取并服从：

1. `AGENTS.md`。
2. `auto-research-contracts/` 下全部三份契约，必须完整读取，不得只读摘要：
   - `autoresearch-bootstrap-contract.md`
   - `local-codex-remote-server-setup.md`
   - `research-writing-recording-contract.md`
3. `docs/AUTORESEARCH_STARTUP.md` 与 `docs/REMOTE_RESEARCH_OPERATIONS.md`，获得 Concept Flow 的固定环境、机器、模型、预算和安全绑定。
4. `docs/RESEARCH_PLAN.md`，获得当前方法身份、不可变协议、禁止 rescue、第一科学硬门槛和有效反证条件。
5. `docs/RESEARCH_STATE.md`，获得唯一当前状态、writer、blocker 和 next safe action。
6. `docs/EXPERIMENT_REGISTRY.md` 与相关 immutable run receipts，核对运行证据。
7. `docs/RESEARCH_WRITING_AND_RECORDING.md` 指向的写作与记录契约，在形成解释、更新状态或起草论文前再次读取。

若这些文件互相冲突，先停止 dispatch，将冲突记录为 blocker；不得自行选择更方便的一版。基础设施 task、reviewer 或 automation 的自然语言汇报都不能覆盖已 push 的权威文件和 immutable receipt。

## 2. Supervisor 的职责边界

Supervisor 是研究控制面，也是推进者，不是另一个并行 writer。只要环境、writer handoff、安全边界和当前注册 gate 已满足，它必须直接恢复或启动下一项注册实验，不能停在状态巡检、报告整理或重复验证。它负责：

- 核验 bootstrap、Git handoff、服务器健康、预算、stop sentinel、tmux 和 reviewer gate；
- 从 `RESEARCH_STATE.md` 的第一个未完成硬门槛继续；
- 将候选想法转化为一次只改变一个研究轴的可证伪 gate；
- 监督服务器 Codex/ARIS 与长实验，接收并审计回报；
- 区分 `PLANNED`、`RUNNING`、`FAILED`、`OBSERVED`，并独立维护 gate disposition；
- 要求内部验证后再走固定 Claude read-only review；
- 只在证据足够时更新论文素材，并让故事强度服从观察结果；
- 在外部 blocker、预算、连续失败或安全边界触发时 fail closed。

Supervisor 不得：

- 与服务器 writer 同时修改同一 checkout；
- 绕过 GitHub 传源码、自动 stash/reset/merge 或 force-push；
- 把缺失实验、失败实现、reviewer 预测或计划写成观察结果；
- 为挽救 headline 临时改变 metric、样本、prompt、locus、dose、decoder 或 control identity；
- 用 Codex 自审代替规定的跨模型 gate；
- 启动第二套 Auto Research pipeline，或启用契约禁止的 Superpowers 工作流。

## 3. 首次接管协议

新 supervisor 必须先完成以下回执，才能派发科学实验：

1. 报告当前本地 branch/HEAD、GitHub upstream 和服务器 checkout SHA；三者不一致时只做 handoff 修复。
2. 读取负责环境搭建的 Codex task `01a0608b-8ee3-7af3-b6bb-883b94085bdc`，确认其最终状态、未完成登录或 blocker；不能从 task 标题推断完成。
3. 运行项目现有的只读/无害 health 与 acceptance 检查，核对 `RESEARCH_STATE.md` 中列出的 receipt 实际存在且对应当前 commit。
4. 核对唯一固定环境、Codex profile、Claude reviewer canonical identity、ARIS full SHA、allowlist、tmux socket、磁盘、预算和 stop sentinel。
5. 核对 server writer 是否已经取得 lease。若服务器是 writer，本地 supervisor 保持只读；要接管修改，先让服务器到安全停止点并完成 commit/push，再在 clean tree 上 `pull --ff-only`。
6. 创建并验证第 4 节的 heartbeat automation。
7. 输出一次启动回执：`run`、`observation`、`gate decision`、`next step`、`writer`、`automation cadence` 和 `evidence paths`。下一步只给一个正向 gate 问题；这些值从权威文件读取，不在本文档中复制维护。

任何 required 项失败时，Supervisor 保持运行但不启动新科学实验；它只推进能够解除 blocker 的安全工作。

## 4. 持续监督 automation

Supervisor 必须创建一个绑定到自身 task 的 heartbeat automation。更新已有 automation，禁止为同一项目重复创建监控任务。

每次 heartbeat：

1. 读取 `docs/RESEARCH_STATE.md`、当前 writer 和最新已 push commit。
2. 通过批准的服务器入口检查本项目 tmux、run receipt、日志尾部、GPU/磁盘、预算、stop sentinel 和连续失败计数；只管理 `/home/qingchan/` 下本项目资源。
3. 将服务器汇报与 immutable receipt、进程状态和 Git SHA 交叉核验。直接复用 dispatcher 已生成的 checksums、metadata、asset/run receipts 和终态验证；除明确终态硬门禁或具体污染证据外，不重新哈希完整 run、shard、模型或数据集。
4. 若运行完成，从 receipt 核对 exit status、metadata、既有 checksum 和 reviewer gate，再决定 `FAILED`、`OBSERVED` 或 `PASS`；validator-only 改动复用原 artifact，不重跑实验或大文件校验。
5. 若没有 active run 且当前 gate 的 dispatch 条件已满足，当轮直接恢复或启动该注册实验；巡检和报告不是终止动作。
6. 有意义的本地源码、契约、状态或论文改动在安全检查点及时 commit 并 push，最终合入主分支；不得积累长期悬空分支。
7. 仅在出现状态变化、异常、需要用户动作或 gate 完成时通知用户；稳定且无变化的轮次保持简短。
8. 根据任务长度和风险调整**同一个** automation 的间隔：
   - 预计 30 分钟内完成、刚启动或状态不稳定：约 10 分钟；
   - 预计 30 分钟至 4 小时：约 20–30 分钟；
   - 预计 4–24 小时且 receipt/进程稳定：约 60–120 分钟；
   - 无运行、处于外部 blocker：约 6–12 小时，只检查 blocker 是否解除；
   - 临近完成、出现错误、磁盘/预算接近门槛或连续失败：立即缩短频率并停止新 dispatch。

自动调频的依据必须写入 automation 最近一次回报。每次回报按“运行—观察—gate 决策—下一步”组织，失败细节只留在 ledger，scope 只通过下一 gate 的一个正向问题陈述。gate `PASS` 后立即进入下一项已授权实验；实验结束后将频率恢复为适合当前 research gate 的巡检周期。用户要求停止时暂停 automation，不留下重复 heartbeat。

## 5. 科学推进方式

任何时刻只有一个 active hard gate。当前 active gate 及其 protocol 只由 `RESEARCH_PLAN.md` 与 `RESEARCH_STATE.md` 决定。执行顺序是：

1. 解除当前 gate 的外部或测量 blocker；
2. 使用最小、最便宜且可判定的 cell 做 implementation/measurement validity；
3. 在 immutable commit snapshot 上运行注册实验；
4. 内部核验后请求固定 Claude read-only review；
5. 将有效结果登记为 `OBSERVED` 并作出 gate 决策；
6. gate `PASS` 后立即进入下一项已授权实验；需要新注册时，只注册由当前观察决定的下一 gate。

以下是候选研究队列，不是已注册实验，也不是观察结果：

- **Utilisation 是否是 model-level routing phenotype**：检验跨 finding、seed、dataset 与 readout 的稳定分群，以及训练因素能否预测它。
- **医疗训练改变 encoding 还是 routing**：优先利用严格匹配的 Qwen/Lingshu 对，比较 decodability、behaviour 与 causal response；LLaVA 对只作描述性复现，除非重新建立严格匹配依据。
- **信息在哪里离开 answer pathway**：沿 vision encoder、connector、LLM visual positions 与 answer position 建立预注册 trajectory，寻找 gap 首次出现与是否恢复。
- **Failure mechanism taxonomy**：区分信息未进入语言模型、未路由到 answer token、以及被语言先验或决策策略压过；prompt、constrained readout、patching 和 intervention 必须分别裁决明确替代解释。
- **Random floor 的来源**：通过 multi-seed random initialisation、blank/mismatched/shuffled image、nuisance 与跨数据集测试，区分视觉数据结构、语言先验和架构效应。
- **跨模态与边界**：第二模态只有在完整复现核心 F/T/B 或 causal protocol 时才支持主现象泛化；仅复现 nuisance/floor 时应作为该子结论的证据。

候选方向只有在写入 `RESEARCH_PLAN.md` 的可证伪 claim、固定 protocol、反证、预算、统计方法和失败动作后才能 dispatch。一次实验只改变一个研究契约轴。

## 6. 测量与证据门槛

- 中心比较必须使用同一批 patient/row IDs；per-image long table 是 F/T/B 和 paired uncertainty 的来源。
- 主分析使用预先定义的 behavioural readout；任何 test-set best-of-N 只能作为明确标注的 optimistic sensitivity，或在 validation 选择后于 test 评估一次。
- Random floor 是 seed distribution，不是单一点；涉及比率时必须报告分母稳定性和 threshold sensitivity。
- 推断优先 patient-clustered paired uncertainty；45 个相关 cells 不能当成 45 个独立样本。
- `U` 可作为 random-floor-adjusted diagnostic contrast；除非另有严格推导，不把 AUROC ratio 写成信息量分解、training causal effect 或 conditional probing 的正式实例。
- 结论限定到被测试的 answer pathway。`decode`、`generate`、`route`、`influence` 和 `steer` 是不同性质的测量，不能互相替代。
- blank/mismatched/shuffled image 等 groundedness control 用于判定视觉证据，而不是把语言或数据集先验误写成视觉使用。

若后续注册 protocol 与本节冲突，必须先显式修改并 review `RESEARCH_PLAN.md`，不能在运行脚本里静默改变。

## 7. 故事与写作推进

论文更新遵循 `docs/RESEARCH_WRITING_AND_RECORDING.md` 路由的源头合同，并额外执行三个独立 gate：

1. **Story gate**：主流 framing 的盲点、具体矛盾、被忽略的分析单位、统一机制和边界是否形成递进发现链条；读者能否用一句话记住新观点。
2. **Claim–evidence gate**：逐条区分 measurement、supported interpretation、unresolved hypothesis 和 excluded claim；标题、摘要、Results 使用一致强度。
3. **Style gate**：报告结构为“运行—观察—gate 决策—下一步”；删除防御性、rebuttal 式预防解释、作者元话语、编辑历史、重复限定与没有识别依据的绝对词。

写作采用 positive story first：先陈述新结构和科学意义，再说明对照排除了什么。主文一段只完成一个推理动作，优先使用“现象 → 对照 → 推论”；完整审计、失败实现和 provenance 下沉到 ledger/appendix。公式后立即解释操作语义，但不借公式复杂度夸大识别范围。

大幅改稿的顺序是：先冻结当前证据支持的主故事和 section logic，再重写摘要、引言与 Results，最后做句子级润色。未完成的计划不得预写成论文贡献。

## 8. 停止、升级与交接

出现以下任一条件时停止新 dispatch：stop sentinel、越出 `/home/qingchan/`、dirty/divergent Git、reviewer 身份或只读性不可证、预算/磁盘门槛、连续三次失败、receipt 不完整、用户需要完成登录/授权，或当前结果将迫使改变方法身份。

只有真正改变研究目标、方法身份、外部授权或资源预算的问题才升级给用户。可以由仓库、服务器证据、官方资料或注册实验解决的问题由 Supervisor 自行解决并记录。

交接给另一个 session 时，先达到安全停止点并 push；交接消息按“运行—观察—gate 决策—下一步”给出当前 commit、writer、automation ID/cadence 和 evidence paths。下一步只陈述一个正向 gate 问题；历史过程留在各自 ledger。

## 9. 正确启动的判据

Supervisor 只有同时满足以下条件才算正确运行：

- 已完整读取本节第 1 项全部权威来源；
- 已读取并核对环境搭建 task 的最新状态；
- 已按“运行—观察—gate 决策—下一步”报告，且与 `RESEARCH_STATE.md` 一致；
- 已创建一个可查看的 heartbeat automation，并说明初始 cadence 与调频依据；
- 未在 blocker 未解除时启动科学实验；
- 已明确接下来从第一个未完成 hard gate 继续，并复用已有终态 receipt。
