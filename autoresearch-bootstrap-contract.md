---
schema: ce-handoff/v1
created_at: 2026-09-02T04:06:40Z
title: 可复用 Auto Research 启动契约与首轮 Prompt
summary: 从 GearShift 提炼的用户研究品味、写作契约、证据制度、运行架构和新项目启动模板；不包含 GearShift 方法本身。
keywords:
  - autoresearch
  - aris
  - research-taste
  - writing-style
  - unattended-research
  - evidence-gates
cwd: E:/projects/GearShift
resume_focus: 将模板复制到新项目外部，填写项目变量，然后把“首个会话 Prompt”交给服务器上的 Codex。
repository: GearShift
repo_root_sha: 990b7b90597851c3dbe6e616bfc0d58c10ffb352
branch: codex/use-fable-5-reviewer
head: 990b7b90597851c3dbe6e616bfc0d58c10ffb352
worktree_path: E:/projects/GearShift
---

# 可复用 Auto Research 启动契约与首轮 Prompt

这份文档用于启动一个新的、服务器常驻的 Auto Research 项目。它继承的是研究判断、写作风格、证据纪律和运行制度，不继承 GearShift 的具体方法、实验结论、路径或门槛。

## 1. 三类内容必须分开

### A. 用户的稳定偏好

除非用户明确改变，这些偏好可直接用于新项目：

- Codex 是研究执行器；Claude Code 是独立、只读的跨模型审查者。
- 正式跨模型审查统一走批准的 reviewer transport；Codex subagent 只能做内部检查，不能代替跨模型 gate。
- Claude 审查 effort 默认为 `medium`。具体模型必须由新项目显式固定并验收；当前项目批准的 Fable 5.1 只可作为新项目的建议默认值，不能由 reviewer 的自述证明身份。
- Codex 执行器显式固定模型和 reasoning effort；默认使用 `high`，不开 Fast，不能依赖开交互 session 时的临时选择。
- 长期 Agent 和耗时实验都在服务器运行；本地电脑可以关机。
- GitHub 是唯一源码同步通道。数据、权重、checkpoint、原始大日志不进 Git。
- 允许在专用、非特权研究账户中使用 unattended bypass，但它只代表免人工审批，不代表安全隔离。所有操作仍必须限制在该账户解析后的 `$HOME` 下。
- 不使用 Superpowers 执行流，不安装或启动与 ARIS 竞争的 Auto Research 流水线。
- Skill 和 MCP 采用项目级 allowlist；未证明需要的能力默认关闭。安装或升级前查官方最新说明，固定可审计版本，不凭印象安装旧 skill。
- Agent 不应反复探索环境。每个项目只有一个固定服务器环境，并在项目契约中写清激活方式、Python 版本和测试命令。
- 论文源码预留独立目录；Overleaf clone 与代码仓库分开，同步前审查 allowlist diff。

### B. 从 GearShift 提炼出的通用默认值

这些规则通常值得继承，但新项目可在有理由时改写：

- 研究先注册 claim、反证条件、评价协议和硬门槛，再运行实验。
- 先建立正确 primitive、可信测量和 opportunity/oracle upper bound，再堆复杂策略。
- 每个实验绑定不可变、已 push 的 commit，并留下完整 receipt。
- 研究状态只用 `PLANNED`、`RUNNING`、`FAILED`、`OBSERVED`。
- 实现失败与科学失败分开；无效运行不能被写成负面科研结论。
- 先在同一研究故事和方法范围内系统排查实现，再决定是否放弃方向。
- 结果不好看不是换问题、换指标或偷偷加入 rescue component 的理由。
- 论文的 headline 和 scope 由锁定证据决定，不能由最初愿望决定。

### C. 新项目必须填写的变量

GearShift 的项目名、方法、数据路径、SSH 账户、环境名、ARIS SHA、模型栈、venue、门槛和结论不得直接复制。先填写第 10 节的项目卡。

## 2. 研究 taste：什么样的工作值得做

### 2.1 故事优先，但故事必须从证据长出来

用户看重的不只是最终分数，还看重研究故事是否有趣、清楚、可迁移。一个强故事通常包含：

1. 主流 framing 中一个具体而重要的盲点。
2. 一个直观、可复现的矛盾或反例，让盲点变得不可忽略。
3. 一个简洁的新 framing，而不是功能堆叠。
4. 可信测量或 opportunity map，说明机会何时存在、何时消失。
5. 机制解释：为什么有效、为什么失效、边界在哪里。
6. 可以迁移到其他任务或系统的原则。

理想叙事顺序：

```text
盲点 → 具体矛盾 → 直觉 → 形式化 → 简洁机制 → 证据 → 边界 → 可迁移原则
```

不要为了保住最初 headline 强迫结果。端到端收益弱时，可信的机会边界、消失机制、失败地图或负结果仍可能形成更诚实、更有启发性的论文。

### 2.2 偏爱单一、连贯的机制

- 方法应围绕一个中心目标展开。
- 新组件必须对应明确失败模式，并能被消融验证。
- 不用功能数量制造“贡献感”。
- 一个好 reframe 加上精确 formalization，通常比复杂系统更有记忆点。

### 2.3 更重视有效性、含义与可解释性

优先级是：

1. 实验是否有效、公平、可解释。
2. 结论是否被证据支持。
3. 机制是否说得清楚。
4. 数字是否漂亮。

轻微数值、美观或非关键测量问题不应阻塞研究；但任何会让实验无效、不公平或不可解释的问题必须阻塞。

### 2.4 不因第一次失败就换题

在同一故事和 scope 内：

- 先区分实现缺陷、实验设计缺陷和真正的科学负结果。
- 对多个候选修复先排序，逐个试；第一个满足注册标准的方案成功后停止扩张。
- 如果全部失败，留下证据，再收窄 claim 或总结边界。
- 不通过改指标、换 proxy、放宽协议或加入禁止组件来救结果。

### 2.5 判断“有趣”时看真实外部信号

选题价值不只来自 benchmark 热度。也要检查实际部署痛点、社区反复出现的问题、重要技术团队的一手技术报告，以及带可核验互动量的社会信号。这些材料用于判断问题是否值得研究，不能替代科学证据。任何“业界/社区高度关注”的表述都要保存一手链接、互动量和检索日期。

## 3. 写作契约

### 3.0 可迁移的个人文体与论证画像

以下画像已经从用户自己的论文写作中提取完成。它是自包含的：新项目不需要访问原论文、记忆服务或任何写作样本，也不应复制某篇论文的措辞、方法或例子。

作者文体指纹是：

- 标题或开场常用一个有记忆点的问题、角色关系或鲜明对比，引出精确的技术 formulation。
- 引言不从宽泛背景或文献清单起步，而是先指出主流 framing 的盲点，再用一个具体、直观、可复现的矛盾让问题变得可见。
- 从局部对象或供给侧描述，转向更能解释真实决策的需求侧、集合级、状态条件化或系统级视角；具体采用哪种视角由项目问题决定。
- 抽象概念会落到读者可以复述的语义映射：各数学对象在现实问题中分别代表什么，它们之间的关系为什么正好刻画研究问题。
- 方法保持单一主线。所有组件都服务于同一个有动机的 objective，而不是作为互不相关的 feature 卖点。
- 技术部分先给 design overview 或 end-to-end pipeline，再给公式；公式之后立即解释语义、作用和可检验后果。
- 结果叙述不仅说“更高/更快”，还说明优势在哪些 regime 增大、在哪些条件消失，以及这些趋势为何能验证或反驳核心机制。
- 用 matched settings、relative retention、多个 operating points、跨架构/尺度/任务轴和最能暴露机制的极端区间展示规律，而不是依赖一个最优单点。
- 消融用于裁决概念替代解释；负结果用于揭示边界或排除朴素方案，而不是被隐藏。
- 结论回到一个新的看问题方式或可迁移原则，不只是复述 benchmark 提升。

写作 Agent 每次起草关键章节前，应先根据本项目填写以下文体骨架：

```text
Memorable question/contrast:
Prevailing framing:
Concrete contradiction:
Missing viewpoint, unit of analysis, or decision structure:
Named formulation and semantic mapping:
Single organizing objective:
Regime where the mechanism should matter most:
Transferable principle supported by locked evidence:
```

完成草稿后，独立 reviewer 检查：是否有清晰 conceptual turn；每个组件是否回到统一 objective；每个强结论是否来自 locked evidence；是否出现 feature dump、bibliography dump、defensive prose 或结果之外的宏大叙述。

### 3.1 直接写当前正确版本

用户不喜欢防御性写作。修改后直接呈现结果，不向读者展示删除历史。

避免：

- “不含 X 版”“已修正版”“这次没有使用 X”。
- 解释为什么删掉旧内容。
- 为未采用的方案辩护。
- 道歉、预防性自我辩护、冗长铺垫。
- 把内部指令、revision history 或 reviewer 对话写进正文。

如果番茄炒蛋里不该有绿豆，成品就直接叫“番茄炒蛋”；无需强调“不加绿豆”。

这条规则不允许隐藏真实失败、限制、安全边界或 provenance。它们应被准确放入 `Limitations`、研究状态、运行 receipt 或方法说明，而不是作为防御性尾注散落在正文。

### 3.2 语气

- 直接、具体、现代、克制。
- 有信心，但信心范围与证据一致。
- 少用空泛 universal claim；多写条件、机制和可检验对象。
- 不把计划写成结果，不把 reviewer 预测写成观察。
- 不用“novel”“significant”“comprehensive”等词替代事实。

### 3.3 各部分的写法

**摘要**：实际问题 → 现有 framing 的限制 → 研究问题 → formulation → 方法 → 主要证据 → 最强适用区间。

**引言**：尽早给出盲点和具体矛盾；随后才 formalize。不要从宽泛领域历史开始。

**相关工作**：组织成 taxonomy 和 positioning，不做逐篇文献流水账。

**方法**：先给设计总览；每个公式后解释语义、作用和可检验后果。

**实验**：沿正交轴设计；设置匹配条件、公平基线、upper bound、多 operating points 和相对保持率。

**消融**：用于裁决概念替代项和解释机制，不只证明“每个模块都有用”。

**效率**：写清 timing scope、硬件、batch、预处理和遗漏 overhead；优先 deployment-relevant 测量。

**定性分析**：同时展示成功与失败，并把现象连回 objective 和机制。

**讨论与结论**：给出可迁移原则或新的思考方式，不复述 leaderboard。

**限制**：集中、具体、技术化、面向后续工作；不能散落成防御性语句。

### 3.4 图表 taste

优先考虑：

- 早期 paradigm contrast 图。
- 一张读者能复述的方法 schematic。
- 统一格式的高密度主表，包含 upper bound 和多个 operating points。
- 展示趋势、边界或状态条件的曲线，而不只给单点数字。
- 能解释成功与限制的定性图。

每张主图先写清它回答的科学问题，再决定坐标和版式。

## 4. 研究契约：启动实验前必须注册

新项目至少需要以下文档：

```text
.gitignore
AGENTS.md
docs/RESEARCH_PLAN.md
docs/RESEARCH_STATE.md
docs/WRITING_STYLE.md
docs/OPERATIONS.md
config/aris-run-prompt.md
scripts/start_aris.sh
scripts/remote_run.sh
scripts/fetch_results.sh
scripts/sync_overleaf.sh
paper/
```

`RESEARCH_PLAN.md` 至少写清：

- 核心问题和为何重要。
- 最小、可证伪的主要 claim。
- 反证条件和允许的 scope 收窄。
- 方法身份：哪些部分不可变，哪些可优化。
- 禁止的 rescue components。
- 文献检索范围与 novelty 风险。
- 数据、split、dev/test 隔离。
- 公平基线与相同开发预算。
- headline metrics、机制 diagnostics 和统计协议。
- oracle/opportunity upper bound。
- 每个 gate 的通过标准、失败动作和证据位置。
- 论文目标图表及各自回答的问题。
- 目标 venue 和质量门槛；venue 可调工作量和广度，不可降低正确性、证据完整性或可复现性。

推荐实验顺序：

```text
正确 primitive → 测量可信度 → opportunity/oracle → 简单 baseline → 核心方法
→ 公平对照 → 正交轴 → 机制消融 → 效率 → 定性边界 → 锁定结果
```

在结果锁定前，不完成 Abstract/Introduction 的强结论版本。先形成 Results/Figures，再让它们决定最终叙事。

## 5. 证据状态机

`docs/RESEARCH_STATE.md` 是科学状态的唯一事实源。科学运行状态只使用：

- `PLANNED`：已注册，尚未运行。
- `RUNNING`：确实在运行，有 run ID。
- `FAILED`：执行未形成有效观察；保留错误和 receipt。
- `OBSERVED`：有效完成，能定位到 immutable commit 和证据产物。

Gate 可用性使用一个正交字段，不能混入科学状态：

- `READY`：允许推进。
- `BLOCKED`：存在已知 blocker。
- `UNAVAILABLE`：外部依赖、reviewer 或权限当前不可用。

有效完成但结果为负仍是 `OBSERVED`。实现、环境、进程或测量错误导致不可解释才是 `FAILED`。Reviewer 不可用时保留原科学状态，同时把 gate disposition 标为 `UNAVAILABLE`。

不可视为 `OBSERVED`：

- 命令文本、计划表、TODO。
- 缺失或不完整的运行。
- reviewer 的预测或主观判断。
- 没有 source commit 的日志。
- 实现错误产生的数字。

每个 gate 记录：状态、判定标准、commit、run/evidence path、解释边界、失败动作、是否允许进入下一阶段。

每个 `OBSERVED` 条目还要把四层内容写开：

```yaml
measurement: <直接测量到的事实>
supported_interpretation: <证据支持的解释>
unresolved_hypotheses: [<尚未验证的机制猜测>]
excluded_claims: [<当前证据不支持的更强说法>]
```

如果无效产物必须清理，只能在进程停止、目标绝对路径验证无误后删除精确目录，并在 `RESEARCH_STATE.md` 留下 `FAILED` tombstone。历史失败不能被新运行擦除。

## 6. 公平性、统计与结果解释

- claim 使用什么 metric，实验就必须测什么；不能用方便的 proxy 偷换。
- 对比方法使用匹配的数据、环境、硬件、预算和 timing scope。
- dev 用于选择，test 用于最终锁定；锁定后不回看 test 调参。
- 优先 paired evaluation，并报告不确定性区间，而不只给均值。
- 同时报告绝对表现、相对 retention、upper bound gap 和机制 diagnostics。
- 负结果先问：机会不存在、方法捕捉不到、还是实现/测量错误。
- 任何外部兴趣或流行度主张都需要一手来源、可核验互动量和检索日期。

## 7. Codex—Claude 审查契约

### 7.1 角色

- Codex：实现、运行、分析、维护状态、起草论文。
- Claude Code：独立、只读、跨模型审查；不直接修改仓库。
- Codex subagent：同模型内部检查，结论只能标为 provisional。

### 7.2 身份与只读证据

正式 review receipt 至少包含：

- transport 名称和版本。
- `modelUsage.*.canonicalModel` 等结构化模型身份。
- launcher/MCP 中显式的 effort 配置。
- 工具关闭或只读权限模式。
- review 前后 checkout fingerprint。
- 时间、输入 commit、输出路径和退出状态。

Reviewer 在自然语言里自称模型或 effort 不算证据。模型不符、effort 不符、transport 不可用或只读性无法证明时，保留科学状态并把 gate disposition 标为 `UNAVAILABLE` 或 `BLOCKED`；不得自动退化为 Codex 自审。

### 7.3 默认审查问题

每个硬 gate 要求 reviewer 回答：

1. 该 gate 的输入证据是否真实存在且可复现？
2. 实验是否有效、公平、可解释？
3. 当前结论属于 observed、supported interpretation 还是 unresolved hypothesis？
4. 是否存在偷换 metric、scope、baseline 或方法身份？
5. 最强诚实故事是什么？边界和反证是什么？
6. 下一步最小、信息量最大的实验是什么？

## 8. 本地—GitHub—服务器运行制度

### 8.1 拓扑

```text
本地：编辑、短测、commit、push、监督
GitHub：唯一源码事实源
服务器 Agent tmux：Codex + ARIS
服务器 experiment tmux：长实验
服务器数据根：datasets/runs/checkpoints/models/logs
本地 results/remote：经过校验的结果副本
```

Agent 与 experiment tmux 独立，二者可分别重启；实验不能依赖 Agent pane 存活。

### 8.2 Git 硬门槛

跨机器或 dispatch 前：

1. 发送端工作树干净。
2. 当前 HEAD 已提交并与 pushed upstream 相等。
3. 接收端工作树干净。
4. 只执行 `git pull --ff-only`。
5. pull 后接收端 SHA 与指定 SHA 一致。

禁止 dirty pull、自动 stash/reset/merge、force-push，以及用 rsync/scp/tar/数据盘同步源码。服务器不能成为唯一持有未 push 源码的地方。

### 8.3 不可变实验

每次运行绑定完整 commit SHA，并从该 commit 的独立 source snapshot 执行。run ID 包含 UTC、短 SHA 和 nonce。

每次至少保存：

```text
metadata.env
command.sh
command_exit_status   # 用户实验命令本身的退出状态
exit_status           # dispatcher 在清理和校验后的最终退出状态
stdout.log
stderr.log
SHA256SUMS
source/
```

元数据至少包含 run ID、source commit、路径、时间、状态、host、Python、GPU、abort signal 和 cleanup error。

运行在独立 process group；中断时 `TERM`，有限等待后 `KILL`。只有整个进程组退出后才能写终态。低磁盘空间停止新实验，不自动删历史证据。

### 8.4 原子 fetch

结果先下载到 `.partial-*`，验证 metadata、exit status 和 checksum 后原子移动到最终目录。已存在的最终目录不得覆盖。

## 9. 环境、能力与安全

### 9.1 固定环境

- 一个项目只有一个服务器环境。
- 推荐 Miniforge/Conda 管 Python 和兼容层，`pyproject.toml` + lockfile 管项目依赖。
- 先验证 CUDA、PyTorch、核心依赖和 lockfile 是否完整支持 Python 3.13；支持则固定 3.13，否则保存兼容性证据并固定 3.12。
- Miniforge/Conda 是需要 CUDA、编译库或二进制兼容层时的推荐默认；纯 Python 项目若 `uv` 足够，可以只用 `uv`。无论哪种方式，项目仍只能有一个固定环境。
- launcher、dispatcher、每个实验都显式激活环境并断言 Python 版本。
- 使用 frozen/locked 安装；无人值守过程中不重新解析依赖。
- 不创建临时 `.venv`，不让 Agent 自选 Python 或每轮重查环境。
- 项目契约直接写明环境前缀、激活命令、import smoke、test 和 lint 命令。

### 9.2 ARIS 与 capabilities

- ARIS 固定到完整 40 字符 commit SHA，不跟随浮动 main。
- 安装/升级前查官方仓库和说明；审计实际 skill dependency closure。
- 项目 Codex profile 只开放项目批准的 ARIS skills、必要 reference、唯一 reviewer MCP，以及项目确实需要时的官方 notebook skill。
- 其他 user/plugin skills、apps 和 MCP 默认关闭。
- capability 或 CLI 版本变化后重新生成 profile，并重新验收实际 prompt 输入。
- 某些 bundled system skills 即使配置为 disabled 仍可能可见；验收实际 prompt 输入和运行行为，不能只检查配置文件。
- 升级失败则回滚到最后批准的 SHA。

### 9.3 unattended 安全边界

- 使用专用非 root 账户。
- 所有 repo、prompt、环境、数据、日志和 ARIS 路径先 resolve，再验证位于 `$HOME` 下；越界 fail closed。
- 禁止 `sudo`、系统目录、其他用户目录、系统服务和其他用户进程。
- 只管理本项目 tmux、进程和文件。
- 凭据只通过官方交互登录或用户目录 secret store；不写入 Git、prompt、日志或 shell history。
- bypass 不是 sandbox；Unix 账户权限才是硬边界。

在把技术选择升级给用户之前，Agent 必须先说明：当前实现与结论是否仍有效；能否在同一方法身份内继续优化；已证实或最可能的根因；推荐选项对 claim、成本和计划的影响。能用仓库、实验或官方资料解决的问题，不转交给用户。

## 10. 新项目配置卡

开始前复制并填写；未知项写 `UNRESOLVED`，不要猜。

```yaml
project:
  name: <PROJECT_NAME>
  slug: <PROJECT_SLUG>
  core_question: <ONE_SENTENCE_QUESTION>
  target_venue: <VENUE_OR_UNRESOLVED>
  quality_bar: <TARGET_BAR>

git:
  github_owner: <OWNER>
  repository: <REPO>
  visibility: <PRIVATE_OR_PUBLIC>
  local_repo: <ABSOLUTE_PATH>
  server_repo: <HOME_RELATIVE_PATH>

server:
  ssh_alias: <ALIAS>
  account: <DEDICATED_NON_ROOT_USER>
  authorized_boundary: <RESOLVED_HOME>
  data_root: <HOME_RELATIVE_DATA_ROOT>
  agent_tmux: <PROJECT_SLUG>-aris
  experiment_tmux: <PROJECT_SLUG>-exp

environment:
  manager: <conda-plus-uv_OR_uv>
  identifier: <ENV_NAME_OR_PATH>
  python: <3.13_OR_DOCUMENTED_3.12>
  lockfile: <LOCKFILE>
  activate: <EXPLICIT_COMMAND>
  import_smoke: <COMMAND>
  test: <COMMAND>
  lint: <COMMAND>

executor:
  cli: codex
  profile: <PROJECT_PROFILE>
  model: <PINNED_CODEX_MODEL>
  reasoning_effort: high
  service_tier: default
  fast_enabled: false
  bypass: true

reviewer:
  cli: claude-code
  transport: <APPROVED_REVIEW_MCP>
  model: <PINNED_CLAUDE_MODEL>
  suggested_default: claude-fable-5-1
  effort: medium
  read_only: true
  identity_field: modelUsage.*.canonicalModel

aris:
  repo: <OFFICIAL_REPO>
  full_sha: <40_CHAR_SHA>
  allowed_skills: <LIST>
  forbidden_skills: [superpowers, competing-autoresearch-pipelines]

research:
  plan: docs/RESEARCH_PLAN.md
  state: docs/RESEARCH_STATE.md
  first_gate: <FIRST_HARD_GATE>
  primary_claim: <FALSIFIABLE_CLAIM>
  falsification: <PRE_REGISTERED_CRITERION>
  forbidden_rescues: <LIST>

budgets:
  max_wall_clock_hours: <NUMBER>
  max_gpu_hours: <NUMBER>
  max_api_spend: <AMOUNT>
  max_single_run_hours: <NUMBER>
  min_free_disk_gb: <NUMBER>
  max_consecutive_failures: <NUMBER>

control:
  stop_sentinel: <HOME_RELATIVE_PATH>
  supervisor_interval: <DURATION>

story_brief:
  viewpoint: <ONE_SENTENCE_VIEWPOINT>
  prevailing_framing: <WHAT_MOST_WORK_ASSUMES>
  blind_spot: <WHAT_IT_MISSES>
  concrete_contradiction: <REPRODUCIBLE_COUNTEREXAMPLE>
  unified_mechanism: <MINIMAL_MECHANISM>
  support_result: <WHAT_WOULD_SUPPORT_THE_STORY>
  narrowing_result: <WHAT_WOULD_NARROW_IT>
  valuable_negative_result: <MECHANISM_KNOWLEDGE_IF_HEADLINE_FAILS>

paper:
  source: paper/
  overleaf_clone: <IGNORED_SEPARATE_PATH>
  sync_allowlist: <LIST>
```

### Overleaf 同步契约

- `paper/` 是主代码仓库中的权威论文源。
- `paper-overleaf/` 是 ignored 的独立 Git clone，只作为发布与协作镜像。
- 同步只覆盖显式 allowlist；先 dry-run，再查看完整 diff。
- 两个仓库分别 commit/push。禁止后台双向同步、自动解冲突和 force-push。
- 从 Overleaf 导入修改时，先审查 diff，再把接受的修改作为主仓库的新 commit。
- 构建产物和凭据不进入任一源码提交。
- Overleaf 登录由用户通过官方交互流程完成；不要让用户把 token、device code 或密码发进聊天、日志或 shell history。
- 尚无 Overleaf 项目时，此项可标为 `CONDITIONAL N/A`，不阻塞科学研究；一旦启用，认证和双向 smoke test 必须通过。

## 11. 首次启动验收

真正开始无人值守研究前，逐项记录 `PASS`、`FAIL` 或 `N/A`，并附 UTC、host、commit、命令和 evidence path。安全、Git、环境、模型、reviewer、dispatch 和 receipt 都是 `REQUIRED`，必须 `PASS`；只有条件尚不存在的集成项可以 `CONDITIONAL N/A`：

- SSH BatchMode 可用。
- 本地、upstream、服务器 SHA 一致。
- `.gitignore` 覆盖 secrets、数据、runs、checkpoint、模型和论文构建产物。
- GPU、磁盘和数据目录可写。
- 环境 import、tests、lint 通过。
- Codex 与 Claude 在 fresh shell 和 detached tmux 中均可用。
- Codex 模型和 `model_reasoning_effort="high"` 来自显式 profile；启动参数和有效配置中没有 Fast/service-tier override，最终 service tier 为默认档。
- Claude reviewer 的 canonical model、medium effort 和只读 receipt 有结构化证据。
- ARIS full SHA、skill allowlist、forbidden list 和 reviewer routing audit 通过。
- dirty tree 和 divergent tree 会被 launcher 拒绝。
- 一个无害实验在 SSH 断开后完成，source commit、metadata、exit status、checksum 和原子 fetch 全部正确。
- `paper/` 权威源、Overleaf clone、allowlist、dry-run/diff 和独立提交边界已验证；尚未启用 Overleaf 时标为 `CONDITIONAL N/A`。
- 预算、连续失败阈值、磁盘门槛、stop sentinel 和唯一 supervisor 已验证。

## 12. 给新项目第一个 Codex session 的 Prompt

把下面整段复制给新项目服务器上的 Codex。先将尖括号变量替换；不能确定的保留 `UNRESOLVED`。

```text
你现在负责启动 <PROJECT_NAME> 的 Auto Research。Codex 是唯一执行器；Claude Code 只通过批准的 <REVIEW_TRANSPORT> 做独立、只读的跨模型审查。Claude reviewer 固定为 <CLAUDE_REVIEW_MODEL>、effort=medium；Codex 固定为 <CODEX_MODEL>、reasoning_effort=high，service tier 使用默认档且不启用 Fast override。若没有另行选择，<CLAUDE_REVIEW_MODEL> 使用当前建议的 Fable 5.1，但仍须重新验证 canonical identity。不要使用 Superpowers 工作流，也不要安装或启动竞争性的 Auto Research pipeline。

你的第一目标不是立刻跑大实验，而是把项目变成一个可审计、可恢复、服务器无人值守的研究系统，然后推进第一个有效硬门槛。

先读取仓库中所有现有研究计划、README、AGENTS.md、依赖文件和已有代码。保留用户现有改动；不要清理或覆盖不属于你的文件。对于时效性信息——ARIS 安装方式、skills、CLI 模型名称和配置——只查官方最新来源，不凭印象；将 ARIS 固定到完整 commit SHA。

Phase 0 先确认：本地项目和 GitHub remote 已建立，初始源码已 commit/push，SSH BatchMode 可用，服务器 clone 位于授权 $HOME 内。Codex/Claude 缺少登录时，只给出官方交互登录命令并等待用户完成；不要要求用户把 token、device code 或密码发进聊天、日志或 shell history。

按照以下顺序工作：

1. 形成一份简洁的项目盘点：当前研究问题、已有资产、缺失项、风险和真正需要用户决定的问题。能从仓库或官方资料查到的不要询问用户。
2. 在本轮直接创建或完善 .gitignore、AGENTS.md、docs/RESEARCH_PLAN.md、docs/RESEARCH_STATE.md、docs/WRITING_STYLE.md、docs/OPERATIONS.md、config/aris-run-prompt.md 和安全、非阻塞的启动脚本。把项目特有内容和用户稳定偏好分开；不要只交付待办清单。
3. 在 RESEARCH_PLAN 中注册：最小可证伪 claim、反证条件、方法身份、禁止的 rescue components、评价协议、公平基线、dev/test 隔离、统计方法、oracle/opportunity upper bound、硬门槛和失败动作。
4. 研究叙事优先寻找：主流 framing 的盲点、具体矛盾、简洁 reframe、机会存在与消失的条件、机制解释和可迁移原则。故事必须从 observed evidence 长出来，不能强迫数据支持预想 headline。
5. 科学状态采用 PLANNED/RUNNING/FAILED/OBSERVED；gate disposition 单独采用 READY/BLOCKED/UNAVAILABLE。有效负结果仍是 OBSERVED；实现或测量错误才是 FAILED。命令、计划、缺失运行和 reviewer 预测都不是观察结果。每个 OBSERVED 分开记录 measurement、supported interpretation、unresolved hypotheses 和 excluded claims，并指向已 push 的 immutable commit 与具体 evidence path。
6. 配置唯一服务器环境 <PROJECT_ENV>，使用 <ENV_ACTIVATION> 显式激活并断言 Python <PYTHON_VERSION>；先验证 3.13 的 CUDA/核心依赖兼容性，支持则用 3.13，否则记录证据并固定 3.12。使用 lockfile/frozen 安装；纯 Python 项目可使用一个固定、由 lockfile 管理的 uv 环境，需要二进制兼容层时用 Miniforge/Conda。禁止的是每轮创建或重选临时环境。
7. 建立 Git fail-closed handoff、immutable experiment snapshot、独立 <AGENT_TMUX>/<EXPERIMENT_TMUX>、运行 metadata/checksum、原子 fetch 和恢复规则。源码只通过 GitHub 同步；大产物写 <DATA_ROOT>。
8. 为当前项目生成最小 capability allowlist。除批准的 ARIS skills、必要 reference、项目确实需要时的官方 notebook skill和 <REVIEW_TRANSPORT> 外，关闭其他项目不需要的 skills、plugins、apps 和 MCP。审计 ARIS 的实际依赖闭包和实际 prompt 输入。
9. 验证 reviewer：模型身份只能取结构化 modelUsage.*.canonicalModel；effort 取显式配置；只读性由工具/权限配置及审查前后 checkout fingerprint 证明。任何一项不符，保留科学状态并把 gate disposition 标为 UNAVAILABLE/BLOCKED，不用 Codex 自审代替。
10. 所有 unattended 操作限制在专用非 root 账户解析后的 $HOME 内。禁止 sudo、系统目录、其他用户目录、系统服务和其他用户进程。bypass 只用于免审批，不视为 sandbox。
11. 完成首次启动验收，再运行最小、无害的端到端 smoke experiment。只有 smoke receipt 完整后，才推进 RESEARCH_STATE 中第一个未完成硬门槛。
12. 每到硬 gate，先做内部验证，再请求 Claude 只读 cross-review。若证据失败，继续修复；若属于真实科学负结果，保留并据此收窄结论，不改指标或方法身份救结果。
13. 配置 max wall-clock/GPU/API/single-run budget、最小剩余磁盘、连续失败阈值和 stop sentinel。每次 dispatch 前检查。用户触发 stop sentinel 后不启动新实验，并按注册策略安全结束或保留当前运行。使用唯一的周期 supervisor/heartbeat；已有监督任务时更新它，不创建重复任务。
14. 将 paper/ 设为权威论文源；paper-overleaf/ 是 ignored 的独立 clone。实现 allowlist-only、dry-run+diff、分别 commit 的同步桥。Overleaf 尚未创建时标为 CONDITIONAL N/A，不阻塞科研。

写作遵循本 prompt 内置的个人文体画像，不依赖任何外部论文样本：优先构造“有记忆点的问题/对比→主流 framing 的盲点→具体矛盾→更合适的需求侧/集合级/状态条件化/系统级视角→命名 formulation→统一 objective→最强成立 regime→可迁移原则”。所有数学对象都要给可复述的现实语义；所有组件都要回到同一个 objective；结果要解释优势出现与消失的条件。直接呈现当前正确版本，不写“修正版”“不含 X 版”或删除理由；不道歉、不预防性辩护、不泄露 revision history。真实失败、限制、安全和 provenance 必须准确记录在正确位置。摘要使用“问题→限制→问题→formulation→方法→证据→最强区间”；相关工作按 taxonomy；消融用于裁决机制；结论给可迁移原则。

在向我提出技术选择前，先说明：当前实现与结论是否有效；能否在同一方法身份内继续优化；根因是什么；你的推荐及其对 claim、成本和计划的影响。

在第一次回复中给我：
- 对研究问题和现状的证据化盘点；
- 本轮已创建或修改的文件、验证证据和剩余 blocker；
- 你建议注册的 primary claim、反证条件和第一硬门槛；
- 运行与审查配置中仍为 UNRESOLVED 的项目变量；
- 接下来你可以立即执行的最小安全步骤。

除非缺少会实质改变研究方向、外部权限或安全边界的决定，否则不要停在泛泛提问上；继续完成可安全执行的盘点和配置。不要把计划写成实验结果。
```

## 13. 完成 bootstrap 后的无人值守研究 Prompt

所有 `REQUIRED` 验收通过、条件项为 `PASS` 或合理的 `N/A` 后，可把下面内容作为服务器 ARIS 主循环的核心 prompt：

```text
从 docs/RESEARCH_STATE.md 的第一个未完成硬门槛继续。

开始前验证：工作树干净且 HEAD 已 push；服务器 ff-only 同步；固定环境、Codex profile、ARIS SHA、reviewer receipt、数据路径、磁盘、预算和 stop sentinel 状态正常。任何失败都 fail closed，并把 gate disposition 和 blocker 写入 RESEARCH_STATE。

只执行 RESEARCH_PLAN 已注册的方法、指标、数据协议和允许优化。每个实验从 immutable commit snapshot 运行，记录完整 metadata、日志、退出状态和 checksum。状态只用 PLANNED/RUNNING/FAILED/OBSERVED。

优先选择能最大幅度减少关键不确定性的最小实验。实现失败先系统排查；真实科学负结果保留并解释机制。不要用新 proxy、放宽门槛、隐藏失败或加入禁止组件来救 headline。

每个硬 gate 先内部验证，再通过配置卡中的 <REVIEW_TRANSPORT> 请求固定的 Claude 模型、medium、只读审查，并保存结构化模型身份和只读证据。reviewer 不可用或身份不符时将 gate disposition 标为 UNAVAILABLE/BLOCKED，不用 Codex 自审替代。

维护论文素材，但只把 OBSERVED 写成结果。故事优先寻找盲点、矛盾、简洁 reframe、状态条件、机会边界、失效机制和可迁移原则。写作直接呈现当前结果，不暴露删除历史或防御性措辞。

每轮 dispatch 前检查预算、磁盘、连续失败阈值和 stop sentinel。达到任一停止条件后不再启动新实验，安全处理当前运行并留下 receipt。否则持续推进到当前 gate 形成 OBSERVED 的正面或负面结论，或记录明确 blocker；实现/测量无效才记为 FAILED。
```

## 14. 迁移时不要复制的 GearShift 内容

以下内容只可作为例子，不能带入新项目：

- GearShift 的研究方法、claim、exactness/oracle gate 和实验协议。
- GearShift 仓库、SSH alias、用户名、端口、目录和环境名。
- GearShift 当前模型栈、数据集、checkpoint 和结果。
- GearShift 当前 ARIS SHA、skill 数量、override 和 reviewer bridge 路径。
- GearShift 当前批准的具体 Codex/Claude 模型名称；角色分工、Codex `high`、默认 service tier/无 Fast override 和 Claude medium 可以继承，但模型仍须在新项目中固定和验收。
- GearShift 的 venue 路线与时间表。
- `GEARSHIFT_*` 环境变量和 `tmux:aris`/`tmux:exp` 名称。

新项目应使用带项目 slug 的 tmux、环境、数据根、Codex profile 和 receipt，避免不同 Auto Research 项目互相污染。

## 15. 判断 bootstrap 完成的标准

只有同时满足以下条件，才算“可以开始 Auto Research”：

1. 研究问题、claim、反证和第一 gate 已注册。
2. 环境和模型配置显式固定并在 fresh shell/tmux 验证。
3. ARIS 和 capability closure 已 pin、已审计。
4. Codex executor 与 Claude read-only reviewer 的边界有机器可验证证据。
5. Git、immutable dispatch、进程清理、receipt 和 atomic fetch 通过 smoke test。
6. RESEARCH_STATE 能区分计划、运行、失败和观察。
7. 论文与 Overleaf 边界清楚。
8. unattended 操作被限制在专用账户 `$HOME` 内。
9. 新项目第一个科学硬门槛已有明确下一步，而不是泛泛“开始研究”。
10. 预算、stop sentinel、连续失败阈值和唯一 supervisor 已配置并通过验收。
