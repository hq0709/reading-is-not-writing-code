---
artifact_contract: "ce-handoff/v1"
created_at: "2026-09-02T06:30:23.4126534Z"
title: "精简版 Auto Research 启动契约"
summary: "只负责新项目初始化、配置卡、验收和首轮启动；研究写作与记录规范由独立契约维护。"
keywords: ["autoresearch", "bootstrap", "startup", "codex", "aris"]
cwd: "E:/projects/GearShift"
resume_focus: "使用精简启动契约初始化新项目，并同时接入独立研究写作与记录契约。"
repository: "GearShift"
repo_root_sha: "e02fb45b58a221940d860c9846e9457cf053b076"
branch: "codex/use-fable-5-reviewer"
head: "990b7b90597851c3dbe6e616bfc0d58c10ffb352"
worktree_path: "E:/projects/GearShift"
---

# 精简版 Auto Research 启动契约

## 1. 本文档只负责启动

本契约用于把一个已有课题初始化为可运行、可恢复、可审计的 AutoResearch 项目。它只规定：项目变量、初始文件、角色、运行门槛、首次验收和首轮动作。

必须与以下两份独立文档一起使用：

1. `research-writing-recording-contract.md`：研究 taste、写作风格、证据语言、AutoResearch 记录卫生和持久化规则。
2. `local-codex-remote-server-setup.md`：本地 Codex、GitHub、SSH、远程服务器、环境、Codex/Claude/ARIS、tmux 和结果同步。

复制进项目后推荐命名：

```text
docs/AUTORESEARCH_STARTUP.md
docs/RESEARCH_WRITING_AND_RECORDING.md
docs/REMOTE_RESEARCH_OPERATIONS.md
```

不要把后两份全文重新塞回启动契约。启动 prompt 只要求读取它们。

## 2. 新项目配置卡

未知项写 `UNRESOLVED`，不能猜：

```yaml
project:
  name: <PROJECT_NAME>
  repository: <OWNER/REPO>
  default_branch: <BRANCH>
  problem_statement: <ONE_PARAGRAPH>
  source_material: <PATHS>

research:
  core_question: <FALSIFIABLE_QUESTION>
  first_gate: <GATE_ID>
  first_decisive_experiment: <EXPERIMENT_ID>
  primary_metrics: <LOCKED_METRICS>
  baselines: <FAIR_BASELINES>
  evidence_root: <PATH>

runtime:
  local_repo: <ABSOLUTE_PATH>
  ssh_alias: <ALIAS>
  server_repo: <HOME_RELATIVE_PATH>
  data_root: <HOME_RELATIVE_PATH>
  environment_activation: <COMMAND>
  python: <PINNED_VERSION>
  agent_tmux: <NAME>
  experiment_tmux: <NAME>

executor:
  cli: codex
  model: <PINNED_CODEX_MODEL>
  reasoning_effort: high
  fast: false
  profile: <PROJECT_PROFILE>

reviewer:
  cli: claude
  transport: <APPROVED_READ_ONLY_TRANSPORT>
  model: <PINNED_CLAUDE_MODEL>
  effort: medium
  read_only: true

aris:
  full_sha: <40_CHARACTER_SHA>
  allowlist: <PATH>
  forbidden_list: <PATH>

control:
  max_wall_clock: <LIMIT>
  max_gpu_budget: <LIMIT>
  max_api_budget: <LIMIT>
  min_free_disk_gb: <NUMBER>
  max_consecutive_failures: <NUMBER>
  stop_sentinel: <HOME_RELATIVE_PATH>
```

## 3. 初始化后的最小仓库结构

```text
AGENTS.md
docs/AUTORESEARCH_STARTUP.md
docs/RESEARCH_WRITING_AND_RECORDING.md
docs/REMOTE_RESEARCH_OPERATIONS.md
docs/RESEARCH_PLAN.md
docs/RESEARCH_STATE.md
docs/EXPERIMENT_REGISTRY.md
config/aris-run-prompt.md
config/aris-skills.txt
config/aris-forbidden-skills.txt
paper/
scripts/start_aris.sh
scripts/remote_run.sh
scripts/fetch_results.sh
```

`AGENTS.md` 只承担索引和硬边界：

- 研究、写作、实验摘要和状态记录遵循 `docs/RESEARCH_WRITING_AND_RECORDING.md`。
- 机器、同步、环境和恢复遵循 `docs/REMOTE_RESEARCH_OPERATIONS.md`。
- 当前问题、协议、门槛和允许优化遵循 `docs/RESEARCH_PLAN.md`。
- 当前阶段、证据与 blocker 遵循 `docs/RESEARCH_STATE.md`。

## 4. 启动前必须注册

在 `RESEARCH_PLAN` 中填写：

- 最小可证伪 claim 和反证条件。
- 方法身份及允许、禁止的优化。
- 数据、切分、指标和统计协议。
- 公平 baseline、oracle/opportunity upper bound 和成本口径。
- dev/test 隔离与最终 test 使用时点。
- 每个 gate 的通过门槛、失败动作和证据位置。
- 预算、磁盘、停止和恢复条件。

这里不重复证据语言和记录模板；它们由 `RESEARCH_WRITING_AND_RECORDING` 统一规定。

## 5. 固定角色

```text
本地 Codex：规划、编辑、短测试、监督和接收结果
服务器 Codex：ARIS 执行器、代码实现、实验投递、证据集成
Claude Code：独立、只读、cross-model reviewer
GitHub：唯一源码同步事实源
服务器外部数据目录：数据、权重、checkpoints、raw runs 和大日志
```

Codex executor 使用项目固定模型、`high` reasoning，且不启用 Fast。Claude reviewer 使用固定模型和 `medium` effort；模型身份来自结构化 transport metadata，effort 来自明确 launcher/MCP 配置。Claude 自述不构成身份凭据，Codex subagent 不替代 cross-model gate。

## 6. 首次验收

所有必需项记录 `PASS` 或 `FAIL`，附 UTC、host、commit、命令和 evidence path：

- 本地到服务器 SSH BatchMode。
- GitHub 仓库访问、服务器所需读写权限和 reversible push probe。
- 本地、upstream、服务器 commit 一致，dirty/divergence fail closed。
- 固定环境在 fresh shell 和 detached tmux 中激活到同一路径。
- Python、核心 import、短测试、lint 和必要 GPU smoke。
- Codex profile 的 model、`high`、Fast disabled 和最小 capabilities。
- Claude reviewer 的 canonical model、medium effort、read-only 和 checkout fingerprint。
- ARIS full SHA、allowlist、forbidden list、dependency closure 和 reviewer routing。
- agent tmux 与 experiment tmux 独立存在且 SSH 断开后继续运行。
- 一个无害实验生成完整 receipt，并能校验后拉回本地。
- budget、disk threshold、stop sentinel 和唯一 supervisor。
- `paper/` 与可选 Overleaf clone 的边界。
- 写作与记录契约的最小验收全部通过。

任何必需项失败时停止正式 AutoResearch，先修复对应基础设施。计划、未运行命令和口头确认不算证据。

## 7. 给新项目第一个 Codex session 的 Prompt

```text
请初始化当前项目的 AutoResearch，但不要立即启动昂贵实验。

先读取并遵循：
1. docs/AUTORESEARCH_STARTUP.md
2. docs/RESEARCH_WRITING_AND_RECORDING.md
3. docs/REMOTE_RESEARCH_OPERATIONS.md
4. 当前课题材料与已有项目文档

填写所有 UNRESOLVED 配置项；检查 Git 状态、SSH、服务器权限、固定环境、Codex profile、Claude reviewer、ARIS pin、capability allowlist、tmux、数据路径、预算和 stop sentinel。只安装项目必需依赖，不创建临时 .venv，不启用未批准的 skill/MCP/app，不使用 Superpowers 或其他自主研究流水线。

从课题材料提炼一个最小可证伪核心问题，在 RESEARCH_PLAN 中注册 claim、反证条件、方法身份、允许与禁止优化、指标、baseline、数据协议、dev/test 隔离、统计方法、成本口径、gate 和失败动作。建立 RESEARCH_STATE 与 EXPERIMENT_REGISTRY，但按独立记录契约保持它们简洁且职责分离。

运行首次验收。验收全绿后，只执行第一项能最大幅度减少关键不确定性的短实验。提交并 push 所有源码和轻量状态；大型 artifact 留在服务器数据目录。最后报告当前 gate、已观察事实、证据位置、决策和下一项动作。
```

## 8. 验收通过后的无人值守 Prompt

```text
读取 docs/RESEARCH_PLAN.md、docs/RESEARCH_STATE.md、docs/EXPERIMENT_REGISTRY.md、docs/RESEARCH_WRITING_AND_RECORDING.md 和 docs/REMOTE_RESEARCH_OPERATIONS.md。

开始前验证：Git checkout clean 且与 upstream 一致；固定环境、Codex profile、ARIS SHA、Claude reviewer、数据路径、磁盘、预算和 stop sentinel 正常。任一硬门槛失败时停止投递并记录 blocker。

只执行 RESEARCH_PLAN 已注册的方法、指标、数据协议和允许优化。每个实验从 immutable commit 运行，保存 run receipt。按独立记录契约维护状态：当前实验直接写它做了什么和观察到什么；失败、blocker、限制和 provenance 只放在指定账本，不把后续方案命名为“不含 X 版”或“修复 A 失败后的版本”。

每个硬 gate 先做 Codex 内部验证，再通过批准的 Claude read-only transport 进行 cross-model review。没有结构化模型身份、medium effort 和只读证据时，cross-model gate 不通过。

持续选择最小且最能裁决关键不确定性的实验，直到当前 gate 得到有效观察、出现明确 blocker，或触发预算、磁盘、连续失败和 stop sentinel 条件。保持源码和轻量状态及时 commit/push；不把大 artifact 和 secret 放进 Git。
```

## 9. 启动完成定义

以下条件同时成立才算完成 bootstrap：

1. 三份契约各自独立，职责没有重复扩张。
2. 配置卡无影响运行的 `UNRESOLVED`。
3. 研究 plan、state 和 registry 已建立。
4. 本地—GitHub—服务器链路通过验收。
5. Codex executor、Claude reviewer 和 ARIS routing 通过结构化验收。
6. 第一项短实验可以从 immutable commit 投递并生成 receipt。
7. 写作和 AutoResearch 记录遵循独立契约，启动 prompt 不再内嵌其全文。

完成后，后续写作偏好和记录规则只更新 `RESEARCH_WRITING_AND_RECORDING`；基础设施只更新 `REMOTE_RESEARCH_OPERATIONS`；本文件只在启动流程本身变化时更新。
