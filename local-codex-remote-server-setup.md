---
artifact_contract: "ce-handoff/v1"
created_at: "2026-09-02T04:26:03Z"
title: "本地 Codex 与远程 Auto Research 服务器配置手册"
summary: "从零配置本地编辑监督端、GitHub 源码通道和服务器常驻 Codex+ARIS 执行端，并规定双方的交接与通信协议。"
keywords: ["codex", "remote-server", "autoresearch", "aris", "ssh", "git", "tmux"]
cwd: "E:/projects/GearShift"
resume_focus: "为新 Auto Research 项目填写配置卡，按 Phase 0 到 Phase 8 配置本地与服务器，并完成端到端验收。"
repository: "GearShift"
repo_root_sha: "e02fb45b58a221940d860c9846e9457cf053b076"
branch: "codex/use-fable-5-reviewer"
head: "990b7b90597851c3dbe6e616bfc0d58c10ffb352"
worktree_path: "E:/projects/GearShift"
---

# 本地 Codex 与远程 Auto Research 服务器配置手册

本手册可以独立完成基础设施配置；与[《可复用 Auto Research 启动契约与首轮 Prompt》](./autoresearch-bootstrap-contract.md)一起分发时，后者补充“研究应如何进行”。复制到新项目后可以删除本文件顶部仅用于临时 handoff 检索的 `ce-handoff` frontmatter。

目标状态：本地电脑可以关机；服务器上的 Codex + ARIS 和长实验继续运行。两端不共享工作目录，不产生两份互相不知道的源码。

## 1. 最终拓扑

```text
┌──────────────────────────────┐
│ 本地 Windows / macOS          │
│ Codex：编辑、短测、监督        │
│ Git / gh / SSH 客户端         │
└──────────────┬───────────────┘
               │
       Git push/pull：源码与研究状态
       SSH：控制、启动、查看、投递任务
       校验 fetch：按 run ID 拉结果
               │
┌──────────────▼───────────────┐
│ GitHub 私有仓库               │
│ 唯一源码事实源                 │
└──────────────┬───────────────┘
               │ deploy key / machine credential
┌──────────────▼───────────────┐
│ 远程 Linux 专用非 root 账户    │
│ tmux:<PROJECT>-aris           │
│   Codex executor + ARIS       │
│ tmux:<PROJECT>-exp            │
│   长实验、独立进程组            │
│ $HOME/data/<PROJECT>/...      │
└──────────────────────────────┘
               │
               └── Claude Code reviewer
                   独立、只读、medium effort
```

这里没有“本地 Codex 直接远程控制另一个 Codex session”的隐式通道。双方通过可审计的工件沟通：Git commit、`RESEARCH_STATE.md`、run receipt 和 reviewer receipt。SSH 只负责控制，不承载唯一源码。

## 2. 五条通信通道

| 通道 | 用途 | 可以传什么 | 不可以传什么 |
|---|---|---|---|
| GitHub | 源码事实源 | 代码、配置、计划、研究状态、轻量论文源 | 数据、权重、checkpoint、raw runs、secret |
| SSH | 控制平面 | 检查、启动 tmux、提交已 push commit 的实验命令 | 未提交源码、聊天中的 token/password |
| tmux | 服务器持久性 | Agent 主循环、长实验 | 机器间同步 |
| Run store | 大产物 | 数据、日志、结果、模型、receipt | 活跃源码的替代副本 |
| Result fetch | 结果回收 | 指定 run ID 的校验后副本 | 整个服务器目录、源码树 |

默认只允许一个源码 writer：

- 远程 ARIS 运行期间，服务器 Codex 是当前 writer；本地只读监督。
- 本地要接管编辑时，先让远程 Codex 到达安全停止点，确认它已 commit + push 且服务器工作树干净，再在本地 `pull --ff-only`。
- 本地完成后 commit + push；服务器确认 clean 后 `pull --ff-only`，再恢复 ARIS。
- 不允许本地和服务器同时在同一分支各改一坨未提交文件。
- 确需并行开发时使用不同分支/PR，而不是同一分支上的两个 dirty tree。

## 3. 项目配置卡

先填写，未知项写 `UNRESOLVED`，不要猜：

```yaml
project:
  name: <PROJECT_NAME>
  slug: <project-slug>
  github_owner: <OWNER>
  github_repo: <REPO>
  default_branch: <main>

git:
  server_writer: true
  server_push_required: true

local:
  os: <windows_OR_macos>
  repo: <ABSOLUTE_LOCAL_REPO>
  ssh_alias: <SSH_ALIAS>

server:
  host: <HOSTNAME>
  port: <PORT>
  user: <DEDICATED_NON_ROOT_USER>
  home: <RESOLVED_HOME>
  repo: <HOME_RELATIVE_REPO>
  data_root: <HOME_RELATIVE_DATA_ROOT>
  aris_repo: <HOME_RELATIVE_ARIS_REPO>
  agent_tmux: <project-slug>-aris
  experiment_tmux: <project-slug>-exp

environment:
  manager: <conda-plus-uv_OR_uv>
  identifier: <ENV_NAME_OR_PATH>
  python: <3.13_OR_DOCUMENTED_3.12>
  activation: <EXPLICIT_ACTIVATION_COMMAND>
  lockfile: <LOCKFILE>
  smoke: <IMPORT_SMOKE_COMMAND>
  test: <TEST_COMMAND>
  lint: <LINT_COMMAND>

codex:
  profile: <PROJECT_PROFILE>
  model: <PINNED_CODEX_MODEL>
  reasoning_effort: high
  service_tier: OMITTED
  fast_enabled: false
  bypass: true

reviewer:
  transport: <APPROVED_REVIEW_MCP>
  model: <PINNED_CLAUDE_MODEL>
  suggested_default: claude-fable-5-1
  effort: medium
  read_only: true

aris:
  full_sha: <40_CHARACTER_SHA>
  allowed_skills: <LIST_FILE>
  forbidden_skills: <LIST_FILE>
  approved_groups_csv: <CSV_OR_EMPTY>
  approved_extra_skills_csv: <CSV_OR_EMPTY>
  forbidden_skills_csv: <CSV_OR_EMPTY>

control:
  stop_sentinel: <HOME_RELATIVE_PATH>
  supervisor_interval: <DURATION>
  min_free_disk_gb: <NUMBER>
  max_consecutive_failures: <NUMBER>
```

## 4. Phase 0：准备与边界

开始前确认：

- 服务器是专用、非 root 研究账户。
- 项目政策要求所有写入、删除、tmux 和项目进程操作位于该账户解析后的 `$HOME` 内；launcher 对所有已声明路径 fail closed。
- bypass 不会把 `$HOME` 变成技术 sandbox。真正的权限边界是该 Unix 账户拥有的全部权限，可能包含 `$HOME` 外的共享路径。若需要机器强制的 home-only 边界，必须另加容器/VM、mount namespace 或操作系统权限隔离，并验证越界写入失败；否则只能把 home-only 作为强制项目契约和可审计 guard，不能声称为硬隔离。
- 不使用 `sudo`、系统目录、其他用户目录、系统服务或其他用户进程。
- GitHub 仓库建议为 private。
- 本地已有 Git、GitHub CLI、OpenSSH client 和 Codex。
- 服务器基础工具至少有 `git`、`tmux`、`ssh`、`curl`、`tar`、`gzip`、Python 和 GPU 驱动工具。
- 需要管理员安装基础包时，由用户在交互 SSH 中处理；无人值守 Agent 不执行 `sudo`。

先把以下内容放进 `.gitignore`：

```gitignore
.env
.env.local
/.agents/skills/
/.aris/
/.venv/
results/remote/
paper-overleaf/
artifacts/raw_runs/
*.ckpt
*.bin
*.safetensors
__pycache__/
.pytest_cache/
.ruff_cache/
```

再加入 `.gitattributes`，避免 Windows checkout 把服务器脚本改成 CRLF：

```gitattributes
*.sh text eol=lf
*.ps1 text eol=crlf
```

服务器脚本的 executable bit 在 Phase 7 创建并 `git add` 文件后设置。

数据根目录使用 `$HOME/data/<PROJECT>` 或其他明确位于 `$HOME` 下的路径，不使用仓库内 `/data` 目录冒充外部数据盘。

## 5. Phase 1：配置本地电脑

### 5.1 Windows

安装并验证：

```powershell
git --version
gh --version
ssh -V
codex --version
```

编辑 `%USERPROFILE%\.ssh\config`：

```sshconfig
Host <SSH_ALIAS>
  HostName <SERVER_HOST>
  User <SERVER_USER>
  Port <SERVER_PORT>
  IdentityFile ~/.ssh/<SERVER_KEY>
  IdentitiesOnly yes
  ServerAliveInterval 30
  ServerAliveCountMax 3
```

PowerShell 验证：

```powershell
ssh -o BatchMode=yes <SSH_ALIAS> true
gh auth status
git -C <LOCAL_REPO> remote -v
```

### 5.2 macOS

```bash
xcode-select --install      # 如果 Git 尚不可用
brew install gh             # 若尚未安装
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/config
chmod 600 ~/.ssh/config
```

先把配置卡中的服务器写入 `~/.ssh/config`：

```sshconfig
Host <SSH_ALIAS>
  HostName <HOSTNAME>
  User <DEDICATED_NON_ROOT_USER>
  Port <PORT>
  IdentityFile ~/.ssh/<LOCAL_PRIVATE_KEY>
  IdentitiesOnly yes
```

然后才运行连接和 GitHub 登录测试：

```bash
ssh -o BatchMode=yes <SSH_ALIAS> true
gh auth status
```

SSH config 与 Windows 相同。macOS 使用仓库中的 `.sh` 脚本；Windows 使用 `.ps1` 入口。两台本地电脑都不需要安装训练环境。

### 5.3 本地 Codex

本地 Codex 直接在本地 clone 中打开，职责限于：

- 研究规划和代码编辑。
- 轻量、短时、无 GPU 或低成本测试。
- commit、push、PR 和服务器监督。
- 拉取已校验的小型结果并分析。

本地不需要复制服务器的 Conda/CUDA 环境。`AGENTS.md` 应明确完整实验只能在服务器执行。

## 6. Phase 2：GitHub 是唯一源码事实源

### 6.1 新建仓库

```bash
gh auth login
gh repo create <OWNER>/<REPO> --private --source . --remote origin --push
```

如果仓库已存在：

```bash
gh repo clone <OWNER>/<REPO>
git remote -v
```

不要把 GitHub 账号 token 复制到服务器 shell history。默认拓扑中服务器 Codex 是源码 writer，因此 deploy key 的 write access 是 `REQUIRED`。只有把 `server_writer=false`、明确禁止服务器改源码时，才可以使用只读 deploy key。GitHub 官方说明 deploy key 只授予单个仓库访问权，适合这种服务器自动化场景。

### 6.2 服务器 deploy key

在服务器交互 SSH 中：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/<PROJECT_DEPLOY_KEY> -C '<project>-server' -N ''
chmod 600 ~/.ssh/<PROJECT_DEPLOY_KEY>
```

把 `.pub` 内容添加到 GitHub 仓库：

```text
Repository → Settings → Deploy keys → Add deploy key
```

默认 `server_writer=true`，必须勾选 write access 并通过 push probe。不要把私钥复制回本地，也不要用 agent forwarding 把本地个人 GitHub 身份长期暴露给服务器。

服务器 `~/.ssh/config`：

```sshconfig
Host github-<project-slug>
  HostName github.com
  User git
  IdentityFile ~/.ssh/<PROJECT_DEPLOY_KEY>
  IdentitiesOnly yes
```

先根据 GitHub 当前官方 SSH host-key 页面核对 host fingerprint，再写 `known_hosts`。然后验证：

```bash
mkdir -p ~/work
auth_output="$(ssh -T github-<project-slug> 2>&1 || true)"
printf '%s\n' "$auth_output"
printf '%s\n' "$auth_output" | grep -F 'Hi <EXPECTED_GITHUB_ID>!'
git ls-remote git@github-<project-slug>:<OWNER>/<REPO>.git HEAD
git clone git@github-<project-slug>:<OWNER>/<REPO>.git ~/work/<PROJECT>
git -C ~/work/<PROJECT> config --local user.name <GIT_NAME>
git -C ~/work/<PROJECT> config --local user.email <GIT_EMAIL>
git -C ~/work/<PROJECT> remote -v
```

`ssh -T` 对 GitHub 成功认证通常仍返回状态 1，因此不能只看退出码；必须同时核对输出中的预期账号，并对精确仓库执行 `git ls-remote`。服务器作为 writer 时，再将当前 HEAD 推送到一次性 probe branch、确认远端存在后删除该 branch，保存这次可逆验收的 receipt。

## 7. Phase 3：服务器目录与环境

所有路径先 `realpath`/resolve，并验证在 `$HOME` 下：

```bash
mkdir -p \
  ~/work/<PROJECT> \
  ~/data/<PROJECT>/{datasets,runs,checkpoints,models,logs} \
  ~/aris_repo \
  ~/.codex/mcp-servers/<REVIEW_TRANSPORT>

chmod 700 ~/data/<PROJECT> ~/.codex/mcp-servers/<REVIEW_TRANSPORT>
```

环境选择：

- 先验证 Python 3.13 是否被 CUDA、PyTorch、核心依赖和 lockfile 支持；支持则固定 3.13。
- 不兼容时保存具体证据并固定 3.12。
- 纯 Python 项目可使用一个固定、lockfile 管理的 uv 环境，并显式设置 `UV_PROJECT_ENVIRONMENT` 到仓库外的项目专属路径。
- 需要 CUDA、编译库或 Conda 二进制包时，用 Miniforge/Conda 建基础环境，再显式把 `UV_PROJECT_ENVIRONMENT` 指向该 Conda prefix；仅仅 `conda activate` 后运行普通 `uv sync` 仍可能创建仓库 `.venv`。
- 禁止 Agent 每轮创建或重选临时 `.venv`。

uv-only recipe：

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/<project-slug>/venv"
uv sync --frozen
uv run python -c 'import sys; print(sys.executable); print(sys.prefix)'
test ! -e .venv
```

Conda + uv recipe：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate <ENV_NAME>
env_prefix="$(python -c 'import sys; print(sys.prefix)')"
export UV_PROJECT_ENVIRONMENT="$env_prefix"
uv sync --frozen <SYNC_MODE>
python -c 'import sys; print(sys.executable); print(sys.prefix)'
test "$CONDA_PREFIX" = "$env_prefix"
test ! -e .venv
```

如果 Conda 环境只负责 Python/pip 基础、其余 Python 包全部由 lockfile 管理，可以使用 exact sync。若 Conda 还持有必须保留的包，先根据当前 `uv sync --help` 选择保留额外包的 inexact 模式，并把该模式写进项目契约；不要让一次 exact sync 静默删除 Conda 管理的兼容层。

每个 launcher 和每个 experiment dispatcher 都重新激活环境；不能信任旧 tmux server 继承的 `PATH`。

推荐在 `docs/OPERATIONS.md` 固定：

```text
Environment manager:
Environment identifier:
Python:
Activation command:
Lockfile:
Import smoke:
Test command:
Lint command:
```

Miniforge 项目可以安装 conda shell completion，但只维护一个幂等 marker block，不改写 Conda 自己的 managed block，并关闭 base 自动激活。

## 8. Phase 4：服务器安装 Codex、Claude 与 ARIS

### 8.1 CLI 安装原则

Codex CLI、Claude Code 和 ARIS 的安装命令可能变化。安装当天必须查看官方当前文档，并记录：来源 URL、命令、版本、时间和 SHA。

当前权威入口：

- [OpenAI Codex 文档](https://learn.chatgpt.com/docs/codex)
- [Anthropic Claude Code CLI reference](https://docs.anthropic.com/en/docs/claude-code/cli-usage)
- [ARIS 官方仓库](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)

安装后验证：

```bash
codex --version
claude --version
git -C ~/aris_repo rev-parse HEAD
```

### 8.2 一次性交互登录

在服务器的交互 SSH session 中运行：

```bash
codex login
claude
```

设备码或浏览器授权由用户本人完成。不要把验证码、token、auth 文件内容或 API key 发到聊天、Git、日志或 shell history。

完成后验证 fresh shell：

```bash
bash -lc 'codex login status'
bash -lc 'claude -p "Reply with exactly READY" --output-format json --tools ""'
```

Reviewer 身份验收另外显式传入配置值：

```bash
claude --help | grep -E -- '--model|--effort|--permission-mode|--tools'

review_model=<PINNED_CLAUDE_MODEL>
env \
  CLAUDE_REVIEW_MODEL="$review_model" \
  CLAUDE_CODE_EFFORT_LEVEL=medium \
  claude --model "$review_model" --effort medium \
    -p 'Reply with exactly READY' \
    --output-format json \
    --permission-mode plan \
    --tools ''
```

先用当前安装版本的 `claude --help` 验证 flags；不支持就停止配置并按该版本的官方文档更新，不能默默省略 model 或 effort。若当前版本提供 `--safe-mode`，也把它固定进 probe 和 wrapper；否则用该版本支持的配置禁用 plugin/hook，并把准确命令记入验收证据。

保存 JSON receipt，并解析 `modelUsage.*.canonicalModel`；同一条包含 `--model "$review_model" --effort medium --permission-mode plan --tools ''` 的 probe 还要在 fresh detached tmux 中运行，不能换成弱化版本。

再在 detached tmux 环境中验证一次，避免“交互 shell 能用、夜间 tmux 失效”。登录过期时停止 ARIS，重新执行对应 CLI 的官方交互登录，重新做 fresh-shell/tmux 验收，再恢复；不要在无人值守 launcher 中塞交互式登录。

### 8.3 ARIS 安装

```bash
git clone https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git ~/aris_repo
git -C ~/aris_repo fetch --tags --prune
git -C ~/aris_repo checkout --detach <ARIS_FULL_SHA>
test "$(git -C ~/aris_repo rev-parse HEAD)" = '<ARIS_FULL_SHA>'
```

使用固定 SHA 中的 Codex-native installer。它先安装 `skills-codex`，再用 Claude overlay 覆盖相应 reviewer route；不要使用 Claude-native 安装入口。配置卡中的 list file 是可审计来源，installer 接收的是 CSV，不接收文件路径；launcher 必须先把去注释、去空行、去重后的条目转换成 CSV，并确认 groups 与 extra skills 至少一个非空：

```bash
bash ~/aris_repo/tools/install_aris_codex.sh --help

groups_csv=<APPROVED_GROUPS_CSV_OR_EMPTY>
skills_csv=<APPROVED_EXTRA_SKILLS_CSV_OR_EMPTY>
exclude_csv=<FORBIDDEN_SKILLS_CSV_OR_EMPTY>
test -n "${groups_csv}${skills_csv}" || { echo 'ARIS allowlist is empty' >&2; exit 1; }

aris_args=(
  <SERVER_REPO>
  --aris-repo "$HOME/aris_repo"
  --with-claude-review-overlay
  --dry-run
)
test -z "$groups_csv"  || aris_args+=(--groups "$groups_csv")
test -z "$skills_csv"  || aris_args+=(--skills "$skills_csv")
test -z "$exclude_csv" || aris_args+=(--exclude "$exclude_csv")
bash ~/aris_repo/tools/install_aris_codex.sh "${aris_args[@]}"
```

审查 dry-run 后，只从数组中去掉 `--dry-run` 执行同一组参数。launcher 应比较 list file 派生值与配置卡中的 CSV，发现漂移就失败。若所固定的旧 SHA 还没有该 overlay flag，则按该 SHA 官方说明依次安装 `skills/skills-codex/*`、覆盖 `skills/skills-codex-claude-review/*`，再注册 `claude-review` MCP；无论哪条路径都审计最终 dependency closure 和 reviewer routes。

不要全量复制所有 skill。项目维护 allowlist、forbidden list 和 dependency-closure audit；升级必须换到新的完整 SHA，重新验收后才能接受。

## 9. Phase 5：反转 ARIS 角色

本项目体系不是 ARIS 的传统“Claude 执行、Codex 审查”，而是：

```text
Codex：实现、实验、状态维护、论文起草
Claude Code：独立、只读、cross-model review
```

正式 reviewer transport 只暴露只读 Claude review：

- model 固定为配置卡中的 Claude 模型。
- effort 固定 `medium`。
- 使用 non-interactive `-p` 和 JSON output。
- 禁用工具或使用 plan/read-only 权限。
- 审查前后记录 Git checkout fingerprint。
- 从结构化 `modelUsage.*.canonicalModel` 验证模型身份。
- reviewer 的自然语言自述不是身份或 effort 证据。
- Codex subagent 是同模型内部检查，不满足 cross-model gate。

Reviewer MCP 或其固定 wrapper 必须显式携带：

```text
CLAUDE_REVIEW_MODEL=<PINNED_CLAUDE_MODEL>
CLAUDE_CODE_EFFORT_LEVEL=medium
```

不要把上游 `claude-review/server.py` 直接当成严格 gate：某些版本的 tool schema 允许每次调用传入 `model` 或 `tools`，从而覆盖默认值。项目要在固定 ARIS SHA 之外再放一层受控 adapter/wrapper，并满足：

- 对 Codex 暴露的 tool schema 不含 `model`、`tools`、effort 或 permission override；如果 transport 无法移除字段，就拒绝任何与固定值不完全一致的覆盖。
- 启动 Claude 时硬编码 `--model <PINNED_CLAUDE_MODEL> --effort medium --permission-mode plan --tools ''`；当前 CLI 支持并验证过 `--safe-mode` 时一并硬编码。
- wrapper 使用绝对路径、mode `0700`，配置和 secret 使用 mode `0600`；receipt 记录 wrapper hash、ARIS SHA、最终 argv、checkout fingerprint 和 structured model metadata。
- 任何一项无法证明时，cross-model gate 为 `UNAVAILABLE/BLOCKED`，不得调用宽松 transport 或退化为 Codex 自审。

参数化注册形态：

```bash
codex mcp add <REVIEW_TRANSPORT> \
  --env CLAUDE_REVIEW_MODEL=<PINNED_CLAUDE_MODEL> \
  --env CLAUDE_CODE_EFFORT_LEVEL=medium \
  -- <ABSOLUTE_CLAUDE_REVIEW_WRAPPER> <WRAPPER_ARGS>
```

实际 adapter 以固定 ARIS SHA 中 `mcp-servers/claude-review` 的官方实现和说明为基础，但项目层负责锁死调用面。Fresh-shell 和 detached-tmux probe 必须显式传入两个变量以及同样的 model/effort/permission/tools flags，再从 JSON receipt 解析 canonical model；不能从配置卡或 reviewer 自述推断。

任何模型、effort、只读性或 transport 验证失败，都将 gate disposition 标为 `UNAVAILABLE/BLOCKED`；不能自动退化为 Codex 自审。

## 10. Phase 6：服务器 Codex profile

OpenAI 官方 Codex 文档规定：用户配置位于 `~/.codex/config.toml`，独立 profile 文件位于 `$CODEX_HOME/<profile>.config.toml`，通过 `--profile <profile>` 选择；项目 `.codex/config.toml` 只在受信任项目中加载。MCP 也由同一 Codex host 配置共享。

不要让无人值守运行继承本地或全局所有 skills/MCP。建议由仓库脚本以 mode `0600` 原子生成项目专属 profile，语义如下：

```toml
model = "<PINNED_CODEX_MODEL>"
model_reasoning_effort = "high"

# service tier 不设置，使用 Codex 默认档；不要启用 Fast override。

[apps._default]
enabled = false

[mcp_servers."<REVIEW_TRANSPORT>"]
enabled = true
required = true
# command/args/env 或 url 由已验收的 reviewer installer 写入；env 必须含
# CLAUDE_REVIEW_MODEL 与 CLAUDE_CODE_EFFORT_LEVEL="medium"
```

生成器还应：

- 固定 server environment 的 PATH 和环境标识。
- 只启用批准的 ARIS skills、必要 shared references、确实需要时的官方 notebook skill，以及唯一 reviewer MCP。
- 禁用其他 discovered MCP、apps、plugin skills 和项目不需要的 global skills。
- 不复制 token 或 secret 到生成后的 profile。
- 每次 CLI、ARIS、skill 或 MCP 变化后重新生成并审计实际 prompt input。

OpenAI 官方文档说明 `codex exec` 适合非交互脚本，默认只读；自动化应显式设置所需权限。这里的 unattended bypass 只在专用、非 root 研究账户中使用。它不构成 sandbox；真正边界是该 Unix 账户的全部权限。Launcher 的 `$HOME` 路径校验只能约束受管配置和发现越界请求，不能阻止任意后续 shell 命令访问该账户原本有权访问的其他路径。

启动时使用当前 CLI 支持的明确 bypass flags；先检查：

```bash
codex exec --help
```

当前已验证的一种启动形态是：

```bash
codex exec \
  --profile <PROJECT_PROFILE> \
  --dangerously-bypass-approvals-and-sandbox \
  -C <SERVER_REPO> \
  - < <ARIS_PROMPT>
```

若 CLI 后续改变 flag，按官方当前 `codex exec` 文档更新 launcher 并重跑验收，不保留静默兼容 fallback。

## 11. Phase 7：仓库内的通信与运行文件

建议仓库至少包含：

```text
AGENTS.md
docs/OPERATIONS.md
docs/RESEARCH_PLAN.md
docs/RESEARCH_STATE.md
docs/WRITING_STYLE.md
config/aris-run-prompt.md
config/aris-skills.txt
config/aris-forbidden-skills.txt
config/reviewer-routing.tsv
scripts/start_aris.sh
scripts/server_run.sh
scripts/remote_run.sh
scripts/remote_run.ps1
scripts/fetch_results.sh
scripts/fetch_results.ps1
scripts/sync_overleaf.sh
scripts/server/dispatch_run.sh
scripts/server/run_command.sh
```

这些脚本必须执行门槛，而不是只相信 prompt。

文件创建后先登记进 index，再设置并验证服务器 `.sh` 的 executable bit。使用 Git Bash 或 macOS/Linux；Windows PowerShell 可逐个写出实际文件名：

```bash
git add scripts
git add --chmod=+x scripts/*.sh scripts/server/*.sh
git ls-files --stage scripts
```

所有服务器 `.sh` 应显示 mode `100755`，并由 `.gitattributes` 保证 LF；PowerShell 入口保持 CRLF。提交前应实际检查没有脚本以 CRLF 落盘。

### 11.1 `start_aris.sh`

顺序：

1. resolve `$HOME` 和所有项目路径。
2. 拒绝任何越界路径或 root 用户。
3. 显式激活固定环境并断言 Python。
4. 检查仓库 clean、upstream 和 ARIS pin。
5. 检查 Codex 登录。
6. 以结构化、无工具请求验证 Claude reviewer。
7. 审计 skill/MCP/capability routing。
8. 确保 experiment tmux 存在。
9. 如果 agent tmux 已存在则返回，不启动第二个 Agent。
10. 在带 UTC 和 commit 的日志中启动 Codex profile。

### 11.2 `remote_run.*`

本地投递实验时：

1. 本地工作树必须 clean。
2. `git fetch origin`。
3. 本地 HEAD 必须精确等于 pushed upstream。
4. 通过 SSH 要求服务器 clean，并 `pull --ff-only`。
5. 服务器 HEAD 必须等于本地指定 commit。
6. 临近真正启动时由 dispatcher 再校验一次 expected commit，关闭外层检查与执行之间的竞态。
7. 命令和 repo path 使用 base64/明确位置参数交给 dispatcher；不要插值拼接进远程 shell。
8. 在 experiment tmux 新建独立 window，返回 run ID。

### 11.3 `dispatch_run.sh`

- 为指定 commit 创建 immutable source snapshot。
- 每次重新激活环境。
- 每个 run 使用独立 process group。
- 记录 command status 与 dispatcher/cleanup final status。
- 写 metadata、stdout/stderr、timestamps、GPU/Python、signal 和 checksum。
- `TERM` 后有限等待，再 `KILL`；整个 process group 退出后才写终态。

### 11.4 `fetch_results.*`

- 只接受安全格式的 run ID。
- 下载到本地 `.partial-*`。
- 验证 metadata、exit status 和 SHA-256。
- 验证通过后原子移动到 `results/remote/<run-id>`。
- 不覆盖已有最终目录。
- Windows PowerShell 优先使用文件级 scp/SFTP，避免二进制内容穿过文本 pipeline；Bash 客户端可使用 tar stream。

## 12. Phase 8：日常交接与沟通

### 12.1 本地把工作交给服务器

```bash
git status --short
git add <EXACT_FILES>
git commit -m '<MESSAGE>'
git push
ssh <SSH_ALIAS> 'cd <SERVER_REPO> && test -z "$(git status --porcelain)" && git pull --ff-only'
ssh <SSH_ALIAS> '<SERVER_REPO>/scripts/start_aris.sh'
```

Windows 最后两步使用 PowerShell 也可以；不要把复杂研究命令手写进多层 SSH 引号，使用 `remote_run.ps1`。

### 12.2 本地投递一个长实验

macOS/Linux：

```bash
./scripts/remote_run.sh 'python scripts/<experiment>.py --config <config>'
```

Windows：

```powershell
.\scripts\remote_run.ps1 -Command 'python scripts/<experiment>.py --config <config>'
```

### 12.3 查看状态

```bash
ssh <SSH_ALIAS> 'tmux ls'
ssh <SSH_ALIAS> 'tmux capture-pane -pt <AGENT_TMUX>:0 -S -120'
ssh <SSH_ALIAS> 'tmux capture-pane -pt <EXPERIMENT_TMUX>:<WINDOW> -S -120'
```

交互进入：

```bash
ssh <SSH_ALIAS>
tmux attach -t <AGENT_TMUX>
# Ctrl-b d 退出但不中止
```

### 12.4 拉取结果

```bash
./scripts/fetch_results.sh <RUN_ID>
```

或 Windows：

```powershell
.\scripts\fetch_results.ps1 -RunId <RUN_ID>
```

拉取后的 `results/remote/` 是 ignored 的结果副本，不是源码同步。

### 12.5 服务器把控制权交回本地

服务器 Codex 应：

1. 停止新 dispatch，安全处理当前实验。
2. 更新 `docs/RESEARCH_STATE.md`。
3. 对有效源码改动运行验证。
4. commit + push。
5. 确认 server checkout clean。
6. 在状态文件中留下 HEAD、active gate、run IDs、blockers 和下一步。

本地随后：

```bash
git status --short             # 必须为空
git pull --ff-only
git rev-parse HEAD
```

如果本地 dirty，不自动 stash、reset 或 pull；先人工判断这些改动属于谁。

## 13. 研究状态就是双方的留言板

`docs/RESEARCH_STATE.md` 是持久通信协议，不依赖聊天历史。至少包含：

```yaml
updated_at: <UTC>
writer_lease: <local_OR_server_OR_none>
source_commit: <FULL_SHA>
active_gate: <GATE>
research_state: <PLANNED_OR_RUNNING_OR_FAILED_OR_OBSERVED>
gate_disposition: <READY_OR_BLOCKED_OR_UNAVAILABLE>
active_runs: [<RUN_ID>]
measurement: <OBSERVED_FACTS_ONLY>
supported_interpretation: <BOUNDED_INTERPRETATION>
unresolved_hypotheses: [<ITEMS>]
blockers: [<ITEMS>]
next_safe_action: <ONE_ACTION>
```

`writer_lease` 是协作约定，不是分布式锁。Launcher 仍须检查 Git clean/upstream。切换 writer 时必须 commit + push + ff-only pull。

## 14. Paper 与 Overleaf

- 主仓库 `paper/` 是权威论文源。
- `paper-overleaf/` 是 ignored 的独立 Git clone。
- 只同步 allowlist；先 dry-run，再看完整 diff。
- 主仓库和 Overleaf 仓库分别 commit/push。
- 禁止后台自动双向同步、自动解冲突和 force-push。
- 从 Overleaf 导回修改时先 review diff，再形成主仓库的新 commit。
- Overleaf 尚未创建时标为 `CONDITIONAL N/A`，不阻塞科学实验。

## 15. 停止、监督与恢复

### 15.1 Stop sentinel

每轮 dispatch 前检查：

```text
<STOP_SENTINEL> 是否存在
剩余磁盘是否低于阈值
GPU/API/wall-clock 预算是否耗尽
连续实现失败是否达到阈值
Git/reviewer/login 是否健康
```

触发 stop 后不启动新实验。当前实验按注册策略安全结束或终止，并留下 receipt。

### 15.2 Supervisor

使用一个周期性 supervisor/heartbeat 检查：Agent 是否存活、实验是否前进、磁盘、登录、reviewer、Git clean/upstream 和停止条件。已有 supervisor 时更新，不创建重复任务。

本地电脑关闭后仍需监督，就让 supervisor 运行在服务器侧；不要依赖本地 Codex Desktop 的 session 保持在线。

### 15.3 常见故障

| 故障 | 正确动作 |
|---|---|
| SSH 失败 | 检查网络、alias、端口和 BatchMode；不改源码 |
| server dirty | 停止 pull/dispatch，确认改动所有者 |
| Git divergence | 停止自动化，在单一 writer 端有意识解决 |
| Codex/Claude 登录过期 | 停止 ARIS，官方交互登录，fresh-shell/tmux 复验 |
| reviewer 身份错误 | gate `UNAVAILABLE`，不改用 Codex 自审 |
| ARIS audit 失败 | 回滚最后批准 SHA |
| 实验实现错误 | `FAILED`，保留 receipt；不写科学结论 |
| 有效负结果 | `OBSERVED`，记录边界和解释 |
| 磁盘不足 | 停止新 run；不自动删历史证据 |
| 本地与服务器都 dirty | 两边停止写入，逐文件确认；不自动 stash/reset/merge |

## 16. 首次端到端验收

所有 `REQUIRED` 项必须 `PASS`：

- 本地到服务器 SSH BatchMode。
- 本地 `gh auth status` 和 private repo 访问。
- `server_writer=true` 时，服务器 deploy key clone、fetch 和可逆 push/delete probe 全部通过；只读 key 只允许用于明确的 `server_writer=false` 拓扑。
- 本地、upstream、server commit 完全一致。
- dirty tree 与 divergence 会被拒绝。
- 所有服务器 `.sh` 在 Git 中为 `100755` 且使用 LF；`.ps1` 使用 CRLF。
- 环境在 fresh shell 和旧 tmux 中显式激活到同一路径。
- Python、import、test、lint。
- Codex model、`high`、`service_tier=OMITTED`，且 profile/启动参数均无 Fast/priority override。
- Claude canonical model、medium effort、无工具/只读和 checkout fingerprint。
- Reviewer MCP schema 不暴露可变 model/tools/effort/permission，或 adapter 对覆盖做 fail-closed 拒绝；receipt 中 wrapper hash 与最终 argv 匹配。
- ARIS full SHA、allowlist、forbidden list、dependency closure 和 reviewer routing。
- `tmux:<AGENT>` 与 `tmux:<EXP>` 可独立重启。
- 从实际需要支持的 Bash/PowerShell 客户端分别投递一个无害实验，并在 SSH 断开后完成。
- receipt 含 source commit、command/final exit status、metadata、logs 和 checksum。
- fetch 到 `.partial-*` 后校验并原子落盘。
- stop sentinel、预算、磁盘阈值和唯一 supervisor。
- `paper/`/Overleaf 边界；未启用 Overleaf 时可 `CONDITIONAL N/A`。
- 反向验收：dirty tree、divergence、wrong commit、受管路径越出 `$HOME`、wrong reviewer 和 missing login 必须 fail closed。除非另有容器/VM/OS 隔离，不把这项路径 guard 宣称为 shell 的 home-only sandbox。

验收证据记录 UTC、host、commit、命令、结果和 evidence path。没有 receipt 的“我试过了”不算通过。

## 17. 新项目配置 Agent Prompt

把本手册与《Auto Research 启动契约》一起交给新项目的本地 Codex：

```text
请依据附带的《Auto Research 启动契约》和《本地 Codex 与远程 Auto Research 服务器配置手册》，把当前项目配置成本地编辑监督、GitHub 唯一源码同步、远程服务器常驻 Codex+ARIS 和独立长实验的拓扑。

先读取现有仓库并填写所有项目变量。时效性内容只查官方当前文档：Codex 查 OpenAI 官方文档，Claude Code 查 Anthropic 官方文档，ARIS 查 wanshuiyin 官方仓库，GitHub SSH/deploy key 查 GitHub Docs。固定实际版本和完整 ARIS SHA。

在本轮完成所有安全、非阻塞的本地仓库配置和脚本；不要只给计划。涉及服务器时先做只读 inventory，再在专用非 root 账户 $HOME 下配置。禁止 sudo、系统目录、其他用户目录、系统服务和其他用户进程。凭据登录只给官方交互命令，由用户本人完成；不要要求用户把 token、device code 或密码发进聊天。

Codex 是执行器，reasoning_effort=high，profile 与启动参数均省略 service tier 且不启用 Fast/priority override。Claude Code 只通过批准的 reviewer transport 做独立只读审查，effort=medium。模型身份必须由结构化 transport metadata 验证。

实现并验证：SSH BatchMode；Git 单一 writer 交接；private repo deploy key；固定环境；项目 Codex profile；ARIS pin 和最小 capability allowlist；独立 agent/experiment tmux；immutable dispatch；run receipt；atomic fetch；stop sentinel；预算和 supervisor；paper/Overleaf allowlist bridge。

所有源码只通过 GitHub。切换 writer 前必须 commit+push，接收端只能在 clean tree 上 pull --ff-only。大产物只进入服务器 data root。不要 rsync/scp 源码，不自动 stash/reset/merge，不 force-push。

第一次回复报告：已完成修改、逐项验证证据、仍需用户完成的交互登录、真正的 blocker，以及下一步最小安全动作。只有外部权限、研究方向或安全授权缺失时才停下。
```

## 18. 当前官方依据

- OpenAI Codex 配置、profile、MCP 与非交互运行：[Codex documentation](https://learn.chatgpt.com/docs/codex)、[configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)、[non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)。
- Claude Code 的 `-p`、JSON output、model、tools 和 permission flags：[Anthropic CLI reference](https://docs.anthropic.com/en/docs/claude-code/cli-usage)。
- ARIS 当前安装器、Codex platform 和 review overlays：[ARIS official repository](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)。
- 单仓库服务器凭据：[GitHub deploy keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys)。

外部文档只说明产品当前支持什么；本手册中的 Git 单一 writer、不可变实验、证据状态机和 `$HOME` 安全边界来自已验证的研究运维实践，应由项目脚本继续机械执行。
