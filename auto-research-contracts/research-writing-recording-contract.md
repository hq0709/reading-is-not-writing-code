# 研究写作与 AutoResearch 记录契约

## 1. 文档定位

本契约是跨项目、长期维护的权威规范，负责四件事：

1. 研究判断与选题 taste。
2. 论文和其他人类可读内容的写作风格。
3. AutoResearch 的状态、实验和证据记录规则。
4. 上述规范的持久化、审查和演进。

它不负责安装 CLI、SSH、Git、Conda、ARIS 或服务器；这些属于配置手册。它也不负责某个项目的具体方法、数据、门槛、路径和结论；这些属于项目的 `RESEARCH_PLAN`、`RESEARCH_STATE` 和实验注册表。

本文件是项目中的源头权威副本。`docs/RESEARCH_WRITING_AND_RECORDING.md` 是稳定入口，只链接到本文件；`AGENTS.md`、AutoResearch 启动 prompt、论文写作 prompt 和 reviewer prompt 都通过该入口引用本契约，不各自复制一套规则。

## 2. 实验推进与证据复用

实验推进优先于重复证明已有 artifact：

- Supervisor 同时是推进者。只要当前环境、writer handoff、安全边界和注册 gate 已满足，就直接恢复或启动下一项注册实验；不得停在状态巡检、报告整理或重复验证。
- immutable dispatcher 已生成的 checksums、metadata、run receipt、asset receipt 和 reviewer receipt 直接复用。
- 完整 run、数据 shard、模型和数据集的哈希只在明确的硬门禁终态验收，或出现具体污染证据时，对裁决所需的最小清单核验一次。
- 已有终态校验是后续 gate 的输入；heartbeat、validator 和报告任务只读取 receipt，不重新计算大文件。
- validator-only 改动在原 artifact 上运行，不重跑实验，也不为了重复证明而重算模型、数据集、shard 或完整 run。
- gate `PASS` 后立即登记决策并进入下一项已授权实验；只有新的硬门禁、具体污染证据或协议要求才能重新验证。
- 产生有意义的本地源码、契约、状态或论文改动后，达到安全检查点即及时 commit 并 push，让 GitHub 上的主分支承载当前权威状态。

数据可信链以一次可审计、可复用的 receipt 为终点。上游未发布同算法的外部摘要时，使用已注册的独立来源、文件身份和内容验证形成可信链；不得把不存在的外部 SHA-256 设为永久 blocker。

## 3. 首要原则：报告运行、观察、决策和下一步

所有人类可读报告——包括 plan、tracker、audit、review、research state、findings、状态汇报、commit、PR 和论文——按同一顺序组织：

```text
运行：执行了什么注册协议或验证
观察：直接证据是什么
Gate 决策：PASS、BLOCKED、FAILED、OBSERVED 或其他注册 disposition
下一步：下一 gate 的一个正向问题与对应动作
```

Scope 只在“下一步”中用下一 gate 的正向问题陈述一次。真实失败集中记录在指定 ledger，其他报告只在失败当前仍改变决策时引用其 run ID 和影响。普通报告删除重复的“未证明／不代表／不建立”清单、防御性解释、编辑痕迹和已解决 blocker。

## 4. 写当前对象，不写它排除了什么

对当前方法、实验分支、结果或文章段落，直接说它是什么、做了什么、观察到了什么。

如果一条要求只是“删除 X”“避免 X”或“不要再做 X”，执行后应让 X 从当前表述中消失。不要把删除动作变成新名字、注释或辩护。

错误模式：

- “不含 X 的版本”
- “移除 X 后的版本”
- “与失败的 A 不同，B……”
- “我们没有采用 X，因为……”——当 X 与当前论证无关时
- 在每个后续分支里反复提醒早先某个分支失败
- 为预防想象中的质疑，添加与当前证据无关的免责声明

正确模式：

- 用正面的技术定义命名当前方案。
- 先给当前实验的目的、干预、测量和结果。
- 只保留改变读者判断所必需的比较、范围和不确定性。
- 删除内容时静默完成，不公开编辑指令和修改历史。

“只写做了什么”约束的是叙述焦点，并不授权隐藏事实。真实失败、blocker、限制、安全边界和 provenance 仍须在其指定账本中准确保存，但不能扩散成所有后续内容的身份标签。

## 5. 记录卫生：AutoResearch 不是流水账

### 5.1 每种信息只有一个权威位置

| 信息 | 权威位置 | 其他位置如何处理 |
|---|---|---|
| 当前阶段、已通过 gate、当前 blocker、下一决策 | `docs/RESEARCH_STATE.md` | 只链接，不复制历史叙述 |
| 待验证 claim、反证条件、协议、门槛 | `docs/RESEARCH_PLAN.md` | 不写成已观察结果 |
| 每次运行的命令、commit、环境、状态、artifact | 实验注册表或 run receipt | 后续分支仅在因果相关时引用 run ID |
| 失败命令、异常、诊断和 partial artifact | failure/run ledger | 不传播到无关实验摘要或论文 |
| 已确认的科学观察和解释边界 | evidence ledger / Results 素材 | 可进入论文，但强度不得超过证据 |
| 写作和叙事原则 | 本契约 | 其他 prompt 只引用 |
| commit、PR、review 的修改原因 | 对应 commit/PR/review artifact | 不进入普通研究叙述 |

### 5.2 `RESEARCH_STATE` 是状态页，不是日记

它只保留当前决策所需的信息：

- 当前 stage 和 gate。
- 已锁定的决定与相应 evidence path。
- 当前仍影响推进的 blocker。
- 下一项最小、可裁决实验。
- writer/runner ownership 等必要运行状态。

已经解决且不再影响决策的故障，从状态页移出，留在对应 receipt、issue、review 或历史 commit 中。不要在每次更新时重述“之前尝试过什么、为什么删掉、现在不再包含什么”。

### 5.3 实验分支用正面身份命名

实验 ID、目录、标题和摘要应按研究变量命名，例如：

- `adaptive-threshold`
- `budget-32`
- `state-conditioned-policy`
- `latency-ablation`

不要按缺失项或旧失败命名，例如：

- `no-X-version`
- `A-failed-now-B`
- `fixed-without-feature`
- `revised-final-corrected`

只有当“移除 X”本身就是注册的干预变量时，`without-X` 才具有科学语义；此时它是消融条件，不是防御性说明。

### 5.4 失败只记录一次，按需引用

执行或测量失败要在对应 run ledger 中留下最小完整记录：

```yaml
run_id: <RUN_ID>
source_commit: <IMMUTABLE_COMMIT>
status: FAILED
failure_class: <IMPLEMENTATION_OR_ENVIRONMENT_OR_MEASUREMENT>
command: <EXACT_COMMAND_OR_RECEIPT_LINK>
evidence_path: <LOG_OR_ARTIFACT>
decision_effect: <BLOCKS_OR_RETRY_REGISTERED_OR_NONE>
```

后续内容仅在该失败仍改变当前选择时引用它：例如它阻塞了 gate，或诊断证明某个协议无效。引用一次 run ID 和决策影响即可，不重写失败故事。

运行失败不等于科学负结果。只有协议有效、测量可信且识别条件成立的结果，才可进入 `OBSERVED`。

### 5.5 科学负结果写成知识，不写成身份

有效负结果可以是贡献，但必须回答至少一个问题：

- 它排除了哪个具体机制或朴素解释？
- 它揭示了什么适用边界或 regime？
- 为什么预期机会消失？
- 它如何改变下一项研究决策？

表述应以观察和机制为中心。例如写“收益在高切换成本区间消失”，不要写“我们失败的切换版本没有收益”。

与核心问题无关、无法解释机制、也不改变决策的失败不进入主叙事。

## 6. 证据语言

每个重要命题必须属于以下一类：

| 类别 | 含义 | 允许的语言 |
|---|---|---|
| `PLANNED` | 已注册但未运行 | “将检验”“计划测量” |
| `RUNNING` | 正在产生证据 | 只能报告运行状态，不报告结论 |
| `FAILED` | 实现、环境或测量无效 | 只报告故障和决策影响，不作科研结论 |
| `OBSERVED` | 协议有效且已有结果 | 精确报告测量值和识别范围 |
| `SUPPORTED INTERPRETATION` | 证据支持但不是直接测量 | “结果支持……解释” |
| `UNRESOLVED HYPOTHESIS` | 仍有替代解释 | 明确标为待检验 |

每个 `OBSERVED` 条目分开写：

```yaml
measurement: <DIRECTLY_OBSERVED_FACT>
supported_interpretation: <EVIDENCE_MATCHED_MEANING>
unresolved_hypotheses: <REMAINING_ALTERNATIVES>
excluded_claims: <CLAIMS_THIS_EVIDENCE_CANNOT_SUPPORT>
evidence: <IMMUTABLE_COMMIT_AND_PATH>
```

`excluded_claims` 属于 evidence ledger 的审计字段，不应机械复制进论文或每次进度汇报。人类可读文本只给使当前 claim 成立所必需的范围限定。

## 7. 研究 taste

### 7.1 优先有启发性的故事

研究价值不只等于最高 benchmark。优先寻找能完成以下任务的证据：

- 重新定义问题或指出主流 framing 的盲点。
- 用可信测量证明机会是否真实存在。
- 画出状态、预算或系统条件下的机会边界。
- 解释优势为什么出现、何时消失。
- 排除看似合理但错误的朴素方案。
- 提炼可迁移的设计原则。

故事必须由锁定证据决定，不为保住最初 headline 改指标、改问题或补无关组件。

### 7.2 围绕一个统一机制

优先采用一条可复述的发现链：

```text
主流 framing 的盲点
→ 一个具体矛盾
→ 被忽略的分析单位或决策结构
→ 统一机制或 objective
→ 机制成立与失效的条件
→ 可迁移原则
```

每个组件都应对应明确失败模式并可被消融裁决。不能形成统一解释的 feature 应删掉或降为工程细节。

### 7.3 先做机会研究，再做复杂方法

先用最小、可证伪的实验确认：问题真实、测量可信、oracle/opportunity upper bound 存在、成本后仍有可用空间。只有这些成立，才投入复杂方法和大规模工程。

下一实验按“单位成本减少多少关键不确定性”排序，而不是按“看起来能产生更多日志”排序。

## 8. 人类可读写作风格

### 8.1 语气

- 直接、当前、具体。
- 自信但受证据约束。
- 技术限定精确，不堆叠泛化免责声明。
- 一段只完成一个推理动作。
- 优先采用“现象 → 对照 → 推论”。
- 不道歉、不自我辩护、不模拟 reviewer 攻击。
- 不泄露 prompt、编辑指令、删除历史或 reviewer 对话。

### 8.2 结构

摘要采用紧凑链条：实际问题 → 现有 framing 的限制 → 核心问题 → 新 formulation → 方法 → 主要证据 → 最强成立区间。

引言先给 dominant framing 的盲点和一个具体矛盾，再引出新的分析视角。这个视角必须由当前问题推导，不能照搬其他论文的名词或例子。

方法部分先给 design overview，再展开技术细节。公式后立即解释数学对象的现实语义以及它如何对应研究问题。

相关工作按 taxonomy 和精确定位组织，不做 bibliography dump。

Results 形成递进发现链。消融用于裁决概念替代和机制，而不是堆表。图表分别回答机会、机制、强 baseline、跨条件边界和代表性现象。

Discussion 和 Conclusion 给出新的思考方式与可迁移原则，不只复述数值。

Limitations 集中、具体地说明真正影响解释的适用范围、成本、失败 regime 和不确定性。不要让限制散落成防御性尾注。

### 8.3 修改规则

修改完成后只展示当前版本。以下信息只出现在其专用 artifact：

- 修改原因：commit、PR、changelog、review response。
- 审查轨迹：review artifact。
- 失败运行：run/failure ledger。
- provenance：receipt 或 provenance record。
- 当前 blocker：`RESEARCH_STATE`。

普通摘要、实验标题、状态消息和论文正文不承担这些审计职责。

## 9. AutoResearch 输出模板

### 9.1 当前实验摘要

```yaml
run: <POSITIVE_TECHNICAL_NAME_AND_LOCKED_PROTOCOL>
observation: <DIRECT_EVIDENCE_OR_NONE>
gate_decision: <PASS_OR_BLOCKED_OR_FAILED_OR_OBSERVED>
next_step:
  question: <ONE_POSITIVE_QUESTION_FOR_THE_NEXT_GATE>
  action: <ONE_AUTHORISED_ACTION>
evidence: <RUN_ID_AND_RECEIPT_PATH>
```

不要添加“未包含的组件”“早先被删除的方案”“为了避免什么而这样做”，除非它们本身是当前实验的注册变量。

### 9.2 进度更新

```text
运行：<REGISTERED_RUN_OR_VALIDATION>
观察：<DIRECT_EVIDENCE_OR_NONE>
Gate 决策：<PASS_OR_BLOCKED_OR_FAILED_OR_OBSERVED>
下一步：<ONE_POSITIVE_GATE_QUESTION>；<ONE_AUTHORISED_ACTION>
证据：<COMMIT/RUN/RECEIPT>
```

不需要“我们没有做……”“之前失败过……”或“这是不含……的版本”。如果存在 blocker，直接写 blocker、影响范围和解除条件。

### 9.3 论文素材条目

```yaml
run: <REGISTERED_EXPERIMENT>
observation:
  measurement: <DIRECT_POSITIVE_FINDING>
  evidence_strength: <DIRECT_OR_SUPPORTED_OR_PRELIMINARY>
  mechanism: <SUPPORTED_EXPLANATION>
gate_decision: <PASS_OR_OBSERVED_OR_BLOCKED>
next_step:
  question: <ONE_POSITIVE_QUESTION_THAT_DEFINES_THE_NEXT_SCOPE>
  action: <ONE_AUTHORISED_ACTION>
evidence: <FIGURE_TABLE_RUN_AND_RECEIPT>
```

## 10. 审查门槛

每个关键写作 artifact 依次通过三个 gate：

1. Story review：是否有清晰发现链、统一机制和读者记忆点。
2. Claim–evidence review：结论、标题和摘要是否超出识别范围。
3. Style/record review：是否存在 revision scar、防御性表述、无关失败、否定式命名、流水账和重复 provenance。

Style/record reviewer 应同时检查两类错误：删掉无语义的防御文字，也防止为了简洁而删掉使结论成立所必需的真实限定。

建议自动检查以下模式，但不能只靠关键词裁决：

```text
without X / X-free / revised / corrected / fixed version
we do not claim / merely / unlike our failed attempt
不含X版 / 去掉X后的版本 / 修正版 / 失败版本 / 我们没有做
```

当否定是实验定义、逻辑事实或真实边界时保留；当它只是编辑历史或自我辩护时删除。

## 11. 持久化与版本管理

- 项目内只有一份权威正文：`auto-research-contracts/research-writing-recording-contract.md`。
- `docs/RESEARCH_WRITING_AND_RECORDING.md` 是稳定入口，只链接权威正文。
- `AGENTS.md` 强制引用稳定入口，并保留实验优先、receipt 复用和统一报告结构这两个源头原则的短摘要。
- AutoResearch prompt 只要求先读取并遵守该文件。
- 跨项目稳定偏好可进入个人记忆；项目事实、门槛和结果不得混入个人风格记忆。
- 规范修改走普通 Git diff 和 review。修改后更新权威文件，不在正文中解释旧规则。
- 若项目需要例外，在项目 `RESEARCH_PLAN` 中精确声明适用范围和原因；不要改写通用契约来容纳单项目特例。

推荐的 `AGENTS.md` 条目：

```text
Read and follow docs/RESEARCH_WRITING_AND_RECORDING.md for research judgment,
human-facing prose, AutoResearch state updates, experiment summaries, and
evidence records. Reuse immutable receipts and terminal verification, advance
passed gates immediately, and report run, observation, gate decision, and one
positive next-gate question. Keep failures and provenance in their ledgers.
```

## 12. 最小验收

新项目接入本契约后，验证：

- 启动 prompt 和 ARIS prompt 引用权威文件，而不是内嵌复制。
- `RESEARCH_STATE` 只含当前状态和决策所需内容。
- failure/run ledger 能保留必要审计信息。
- 一个失败运行不会自动污染后续实验名称和摘要。
- 一个真实科学负结果能以观察、边界或机制知识呈现。
- 删除一个组件后，普通输出不会出现“无该组件版”或删除理由。
- reviewer 同时捕获过度辩护和事实性限定被误删。
- 论文、实验记录、commit/PR 和 provenance 各自承担清晰、不同的职责。
- heartbeat 只读取已有终态 receipt，不重复哈希完整 run、shard、模型或数据集。
- heartbeat 在环境与 gate 满足时恢复或启动下一项注册实验，不以巡检或报告替代推进。
- validator-only 改动复用原 artifact；gate `PASS` 后下一项已授权实验可立即 dispatch。
- 有意义的本地改动在安全检查点及时 commit、push，并最终进入主分支。
- 所有报告都按“运行—观察—gate 决策—下一步”组织，并只陈述一个正向下一 gate 问题。

本契约验收通过后，启动契约只需引用它，无需再次展开写作画像和记录规则。
