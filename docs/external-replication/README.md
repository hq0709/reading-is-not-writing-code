# Concept Flow：跨模型、规模与领域的实验代跑任务包

协议：`cf-transfer-v1`。状态：`PLANNED`。日期：2026-09-06。

## 1. 转发给执行方的任务说明

我们研究的是一个可检验的现象：模型内部能读出一个概念、沿这个概念方向干预能改变回答，是否足以说明该方向具有概念特异性？需要用统一实验在多模型、多规模和多领域中检验，而不是训练一个新方法或刷问答准确率榜。

完整目标是 **22 个模型检查点 × 3 个数据集 × 每个数据集 6 个概念**，形成三张主实验大表。每个检查点独立拟合相同容量的线性探针，比较概念方向、其他概念方向、119 个随机方向和 sham，并保留逐样本输出。不同人可以各负责一个模型家族或若干大模型；请按本包的数据、测量与返回格式执行，机器路径、容器、调度、批大小和并行方式由执行方安排。

接任务时回复：能承担的 `model_key`、GPU 型号/数量/显存、内存/磁盘、预计可用时段、数据与模型访问权限。先交一次接口验证及吞吐估算；共同确认锁定配置后批量运行。完成整块再交付，不需要周期性汇报每个 batch。

**本包是完整实验规格和交付契约，不是一键适配好所有模型的 runner。** 现有代码可复用，但新模型家族、分片加载及另外两个数据集还需执行方或项目方完成适配。验收这些适配后才能采集正式 test 结果。这里没有新实验结果。

文件入口：

- [models.csv](models.csv)：全部模型的准确 ID、分批顺序、规划硬件及官方模型卡。
- [protocol.json](protocol.json)：机器可读的固定配置和工作量。
- [cohorts.csv](cohorts.csv)：NIH 的固定 16/400/600 人接口验证、校准、测试名单。
- [return-format.md](return-format.md)：返回目录、字段、数组和完整性规则。
- [main-tables.csv](main-tables.csv)：三张主表的单元格级结果模板，全部 `TBD`。
- 代码参考基线：[`763072d`](https://github.com/wy-coliney/concept-flow/tree/763072d39dfedb34d42432415408de4e6f547f9b)。私有仓库访问权限由项目方提供；数据许可由各执行方自行取得。

## 2. 三张主实验大表

### 表 1：跨模型与规模的概念可读性—特异性矩阵

22 行检查点，按同一家族从小到大分组。NIH 六个概念各有两个主列：校准集 controlled selectivity `S`，测试集匹配方向相对其他概念方向的优势 `O`。再列参数规模、回答能力合格数和有确证竞争者胜出的概念数。单元格提供估计和区间/状态，正文表约 16–18 列，完整 6×6 方向表放附录。

这张表回答：现象是否只出现在一个小模型或一个概念？同代增大模型后，哪些概念发生变化？参数名义大小之外另记录 vision/LLM/total 参数量；同系列权重的训练数据和视觉塔可能也变化，因此规模趋势是检查点关联，不直接等于参数量因果效应。

### 表 2：跨数据集、跨领域的现象覆盖

仍为 22 行；NIH、CheXpert、COCO 各有四个列组：可读概念数 `n_read/6`、可回答概念数 `n_answer/6`、满足参考 steering 条件数 `n_steer/6`、已确证其他概念方向更强的数量 `n_competitor/n_read`。完整数值和分母一并给出；零分母写 `NA`。末尾按模型家族汇总，保留每一行。

NIH 与 CheXpert 检验胸片数据来源迁移；COCO 检验通用物体概念。每个数据集都重拟合自己的概念探针，因此这是**同一研究协议的跨数据集复现**，不是把 NIH 的方向零样本搬到 COCO，也不是跨疾病标签的严格一一映射。

### 表 3：结论对测量选择的稳定性

22 行，NIH/CheXpert/COCO 三个面板；各自显示：措辞变化的配对优势差、A/B 映射变化、负/正剂量曲线、3 次训练样本重拟合的方向/结论一致性、视觉连接器位置的结果、阳性减阴性位移差。分别展示 Effusion/Mass、Effusion/Edema 和 person/bottle 的 prompt 结果；完整六概念剂量与位置结果可在附录展开。

这张表回答：观察是稳定的概念竞争现象，还是依赖一个 prompt、一次探针拟合或一个 hook？显著/不显著翻转不等同于两个条件之间有显著差异；配对差区间必须另外计算。后续排版允许宽表或分面，不以不可读的小字塞满一页。

三张表的指标互补，不重复同一个均值。完整目标要求全部计划行；分批结果只能标阶段性覆盖。无法执行、能力不合格、测量无效及统计未决分别保留，不能删掉不利模型。

## 3. 模型与资源

| 系列 | 必须覆盖的名义规模 | 比较意义 |
| --- | --- | --- |
| Qwen2.5-VL-Instruct | 3B、7B、32B、72B | 同代规模；7B 与已有证据衔接 |
| Qwen3-VL-Instruct | 4B、8B、32B | 新一代、不同视觉融合方式 |
| InternVL3.5 HF | 8B、14B、38B | 另一主流视觉架构系列 |
| Gemma 3 IT | 4B、12B、27B | 非 Qwen 系列规模对照 |
| MedGemma IT 多模态 | 4B、27B | 与 Gemma 的医学/通用配对 |
| Llama 3.2 Vision Instruct | 11B、90B | 大模型与 cross-attention 架构 |
| LLaVA 1.5 HF | 7B、13B | 与既有 LLaVA 结果衔接 |
| Lingshu | 7B、32B | Qwen 系医学专用模型 |
| LLaVA-Med v1.5 Mistral | 7B | 既有医学模型的正式接口复现 |

总计 22 个检查点。`phase1/phase2` 仅是调度顺序，两批都属于完整证据目标。MedGemma 必须选多模态版本；LLaVA-Med 与 Vicuna 基础的 LLaVA-1.5 不是仅改变医学训练的一对，不把差异全部归因于医学专用化。各模型卡在 `models.csv`；Qwen3-VL 这里选 Instruct 而不是 Thinking 或量化版。闭源 API 无法提供内部激活和受控 hook，不属于这套主实验的替代品。

### 硬件规划值

以下是 BF16、单张图、短问题、直接读下一 token logits 的**保守规划建议，不是实测最低显存或工时保证**。B 是模型名义档位，实际总参数另记。权重约为 `2 × total_parameters` 字节，还需视觉输入、activation、KV/临时计算与分片开销。

| 档位 | 建议设备 | 主机与存储建议 |
| --- | --- | --- |
| 3–8B | 1×48GB；1×80GB 更宽裕 | RAM 64–128GB；每模型预留 100GB |
| 11–14B | 1×80GB | RAM 128GB；每模型 150GB |
| 27–38B | 2×80GB 同节点，优先高速互联 | RAM 256GB；每模型 200GB |
| 72B | 4×80GB 或 2×141GB | RAM 512GB；每模型 350GB |
| 90B | 4×80GB，先测视觉峰值和分片 | RAM 512GB；每模型 450GB |

这些预算之外另留数据盘，整套建议 1–2TB 可用空间，依实际下载版本确认。不需 GPU 训练整个 VLM；线性探针主要是 CPU 工作。不要因显存不足静默改成 4bit/8bit，量化是单独的敏感性条件。32B/38B/72B/90B 和医学 27B/32B 优先交给大显存节点；小模型可由其他节点并行承担。

先在固定 preflight 图像上计时包含处理器、视觉前向、干预、logits 和写盘的 256 个 image-condition，分别覆盖 baseline、概念与 random。报总吞吐 `r`（已含所有 GPU）、峰值显存及所用 GPU 数 `g`。预计 wall-hours=`N/r/3600`，GPU-hours=`g×wall-hours`，另外加加载/抽特征和约 25% 调度余量。不能把旧 7B 的吞吐直接代入 90B。

完整计划含约 **93,605,600 个 image-condition 输出**，其中主 CORE 约 **30,175,200**，其余是 prompt、剂量、重拟合、第二位置和校准；这不是独立样本数。若总吞吐 10/100/500 条每秒，纯条件评分约为 2,600/260/52 小时（未计抽特征等开销；这只是算术示例）。多机并行可缩短历时，不改变实际 GPU-hours。预算按整块预估后确认。

## 4. 数据和固定样本

### NIH ChestX-ray14

使用仓库 `data/manifest.csv`，保持已有 patient split：18,212 train 图/5,783 人；val 2,674 图/855 人；test 5,343 图/1,659 人。图像由执行方合法获取；用 `row_id + '.png'` 解析本地路径。路径可变，row_id 和患者归属保持不变。

本包 `cohorts.csv` 已给出 16 人 preflight、400 人 calibration、600 人 test，每人一图且三者与 train 患者互斥。名单先按 `SHA256('cf-transfer-v1-row:'+row_id)` 为每患者选一图，再按 `SHA256('cf-transfer-v1-patient:'+patient_id)` 排患者；val 前16人为 preflight、接400人为 calibration，test 前600人为正式测试。哈希用于固定抽样顺序，不是反复校验大文件。全部 train 行用于拟合。剂量实验用文件中 test 的前200行。

六个概念顺序为 Effusion、Atelectasis、Pneumothorax、Cardiomegaly、Mass、Nodule。校准集阳性数依次为 20/33/13/15/15/27；测试集为 46/57/17/23/32/47。Pneumothorax 等少阳性区间可能较宽，照实报告。本队列来自已有项目 manifest，可能与历史实验患者重叠，定位是共享样本上的跨模型复现，不声称全新患者独立确认。

### CheXpert：另一胸片来源

使用正式训练发布包的标签 CSV 和 frontal 图像；保留官方验证集不用于拟合，不冒称这里的内部 held-out 是官方 leaderboard test。数据申请和许可见 [官方入口](https://stanfordmlgroup.github.io/competitions/chexpert/)。可以选同一个明确发布版本，但所有模型共享该版本与原始图像；版本和分辨率在 manifest 中锁定。

由一个数据负责人统一构建：从 patient ID 提取患者；每患者按 `SHA256('cf-transfer-v1-chexpert-row:'+原始相对路径)` 选一张 frontal 图，再按 `SHA256('cf-transfer-v1-chexpert-patient:'+patient_id)` 排序。前20,000人 train，接16人 preflight，接400人 calibration，接600人 test。标签不参与排序。名单及标签掩码先交回项目方冻结，其他人复用同一份。

概念依次为 Pleural Effusion→Effusion、Atelectasis、Pneumothorax、Cardiomegaly、Consolidation、Edema。对每概念，1为阳性、0为阴性，-1和空白是未知，保留原值和 `label_known`，仅在该概念的 probe 拟合与标签指标中排除未知。回答位移依然在完整600图上估计。六个概念可能有不同的已知标签分母。随机标签 control 对该概念使用同一已知标签子集，与真实 probe 的样本容量匹配。

### COCO 2017：通用视觉概念

使用 [COCO 官方 train2017、val2017 和 instances annotations](https://cocodataset.org/#download)，不使用模型生成标签。概念依次 person、dog、car、chair、bottle、bicycle；对应 COCO category ID 1、18、3、62、44、2。某图存在该类别任一实例（包括 crowd）为1，否则为0。

train2017 按 `SHA256('cf-transfer-v1-coco:'+十二位image_id)` 排序：前20,000图 train，接16图 preflight，接400图 calibration；val2017 同样排序取前600图 test。这里统计独立单元是 image_id，没有患者。保留官方 split 并检查精确图像 ID 互斥；这些公开数据可能出现在模型预训练中，报告为公开 benchmark 复现而不是训练污染已排除的证据。

先构建一次 manifest，记录 image_id、relative_path、role、width、height、六标签，提交冻结后各人复用。六个概念的 train/calibration/test 阳阴计数必须交回；若标签不足以拟合或估计，标记具体状态而不基于模型结果补抽图片。

### 共同原则

每个数据集的名单全模型共享，抽样在模型输出前固定；模型不得各自挑图。NIH 与 CheXpert 对患者独立采样；COCO 按图片采样。医疗数据、患者标识及原始图像按许可通过授权通道交付，不放进公开 PR；统计 join 使用数据集内稳定 ID。

## 5. 固定实验口径

### 5.1 位置、输入和接口

主位置是**实际被连接器消费的最后一处视觉特征块输出**，并非按模块名字猜最后一层。LLaVA 的 selected feature layer 可能为倒数第二层；Qwen3 的 DeepStack 存在多路注入，必须在配置中明确本次只改哪一路、哪些旁路保持原样。Llama cross-attention 的视觉状态不是简单拼接进文本的 token，应按真实消费位置适配。

第二位置为实际流入语言模型的 connector/image-embedding 输出。没有同构单一位置的模型，明确记录已选择的消费边界及其与主位置的关系；多路干预不能混标为单路。每个位置独立拟合探针。位置结果按语义边界分组，不能把不同架构的层号当成可直接比较的深度。

使用官方 processor/chat template，固定其 revision、resize/crop、max tiles/max pixels、特殊 token 和 attention 实现。同家族规模实验共享一套可支持的图像策略；Qwen2.5 的桥接配置保留已有336×336设置，其他家族用选定官方图像策略。不要强行将所有模型的像素数设成同一个值而破坏原生处理器。报告实际 token 数及有效分辨率，原生分辨率差异属于检查点比较的条件。

`eval()`、关闭 dropout、推理禁用梯度，主条件 BF16；投影和logits保存FP32、统计FP64。只读取固定回答起点的下一 token 分数，不做自由生成、采样或思维链。完整聊天前缀和回答起点必须存档。

每个新家族/位置做一次接口验证：固定16张preflight中的clean与alpha0一致性、改动确实到达connector及answer logits、只有预定视觉token改变、批处理/分片与单例结果在预先声明容差内一致；另外用简单存在/不存在陈述验证6种模板的语义方向。失败表示测量未就绪，不是模型科学负结果。允许优化batch/sharding/cache，只要保持相同数学操作和可验证等价性。

### 5.2 线性探针和方向

从选定位置的有效、被消费图像token平均池化得到 `x∈R^D`，padding/文本/未消费CLS排除；每张图的多tile按实际有效token平均。`P=PCG64(seed=0).standard_normal(D,512)/sqrt(512)`，存FP32。全部方向共享同一投影和 train-only StandardScaler；分类器为 `LogisticRegression(C=1,max_iter=2000,solver=lbfgs,class_weight=None,random_state=0)`。不同数据集分别拟合。

每概念真实探针之外拟合20个固定type→随机标签control（seeds0–19）、view/sex等可用 nuisance probes。胸片type为view×sex×age-decade，遵循 `src/probe.py:build_types/control_labels`；缺失元数据为`na`。COCO type为宽高比桶（<0.8、0.8–1.25、>1.25）×原始像素面积桶（<65536、65536–262143、262144–1048575、≥1048576），至少8种；若实际少于8种则controlled-selectivity记不合格，原始AUROC仍交付。两领域control任务不同，selectivity绝对值不能当统一能力标尺。

完整保存 P、训练均值μ、scale s、每个真实/control分类器的系数w与截距b、control assignment和逐图分数。主干预单位方向为：

`v_c = normalize(P @ (w_c / max(s,1e-8)))`。

不要直接使用512维系数，也不要遗漏标准化逆映射。每个有效token干预为 `h'_t=h_t+alpha*||h_t||₂*v_c`，范数来自当次该token原始activation；不是平均池化向量的范数。

随机对照：用 PCG64 seed0生成119×D标准高斯数组，逐行单位化，在同模型/数据集/位置的全部问题、prompt和剂量共享。生成后按概念顺序对每个v_c做一次coordinate permutation得到该问题sham，保存实际向量和permutation。主比较固定`alpha=+0.25`，不从test中挑最佳剂量。

REFIT seed1/2改变的是训练独立单元的bootstrap重抽样（胸片患者、COCO图片），保留抽中患者全部训练行及重复次数；重新拟合scaler与六个probe，固定P及其seed0。不能仅改确定性logistic solver的random_state冒充独立方向重拟合。三次结果分开报告。

### 5.3 回答与模板

模板精确文本在 `protocol.json`，I=“Is there”，W=“Does ... show”，Y=yes/no，A/B=交换正负含义的字母回答。胸片 `{image_phrase}` 为“this chest radiograph”，COCO为“this photograph”。句法与finding短语固定，不由执行方润色。

收集 yes/no 的 plain、首字母大写及leading-space单token候选，A/B的plain及leading-space候选；去重ID，正负集合非空且互斥，返回每个候选的token ID和logit。主分数 `m=max(logit_present)-max(logit_absent)`，`p=sigmoid(m)`；另外保留log-sum-exp聚合、完整候选总softmax质量、原始A−B差。p是两类候选归一化分数，不是模型全词表校准概率。无法在相同起点得到有效候选集合时，标接口待适配，不静默换成生成文本准确率。

### 5.4 工作块及完整输出数

下表每个数量以一个模型、一个数据集计算。各块固定，允许重用同一baseline而不重复推理。

| 块 | 数据集与配置 | 逐图输出数 |
| --- | --- | ---: |
| CORE | 三数据集各6概念，IY，alpha+.25；baseline+6概念+119random+问题sham=127，600图 | 每数据集457,200 |
| CALIBRATION | NIH/COCO：6概念IY，加两个指定概念另5模板；400图clean | 每数据集6,400 |
| CALIBRATION | CheXpert：6概念IY；400图clean | 2,400 |
| PROMPT | NIH Effusion/Mass；CheXpert Effusion/Edema；COCO person/bottle；IY之外5模板，各完整127配置，600图 | 每数据集762,000 |
| DOSE | 三数据集全部6概念，前200test图；-.5/-.25/-.1/+.1/+.5；6概念+前20random+sham | 每数据集162,000 |
| REFIT | 三数据集全部6概念，seed1/2各6概念方向+问题sham，+.25，600图 | 每数据集50,400 |
| LOCUS | 三数据集第二位置完整CORE，另400图×6概念clean校准 | 每数据集459,600 |
| ALTDIR | 三数据集全部6概念，IY，+.25，600图；主位置seed0的四个替代方向族各6概念：dom（类均值差）、pattern（Haufe Σw）、orth（logistic normal去除其余五概念张成空间分量）、resid（对其余五标签残差化特征后重拟合的logistic normal），同一投影/缩放/映射；baseline复用CORE；后加模块，仅显式入队 | 每数据集86,400 |
| EXTCOMP | NIH/CheXpert全部6概念，IY，+.25，600图；扩展竞争方向族：数据集其余标签（训练行已知阳性、阴性各≥100：NIH Consolidation/Edema/Infiltration；CheXpert Enlarged Cardiomediastinum/Fracture/Lung Lesion/Lung Opacity/Pneumonia/Support Devices）按同一投影/缩放/C/seed0拟合的logistic方向；baseline复用CORE；后加模块，仅显式入队 | NIH 10,800；CheXpert 21,600 |
| TOKENW | 三数据集全部6概念，IY，+.25，600图；六个logistic方向按逐token权重写入：tokenw（token探针分数的softmax×T，均值1）与topq（分数前25%的token权重4，其余0，均值1），总剂量同CORE；baseline复用CORE；后加模块，仅显式入队 | 每数据集43,200 |
| PRECISION | NIH/COCO全部6概念，IY，+.25，前200test图；完整CORE网格（127配置，自带baseline）在两种数值设置下各跑一遍：fp32（权重与前向float32，批组成同CORE）、batch1（bf16，每个条件单独batch=1）；outcomes多一列numerics；后加模块，按设置各一个任务显式入队 | 每数据集304,800（每设置152,400） |
| ANSDIR | 三数据集全部6概念，IY，+.25，600图；回答方向oracle：对前3,000训练行做clean前向，取主模板回答margin，在同一投影/缩放的512维特征上按问题做ridge回归（5折CV选alpha），按logistic同一规则映射为a_q；每问题写入6个a_d加a_q的坐标置换sham，baseline与random族复用CORE；prep为GPU队列任务，后加模块，仅显式入队 | 每数据集25,200 |

训练/测试pooled features不包含在以上计数。正向+.25全随机主比较和其他剂量20random敏感性是不同证据规格，不能拼起来声称整个dose sweep均做过119随机校正。

## 6. 分析与结论规则

执行方最重要的是完整返回原始分数和拟合产物；项目方据同一份数据做最终CPU统计，减少多人各算一套。

1. 校准集：每概念报告真实probe AUROC、control均值/范围、selectivity S、回答AUROC/Brier、阳阴计数。至少10阳性和10阴性、2,000次按独立单元bootstrap至少1,900次有效；S的one-sided95%下界>0标`readable`，回答AUROC同类下界>0.5标`answer-capable`。两者分开，全部模型仍进入CORE，不按校准成功与否删除实验。CheXpert的标签指标只用known rows。
2. CORE：`W_qd=mean_i[p_i(q,d)-p_i(q,baseline)]`；`O_q=W_qq-max_(d≠q)W_qd`。返回完整6×6、每个random/sham效应、匹配方向在固定随机族中的rank和逐样本向量。steering-reference条件为匹配方向W>0、超过同剂量119random的经验p95和abs(sham)；这是固定参照规则，不自动解释为5%假设检验。
3. 对`C_qd=W_qq-W_qd`做5,000次独立单元bootstrap；同数据集所有模型共享抽样索引。每个对比固定SD为其bootstrap SD，中心化Z，取每draw全部对比`max(abs(Z))`的95%线性分位数形成双侧同时区间。22×3×6×5=1,980个计划对比；不同数据集独立抽样后在同一draw编号组合最大值。取得完整数据后统一max-T；未取得全部模型时只交点估计与描述性区间，缺失项列为未完成，不能临时缩成有利模型族。5000次bootstrap也不足以直接用经验分位估计0.05/1980这样的极小尾概率。
4. 若某q的任一`C_qd`同时区间上界<0，可判有更强竞争者；若全部五个下界>0，可判固定临床/物体族正优势；其余为未决。O的直接percentile区间另在每draw重新求max，不与max-T混写。范围或失败率的分母固定为所有可测单元，能力条件下的比例另外给出。
5. 逐概念返回阳性/阴性平均margin位移、两组差、correct-label margin位移、AUROC变化、Brier变化及pointwise CI。35/36等点估计计数不能当显著性计数。CheXpert未知标签单独记数。
6. PROMPT报告配对O差、映射交换的semantic/raw分量；DOSE报告signed effect曲线和对应参照；REFIT报告方向余弦与O分布；LOCUS报告位置配对差。敏感性报告完整预定网格和pointwise区间；若要对其中的特定差做正式显著性宣称，在该块固定对比集合内统一多重校正，不能事后挑一个最低p值。
7. 跨模型先给每个检查点和家族结果，不把22个模型视为独立随机抽样的“所有VLM”。患者bootstrap反映样本不确定性，不是模型家族抽样误差。NIH/COCO的6概念也不是领域内随机抽取的所有概念。

计算细节：CORE bootstrap用PCG64(seed=2026090601)，以nih、chexpert、coco顺序各生成5000×600的有放回索引，按冻结test顺序取样。固定SD采用ddof=1，SD≤1e-12的Z分量置零，所有分量退化时critical=0，区间仍使用实际SD。这是bootstrap近似同时区间，不是有限样本精确检验。校准seed=2026090602、诊断seed=2026090603，按相同数据集顺序共享索引；标签缺失/单类别draw排除仅适用于对应AUROC，报告有效draw数。

`main-tables.csv`的T3聚合定义：先在各概念内计算signed dose O（负剂量对所有方向乘sign(alpha)），取六个剂量-.5/-.25/-.1/+.1/+.25/+.5上的极差，再对六概念取中位数；+.25用CORE前200图，与DOSE共享样本。REFIT取seed0/1/2的O样本SD，再对六概念取中位数。connector_median_O为第二位置六个O的中位数。median_label_gap为主位置匹配方向的阳性减阴性margin位移差的六概念中位数。聚合区间在每次配对bootstrap内重算全部聚合，不把六个概念当独立随机样本。prompt两个指标分别为O(IY)−O(WY)、O(IA)−O(IB)。

最终可支持的说法由数据决定：多家族/规模/领域一致时，报告这些检查点和概念覆盖中的复现范围；若只在特定条件成立，就明确条件。随机与临床族优势、因果语义身份和诊断收益是不同层次。原始随机初始化floor/utilisation不在这轮主endpoint中，不能将这套实验转述成已完成所有模型的随机初始化因果研究。

## 7. 如何分工执行

建议先做家族接口，再分给不同机器的同家族规模；各机器完整跑一个`model×dataset×module`块，保留同一模型revision。大模型优先级：Qwen2.5-72B、Llama90B、InternVL38B、Qwen3-32B、Lingshu32B、MedGemma27B；通用配对行同时安排，以便比较。

1. 数据负责人统一生成/冻结CheXpert和COCO名单，确认NIH名单可访问；执行方取得合法图像与模型。名义模型ID已核验，具体model/processor revision解析为完整commit后写进锁定配置。
2. 家族负责人实现模型加载、feature pooling、intervention和logit scorer四个适配接口；可参考本仓库`src/extract.py`、`src/first_gate.py:capacity_direction`、`src/intervene.py:Steerer`与`src/qwen_answer_encoding.py`。原`run_*gate.sh`绑定我们服务器路径/receipts，不能直接当可移植命令。`probe.py --bootstrap`目前不是完整的注册分析入口，正式拟合必须保存本包要求的产物。
3. 提交16图接口验证、小样本输出文件和吞吐/资源估算，项目方一次确认协议ID、cohort版本、revision、locus与返回字段。这里锁的是科学接口，不是机器上的每条命令。
4. 拟合探针并完成校准，再依计划完成CORE、PROMPT、DOSE、REFIT和LOCUS。模型/某概念能力差仍保留结果；OOM/NaN/接口不成立标技术状态，在修复同一条件后续跑。更换checkpoint、位宽、样本或prompt需新配置ID。
5. 整块交回源码commit或git bundle、环境锁定、元数据、逐样本表、拟合产物与coverage。CPU汇总可由项目方完成；原GPU产物保留到验收。模型权重和原始医学图像不需回传。

既有服务器仍采用项目ARIS流程；外部执行方无需安装Codex/ARIS或复刻我们的账号、路径、tmux。项目方在接收时统一做独立证据核验。下载链接、存储路径、调度系统和批大小由执行方决定。

## 8. 验收及当前就绪程度

每个run包含：完整且唯一的网格行；与共享manifest一致的样本ID；完整baseline和指定对照；可重建的normal/probe；候选logits与临床方向正确；实际资源与结束状态；所有偏差和缺失块。保留未决/负结果。通过这些要求后，项目方能直接在CPU重算三张表，不需请求重跑VLM。

当前已提供：模型清单、固定协议、NIH名单、三表结果模板与完整返回契约。正式开跑前还需要：各家族适配验证，CheXpert/COCO共享manifest，模型revision与资源分配。它们属于明确的执行准备任务，而不是默认已经完成的实验。涉及新家族的token候选、DeepStack/cross-attention位置或数据许可不能通过猜测填空。

来源核验日期为2026-09-06；官方模型卡链接逐项在models.csv。硬件与时间是本包规划估算，数据计数除已生成NIH名单外均为预定规模，结果模板保持TBD。
