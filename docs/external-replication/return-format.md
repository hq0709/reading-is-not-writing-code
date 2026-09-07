# 代跑结果返回格式

版本 `cf-transfer-v1`。目标是收到后能直接用CPU重新拟合检查、重算统计和填主表。Parquet是推荐格式；无Parquet依赖时允许同字段UTF-8 CSV.gz，数组字段用JSON字符串。不要只交PDF、截图、聚合均值或控制台日志。

## 目录

```text
<model_key>/<dataset_id>/
  run.json
  environment.txt
  source.bundle                 # 或可访问的完整源码commit
  manifests/cohort.csv
  manifests/labels.csv
  prompts.json
  features/<locus>.npz
  fits/<locus>/seed0.npz
  fits/<locus>/seed1.npz          # REFIT适用时
  fits/<locus>/seed2.npz
  probe_scores.parquet
  outcomes/CORE.parquet
  outcomes/CALIBRATION.parquet
  outcomes/PROMPT.parquet
  outcomes/DOSE.parquet
  outcomes/REFIT.parquet
  outcomes/LOCUS.parquet
  preflight.json
  coverage.csv
  summary.json                  # 可以由项目方接收后生成
```

不适用的模块在coverage中写`NOT_REQUESTED`而非伪造空成功文件；计划内但未执行写`NOT_STARTED`，正在执行为`RUNNING`，完整为`COMPLETE`，失败为`FAILED`并解释。结果状态另用`SUPPORTED/UNRESOLVED/INELIGIBLE`，与执行状态分开。

## run.json

必须含：`protocol_id,run_id,model_key,model_id,model_revision,processor_revision,dataset_id,dataset_release,source_commit,adapter_path,python_version,torch_version,transformers_version,numpy_version,sklearn_version,weight_dtype,activation_dtype,logit_dtype,attention_backend,device_map,gpu_models,gpu_count,batch_size,started_utc,ended_utc,wall_seconds,gpu_hours,peak_gpu_memory_bytes,processing_settings,loci,fit_seeds,random_seed,projection_seed,cohort_file,completed_modules,status,deviations`。

- revision是实际加载的完整commit，不是`main/latest`；转换模型附源revision、映射代码和strict-load/equivalence验证。多块复用相同run元数据可用run_id关联。
- `loci`每项含逻辑名、实际module_path、输出tensor选择、hidden_dim、token mask/pooling、影响的视觉分支、旁路、消费者、是否prompt-independent。
- `processing_settings`保存resize/crop/tile/pixels、chat template原文、模型配置、tokenizer特殊ID、执行precision、完整依赖锁定文件位置。
- `gpu_hours`为真实占用GPU数×wall时间，跨节点则求和。CPU拟合时间另列。不得把计划估计填为实测耗时。
- `deviations`空列表表示按协议执行，变化填写实际内容和受影响块。模型性能低不是技术偏差。
- 不包含token、密码、服务器私钥、完整私密环境变量或未授权患者文件。

## cohort.csv 与 labels.csv

`cohort.csv`：`dataset_id,row_id,unit_id,role,order,relative_image_path,original_split`。NIH/CheXpert unit_id为患者标识，COCO为image_id。role为train/preflight/calibration/test；order对应冻结顺序。训练中的重复患者允许多图；正式测试每独立单元一图。

`labels.csv`长表：`dataset_id,row_id,concept,label,label_known,label_raw,view,sex,age,width,height,type_id`。label为0/1或空，未知必须`label_known=false`。医疗不把未知写为0。COCO无临床元数据用空。保留六个概念完整标签及可用的其他共现标签；NIH还保留manifest的全部九标签和no_finding。

NIH本包只额外提供评价名单，train从代码仓库原manifest读取。执行方可加本地image_path列供runner使用；返回稳定relative path即可，不需要上传原始图像。CheXpert/COCO名单先构建一次并由项目方冻结，共享给所有执行人。

## prompts.json

每个`dataset_id,concept,template_id`保存：准确问题文本、完整渲染后的聊天字符串/消息结构、generation prompt、回答logits的sequence index规则、positive/negative候选的字符串与token_id列表、raw A/B集合、score聚合方式。

IY/WY候选源为`yes,Yes, yes, Yes`与`no,No, no, No`；IA/WA为A正B负，IB/WB为B正A负，候选为plain及leading-space。只保留真正单token且按ID去重的候选。若集合为空、相交或prefill位置不对，先修接口再采样。保存每个候选分数能让我们比较max与log-sum-exp，不必重跑GPU。

## features/<locus>.npz

必需数组：`row_id`（Unicode字符串，无pickle）、`x`（N×D池化raw features，FP16或FP32）、`valid_token_count`（N）、`fit_role`（N）。包含全部拟合与校准/test样本。prompt-dependent位置须按实际prompt保存独立数组并注明，不能借用一个prompt的feature代表其他prompt。

主实验使用prompt-independent视觉边界；第二位置也优先选择prompt-independent图像输入。如果该架构不能提供这样的边界，在开跑前登记按问题拟合方案与额外提取量，作为独立适配条件，不静默复用。

池化数组通常远小于全token缓存：约`N×D×dtype_bytes`；19,212图、D=4096、FP16约157MB/位置。无需回传所有层/所有token的大型激活；16张preflight可留token片段用于核对。模型权重无需回传。

## fits/<locus>/seed*.npz

必需数组：

- `projection` D×512、`scaler_mean` 512、`scaler_scale` 512。
- `concept_names` 6、`coefficients` 6×512、`intercepts` 6、`clinical_vectors` 6×D。
- `random_vectors` 119×D（seed0文件），`sham_vectors` 6×D、`sham_permutations` 6×D。
- `control_coefficients` 6×20×512、`control_intercepts` 6×20（不同已知标签掩码可能导致各概念control模型不同）。NIH/COCO若完全共享control可存20×512并明确broadcast映射。
- `type_names`、`control_type_labels` 20×T；真实/控制probe的训练行掩码；`train_unit_ids`和`train_unit_multiplicities`（REFIT必需）。
- `projection_seed,fit_seed,C`；分类器其他参数在run.json中。nuisance拟合产物可放相同格式的独立数组并注明目标。

主clinical vector必须可由`normalize(P@(w/max(s,1e-8)))`重建。必须保存mean和intercept：旧directions.npz只有normal所需字段，不足以重建完整probe评分。REFIT每个seed都保存自己的scaler、coef、intercept和direction；主随机族保持seed0。

## probe_scores.parquet

一行一个样本×概念×probe：

`run_id,dataset_id,row_id,unit_id,role,locus_id,fit_seed,concept,probe_kind,control_seed,target_label,target_known,logit,probability`。

probe_kind为`real/control/nuisance`；非control的control_seed为空。至少包括calibration和test全部真实与20control评分；未知真实标签仍可存分数但target_known=false，统计时mask。control的target_label是对应type随机标签，不是疾病标签。

## outcomes/*.parquet

必需列：

| 字段 | 类型与含义 |
| --- | --- |
| protocol_id,run_id,model_key,dataset_id | string，身份 |
| module,role | string，工作块与cohort |
| row_id,unit_id | string，稳定样本与独立单元 |
| concept,template_id,locus_id | string，问题、模板、位置 |
| fit_seed | int，主条件0，REFIT为1/2 |
| direction_id,direction_kind | string；baseline、concept:<name>、random:000..118、sham:<question>；kind为baseline/concept/random/sham |
| alpha | float64，baseline为0，其他为协议的有符号剂量 |
| positive_token_ids,negative_token_ids | list<int>，与prompts.json一致 |
| positive_logits,negative_logits | list<float32>，逐候选，不取整 |
| vocab_logsumexp | float32，完整词表logsumexp，用于重建候选总概率质量 |
| semantic_margin,p_present | float32/64，max正减max负及sigmoid |
| lse_margin,raw_ab_margin | float32；raw_ab仅AB模板适用，其余空 |
| valid_token_count,input_token_count | int，实际图像token及完整输入长度 |
| token_norm_mean,token_norm_median,delta_norm_mean | float32，hook测得；baseline delta=0 |
| sample_status,error_reason | OK或FAILED；成功原因空 |

唯一键为`run_id,module,row_id,concept,template_id,locus_id,fit_seed,direction_id,alpha`。同一元数据scope可按partition存储，但导出时必须可完整恢复这些列。

baseline是真正不加hook的clean forward。alpha0 no-op只属于preflight。DOSE/REFIT复用相同model/dataset/row/question/IY的CORE baseline；在coverage明确`baseline_module=CORE,baseline_fit_seed=0`。LOCUS自身包含一套clean baseline并独立报告其位置拟合。CORE其他条件完整，不以sham替baseline。

允许分片，但每块都交`expected_rows,actual_unique_rows,failed_rows`；failed行保存键和失败原因，分数为空，不用0或上次输出补齐。最终统计只在完整块验收后进行，缺一条不能默默换分母。

## preflight.json 与 coverage.csv

preflight包含：16固定样本身份；实际tensor路径/shape/mask；clean、alpha0、+.25的最大logit差；消费者tensor变化；batch/sharding比较误差；语义正负陈述的候选分数；容差及接口结论；256条件测速与资源峰值。只验证改变过的家族/位置路径，证据不混入正式600图。

coverage列：`run_id,model_key,dataset_id,module,locus_id,fit_seed,template_id,concept,expected_rows,actual_unique_rows,failed_rows,execution_status,baseline_module,baseline_fit_seed,reason`。可汇总到question块，不能只给一个“全部完成”的boolean。

## 接收后的验收

1. 样本、标签、完整网格与固定配置对应；重复键、丢失baseline或意外删样本会阻止合并。
2. 由候选logits重算margin/p；由fit参数重建probe和normal；确认别把control标签当真标签。
3. 患者/图片配对bootstrap及各分母可从原始表重建；重新计算所有表格列，不采信截图中的显著性星号。
4. 输出完整但不符合发现预期仍验收为有效结果；程序或测量无效保留FAILED。

这样收到数据后可直接完成统计归并、三张主表和必要的附录，不依赖执行方的特定服务器、会话或绘图脚本。
