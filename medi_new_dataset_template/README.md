# MeDi 新数据集实验模板

这个目录整理的是一套可复用的实验思路：当拿到一个新的医疗图像数据集时，怎么按现在这条 MeDi 主线去做数据整理、缺失子群构造、合成补样、下游评估和 fairness 分析。

这里放的不是某一个具体数据集的完整训练代码，而是更通用的骨架：

- 一份实验思路说明
- 一个通用的缺失子群计划生成脚本
- binary fairness 评估脚本
- single-label multiclass 的 one-vs-rest fairness 评估脚本
- 当前在用的 `fairness_eval.py`

## 整体思路

现在这条主线可以概括成 7 步。

1. 先把新数据集整理成图像级的标准表。
2. 选定主任务类别和候选 metadata。
3. 构造训练中缺失的 `类别 × metadata` 子群。
4. 只对这些训练中缺失的子群生成图像。
5. 让每个缺失子群的生成张数和测试目标集里的真实张数一致。
6. 用真实数据和真实+合成数据分别做下游分类。
7. 比较主结果和 fairness 结果。

## 第一步：先整理成统一 CSV

建议至少整理出下面这些列：

- `sample_id`
- `image_path` 或 `image_path_processed`
- `primary_class`
- 一个或多个 metadata 列
  例如 `sex`、`age_bin`、`race`
- 如果需要分层切分，还可以保留：
  - `patient_id`
  - `official_split`

如果是单标签任务，`primary_class` 就直接是主类。  
如果原始数据是多标签，但你准备按 single-label 主线做实验，那就先明确好“主类”的整理规则。

## 第二步：选主类和 metadata

这条主线最关键的一点是：

- `class` 是主任务标签
- `metadata` 是你想研究的 subgroup 维度

最常见的组合是：

- `MeDi(类别)`
- `MeDi(类别+性别)`
- `MeDi(类别+年龄)`
- `MeDi(类别+种族)`
- `MeDi(类别+多个 metadata)`

一般建议先从最稳、最可解释的 metadata 开始，不要一上来把所有列都塞进去。

## 第三步：怎么构造缺失子群

严格按现在这条 MeDi 主线做的话，核心不是“随机多生成一点图”，而是：

- 先定义训练中缺失的 subgroup
- 再只补这些缺失 subgroup

这里常见有两种做法。

### 做法 A：随机候选子群

这和我们在 ODIR 上的做法一样。

- 先统计所有 `类别 × metadata` 子群
- 只保留样本数和患者数达到阈值的候选组
- 每个类别随机选 1 个或若干个子群
- 用固定随机种子保证可复现

这种做法适合：

- 你拿到的是一个原始数据集
- 没有官方预定义的 fairness split
- 你希望避免人工主观挑组

### 做法 B：预先指定缺失子群

这和 Harvard-GF 当前这条线更像。

- 直接指定哪些 `类别 × metadata` 组合是训练缺失子群
- 然后从 train / val 中移除这些组
- test 里保留这些组，作为目标子群

这种做法适合：

- 你已经有很明确的 fairness 问题
- 你想控制实验难度
- 你想让不同实验轮次严格可比

## 第四步：生成数量怎么定

严格 MeDi 主线里，生成数量不是手工拍脑袋定的，而是：

`n_generate = 该缺失子群在 test_target 中的真实样本数`

也就是说：

- 先定缺失子群
- 再在 `test_target` 里统计这些组真实有多少张
- 然后生成模型就按这个数量去补

所以单个实验组的总合成张数，等于所有缺失子群的 `n_generate` 之和。

## 第五步：实验组怎么设

最常见的一组主线对比就是：

- `仅真实数据`
- `MeDi(类别)`
- `MeDi(类别+metadata1)`
- `MeDi(类别+metadata2)`
- `MeDi(类别+多个 metadata)`

这样做的好处是很直接：

- 能看只加类别条件有没有帮助
- 能看哪个 metadata 最有用
- 也能看多个 metadata 一起加时是互补还是互相干扰

## 第六步：下游评估怎么做

现在这条线默认还是：

- 冻结一个图像 encoder
- 提 embedding
- 再用线性分类器做下游分类

主结果建议至少保留：

- `Accuracy`
- `Balanced Accuracy`
- `Macro-F1`
- `MCC`

如果还要看图像分布质量，可以再加：

- `FID`

## 第七步：fairness 怎么看

fairness 这里最容易混淆的一点是：

fairness 指标一定要先指定“按什么维度分组”。

比如：

- 按 `sex`
- 按 `age_bin`
- 按 `race`
- 按 `sex × age_bin`
- 按 `race × age_bin`

比较稳妥的原则是：

- 主 fairness 结果，看你构造缺失子群时用的联合分组
- 单个维度的 fairness 结果，当辅助分析

例如：

- 如果你是按 `sex × age_bin` 构造缺失子群
  - 主 fairness 看 `sex × age_bin`
  - `sex` 和 `age_bin` 单独看作辅助
- 如果你是按 `race × age_bin` 构造缺失子群
  - 主 fairness 看 `race × age_bin`
  - `race` 和 `age_bin` 单独看作辅助

## 任务类型和 fairness 工具怎么选

### 1. Binary 任务

比如：

- `control` vs `glaucoma`

直接用：

- `binary_fairness_eval.py`

### 2. 多类单标签任务

比如：

- `normal / dr / glaucoma / cataract / ...`

这种不属于 multilabel。  
建议按 one-vs-rest 转成多个二分类，再聚合 fairness。

直接用：

- `ovr_fairness_eval.py`

### 3. 真正的 multilabel 任务

如果你的数据真的是 multi-hot 多标签任务，再直接用 `fairness_eval.py` 的 multilabel 模式。

## 这个目录里有哪些文件

- `fairness_eval.py`
  当前使用的 fairness 指标主文件。
- `binary_fairness_eval.py`
  binary 任务的命令行封装。
- `ovr_fairness_eval.py`
  single-label multiclass 任务的 one-vs-rest fairness 封装。
- `build_missing_plan.py`
  从 `train.csv` 和 `test.csv` 里自动找训练缺失子群，并生成 `missing_plan.csv`。

## 建议的数据目录结构

一个新数据集接进来以后，建议至少整理成：

```text
dataset_name/
  metadata/
    metadata.csv
  splits/
    scenario_name/
      train.csv
      val.csv
      test.csv
      test_target.csv
      missing_plan.csv
```

这样后面的训练、采样、下游评估和 fairness 统计都会比较顺。
