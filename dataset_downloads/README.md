# 数据集下载脚本

这个目录只保留数据集下载相关的脚本，没有放训练、评估、画图之类的实验脚本。

整体结构很简单：

- `download_dataset_common.sh`
  公共 helper。主要负责日志输出、下载根目录解析、Kaggle / Hugging Face / gdown / Dataverse 这类常用下载函数，以及 `.done` 标记和手动下载说明文件。
- `download_oph_datasets.sh`
  眼科数据集下载脚本。
- `download_chest_datasets.sh`
  胸片数据集下载脚本。
- `download_skin_datasets.sh`
  皮肤图像数据集下载脚本。
- `download_pathology_datasets.sh`
  病理数据集下载脚本。
- `download_neuro_datasets.sh`
  神经影像数据集下载脚本。
- `download_physiology_datasets.sh`
  生理信号 / 生理测量数据集下载脚本。
- `download_general_datasets.sh`
  通用数据集下载脚本。

## 使用方式

每个脚本都支持 `--help`，可以先看它支持哪些数据集和需要什么环境变量。

例如：

```bash
bash dataset_downloads/download_oph_datasets.sh --help
bash dataset_downloads/download_oph_datasets.sh harvard-gf fairfedmed-oph
bash dataset_downloads/download_skin_datasets.sh isic2019 ham10000
DOWNLOAD_ROOT=/data/datasets bash dataset_downloads/download_chest_datasets.sh chexpert
```

## 下载目录

这些脚本默认会把数据下载到：

```bash
${PWD}/datasets/<domain>
```

如果想改位置，可以设置：

```bash
DOWNLOAD_ROOT=/your/path
```

脚本会自动在这个根目录下再接对应的 domain 子目录。

## 认证和手动下载

有些数据集可以直接自动下载，有些不行。

- 如果数据源支持命令行自动下载，脚本会直接下载。
- 如果数据源需要先登录、先申请权限，或者只能在官网手动点下载，脚本不会假装成功，而是会在目标目录里写一个 `ACCESS_INSTRUCTIONS.txt`，告诉你应该去哪个官网、下一步该怎么做。

## 这次整理里顺手做的清理

- 只保留了下载脚本本身，没有把实验脚本和中间结果一起带进来。
- 把眼科下载脚本也统一成和其他域脚本一样，复用 `download_dataset_common.sh`，避免 helper 重复维护。
- 所有脚本都重新跑过 `bash -n`，至少保证了基本的 shell 语法是通的。
