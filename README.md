# NPJ-Fairness

这个仓库当前先整理了一版数据集下载脚本，放在：

- [dataset_downloads](./dataset_downloads)

里面包含各个 domain 的下载脚本、一个公共 helper，以及简单的使用说明。  
如果你只是想下载数据，优先看这里。

另外还整理了一版把 MeDi 接到新数据集上的通用实验模板，放在：

- [medi_new_dataset_template](./medi_new_dataset_template)

里面包含：

- 新数据集接入 MeDi 的整体实验思路
- 通用的缺失子群计划生成脚本
- fairness 指标评估代码

如果你是想把 MeDi 接到一个新数据集上，优先看这里。

另外也单独放了一份原始 MeDi 主干代码，放在：

- [medi_original](./medi_original)

这里保留的是原始 MeDi 的训练、采样、嵌入、线性评估和 shell 启动脚本，不混入后面为 `ODIR-5K` 和 `Harvard-GF` 做的特化版本。

如果你只是想参考原始实现细节，再看这里就行。
