# Group 3 · Convex Optimization for Sparse Signal Reconstruction

**GitHub:** https://github.com/DongkuanWu/COMP6704-Group3-LASSO

**扩充版技术报告：** [LaTeX 源码](reports/COMP6704.tex)，包含八幅分析图；配套样式为 `reports/neurips_2026.sty`。在 `reports/` 中运行两次 `pdflatex COMP6704.tex`，或将两文件上传 Overleaf 并以 `COMP6704.tex` 为主文件编译。绘图数据内嵌于 LaTeX 中，原始记录见 `data/reference/extended/`。下面的 HTML 是较早的四图版本。

这是课程项目的独立实验仓库。给定同一组带噪线性测量，ISTA、FISTA 和 ADMM 求解同一个 LASSO 模型。我们分别评估目标函数值、信号重建误差、迭代次数和实际耗时。随仓库提供历史原始数据，可以先阅读报告和核对统计结果，再自行运行算法。

**阅读报告** — [英文技术报告（离线 HTML）](reports/technical-report.html)。用浏览器打开即可查看四幅图和十种配置的完整对比表。正文、图表与样式都在同一文件内，可直接发送给组员，也可通过浏览器打印。

## 1. 快速开始

在解压后的仓库根目录打开终端。历史环境为 Windows 11、Python 3.12.14 和 NumPy 2.3.5。依赖文件固定了 NumPy 版本。实验在 CPU 上执行，无需 GPU 或下载数据集。Python 3.12 是已验证的版本。

Windows PowerShell：

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools/verify.py
.\.venv\Scripts\python.exe experiments/extended_experiments.py
.\.venv\Scripts\python.exe tools/verify.py runs/extended
~~~

macOS / Linux：

~~~bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/verify.py
.venv/bin/python experiments/extended_experiments.py
.venv/bin/python tools/verify.py runs/extended
~~~

首次核对应输出 3,000 条运行记录、600 个实例与算法组合、30 条配置汇总，归档数据的最大同实例目标值差约为 2.299e-06。新机器的时间和时间区间会变化。检查程序使用正在验证的那批运行数据重算结果。

## 2. 目录和入口

~~~text
experiments/
  solvers.py                 数据生成、目标函数、残差和三个求解器
  run_experiments.py         初步实验，6 配置 × 10 实例 × 3 算法
  extended_experiments.py    主实验，10 配置 × 20 实例 × 3 算法 × 5 次计时
data/reference/
  pilot/                     180 次初步实验的原始数据、汇总、轨迹和环境记录
  extended/                  汇报和报告使用的 3,000 次运行及统计结果
tools/
  verify.py                  从逐次运行记录复核分组、均值和 bootstrap 区间
  build_report.py            从历史 CSV 生成报告、表格和四幅 SVG
  report_template.html       英文技术报告的正文和版式
reports/
  technical-report.html      可独立分发的完整报告
  figures/                   可单独插入 PPT 或文档的图
requirements.txt             NumPy 版本
runs/                        首次运行时生成，保存本机结果
~~~

在已激活的虚拟环境下，可以执行以下命令：

~~~bash
python experiments/run_experiments.py
python experiments/extended_experiments.py
python tools/verify.py runs/extended
python tools/build_report.py
~~~

前两个命令分别写入 `runs/pilot/` 和 `runs/extended/`。同一命令再次运行会更新该目录里的结果，需要保留多次实验时，请先将上一次运行目录另存。报告构建始终读取 `data/reference/extended/`，确保提交版与 PPT 的数字使用同一批记录。图表可离线重建，构建器只使用 Python 标准库。

## 3. 问题、输入与输出

观测模型为 **b = Ax⋆ + ε**，其中 A 的维度为 m × n，b 长度为 m，待恢复信号 x⋆ 长度为 n，含 s 个非零系数。三种求解器从零向量开始，共同求解

~~~text
minimize over x:  ½ ‖Ax − b‖₂² + λ ‖x‖₁
λ = α ‖Aᵀb‖∞      L = ‖A‖₂²      gradient step = 1/L
~~~

真实信号只参与生成模拟观测和事后计算恢复误差。求解器的输入是 A、b、λ、L 和残差归一化因子。每个求解器返回估计向量、按检查点记录的轨迹、单独记录的矩阵准备时间。ADMM 返回分裂变量 z。ISTA/FISTA 的 setup 字段固定为 0，该字段专门记录 ADMM 求逆准备阶段。

| 方法 | 每次更新 | 对耗时的影响 |
|---|---|---|
| ISTA | 在当前点求梯度，按 1/L 前进，再以 λ/L 软阈值化 | 每步主要为两次稠密矩阵向量乘法 |
| FISTA | 在外推点做相同的近端梯度步，并更新动量 | 通常减少更新次数，每步增加向量运算 |
| ADMM | 引入 x=z，依次更新 x、z 和缩放对偶变量 u | 固定 ρ=1，每次运行先显式求逆，再复用逆矩阵 |

共同停止量为 L 乘以一次近端梯度更新前后的差的无穷范数，再除以 max(1, ‖Aᵀb‖∞)。阈值为 10⁻⁴。ADMM 还要求归一化的原始、对偶残差均达到阈值。在第 1 轮、每 10 轮以及上限 1,200 轮检查。不同算法满足停止条件的迭代数包含检查粒度的影响。

## 4. 数据生成和配置

每个实例从标准正态分布生成 A 并逐列做单位范数归一化。随机无放回抽取 s 个支撑位置，符号随机，幅度均匀分布在 [0.8, 1.2]。加性噪声标准差为 σ。每个配置使用种子 0–19，每个种子内三个求解器共用一份输入。

| case | m | n | s | σ | α |
|---|---:|---:|---:|---:|---:|
| baseline | 96 | 192 | 10 | 0.02 | 0.05 |
| fewer_measurements | 64 | 192 | 10 | 0.02 | 0.05 |
| more_measurements | 128 | 192 | 10 | 0.02 | 0.05 |
| more_sparse | 96 | 192 | 5 | 0.02 | 0.05 |
| less_sparse | 96 | 192 | 24 | 0.02 | 0.05 |
| no_noise | 96 | 192 | 10 | 0 | 0.05 |
| more_noise | 96 | 192 | 10 | 0.08 | 0.05 |
| lower_lambda | 96 | 192 | 10 | 0.02 | 0.01 |
| higher_lambda | 96 | 192 | 10 | 0.02 | 0.15 |
| larger_problem | 192 | 384 | 20 | 0.02 | 0.05 |

要扩展设置，修改主实验的 `CASES`。每个元组按 (m, n, s, σ, α) 排列。尺寸对照同时增加 m、n、s，保持采样和稀疏比例。跨配置复用种子时，改变维度会改变随机抽样序列。正则化对照保持同一种子的数据固定，只改变 α。

## 5. 如何解释毫秒级计时

1. 每个实例内打乱三种算法及其五次重复运行的执行顺序，使用 `time.perf_counter()` 记录耗时。
2. 对同一配置、种子和算法的五次时间取中位数，再对二十个种子求平均。3,000 次运行对应 200 个配置与种子组合。
3. 配对比较先在同一种子上求两个时间中位数之差。再对二十个差值进行 2,000 次有放回的 bootstrap，取均值分布的 2.5% 和 97.5% 分位数。以 FISTA 减 ISTA 时，负数代表 FISTA 较快。
4. 数据生成和计算 L 在计时区间外。ISTA/FISTA 在计时前生成 Aᵀb，ADMM 将 Aᵀb 和显式求逆计入耗时。区间反映当前机器和合成实例分布上的变异。实验没有专门预热或固定 CPU 亲和性，各配置的探索性区间没有多重比较校正。

历史元数据记录操作系统、Python 和 NumPy 版本。CPU 型号、BLAS 线程数和后台负载未在原实验中记录。重新计时时建议记录它们并保存自己使用的线程设置。

## 6. 输出数据字典

| 扩展实验文件 | 一行表示什么 | 主要字段及单位 |
|---|---|---|
| extended_all_runs.csv | 一次运行，共 3,000 行 | 联合键 case、seed、algorithm、repeat。seconds、setup_seconds 单位为秒。iterations、objective、pg_residual、relative_reconstruction_error、converged 描述终点 |
| extended_per_seed.csv | 每个配置、种子、算法，共 600 行 | median_seconds、五次时间的最小/最大值、median_setup_seconds、最终质量指标 |
| extended_summary.csv | 每个配置和算法，共 30 行 | 20 个实例的时间中位数均值及区间，平均迭代数、平均恢复误差及区间 |
| extended_paired_comparisons.csv | 每配置三种有序配对，共 30 行 | algorithm_a − algorithm_b 的平均时间差及 95% 区间，单位秒 |
| extended_traces.csv | 每配置种子 0、重复标签 0 的各检查点 | iteration、seconds、objective、pg_residual。ADMM 另有 primal_residual、dual_residual |
| extended_metadata.json | 原运行环境与实验设置 | 配置、种子、重复次数、阈值、重采样次数、计时顺序 |

恢复误差是相对二范数误差，CSV 中的 0.0846 表示 8.46%。绘图和报告会将秒乘 1,000 转为毫秒。设置时间中位数与总时间中位数可能来自不同的运行，报告计算 ADMM 准备占比时使用同一次运行的组成项。重复标签 0 是打乱顺序前的编号。

初步实验保存 `results_raw.csv`、`results_summary.csv`、`traces.csv` 和 `run_metadata.json`。它每个实例只测量一次，主要用于保留从初步比较到扩展实验的过程。

## 7. 对照汇报中的结果

| 基准条件 | ISTA | FISTA | ADMM |
|---|---:|---:|---:|
| 平均迭代数 | 148.0 | 95.0 | 40.5 |
| 时间（ms） | 2.12 | 1.74 | 48.83 |
| 时间 95% 区间（ms） | [1.84, 2.45] | [1.56, 1.97] | [29.70, 71.97] |
| 平均相对恢复误差 | 8.47% | 8.46% | 8.46% |

共享实验中，FISTA 在十种条件中的九种比 ISTA 平均更快；测量数增至 128 时，ISTA 平均快 0.172 ms。全部 3,000 次运行达到各自的停止条件。同一实例三个终点目标值的最大差距为 2.30 × 10⁻⁶。详细时间分解、成对区间、测量条件、噪声和 λ 灵敏度见报告。原汇报中 FISTA 部分另有一组独立的 2,000 次补充实验，目前可用材料未包含其完整源码和逐次记录。仓库中的 FISTA 是三算法共享研究的可运行实现，报告只用共享研究作三方比较。

## 8. 复现与拓展

`tools/verify.py` 从逐次运行记录检查组合覆盖、停止状态、同实例计时中位数，并重建汇总表与配对表的均值和 bootstrap 区间。数值输出因浮点库而可能存在微小差异，运行时间更依赖硬件环境。若要研究 ADMM 的工程优化，可在相同实例上比较显式求逆、Cholesky 求解以及跨右端项复用分解的耗时。另一个独立问题是噪声敏感性，可固定 λ 后只改变 σ，与现有按 b 缩放 λ 的策略对照。

