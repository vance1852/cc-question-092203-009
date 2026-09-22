# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。

## 检查点与断点恢复（GA / PSO）

大型机位搜索可能跨多个算力窗口。启用检查点后，优化器每隔 N 代/次迭代把当前状态**原子写入**一个 `.npz` 文件：

- GA：种群、适应度、全局最佳、迭代位置（已完成代数）、收敛/均值历史、随机数状态；
- PSO：粒子位置、速度、当前/个体/全局最佳、迭代位置、收敛/均值历史、随机数状态。

```bash
# 第一跑：每 5 代保存一次（也可不指定路径，默认放在输出目录）
python -m wind_farm_opt --algorithm ga --checkpoint ckpt/ga.npz --checkpoint-interval 5

# 窗口被回收后：从同一断点继续，直到跑满 --iterations
python -m wind_farm_opt --algorithm ga --checkpoint ckpt/ga.npz --resume
```

**兼容性指纹。** 每个检查点都带有对以下配置计算的 SHA-256 指纹：场地边界顶点、风机台数/直径/功率曲线/推力系数、尾流模型及参数、尾流叠加方式、风速积分参数、风资源扇区以及算法参数（不含可在恢复时调整的总迭代数）。换场地、换风机、换模型或改算法参数后用旧断点续算会被**直接拒绝**，防止错误续算。

**安全性。** 写入流程为「同目录临时文件 → 回读校验 SHA-256 校验和 → `os.replace` 原子替换 → 目录 fsync」；写入失败、文件损坏、截断、非检查点文件、格式版本/状态版本不兼容、跨算法（GA↔PSO）或维度不符，均报错且**不覆盖**现有断点。恢复时 RNG 状态逐位还原，因此分段续算到相同总迭代数与不中断运行的产物（位置、适应度、历史）**完全一致**。

**摘要标注。** 控制台与 `results.json` 的 `optimization_run` 段会标明 `new`（新跑）/ `resumed`（那个检查点、从第几代、最初来源运行 ID、第几次续算）；不启用检查点的默认流程不受任何影响，也不会出现该字段。

**注意：** `--resume` 若文件不存在则提示后从头新跑；已完成的断点再次 `--resume` 直接返回既有结果。另外，指纹保证物理与算法配置一致，但不校验 Python/numpy 版本——跨版本恢复虽然可用（BitGenerator 状态可重建），浮点数代码路径可能产生极小差异。

