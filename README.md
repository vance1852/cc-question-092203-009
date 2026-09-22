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

## 周期性检查点与断点恢复

GA 和 PSO 在大型机位搜索中耗时数小时，共享算力窗口若被回收，可用检查点续算而不必从头开始。该功能默认关闭，不配置 `--checkpoint` 时行为与原来完全一致。

```bash
# 新跑：每 5 代/次迭代原子保存一次完整状态
python -m wind_farm_opt --algorithm ga --iterations 100 \
  --checkpoint output/ga.ckpt.json --checkpoint-interval 5

# 算力窗口被回收后，用同一命令即可从最近断点自动续算
# （存在检查点则恢复，否则新跑）
python -m wind_farm_opt --algorithm ga --iterations 100 \
  --checkpoint output/ga.ckpt.json --checkpoint-interval 5

# 显式要求恢复（断点缺失/损坏/不兼容时报错，绝不静默重跑）
python -m wind_farm_opt --checkpoint output/ga.ckpt.json --resume

# 忽略旧断点强制新跑，首次保存时原子替换旧文件
python -m wind_farm_opt --checkpoint output/ga.ckpt.json --fresh
```

检查点完整保存继续计算所需的全部状态：

- **GA**：种群、适应度、全局最佳位置/适应度/出现代数、收敛与均值历史、NumPy 随机数发生器位状态；
- **PSO**：粒子位置与速度、适应度、个体历史最佳位置/适应度、全局最佳位置/适应度/出现迭代、收敛与均值历史、随机数位状态。

此外记录已完成迭代位置、运行 ID、恢复次数等台账信息。初始状态（第 0 代/次）与最终状态会额外强制落盘，因此昂贵的初始种群/粒子生成也不会因回收而丢失。

**稳定指纹阻止错误续算。** 每个检查点对场地边界顶点、风机型号与功率曲线/推力系数、尾流模型及参数、风资源扇区、叠加方式、目标函数身份，以及算法全部行为参数（种群规模、总迭代数、交叉/变异/惯性系数、种子等）计算 SHA-256 指纹。恢复时用当前配置重算指纹比对：场地或模型有任何不一致都会拒绝恢复并报错，避免在错误的断点上续算。总迭代数也纳入指纹——只允许恢复到相同总迭代数，保证结果可复现。

**原子写入，损坏不覆盖现有结果。** 写入采用“同目录临时文件 → fsync → `os.replace`”，崩溃、掉电或磁盘写满都只会留下临时文件（会被清理），既有检查点保持完整。文件还带整包完整性摘要，截断、JSON 损坏或任何字段被篡改都会在加载时被识别并拒绝，且拒绝加载不会触发写回。

**一致性保证。** 从断点恢复到相同总迭代数，与不中断运行得到逐位一致的产物（种群/粒子、适应度、最佳解、收敛历史一致），因为随机数位状态也被完整还原。

`results.json` 中的 `optimization_run` 字段标明本次是新跑（`fresh`）还是恢复（`resumed`）、运行 ID（同一次搜索跨多次恢复保持不变）、恢复次数、检查点来源路径以及续算起始迭代；未启用检查点时该字段不存在。

程序化使用时，`GAConfig`/`PSOConfig` 暴露 `checkpoint_path`、`checkpoint_interval`、`resume`（`None`=自动、`True`=必须恢复、`False`=必须新跑），返回的 `OptimizeResult.run_provenance` 与上述台账对应；自定义目标函数可在其绑定对象上实现 `checkpoint_fingerprint()` 方法返回稳定字典来参与兼容性校验。

