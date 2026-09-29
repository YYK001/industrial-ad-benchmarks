# INP-Former MVTec AD / VisA 单类 full-shot 复评

此入口复用已验证的 `inpformer_btad.run` 运行逻辑、`inpformer_external.run.build_model/train_model`，
以及固定版本官方 INP-Former。没有重写模型或训练数学过程。两个数据集分别汇总，不混成一个27类macro。

## 固定协议

- MVTec AD：15类，各自使用全部 `train/good` 训练独立模型；不使用 MVTec AD 2。
- VisA：12类，官方单类 `split_csv/1cls.csv` 划分，或已按该划分整理的 `train/good, test/good|bad, ground_truth/bad` 格式。
  原始格式直接读取，无需复制数据或占用临时磁盘。整理版没有原始CSV时，无法单靠文件夹证明整理过程使用了官方划分，需保留数据来源。
- seed=1、dinov2reg_vit_base_14、448→392中心裁剪、INP_num=6、batch=16、FP32、shuffle/drop_last、200轮。
  保留官方目标层、重建分组、初始化、编码器冻结、StableAdamW、WarmCosineScheduler、损失和梯度裁剪。
- 每20轮原子覆盖最新恢复点；不累积历史checkpoint；第200轮模型用于评价。不使用早停、测试集调参或择优。
- 预测输入只有图片路径。VisA CSV预测只读取类别、split和图片路径，不使用label/mask；训练只检查训练行normal状态。
- 评价阶段才使用测试标签和mask。原始VisA非零mask缺陷ID映射为前景，保持二值；不混入真实异常训练。
- 256×256异常图、官方高斯kernel=5/sigma=4、最高655像素均值图像分数。
  GT与预测使用相同中心裁剪视野，空mask异常保留图像异常标签。
- 五指标与固定FPR函数完全复用既有实现，AP是average precision；AUPRO=CUDA/200阈值/FPR≤0.3。
  固定FPR=1%/5%，四连通、小区域≤256×256面积0.1%，不称为部署误报率。
- 这是当前环境和统一指标口径下的官方配置复评，不宣称与原论文/原环境指标逐位一致。
  不可与此前完整视野结果直接作差。

## 文件与代码复用

- MVTec AD清单/配对：既有 `DINOv3.relation_reliability.datasets.mvtec_ad_records`。
- 原始VisA清单/配对：既有 `DINOv3.MADEqual.visa_task_decoupled.dataset.visa_one_class_records`，源码原样纳入。
- `data.py`：图片专用清单、VisA整理版配对、原始多缺陷ID mask二值化、Kaggle根路径探测。
- `run.py`：在单独CLI进程中将数据函数绑定到共享运行器；测试结束恢复BTAD绑定。不能在同一解释器多线程中切换数据绑定。
- `pipeline.py`：两张GPU各一个独立子进程；按训练步数从大到小调度类别；每类训练→预测→评价。
  全部所选类别完成后分别生成每个数据集的macro和ZIP。

## 检查与单类 smoke

以下命令从仓库根目录运行；继续使用BTAD已跑通的环境、DINOv2权重和独立adeval目录。

```bash
# 只读探测两个挂载数据集、列出布局/类别/正常训练数/测试数/总步数。不会训练。
python -m external_baselines.inpformer_benchmarks.pipeline inspect --dataset both --input-root /kaggle/input

# 若探测到多个副本，通过 --mvtec-root /实际根目录 --visa-root /实际根目录 明确选择。

# 完整数据、GT配对和裁剪审计：DATA_MVTEC/DATA_VISA替换为inspect输出的根目录。
python -m external_baselines.inpformer_benchmarks.run --dataset mvtec check \
  --dataset-root "$DATA_MVTEC" --output-dir /kaggle/working/inpformer_mvtec_check
python -m external_baselines.inpformer_benchmarks.run --dataset visa check \
  --dataset-root "$DATA_VISA" --output-dir /kaggle/working/inpformer_visa_check

# 32张正常图片，两步FP32 batch16，确认非零LR更新与compact checkpoint重载。
python -m external_baselines.inpformer_benchmarks.run --dataset mvtec smoke \
  --dataset-root "$DATA_MVTEC" --category bottle --workers 2 --device cuda:0 \
  --backbone /kaggle/working/weights/dinov2_vitb14_reg4_pretrain.pth \
  --output-dir /kaggle/working/inpformer_mvtec_smoke
python -m external_baselines.inpformer_benchmarks.run --dataset visa smoke \
  --dataset-root "$DATA_VISA" --category candle --workers 2 --device cuda:0 \
  --backbone /kaggle/working/weights/dinov2_vitb14_reg4_pretrain.pth \
  --output-dir /kaggle/working/inpformer_visa_smoke
```

单阶段入口的输出目录必须不存在。check/smoke不会自动启动正式训练。

## 正式实验：同一cell训练、评价、打包

在Kaggle数据检查和对应smoke通过后使用。此cell会显式启动正式训练；不要为重新运行而更换时间戳目录。
同一个输出根目录下，脚本自动跳过已完成阶段、从最新恢复点继续；全部完成时只重新生成汇总/ZIP。

```python
import os, sys, subprocess
from pathlib import Path
from IPython.display import FileLink, display

repo = Path('/kaggle/working/industrial-ad-benchmarks')
out = Path('/kaggle/working/inpformer_mvtec_visa_v1')  # 续跑保持这个目录不变
env = os.environ.copy()
env['PYTHONPATH'] = os.pathsep.join([
    '/kaggle/working/inpformer_deps', str(repo), env.get('PYTHONPATH', '')
])
subprocess.run([
    sys.executable, '-u', '-m', 'external_baselines.inpformer_benchmarks.pipeline', 'run',
    '--dataset', 'both', '--input-root', '/kaggle/input',
    '--backbone', '/kaggle/working/weights/dinov2_vitb14_reg4_pretrain.pth',
    '--output-root', str(out),
], cwd=repo, env=env, check=True)
os.chdir('/kaggle/working')
for path in sorted(out.glob('INPFormer_*_objective_results.zip')):
    display(FileLink(path.relative_to(Path.cwd()).as_posix(), result_html_prefix='下载结果：'))
```

也可先只跑一个数据集：`--dataset mvtec` 或 `--dataset visa`，分别使用不同输出根目录。
需要按类别分批时，单数据集可加 `--categories bottle cable` 等；部分结果明确为selected-category macro，
不会冒充完整数据集结果。同一输出根目录不能随意更改类别选择、输入路径或backbone路径。

## 中断与存储

- 再次运行完全相同命令：检测已有完成标记，跳过完成的训练/预测/评价；未完成训练恢复到最近20轮保存点。
  在第20轮之前中断且没有checkpoint时，会明确打印fresh model并从头训练该类。
- 每个恢复尝试使用新的 `train_NNN` 目录，保留损失、配置和日志；只保留最高轮次的恢复checkpoint。
  完成训练后删除该类旧恢复checkpoint。预测/评价若中断，在新的尝试目录重做该阶段，不覆写完成结果。
- Linux输出目录锁由launcher和子进程共同持有；旧进程仍在运行时拒绝重复启动。
  捕获到中断/失败时终止该launcher的子进程组。浏览器断开不等于后台进程退出。
- Kaggle停止会话或临时磁盘被清理后，目录不存在就不能恢复。须由你保存/取回输出并恢复到原目录；本脚本不上传。
- MVTec AD和VisA的 `last.pt` 只保存模型中非encoder状态；加载时重新加载同一已核验DINOv2预训练权重，
  再严格检查全部其余key。恢复checkpoint仍保存优化器、调度器和随机状态。BTAD原来的完整last.pt格式保持支持。
- 两个数据集27个最终checkpoint不再重复保存27份编码器。连续预测保留在本地，不放进下载ZIP。
  不自动删除旧BTAD结果或其他实验；若多次中断预测留下多个不完整预测尝试，也需要考虑这些目录的空间。
- 27类200轮可能需要多个Kaggle会话；脚本不承诺一次会话内完成，也不会为了时限缩短训练或改变batch/精度。

## 结果包

输出根目录中分别生成 `INPFormer_mvtec_objective_results.zip`、`INPFormer_visa_objective_results.zip`。
各自包含全部类别五指标、类别等权macro、固定FPR及macro、空mask统计、逐图分数、空间信息、每轮损失、
各阶段资源/配置/完成记录和一份实际源码快照。配置记录外部仓库revision及dirty状态。
不包含checkpoint或连续像素预测；报告不能将ZIP里没有的模型/像素图说成已独立复核。
若有中断，成功训练尝试的耗时不是从头训练总耗时；保留各次尝试记录供合并。

## 本地验证范围

BTAD原回归测试与新增MVTec AD/VisA适配测试一起运行：

```bash
python -m pytest external_baselines/inpformer_btad/test_local.py external_baselines/inpformer_benchmarks/test_local.py -q
```

覆盖：官方CSV与整理版清单、不同扩展名配对、VisA非零缺陷ID二值化、空mask保留、测试label/mask不参与训练清单与预测输入、
compact checkpoint重载、跨数据集checkpoint拒绝、按最高轮次选恢复点、15/12类别等权macro、轻量ZIP排除模型与像素图。
本地MVTec AD完整15类已做真实数据检查；VisA只有合成双格式检查，需在Kaggle挂载数据上执行check。
新数据集真实CUDA smoke、27类正式训练及Linux进程中断恢复尚未在本地执行，不将CPU测试通过表述为实验完成。
