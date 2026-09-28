# INP-Former官方单类full-shot配置的BTAD外部复评

GitHub 同步版请使用根目录 README 中的 `git clone --recurse-submodules`。
版本读取支持 submodule 的 `.git` 指针文件，官方源码保持固定版本。
下文 `validation/` 路径是原研究工作区的本地验证记录，不随源码仓库发布；可按命令重新生成。

本目录是独立 BTAD 入口。复用已有官方 INP-Former 和已跑通的外部模型/训练函数，
不改官方源码，不重写模型。不称为原论文 BTAD 结果复现，不混用 INP-Former++、
多类、few-shot 或 zero-shot。所有命令只操作本地文件，没有服务器连接、上传或自动正式训练逻辑。

## 固定协议与复用边界

- 官方实际 commit：`17d265381d9b323a2ef6e05aab0665a85edebe84`；本地检查时 checkout 无修改。
- 官方对照：`external_baselines/INP-Former/INP_Former_Single_Class.py`。
- 直接调用 `external_baselines.inpformer_external.run.build_model`、`train_model`、
  `TrainDataset`、`data_loader_options`。复用官方 `INP_Former`、`Mlp`、
  `Aggregation_Block`、`Prototype_Block`、DINOv2 encoder loader、初始化及编码器冻结行为。
- 目标层 `[2,3,4,5,6,7,8,9]`，encoder/decoder 重建组均为 `[[0,1,2,3],[4,5,6,7]]`。
- 01/02/03 各自初始化、独立训练，seed=1。使用各类全部 `train/ok`，没有校准划分。
  `shuffle=True, drop_last=True` 意味着每轮不足 16 的尾批被丢弃，但不会预先排除训练图片。
- `dinov2reg_vit_base_14`，448×448 resize、392×392 center crop、INP_num=6；
  batch=16，FP32，200 轮，只有最后一轮 checkpoint；无早停、无测试集择优。
- 仅从显式本地 DINOv2 预训练权重初始化编码器。INP、bottleneck、extractor、decoder
  使用官方初始化；训练入口没有检测模型 checkpoint 参数，不支持 RobustAD 检测模型续训。
- 官方 `StableAdamW(lr=1e-3, betas=(0.9,0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)`。
  官方 `WarmCosineScheduler(base_value=1e-3, final_value=1e-4, warmup_iters=100)`，
  总步数为 `200 * floor(正常训练数/16)`。注意官方 warmup 的初始 LR 是 0，随后升至 1e-3；
  没有改动其 scheduler 初始化或 step 顺序。
- 官方 `global_cosine_hm_adaptive(y=3) + 0.2*g_loss`；梯度裁剪 max_norm=0.1。
  保留现有 pinned memory、persistent workers、prefetch=2、non-blocking copies；默认 workers=4，
  可按资源调整。关闭 `CUDA_LAUNCH_BLOCKING=1` 和 TF32，不启用 AMP。
- 仅复用 DINOv3 项目中的数据/指标工具，不构建 DINOv3、memory、MAD、LOCAL 或 GUIDED 模型。
  不继承旧 RobustAD SCRS、校准划分、mask 非空拒绝规则或 ADEval 评价路径。

## 数据与空间口径

BTAD 根目录可以直接是 `BTech_Dataset_transformed`，也可以是它的父目录 `BTAD`：

```text
BTech_Dataset_transformed/
  01/  02/  03/
    train/ok/*
    test/ok/*
    test/ko/*
    ground_truth/ko/*
```

复用 `DINOv3.MADEqual.btad_validation.dataset` 的 `find_root`、`images`、`records`、
`mask_info`：支持 jpg/jpeg/png/bmp/tif/tiff；按 stem 配对不同扩展名 mask，类别保留前导零。
`check` 对照旧 BTAD records 验证图片集合及顺序一致，不采用其 memory/calibration 角色字段。
训练和推理另用只有 `path` 的 `ImageRecord`，不调用带标注的 `records`，不读取测试标签或 mask；
所有图片转 RGB。`check` 是显式只读数据审计，正式 GT 读取发生在独立 `evaluate` 阶段。

预测保持 `score_records` 的计算：官方 `cal_anomaly_maps` → 双线性变为 256×256
(`align_corners=False`) → 官方 `get_gaussian_kernel(kernel_size=5,sigma=4)` →
平滑后像素排序、最高 `floor(65536*0.01)=655` 个像素均值作为图像分数。
预测函数不再读取或返回 GT。

每图保存原始尺寸和变换：原图→448×448→裁剪 xyxy=`[28,28,420,420]`→256×256。
相对于原图，裁剪范围是 x=`[W/16,15W/16]`、y=`[H/16,15H/16]`。
GT 的 0/1、0/255 均先转二值，再最近邻 resize/crop/resize；不插值出灰度标签。
这是对 BTAD 二值 GT 的明确适配，不继承官方通用 GT transform 的默认双线性插值。
原始空 mask 和裁剪后空 mask 的异常均保留 `label=1`，只对没有 mask 的正常图片构造全零 GT。
另记录最终 256×256 重采样后的空 mask 数。

**报告是官方裁剪视野下的指标，不能与旧 BTAD 完整视野指标直接计算差值。**
新异常图不拉伸回完整原图。若以后公平对照，可单独将旧连续预测按相同原图裁剪范围
重采样到本次评价坐标，再用同一 GT/指标离线重评；不重训旧模型、不覆盖旧报告。
本轮没有执行或生成这个额外对照。

## 指标与保存内容

直接复用 `DINOv3.MADEqual.mvtec_broad6_compose2.metrics.evaluate_fast`：
Pixel AUROC、Pixel AP、快速 CUDA AUPRO@0.3（全局分数范围 200 阈值）。
该函数默认的 image max 指标被保存的官方 top-1% 图像分数重算覆盖；
Image AUROC 用 `roc_auc_score`，Image AP 用 `average_precision_score`。
CSV 为兼容现有函数保留 `image_AUPR`、`pixel_AUPR` 字段名，二者都是 AP，不是梯形 PR AUC。
正式评价强制 CUDA，`allow_cpu_fallback=False`；不调用旧 ADEval CPU 路径。

固定 FPR 复用 `hard_sample_discrimination.scoring.fixed_fpr_diagnostics`：1%/5% 下
缺陷像素召回、区域平均覆盖、小区域平均覆盖。四连通，小区域≤评价图像面积的 0.1%，
在本次 256×256 坐标系下即至多 65 个像素。测试负像素阈值整组处理 ties，实际 FPR≤上限。
这些是事后评价诊断，**不代表经过独立校准的部署误报率**。

- `config.json`：实际官方 commit、完整固定配置、参数、实际包版本、性能设置。
- `source_snapshot/`：官方 Python 源码、现有外部适配、新入口及关键数据/指标文件的轻量快照；
  不做数据/权重哈希闭包，不保存中间特征。
- `check`：`counts.csv`、带图片/mask 标识与空间变换的 `input_manifest.json`。
- `train`：每类 `training_inputs.json`、`epoch_losses.csv`、`last.pt`、完成状态。
- `predict`：逐图连续 FP32 `NNNN.npy`、`predictions.json`（标识/变换/分数）、`image_scores.csv`。
- `evaluate`：`category_metrics.csv`、`macro_metrics.csv`、`fixed_fpr.csv`、
  `fixed_fpr_macro.csv`、`mask_audit.csv`、`counts.csv`、`report.md`。
- 三类使用类别等权 macro；只选一类时明确标为 selected-category macro，不能称三类结果。
  小区域不存在时输出 N/A，该项 macro 只平均有定义类别，并保存参与类别数。
- `resources.csv`：初始化、训练、推理（包括预测写盘）、GT/预测读取、指标评价分开计时；
  各阶段 CUDA 同步后计时，并记录峰值 allocated/reserved 显存。

输出目录必须不存在，拒绝覆盖已有实验结果。checkpoint 校验协议、类别、固定配置和
epoch=200，并拒绝 smoke checkpoint。无需 barrier 或文件哈希系统。

## 环境与资源

优先复用已跑通的 INP-Former 兼容环境，不自动安装包、不修改服务器基础环境。
已有外部环境参考：Python 3.8.12、torch 2.0.0+cu118、torchvision 0.15.1+cu118、
timm 0.9.12、kornia 0.7.3、adeval 1.1.0，以及官方 requirements 中的 NumPy、SciPy、
scikit-learn、scikit-image、Pillow、pandas、opencv、matplotlib、tqdm。
官方 utils 在导入时依赖 adeval，即使本入口不调用其评价器，也需保证依赖可导入。
本入口不强制逐个包与旧环境版本完全相同，但记录实际版本；Kaggle 兼容性必须先 smoke 验证。

必须准备：全部正常训练图片、官方源码、现有外部适配、本目录及项目已有数据/指标工具，
以及 `dinov2_vitb14_reg4_pretrain.pth`。正式评价另需完整固定测试集合及 mask。
权重通过 `--backbone` 指向你上传的只读文件；缺失直接报错，不访问网络。
仅对这一份预训练权重复用现有外部适配已核验的 SHA256
`73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71`，
防止把旧检测模型误作为编码器输入后被官方 `strict=False` 静默接受；没有新增数据哈希系统。
官方 loader 的文件解析被限定到这份本地权重，模型加载过程保持官方方式。
源码版本读取需要随源码保留 `.git/HEAD` 及其指向的 ref（或 `packed-refs`）；
可以在上传源码压缩包时一并保留这些小文件。不要只上传不带版本记录的散装源码。
官方导入时创建空缓存目录的副作用被隔离在临时目录，源码可以放在只读输入区。

主要资源约束是 Kaggle 双 T4、30 GB RAM、20 GB 临时可写空间、约 40 GB 由你上传的持久输入。
默认在一张 T4 上逐类运行，保持官方 batch=16；不会把两张卡当成统一显存，也不自动改成 DDP。
如果单卡 FP32 batch=16 不足，smoke 应报 OOM，不能自行改 batch/精度冒充本协议。
本轮尚未实测 T4 显存和耗时。每个最终模型约数百 MB，三类 741 张 256×256 FP32
连续预测约 185.25 MiB（不含小量文件头），不写重复特征缓存。
临时空间中的 checkpoint/结果在会话结束后可能丢失，运行后由你安排取回；本入口不上传。

## 独立命令（在项目根目录运行）

下面是供后续 Kaggle 使用的 Bash 示例。使用已有兼容 Python；源码根目录在 `PYTHONPATH`
中。每一步是显式独立命令，smoke 完成不会启动 train。修改路径后再运行。

```bash
export PYTHONPATH=/kaggle/working/experiment
cd /kaggle/working/experiment
DATA=/kaggle/input/btad/BTech_Dataset_transformed
WEIGHT=/kaggle/input/dinov2/dinov2_vitb14_reg4_pretrain.pth
OUT=/kaggle/working/inpformer_btad_v1

# 只读检查：不建模型；会检查完整图片集合、配对、原始/裁剪后空 mask。
python -m external_baselines.inpformer_btad.run check \
  --dataset-root "$DATA" --output-dir "$OUT/check"

# 资源 smoke：真实官方模型，两个正常 batch 的训练与正常图推理、保存重载比较。
# 调度总步数仍按该类完整训练 loader×200 计算；无测试效果、无参数选择。
python -m external_baselines.inpformer_btad.run smoke \
  --dataset-root "$DATA" --backbone "$WEIGHT" --category 01 \
  --device cuda:0 --workers 4 --output-dir "$OUT/smoke_01"

# 独立指标 smoke：只用合成数组，CUDA 与既有 CPU 指标分支对照，不读取 BTAD 测试集。
python -m external_baselines.inpformer_btad.run metrics-smoke \
  --device cuda:0 --output-dir "$OUT/metrics_smoke"

# 正式训练：三类独立、200轮、最后 checkpoint。仅供后续显式执行，本轮未运行。
python -m external_baselines.inpformer_btad.run train \
  --dataset-root "$DATA" --backbone "$WEIGHT" \
  --device cuda:0 --workers 4 --output-dir "$OUT/train"

# 独立推理：只读取图片，连续预测与分数落盘，完全不读取测试 GT。
python -m external_baselines.inpformer_btad.run predict \
  --dataset-root "$DATA" --backbone "$WEIGHT" --checkpoint-root "$OUT/train" \
  --device cuda:0 --workers 4 --output-dir "$OUT/predictions"

# 正式评价：读取固定预测后加载 GT，复用已有 CUDA 指标。
python -m external_baselines.inpformer_btad.run evaluate \
  --dataset-root "$DATA" --prediction-root "$OUT/predictions" \
  --device cuda:0 --output-dir "$OUT/evaluation"
```

可用 `--category 01/02/03` 按类运行，默认 all。每个命令可指定 `--official-root`。
两步 smoke 使用前 32 张正常训练图片、batch=16，保留完整训练的调度总步数。
记录两步实际学习率，要求首步为 0、第二步大于 0、参数有限且确实发生更新；
随后严格检查 checkpoint 重载预测一致性。它不会自动触发正式训练。
`metrics-smoke` 的 CPU 分支仅用于合成数据对照；正式 `evaluate` 仍禁止 CPU fallback。
若手动使用第二张 T4，各类进程必须使用不同的输出目录；随后将已完成的各类目录
置于统一 checkpoint/prediction 根目录再评价。入口本身不调度并行训练。
200 轮必须能在当前运行窗口内完成；当前入口未提供中断续训，也不保存优化器恢复状态。

## 本轮本地验证

本地未连接服务器、未上传、未正式训练，也未修改已有报告/实验结果。
默认 Python 缺少 torch，因此使用已有 `D:/anaconda/miniconda/python.exe` 的 CPU 环境：
torch 2.4.1+cpu、torchvision 0.19.1+cpu。缺少 timm/kornia/adeval 和真实 CUDA，未安装新包。

```powershell
& D:/anaconda/miniconda/python.exe -m pytest external_baselines/inpformer_btad/test_local.py -q --disable-warnings
& D:/anaconda/miniconda/python.exe -m external_baselines.inpformer_btad.run check --dataset-root datasets/BTAD --output-dir external_baselines/inpformer_btad/validation/new_data_check
```

上面的重跑检查命令使用新目录名；目录已存在时请另选新名称。
本轮最终 CPU 测试为 **8 passed**，记录在 `validation/local_tests.xml`。
现有真实 BTAD 数据检查结果（`validation/final_data_check/counts.csv`）：

| 类别 | 正常训练 | 正常测试 | 异常测试 | 测试总数 | 原始空 mask 异常 | 裁剪后空 mask 异常 |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 400 | 21 | 49 | 70 | 0 | 1 |
| 02 | 399 | 30 | 200 | 230 | 1 | 8 |
| 03 | 1000 | 400 | 41 | 441 | 10 | 10 |

训练每轮步数为 25/24/62，总 scheduler 步数为 5000/4800/12400。
原始空 mask 11 张，裁剪后空 mask 共 19 张（包含原始空 mask）；最终 256×256 的空数也是 19。

CPU 测试覆盖：不同图片/mask 扩展名及 0/1/255 编码、空异常保留、矩形图裁剪对应、
边界异常裁掉、RGB 转换、两步外部训练与直接执行官方训练代码块的参数/损失精确相等、
checkpoint 重载预测精确相等、测试标签翻转及 mask 损坏不影响训练/预测、
预测数组/图像分数与原 `score_records` 精确相等、既有指标 CPU 分支及合成评价产物/macro。
缺失权重和误传检测 checkpoint 在任何官方可选依赖导入之前被拒绝。
测试通过 AST 提取执行官方函数以绕开缺少的可选导入，不复制实现；使用小型测试模型，
不是完整预训练 INP-Former 模型验证。CPU 指标分支只在测试中显式注入，正式入口仍强制 CUDA。

待完成的真实验证：兼容环境完整官方导入、本地 DINOv2 权重加载、双 T4 环境单卡
batch=16 FP32 的前后向/显存/耗时、真实模型 checkpoint 重载一致性、CUDA 五指标与固定 FPR
的完整运行。**本地测试通过不代表正式复现成功，也没有产生正式 BTAD 模型效果结论。**
