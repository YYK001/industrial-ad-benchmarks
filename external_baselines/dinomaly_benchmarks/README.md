# Dinomaly 官方单类训练和推理配置与项目快速 CUDA 指标复评

用于硕士论文第一章的独立对比入口：MVTec AD 15 类、VisA 12 类，每类正常训练集独立训练一个模型。
本轮交付代码与小测试，没有启动正式训练、全量推理或评价。不是 Dinomaly2、多类模型或 DINOv3 替换版本。

## 官方版本和代码复用

官方仓库：[guojiajeremy/Dinomaly](https://github.com/guojiajeremy/Dinomaly)，实际锁定
`1f252be03a918789b19848f0ca37166e7c28dada`，Apache-2.0 原许可证位于相邻 `dinomaly_official/LICENSE`。
官方源码在本同步分支中作为固定版本 submodule；研究工作区的独立 checkout 保持不变。
未调用此前 normal_reconstruction 模型或实验入口。

`official.construction_ast` 从两个官方 `train()` 函数原样提取 encoder_name 至 lr_scheduler 的完整构建段并执行，
直接使用官方 ViTill、bMlp、8 个 Block/LinearAttention2、trunc_normal_ 初始化、StableAdamW 和 WarmCosineScheduler。
没有重新编写网络、损失或调度公式，也不执行原脚本的训练循环或 CPU compute_pro。
`symbols` 在独立 CLI 进程导入官方脚本，检查 models/utils/dataset 等模块的真实来源，遇到其他方法的同名缓存模块拒绝继续。
编码器路径只映射到显式本地权重，不自动下载；在官方 load 的 strict=False 之外增加严格重载，缺键或不匹配即失败。
仅校验一个冻结编码器文件的 SHA256，不建立大型哈希闭包。

项目复用如下，既有文件均未修改：

- 数据清单：`inpformer_benchmarks.data.Adapter.image_records/records`，进一步复用 MVTec AD `mvtec_ad_records`
  和 VisA `visa_one_class_records`；只复用清单/配对，**不调用 Adapter.load_gt**。
- 写文件：`patchcore_official_eval.storage` 的原子 JSON、CSV、NPZ 写入。
- 五指标、固定 FPR 与等权汇总：`patchcore_official_eval.evaluation.calculate/summarize`。
  这些函数中的 DINOv3 目录引用只提供通用指标，不构建、训练或运行 DINOv3、MAD、LOCAL、GUIDED。
  其中 fixed_fpr_diagnostics 是既有通用函数；不调用 normal_reconstruction 的模型或实验运行器。
- 资源和环境：`destseg_mvtec_pretrained.resources.AllocatorResources`、`patchcore_official_eval.resources.environment`。
- 双进程日志、heartbeat、取消：`destseg_visa.processes.run_process`。没有 DDP、跨卡切分或新的实验管理框架。

## 固定配置和评价视野

两个官方单类脚本的实际配置与用户提供记录一致：seed=1 每类重设；5000 次 optimizer.step；batch=16、shuffle/drop_last；
RGB Resize448→ToTensor→CenterCrop392→ImageNet normalize；dinov2reg_vit_base_14；target_layers 2..9；
encoder/decoder 两个融合组均为 `[0,1,2,3]` 和 `[4,5,6,7]`；bMlp drop=0.2；8 个 LinearAttention2 解码 Block；mask_neighbor_size=0。

仅官方瓶颈和解码器进入优化器，不更改编码器原有 no_grad forward 行为或 requires_grad 标志。
StableAdamW lr=2e-3、betas=(.9,.999)、weight_decay=1e-4、amsgrad=True、eps=1e-8；
WarmCosineScheduler base=2e-3、final=2e-4、total_iters=5000、warmup_iters=100。
损失直接调用 global_cosine_hm_percent，p=min(.9*已完成更新数/1000,.9)、factor=.1；梯度裁剪 .1；FP32，无 AMP/TF32。
保持 optimizer.step→scheduler.step 的官方次序。构造调度器后第一个更新使用零学习率，smoke 至少执行两步并验证非零更新。
固定第 5000 步评价，不按测试指标选权重；目录名 it10k 不代表实际步数。不使用 INP-Former 的 200 轮设置。

推理顺序与锁定的 evaluation_batch 一致：官方模型 en/de→cal_anomaly_maps(out_size=392)→
bilinear 到 256（align_corners=False）→官方 Gaussian kernel5 sigma4→降序最高 int(65536*.01)=655 像素均值。
不加 sigmoid、MAD、平滑以外的额外处理、逐图归一化或自定义分数。

**GT 的初始 Resize 是 PIL 双线性，随后 PIL CenterCrop392→ToTensor→nearest256→bool(nonzero)，多通道用 max 合并。**
这与 INP-Former 适配器全 nearest 的 GT 不同。VisA 原始多缺陷 ID 先按非零前景转为 0/255，再走官方 GT 变换；
这复用官方 VisA 整理的语义，不把 ID 当概率阈值。保存原始、裁剪后、最终评价空 mask 统计；异常图像标签保持 CSV/目录标签，
不因缺陷被裁掉而改成正常。评价始终保留裁剪视野和 256×256 网格，不将裁剪图拉伸到全图，也不补裁剪外边界。

指标命名为“官方单类训练和推理配置 + 项目快速 CUDA 指标复评”：Image AUROC/AP、Pixel AUROC/AP、AUPRO@0.3。
AP=average precision；AUPRO=既有快速 CUDA、200 个阈值、FPR≤.3、既有四连通。
不会自动调用官方 CPU compute_pro 或 ADEval，也不会隐式回退到 CPU。
固定 FPR 1%/5% 的实际 FPR、缺陷像素召回、区域/小区域覆盖沿用项目工具；小区域≤评价图像素数的 .1%。
无小区域为 N/A，只按有效类别等权平均；阈值是测试集事后诊断，不用于训练或部署阈值选择。

## 1 环境和冻结权重准备

以下是供服务器执行的 Bash 命令，本轮没有执行安装或权重下载。各阶段从当前项目根目录运行。
这里选独立 Python3.10、torch2.1.2+cu118、torchvision0.16.2，与官方历史 torch1.12+cu113 环境不同，
属于待 smoke 验证的兼容环境；官方 requirements.txt 完整保留。配对参考 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。
不修改既有 DeSTSeg 或其他方法环境。不额外安装 xformers，采用官方 attention 的非 xformers 路径。
兼容依赖清单补充 `colorama==0.4.6`：官方 optimizer 包导入时会连带导入 ACProp，
其顶层依赖 colorama，即使实际训练只使用 StableAdamW 也需安装；官方源码和训练算法不变。

```bash
REPO=/kaggle/working/industrial-ad-benchmarks  # 改成代码所在根目录
VENV=/kaggle/working/dinomaly-venv
OFFICIAL="$REPO/external_baselines/dinomaly_official"
WEIGHT=/kaggle/working/dinomaly_assets/dinov2_vitb14_reg4_pretrain.pth
OUT=/kaggle/working/dinomaly_single_v1    # 正式输出独立；续跑保持此路径
MVTEC=/kaggle/input/path/to/mvtec_anomaly_detection
VISA=/kaggle/input/path/to/VisA           # 含 split_csv/1cls.csv 的原始根目录
cd "$REPO"

# 如果同步代码时未带独立官方 checkout，仅在目录不存在时下载。
if [ ! -d "$OFFICIAL" ]; then
  git clone https://github.com/guojiajeremy/Dinomaly.git "$OFFICIAL"
  git -C "$OFFICIAL" checkout --detach 1f252be03a918789b19848f0ca37166e7c28dada
fi

uv venv --seed --python /usr/bin/python3.10 "$VENV"
PY="$VENV/bin/python"
uv pip install --python "$PY" torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu118
uv pip install --python "$PY" -r external_baselines/dinomaly_benchmarks/requirements.server.txt
"$PY" -m pip check
"$PY" -m external_baselines.dinomaly_benchmarks source --dataset mvtec --official-root "$OFFICIAL"

# 只下载公开冻结编码器，不下载多类 detector 或其他方法权重。
mkdir -p "$(dirname "$WEIGHT")"
if [ ! -f "$WEIGHT" ]; then
  curl -fL --retry 3 \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth \
    -o "${WEIGHT}.part"
  mv "${WEIGHT}.part" "$WEIGHT"
fi
"$PY" -c 'import sys; from external_baselines.dinomaly_benchmarks.official import weight_identity; print(weight_identity(sys.argv[1]))' "$WEIGHT"
"$PY" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==2; print(torch.__version__); print([torch.cuda.get_device_name(i) for i in range(2)])'

# 标准库检查 + 合成科学运行时检查；不下载权重、不训练真实模型。
"$PY" -m unittest discover -s external_baselines/dinomaly_benchmarks/tests -v
```

SHA256 必须为 `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71`。
真实权重若不匹配，明确失败，不随机初始化编码器继续实验。

## 2 数据检查

```bash
"$PY" -u -m external_baselines.dinomaly_benchmarks check --dataset mvtec \
  --dataset-root "$MVTEC" --categories all --output-dir "$OUT" --official-root "$OFFICIAL"
"$PY" -u -m external_baselines.dinomaly_benchmarks check --dataset visa \
  --dataset-root "$VISA" --categories all --output-dir "$OUT" --official-root "$OFFICIAL"
```

官方目录必须保留其 git checkout 元数据；若只拷贝了源码没有 `.git`，请在独立目录获取锁定 checkout，
然后通过 `--official-root` 指向它。版本核对不会静默接受未知来源的源码副本。

MVTec AD 参考 3629 张正常训练、1725 张测试；VisA 参考 8659 张正常训练、2162 张测试。
`data_check/counts.csv`、`manifest.json` 保存实际数目、配对及几何和空 mask；complete.json 列出实际与参考差异。
不补删图片凑数、不划 calibration。训练图片不读取测试标签/mask。
原始 VisA 直接读取 CSV，不复制图像；训练按文件名排序，与单类 ImageFolder 的索引顺序对齐。
若使用整理版，根目录为 1cls 下的 `train/good,test/good|bad,ground_truth/bad`，
整理版没有 CSV 时只能记录文件夹来源，无法凭文件夹独立证明原划分；请保留原 CSV/官方准备过程。

## 3 单类别 CUDA smoke

```bash
"$PY" -u -m external_baselines.dinomaly_benchmarks smoke --dataset mvtec \
  --dataset-root "$MVTEC" --categories bottle --official-root "$OFFICIAL" --backbone "$WEIGHT" \
  --device cuda:0 --workers 2 --smoke-images 32 --smoke-steps 2 --inference-batch 1 \
  --output-dir /kaggle/working/dinomaly_smoke_mvtec_v1

"$PY" -u -m external_baselines.dinomaly_benchmarks smoke --dataset visa \
  --dataset-root "$VISA" --categories candle --official-root "$OFFICIAL" --backbone "$WEIGHT" \
  --device cuda:1 --workers 2 --smoke-images 32 --smoke-steps 2 --inference-batch 1 \
  --output-dir /kaggle/working/dinomaly_smoke_visa_v1
```

smoke 新目录与正式输出完全分开，不允许复用已存在目录。仍按 5000 步的官方调度曲线执行前两步，
检查有限损失/梯度、非零学习率、真实参数变化、compact 权重严格重载、真实推理重载前后预测一致和 NPZ 保存重载。
只预测两张测试图，不计算全类别指标；训练及推理峰值分别写入 resources.csv。
单卡 OOM 会写 failure.json 并退出，绝不静默降低 batch16、分辨率、步数或改 AMP。是否调整工程配置由用户决定。

## 4 单类别正式训练和评价

以下命令会启动正式实验，先在服务器完成上面的检查和 smoke，再由用户执行。

```bash
"$PY" -u -m external_baselines.dinomaly_benchmarks train --dataset mvtec \
  --dataset-root "$MVTEC" --categories bottle --official-root "$OFFICIAL" --backbone "$WEIGHT" \
  --device cuda:0 --workers 4 --checkpoint-every 250 --output-dir "$OUT"
"$PY" -u -m external_baselines.dinomaly_benchmarks predict --dataset mvtec \
  --dataset-root "$MVTEC" --categories bottle --official-root "$OFFICIAL" --backbone "$WEIGHT" \
  --device cuda:0 --workers 4 --inference-batch 16 --output-dir "$OUT"
"$PY" -u -m external_baselines.dinomaly_benchmarks evaluate --dataset mvtec \
  --dataset-root "$MVTEC" --categories bottle --official-root "$OFFICIAL" \
  --device cuda:0 --output-dir "$OUT"
"$PY" -m external_baselines.dinomaly_benchmarks summarize --dataset mvtec \
  --categories bottle --output-dir "$OUT"
```

VisA 将 dataset 改为 visa、root 改为 `$VISA`、category 改为 candle 等。训练固定 batch16；推理 batch 可单独修改。
每个类别重新 seed=1、重新加载同一冻结编码器、初始化瓶颈/解码器，不能从别类 final.pt 开始。

## 5 双 T4 按类别运行两个数据集

```bash
# 每 GPU 同时最多一个类别，模型完整在单卡；每类训练→预测→复评后释放。
"$PY" -u -m external_baselines.dinomaly_benchmarks.pipeline --dataset mvtec \
  --dataset-root "$MVTEC" --categories all --official-root "$OFFICIAL" --backbone "$WEIGHT" \
  --devices cuda:0 cuda:1 --workers 4 --inference-batch 16 --checkpoint-every 250 --output-dir "$OUT"

"$PY" -u -m external_baselines.dinomaly_benchmarks.pipeline --dataset visa \
  --dataset-root "$VISA" --categories all --official-root "$OFFICIAL" --backbone "$WEIGHT" \
  --devices cuda:0 cuda:1 --workers 4 --inference-batch 16 --checkpoint-every 250 --output-dir "$OUT"
```

两个数据集顺序运行，不能同时各起一个双卡 launcher。类别按既有清单 `[::2]`/`[1::2]` 分配，
`--categories bottle cable` 等支持分批。每分钟 heartbeat，训练每 50 步打印；缺少子进程输出超过 900 秒报错，评价允许 3600 秒。
子进程失败或中断会取消本 launcher 的其他子进程，并导出已有轻量记录。不承诺 27 类在一个 Kaggle 会话内完成。

## 6 中断后基本恢复

```bash
# 保持同一 OUT、数据与编码器路径、workers 和其他设置，只增加 --resume。
"$PY" -u -m external_baselines.dinomaly_benchmarks.pipeline --dataset mvtec \
  --dataset-root "$MVTEC" --categories all --official-root "$OFFICIAL" --backbone "$WEIGHT" \
  --devices cuda:0 cuda:1 --workers 4 --inference-batch 16 --checkpoint-every 250 \
  --output-dir "$OUT" --resume

# 或只恢复单类训练，已完成类别可跳过。
"$PY" -u -m external_baselines.dinomaly_benchmarks train --dataset visa \
  --dataset-root "$VISA" --categories candle --official-root "$OFFICIAL" --backbone "$WEIGHT" \
  --device cuda:0 --workers 4 --checkpoint-every 250 --output-dir "$OUT" --resume --skip-completed
```

未完成类每 250 步原子覆盖一个 latest.pt，含全部非 encoder 模型状态、optimizer、scheduler、更新数、
Python/NumPy/torch/当前 GPU 随机状态、epoch、已消费批数和 epoch 迭代前 RNG。
重建官方 shuffle 顺序并跳过已消费批次，然后恢复 dropout 等当前随机状态；不是只加载模型的“恢复”。
checkpoint 间未保存的更新会重做，早于首次 250 步中断且无 latest 时明确从头初始化。
数据变换确定性，无新增增强；num_workers>0 使用 spawn、无 persistent_workers（运行工程选择，区别于原脚本默认 Linux fork）。
可改到同一机器的另一张 CUDA 卡恢复随机状态；需要保留相同 worker 数和数据清单。
已提供中途和 epoch 边界的 CPU 合成测试，本机因缺科学环境尚未执行；真实 CUDA/多 worker 中断恢复同样待验证。

完成后保存固定 final.pt，写 complete.json，再删除**仅本类别自己的滚动 latest.pt** 以减少重复 optimizer 存储；
最终权重、连续预测和其他实验文件保持保留。多个训练尝试分别留在 attempt_NNNN，有独立日志和资源统计。
latest 中已含完整恢复状态，而最终 compact 权重只包含非 encoder 状态；encoder 用相同 SHA256 的公开预训练权重严格重载。
smoke 检查严格重载后的预测一致。取消后先确认旧进程已退出再启动同类别；不要同时写同一个类别输出。
Kaggle 临时磁盘丢失时必须取回并解压已保存的恢复包与轻量记录到相同路径；代码不自动上传。

## 7 缓存复评、分批汇总和导出

```bash
# 第一次评价需数据；GT 缓存完整后，可省略数据根和 backbone，不重新前向。
"$PY" -u -m external_baselines.dinomaly_benchmarks evaluate --dataset mvtec --categories all \
  --official-root "$OFFICIAL" --output-dir "$OUT" --device cuda:0 --evaluation-name reeval_v2

# 全部类别评价完成才生成完整 macro；缺类直接失败，不填零或冒充完整结果。
"$PY" -m external_baselines.dinomaly_benchmarks summarize --dataset mvtec --categories all --output-dir "$OUT"
"$PY" -m external_baselines.dinomaly_benchmarks summarize --dataset visa --categories all --output-dir "$OUT"

# 明确选择子集时，scope=selected_categories_macro_NOT_full_dataset。
"$PY" -m external_baselines.dinomaly_benchmarks summarize --dataset mvtec \
  --categories bottle cable --output-dir "$OUT"

# 重评不同名称后汇总对应名称。
"$PY" -m external_baselines.dinomaly_benchmarks summarize --dataset mvtec \
  --categories all --output-dir "$OUT" --evaluation-name reeval_v2

# 分别保存轻量记录、最终权重、连续预测+GT、未完成类别恢复点。
for DATASET in mvtec visa; do
  for KIND in report weights predictions resume; do
    "$PY" -m external_baselines.dinomaly_benchmarks export --dataset "$DATASET" \
      --output-dir "$OUT" --export-kind "$KIND"
  done
done
```

子集汇总保存于 `summaries/<评价名>/selected/<类别列表>/`，不覆盖已完成的全数据集 macro。
同一数据集的分批结果使用相同 OUT 下的类别目录，最后 categories all 读取完整逐类结果重新汇总，
绝不平均不同大小批次的 macro。数据集目录独立，永远不生成 27 类总 macro。
输出 `${OUT}/Dinomaly_<dataset>_<kind>.zip`，仅 report 包包含 JSON/JSONL/CSV/log/txt。
predictions 包是 FP32 NPZ 和 bool GT NPZ；它的逐图索引/空间信息在 report 包。
恢复时按原目录层级解压报告、对应 weights/predictions/resume 包；最终推理还需要独立冻结编码器文件。
不打包虚拟环境、原始数据或热力图，不自动删除预测和最终权重。

```text
OUT/
  mvtec/ 或 visa/
    类别/
      train/identity.json, progress.json, complete.json, final.pt
      train/latest.pt                       # 仅未完成类别保留
      train/attempt_NNNN/config.json, environment.json, resume.json, losses.csv, resources.csv
      predict/v1/identity.json, sample_scores.csv, samples.json, resources.csv, complete.json
      predict/v1/maps/000000.npz             # FP32 连续异常图与固定图像分数
      predict/v1/ground_truth/*.npz, manifest.json
      evaluate/v1/result.json, category_metrics.csv, fixed_fpr.csv, mask_audit.csv, resources.csv, complete.json
    summaries/v1/category_metrics.csv, macro_metrics.csv, fixed_fpr.csv, fixed_fpr_macro.csv, complete.json
    data_check/counts.csv, manifest.json, complete.json
    launcher/config.json, *.log, progress.json, pipeline_status.json
```

## 本轮本地验证范围

本机 Python3.12，无 torch/numpy/Pillow/scipy/pytest/CUDA。没有下载编码器权重，没有安装科学环境，没有正式数据实验。
已执行 `python -m unittest discover -s external_baselines/dinomaly_benchmarks/tests -v`：
5 个标准库检查通过，5 个科学运行时检查明确跳过。
标准库检查覆盖实际官方 commit/构建段、同名模块碰撞、15/12 类双分片、恢复游标边界、CLI 和轻量打包。
CLI help、官方 source 检查通过，官方 tracked 源码未修改。

科学运行时测试已编写，需在上述独立环境实际运行：官方 RGB/GT 精确对照、非方形几何和 VisA 非零 ID/空裁剪、
官方 evaluation_batch 原始推理段与封装异常图/655 像素评分一致、compact 保存重载及缓存输入、
官方 optimizer/scheduler 与 dropout/shuffle 的中途和 epoch 边界恢复一致、更新 5000 边界、类别宏平均/N/A/子集标记。
这些合成测试不代表真实模型已复现成功。待验证：真实编码器严格加载、ViTill CUDA forward/backward、
T4 batch16 显存、真实 smoke 重载一致、Linux 多 worker 中断恢复、完整 15/12 类数据检查和正式实验。

本地实现阶段没有修改已有方法源码、结果或报告，没有连接服务器或启动正式实验。
本同步分支为 `dinomaly-official-single-class`，只包含新入口、固定官方 submodule 和必要的公共源码依赖；
既有方法源码保持不变，不收录数据集、权重、预测、报告或虚拟环境。
