# industrial-ad-benchmarks

工业异常检测外部基线、数据适配和评价代码，面向 Kaggle 环境同步。

当前交付：**INP-Former官方单类full-shot配置的BTAD外部复评**。
MVTec AD / VisA 已有独立数据入口、续跑流程和结果打包；正式运行结果尚待Kaggle执行。
说明见 [MVTec AD / VisA README](external_baselines/inpformer_benchmarks/README.md)。

本分支新增 **Dinomaly 官方单类版本（MVTec AD / VisA）**，说明与分阶段命令见
[Dinomaly README](external_baselines/dinomaly_benchmarks/README.md)。
官方源码作为独立固定版本 submodule；已有 INP-Former 源码与入口保持不变。

获取 Dinomaly 独立分支：

```bash
git clone --branch dinomaly-official-single-class --recurse-submodules https://github.com/YYK001/industrial-ad-benchmarks.git
cd industrial-ad-benchmarks
```

## 获取代码

```bash
git clone --recurse-submodules https://github.com/YYK001/industrial-ad-benchmarks.git
cd industrial-ad-benchmarks
```

已有 checkout 更新：

```bash
git pull --ff-only
git submodule update --init --recursive
```

官方 INP-Former 以 submodule 固定在
`17d265381d9b323a2ef6e05aab0665a85edebe84`，保留官方源码不变。
不要使用 `git submodule update --remote` 漂移到其他版本。

## Kaggle 使用

数据集和 `dinov2_vitb14_reg4_pretrain.pth` 由你单独上传到 Kaggle Input。
使用已有兼容环境，不自动安装依赖、不自动启动训练。

先检查数据（路径按实际 Kaggle Input 修改）：

```bash
python -m external_baselines.inpformer_btad.run check \
  --dataset-root /kaggle/input/btad/BTech_Dataset_transformed \
  --output-dir /kaggle/working/inpformer_check
```

完整依赖、固定协议和独立 `check / smoke / metrics-smoke / train / predict / evaluate` 命令见
[INP-Former BTAD README](external_baselines/inpformer_btad/README.md)。
其中旧示例的项目目录 `experiment` 应替换成你实际克隆的 `industrial-ad-benchmarks`。

测试：

```bash
python -m pytest external_baselines/inpformer_btad/test_local.py -q
```

本地仅验证 CPU 小规模逻辑；完整预训练模型、T4 FP32 batch=16 的资源开销和真实 CUDA
指标仍待验证。测试通过不代表正式复现成功。

## 仓库范围

- `external_baselines/dinomaly_benchmarks`：Dinomaly 单类训练、恢复、缓存复评与双 T4 入口。
- `external_baselines/dinomaly_official`：锁定 `1f252be03a918789b19848f0ca37166e7c28dada` 的官方 submodule。
- 新增的 PatchCore 评价工具、DeSTSeg 资源/进程工具和 DINOv3 路径下的公共依赖只供复用函数，
  不代表启用这些方法的检测模型；详细复用关系见 Dinomaly README。
- `external_baselines/inpformer_btad`：独立 BTAD 入口与测试。
- `external_baselines/inpformer_external`：复用已有模型构建、训练和工具函数。
- `external_baselines/INP-Former`：固定版本官方 submodule。
- `DINOv3`、`DINOv2` 和 `external_baselines/superadd_external`：现有数据/指标模块所需的
  Python 导入依赖。保留原模块路径，不代表在本协议中启用 DINOv3 或 SuperADD 检测模型。
- `tools/sync_sources.py`：从研究工作区按本地 Python 导入关系导出源码；会覆盖同名源码。

不收录数据集、预训练权重、checkpoint、连续预测、历史报告或本地验证输出。
官方代码的版权/许可见 submodule 中的原始说明；本仓库不为第三方代码重新授权。
