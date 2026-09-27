# INP-Former single-class full-shot external-dataset evaluation

This directory evaluates the unmodified official INP-Former implementation on
RobustAD. It is an **INP-Former official single-class full-shot protocol
evaluation on an external dataset**, not an original paper benchmark and not a
cross-dataset zero-shot transfer.

## Locked upstream protocol

- official repository commit: `17d265381d9b323a2ef6e05aab0665a85edebe84`;
- official seed: `1` (hard-coded by `INP_Former_Single_Class.py`);
- encoder: `dinov2reg_vit_base_14`;
- resize 448, center crop 392, six INP tokens;
- batch size 16 and 200 epochs;
- official loss, optimizer, scheduler, decoder and frozen encoder behavior;
- official 256x256 Gaussian-smoothed anomaly map and mean top-1% image score;
- official `adeval==1.1.0` pixel AUROC/AUPR/AUPRO implementation.

Formal runs retain FP32, deterministic cuDNN behavior and the official batch
size 16. Performance-only plumbing uses eight DataLoader workers, pinned host
memory, persistent workers, prefetching and non-blocking host-to-device copies.
The official script's debugging-only `CUDA_LAUNCH_BLOCKING=1` is disabled and
the choice is recorded in `protocol.json`.

The external runner imports these components from the clean upstream checkout.
It does not edit upstream source. Git commands are intentionally absent from
the runner; commit identity is read directly from `.git/HEAD` metadata.

## Two separate result families

### Native full-shot (primary ranking result)

Every source-domain normal training image is used to train a separate model for
MetalParts, PCB and PiledBags. Report Image AUROC/AUPR for all groups and Pixel
AUROC/AUPR/AUPRO only for MetalParts and PCB. This protocol does not define or
report deployment normal FP.

### SCRS audit (thresholded robustness result)

Before training, each category's normal training pool is split deterministically
into 80% fit and 20% calibration using split seed 42. The model still uses the
official training seed 1. Only the fit split trains the model. The q95 threshold
comes only from the unseen source-normal calibration split, and test-domain
normal FP is reported as `normal_image_FP_calibration_q95`.

Native and SCRS results must never be pooled into one table column or described
as the same training protocol.

## RobustAD rules

- each category has an independent model;
- all source and target shifts remain separate evaluation groups (19 total);
- PiledBags is image-only and never enters a pixel metric or pixel aggregate;
- `metadata.jsonl` is authoritative for image labels and mask pairing;
- the audit rejects unreadable files, missing/raw-empty masks, normal rows with
  masks, and exact train/evaluation content overlap;
- anomalies whose valid mask is completely removed by the official center crop
  remain in evaluation and are recorded as preprocessing warnings;
- Native output uses no test-derived threshold metrics such as F1-max.

## Server prerequisites

Do not install into the server base environment. Before creating an isolated
environment or starting a run, inspect resources:

```bash
df -h
free -h
nvidia-smi
```

The upstream requirements request Python 3.8.12, PyTorch 2.0.0+cu118 and
torchvision 0.15.1+cu118. Environment creation and installation should happen
only after disk-space confirmation.

The required backbone is:

```text
external_baselines/INP-Former/backbones/weights/dinov2_vitb14_reg4_pretrain.pth
```

Official download URL:

```text
https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
```

Its SHA256 must be computed after download and passed as
`INPFORMER_BACKBONE_SHA256`. The runner refuses an unverified checkpoint and
will not trigger the upstream automatic download path.

Approved official checkpoint SHA256:

```text
73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71
```

## Run order

The audit is CPU-only. It hashes image content and may take several minutes:

```bash
export INPFORMER_PYTHON=/root/miniconda3/envs/inpformer/bin/python
bash /root/autodl-tmp/DINO-NVS/external_baselines/inpformer_external/run_inpformer.sh audit
```

Do not run a GPU smoke while SubspaceAD is active. After it finishes, run the
single-category Native smoke first:

```bash
export INPFORMER_BACKBONE_SHA256=<approved_sha256>
bash /root/autodl-tmp/DINO-NVS/external_baselines/inpformer_external/run_inpformer.sh native_smoke
```

After reviewing the smoke output, run Native. SCRS is a later, separately
labelled experiment:

```bash
bash /root/autodl-tmp/DINO-NVS/external_baselines/inpformer_external/run_inpformer.sh native
bash /root/autodl-tmp/DINO-NVS/external_baselines/inpformer_external/run_inpformer.sh scrs
```

The shell wrapper deliberately does not use `set -e` or remote PowerShell.
