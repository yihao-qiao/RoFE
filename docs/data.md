# Data preparation

Every dataset and weight file is obtained by the user under its own licence. Steps 1 to 3 cover every configuration
in `configs/`.

## 0. Third-party sources and weights

Clone the third-party sources at the pinned commits:

```bash
git clone https://github.com/alanarazi7/MulTaBench && git -C MulTaBench checkout 54e2466006da11504e4ea7583ebb056f5b3d7343
git clone https://github.com/too-z/MultiModalPFN && git -C MultiModalPFN checkout e7a8f93205a9901f199dead7d39b9afc78d1471d
git clone https://github.com/facebookresearch/dinov3 && git -C dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
```

Meta distributes the DINOv3 ViT-B/16 LVD-1689M weights on request. The file is
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`, SHA-256
`73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`.

The TabPFN 2.5 and 3.5 classifiers (`docs/third_party.md`) go in `weights/`, next to the TabICLv2 checkpoint that
`scripts/download_weights.py` writes there.

## 1. MulTaBench image and text tasks

CPU, about 30 to 60 minutes plus downloads. Run in `envs/multabench`, with TabSTAR 1.1.15 from PyPI:

```bash
echo "tabstar==1.1.15 --hash=sha256:9c4bf7392101813aac9c0209edbe664e1091c4463ddb4628869a9569d75c1d4e" > tabstar.txt
uv pip install --no-deps --require-hashes -r tabstar.txt
export KAGGLEHUB_CACHE=/path/to/kagglehub
python scripts/prepare_data.py multabench --set all --source MulTaBench --data DATA
```

Each task keeps MulTaBench's first five splits and at most 10,000 training rows per fold. A fold with more than ten
classes keeps ten classes, drawn with a fixed seed per task and fold.

## 2. The four MultiModalPFN tasks

CPU, a few minutes each, main environment. Kaggle credentials are needed, and PetFinder requires accepting the
competition rules.

```bash
python scripts/prepare_data.py download --dataset pad --output raw/pad
python scripts/prepare_data.py download --dataset cbis --output raw/cbis
python scripts/prepare_data.py download --dataset petfinder --output raw/petfinder
python scripts/prepare_data.py mmpfn --task pad_ufes_20 --root raw/pad --data DATA
python scripts/prepare_data.py mmpfn --task petfinder --root raw/petfinder --data DATA
python scripts/prepare_data.py mmpfn --task cbis_mass --root raw/cbis --data DATA
python scripts/prepare_data.py mmpfn --task cbis_calc --root raw/cbis --data DATA
```

The PAD step fetches the authors' metadata (SHA-256 pinned) and checks it against the Kaggle copy. The two CBIS-DDSM
tasks assign their three views by DICOM role (`--cbis-policy dicom_metadata`, the default).

## 3. Embeddings

One GPU, about 1 to 2 GPU-hours per encoder for all tasks.

```bash
python scripts/extract_features.py --encoder dinov3 --dinov3-source dinov3 \
    --checkpoint dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --data DATA --tasks <image tasks>  # envs/dinov3
python scripts/extract_features.py --encoder siglip --data DATA --tasks <image tasks>            # envs/features
python scripts/extract_features.py --encoder electra --data DATA --tasks <text tasks>            # envs/features
python scripts/extract_features.py --encoder e5 --data DATA --tasks <text tasks>                 # envs/features
```

- DINOv3 gives the class token after the final norm. Images are resized to 256 x 256 with ImageNet normalization.
- SigLIP so400m-patch14-384 gives its pooled output.
- ELECTRA-base and E5-base-v2 give the mean of the last hidden states over the tokens. Texts have at most 512 tokens,
  and E5 gets the prefix `query: `.
- Everything runs in FP32 with TF32 disabled.

## Layout

The prepared layout is

```
DATA/tasks/<TASK>/seed{0..4}.json, audit.json, images.jsonl or texts.jsonl
DATA/features/<encoder>/<TASK>/features.npy, row_ids.json, manifest.json
```
