# Intel Scene Classification Dataset

This homework needs a scene classification image dataset. The best fit is the Intel Image Classification / Natural Scene Classification dataset:

- classes: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`
- usual layout: `seg_train/seg_train/<class>`, `seg_test/seg_test/<class>`, `seg_pred/seg_pred`
- image size: about `150 x 150`

The original dataset is commonly available from Kaggle, but Kaggle often requires an account and API token. For this local environment, the tested no-login mirror is:

```text
https://hf-mirror.com/datasets/resolverkatla/Intel-Image-Classification/resolve/main/intel-image-classification.zip
```

The full zip is about 346 MB. In the current network it connects without a proxy, but download speed can be unstable, so use the resumable script:

```bash
bash scripts/download_intel_scenes.sh
```

If a faster mirror is available, override the URL:

```bash
INTEL_SCENES_URL="https://example.com/intel-image-classification.zip" bash scripts/download_intel_scenes.sh
```

Expected training directories after extraction:

```text
data/intel-scenes/seg_train/seg_train/buildings
data/intel-scenes/seg_train/seg_train/forest
data/intel-scenes/seg_train/seg_train/glacier
data/intel-scenes/seg_train/seg_train/mountain
data/intel-scenes/seg_train/seg_train/sea
data/intel-scenes/seg_train/seg_train/street
data/intel-scenes/seg_test/seg_test/buildings
data/intel-scenes/seg_test/seg_test/forest
data/intel-scenes/seg_test/seg_test/glacier
data/intel-scenes/seg_test/seg_test/mountain
data/intel-scenes/seg_test/seg_test/sea
data/intel-scenes/seg_test/seg_test/street
```
