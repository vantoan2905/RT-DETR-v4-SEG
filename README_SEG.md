# RT-DETRv4-Seg: instance segmentation on RT-DETRv4

Fork of [RT-DETRv4](https://github.com/AIMIA-PKU/RT-DETRv4) at upstream commit
`55fefaa` (git history removed). Adds a query-based mask head; with
`DFINETransformer.mask_dim: 0` the detector behaves exactly as upstream.

## How it works

```
backbone ─► HybridEncoder ─► [S3, S4, S5] ─► DFINETransformer ─► boxes, scores
                               │                     │ query hidden states (per layer)
                               ▼                     ▼
                      MaskFeatureHead           mask_embed MLP
                      (S3 ↑2 → stride 4)         (B, Q, D)
                        (B, D, H/4, W/4)  ──einsum──►  mask logits (B, Q, H/4, W/4)
```

* Mask features come from the encoder's highest-resolution output (stride 8),
  upsampled once to stride 4: 160×160 for a 640 input, the grid
  DeepStream-Yolo-Seg's parser reads.
* Every decoder layer's queries get a mask embedding (one shared MLP), so masks are
  supervised on the final layer and on each auxiliary layer. Denoising queries,
  encoder proposals and the first-layer "pre" head have no masks.
* Masks are computed only where needed: for matched queries in the loss, and for
  the top-k detections in the post-processor.
* Loss `masks`: per-pixel BCE (`loss_mask`) + soft dice (`loss_dice`) against the GT
  masks resized to the mask grid, weights 5 / 5. Matching is unchanged (class + box).

## Files changed / added

| File | Change |
|---|---|
| `engine/rtv4/mask_head.py` | **new**: `MaskFeatureHead`, mask logits, BCE / dice |
| `engine/rtv4/dfine_decoder.py` | decoder returns query hidden states; `mask_dim`, `mask_act` params; `pred_mask_embed` + `mask_features` in outputs |
| `engine/rtv4/rtv4_criterion.py` | `masks` loss |
| `engine/rtv4/postprocessor.py` | top-k masks (eval: binary at original size; deploy: logits). `num_top_masks`, `mask_threshold`. Also fixes the softmax branch gathering un-converted boxes |
| `engine/rtv4/hybrid_encoder.py` | upstream bug fix: the cached eval `pos_embed` is not a buffer and stayed on the CPU |
| `engine/data/dataset/coco_dataset.py` | masks from polygons, uncompressed **and compressed RLE**, empty segmentations |
| `engine/data/transforms/mosaic.py` | masks are pasted onto the mosaic canvas (they were concatenated unplaced) |
| `engine/data/dataloader.py` | mixup concatenates masks; multi-scale resizes masks (was `NotImplementedError`) |
| `configs/base/rtv4_seg.yml` | **new**: turns on the head, losses, `segm` evaluation |
| `configs/dataset/handoff_fullframe_seg.yml` | **new**: avision handoff full-frame dataset |
| `configs/rtv4_seg/rtv4_seg_hgnetv2_s_handoff.yml` | **new**: RT-DETRv4-S-Seg, 80 epochs |
| `tools/deployment/export_onnx_seg_deepstream.py` | **new**: ONNX for `NvDsInferParseYoloSeg` |

## Train

Uses the avision venv (`calflops` was added to it). HGNetv2 ImageNet weights are
downloaded into `weight/hgnetv2/` on first run.

```bash
cd RT-DETRv4
source ../avision/.venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python train.py -c configs/rtv4_seg/rtv4_seg_hgnetv2_s_handoff.yml --use-amp --seed 0
```

Evaluation reports COCO `bbox` and `segm` AP each epoch. Other sizes: copy the config
and swap `../dfine/dfine_hgnetv2_s_coco.yml` for `_m_` / `_l_` / `_x_`.

The DINOv3 teacher of RT-DETRv4 is off (its repo and weights are not here; the distill
loss is 0). Enable it by copying the `teacher_model`, `HybridEncoder.distill_teacher_dim`
and `RTv4Criterion.distill_*` blocks from `configs/rtv4/rtv4_hgnetv2_s_coco.yml`.

### Dataset notes

* COCO layout with masks in `segmentation` (polygons or RLE).
* Category ids are used as class indices (`remap_mscoco_category: False`), so
  `num_classes` = max category id + 1. For the handoff set (1 = person, 2 = clothing)
  that is 3, with class 0 unused; COCO evaluation then works on the original ids.

## Export for DeepStream

```bash
python tools/deployment/export_onnx_seg_deepstream.py \
    -c configs/rtv4_seg/rtv4_seg_hgnetv2_s_handoff.yml \
    -r outputs/rtv4_seg_hgnetv2_s_handoff/best_stg2.pth \
    --labels person clothing
```

Output `output` (B, 300, 6 + 160·160): `x1 y1 x2 y2` in network pixels, score, class
id (category id − `--label-offset`, default 1 → 0-based), and the sigmoid mask
ROI-aligned to the detection's own box. Input `input` is RGB in [0, 1]
(`net-scale-factor=1/255`, no mean/std). The script checks onnxruntime against torch
per detection.
