"""Export an RT-DETRv4-Seg checkpoint to an ONNX that DeepStream-Yolo-Seg's
``NvDsInferParseYoloSeg`` parser reads (same contract as avision's
training/export_ltdetrv2_seg.py and training/export_rfdetr_seg.py):

    input   "input"   float32 (B, 3, H, W), RGB, scaled to [0, 1] (net-scale-factor=1/255)
    output  "output"  float32 (B, K, 6 + H/4 * W/4)
        [0:4] x1, y1, x2, y2 in network-input pixels
        [4]   score
        [5]   class id, 0-based against the labels file (category id - --label-offset)
        [6:]  sigmoid mask of *this detection's box*, row-major, H/4 x W/4

    python tools/deployment/export_onnx_seg_deepstream.py \
        -c configs/rtv4_seg/rtv4_seg_hgnetv2_s_handoff.yml \
        -r outputs/rtv4_seg_hgnetv2_s_handoff/best_stg2.pth \
        --labels person clothing

RT-DETR takes [0, 1] input with no mean/std, so nothing is folded into the graph. The
model's masks cover the whole network input; DeepStream stretches a detection's mask
over that detection's box, so each mask is ROI-aligned onto its own box here.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from engine.core import YAMLConfig

PARSER_MASK_STRIDE = 4  # nvdsparseseg_Yolo.cpp: mask_width = netW / 4


class DeepStreamSegWrapper(nn.Module):
    def __init__(self, model, postprocessor, input_size, label_offset):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor
        self.input_h, self.input_w = input_size
        self.label_offset = label_offset

    def forward(self, images):
        batch = images.shape[0]
        # boxes come out in network-input pixels when the "original" size is the input size
        sizes = torch.tensor([[self.input_w, self.input_h]], device=images.device).expand(batch, 2)
        labels, boxes, scores, mask_logits = self.postprocessor(self.model(images), sizes)
        masks = self.crop_masks_to_boxes(mask_logits, boxes)
        return torch.cat([
            boxes,
            scores.unsqueeze(-1),
            (labels - self.label_offset).to(boxes.dtype).unsqueeze(-1),
            masks.sigmoid().flatten(2),
        ], dim=-1)

    def crop_masks_to_boxes(self, mask_logits, boxes):
        """(B, K, h, w) whole-input masks -> (B, K, H/4, W/4) masks of each own box."""
        batch, queries, mask_h, mask_w = mask_logits.shape
        flat = mask_logits.reshape(batch * queries, 1, mask_h, mask_w)
        index = torch.arange(batch * queries, device=boxes.device, dtype=boxes.dtype).unsqueeze(1)
        rois = torch.cat([index, boxes.reshape(batch * queries, 4)], dim=1)
        out_h, out_w = self.input_h // PARSER_MASK_STRIDE, self.input_w // PARSER_MASK_STRIDE
        cropped = roi_align(flat, rois, output_size=(out_h, out_w),
                            spatial_scale=mask_w / self.input_w,  # network px -> mask grid
                            sampling_ratio=-1, aligned=False)
        return cropped.reshape(batch, queries, out_h, out_w)


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)
    if not getattr(cfg.model.decoder, 'mask_dim', 0):
        raise SystemExit('this config has no mask head (DFINETransformer.mask_dim is 0)')

    input_h, input_w = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
    wrapper = DeepStreamSegWrapper(cfg.model.deploy(), cfg.postprocessor.deploy(),
                                   (input_h, input_w), args.label_offset).eval()

    dummy = torch.rand(args.batch_size, 3, input_h, input_w)
    with torch.no_grad():
        probe = wrapper(dummy)
    expected = 6 + (input_h // PARSER_MASK_STRIDE) * (input_w // PARSER_MASK_STRIDE)
    print(f'output {tuple(probe.shape)} (parser expects last dim {expected})')
    assert probe.shape[-1] == expected

    out = args.out or args.resume.replace('.pth', '_ds.onnx')
    torch.onnx.export(
        wrapper, dummy, out,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}} if args.dynamic_batch else None,
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )
    print(f'wrote {out}')

    if args.labels:
        labels_out = out.replace('.onnx', '_labels.txt')
        with open(labels_out, 'w') as f:
            f.write('\n'.join(args.labels) + '\n')
        print(f'wrote {labels_out}')

    if args.simplify:
        import onnx
        import onnxslim
        onnx.save(onnxslim.slim(onnx.load(out)), out)
        print('simplified with onnxslim')

    check_parity(out, dummy, probe)


def check_parity(onnx_path, dummy, probe):
    """Compare onnxruntime with torch per detection. Rows are paired first: detections
    whose scores differ by ~1e-6 can swap places between the two runtimes, which makes
    a position-by-position diff look huge although every detection is identical."""
    import numpy as np
    import onnxruntime as ort
    from scipy.optimize import linear_sum_assignment

    ort_out = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']) \
        .run(None, {'input': dummy.numpy()})[0]
    worst = {'box': 0.0, 'score': 0.0, 'mask': 0.0}
    for t, o in zip(probe.numpy(), ort_out):
        cost = (np.abs(t[:, None, :4] - o[None, :, :4]).sum(-1)
                + 1e3 * np.abs(t[:, None, 4] - o[None, :, 4])
                + 1e3 * (t[:, None, 5] != o[None, :, 5]))
        rows, cols = linear_sum_assignment(cost)
        worst['box'] = max(worst['box'], float(np.abs(t[rows, :4] - o[cols, :4]).max()))
        worst['score'] = max(worst['score'], float(np.abs(t[rows, 4] - o[cols, 4]).max()))
        worst['mask'] = max(worst['mask'], float(np.abs(t[rows, 6:] - o[cols, 6:]).max()))
    print('onnxruntime vs torch, per detection: '
          f"box {worst['box']:.2e} px, score {worst['score']:.2e}, mask {worst['mask']:.2e}")
    if worst['box'] > 0.5 or worst['score'] > 1e-3 or worst['mask'] > 1e-3:
        print('WARNING: the exported graph does not reproduce the torch model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', '-c', required=True)
    parser.add_argument('--resume', '-r', required=True, help='training checkpoint (.pth)')
    parser.add_argument('--out', '-o', default=None)
    parser.add_argument('--label-offset', type=int, default=1,
                        help='subtracted from class ids (1 when category ids start at 1)')
    parser.add_argument('--labels', nargs='*', default=None, help='class names written to <out>_labels.txt')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--dynamic-batch', action='store_true')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--simplify', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    main(args)
