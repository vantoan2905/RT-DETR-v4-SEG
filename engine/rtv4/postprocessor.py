"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register
from .mask_head import compute_mask_logits


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        num_top_masks=100,
        mask_threshold=0.5,
    ) -> None:
        """num_top_masks / mask_threshold apply only when the model has a mask head: the
        highest-scoring ``num_top_masks`` detections get a binary mask at the original
        image size (COCO evaluates at most 100 detections per image)."""
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.num_top_masks = num_top_masks
        self.mask_threshold = mask_threshold
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            index = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).repeat(scores.shape[0], 1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        has_masks = 'pred_mask_embed' in outputs

        # TODO for onnx export
        if self.deploy_mode:
            if has_masks:
                # (B, K, H/4, W/4) logits over the whole network input, one per detection
                embed = outputs['pred_mask_embed'].gather(
                    dim=1, index=index.unsqueeze(-1).repeat(1, 1, outputs['pred_mask_embed'].shape[-1]))
                return labels, boxes, scores, compute_mask_logits(embed, outputs['mask_features'])
            return labels, boxes, scores

        masks = self._masks_for_eval(outputs, index, orig_target_sizes) if has_masks else None

        # TODO
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor(
                [mscoco_label2category[int(x.item())] for x in labels.flatten()],
                device=boxes.device
            ).reshape(labels.shape)

        results = []
        for i, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            result = dict(labels=lab, boxes=box, scores=sco)
            if masks is not None:
                # top-k is sorted by score, so masks[i][k] belongs to detection k
                result['masks'] = masks[i]
            results.append(result)

        return results

    @torch.no_grad()
    def _masks_for_eval(self, outputs, index, orig_target_sizes):
        """Binary masks (K, 1, H, W) on CPU at each image's original size, for the
        ``num_top_masks`` best detections. The network input is the whole image resized
        (no letterbox), so the mask grid maps onto the original image by plain resizing."""
        k = min(self.num_top_masks, index.shape[1])
        embed_dim = outputs['pred_mask_embed'].shape[-1]
        embed = outputs['pred_mask_embed'].gather(dim=1, index=index[:, :k].unsqueeze(-1).repeat(1, 1, embed_dim))
        logits = compute_mask_logits(embed, outputs['mask_features'])
        masks = []
        for logit, size in zip(logits, orig_target_sizes):
            w, h = int(size[0]), int(size[1])
            prob = F.interpolate(logit[None].float(), size=(h, w), mode='bilinear', align_corners=False)[0]
            masks.append((prob.sigmoid() > self.mask_threshold).unsqueeze(1).cpu())
        return masks


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
