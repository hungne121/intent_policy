"""Perception models, registered by name for configs/pif/habit.yaml `habit.perception`.

pose   (rgb) -> keypoints [133, 2] px, scores [133]        COCO-WholeBody layout
depth  (rgb) -> depth [H, W] m (NaN = no value), focal px  metric, camera intrinsics estimated by the model
detect (rgb) -> [{box [4], score, label}]                   open-vocabulary, `prompts` given per task
Models are used as released (no fine-tuning).
"""
from __future__ import annotations
import numpy as np

from pif.config import register


@register('pose', 'rtmw')
class RTMW:
    """RTMW (Jiang, Xie, Li 2024) via rtmlib, ONNX weights of OpenMMLab. Mode `performance`: YOLOX-m + RTMW 384x288.
    The person kept = the detection with the highest summed body score (HABIT has one human)."""

    def __init__(self, mode: str = 'performance', device: str = 'cpu'):
        from rtmlib import Wholebody
        self.model = Wholebody(mode=mode, backend='onnxruntime', device=device)

    def __call__(self, rgb):
        kp, sc = self.model(rgb[..., ::-1])               # rtmlib expects BGR
        if len(kp) == 0:
            return np.zeros((133, 2), np.float32), np.zeros(133, np.float32)
        best = int(np.argmax(sc[:, :17].sum(1)))
        return kp[best].astype(np.float32), sc[best].astype(np.float32)


@register('depth', 'moge2')
class MoGe2:
    """MoGe-2 (Wang et al. 2025, NeurIPS; github.com/microsoft/MoGe, MIT). ViT-S: 35M parameters, 1.2 GB VRAM."""

    def __init__(self, model: str, resolution_level: int = 9, fp16: bool = True, device: str = 'cuda'):
        import torch
        from moge.model.v2 import MoGeModel
        self.torch, self.device, self.level = torch, device, resolution_level
        self.fp16 = fp16 and device != 'cpu'
        self.model = MoGeModel.from_pretrained(model).to(device).eval()

    def __call__(self, rgb):
        x = self.torch.tensor(rgb / 255.0, dtype=self.torch.float32, device=self.device).permute(2, 0, 1)
        with self.torch.inference_mode():
            out = self.model.infer(x, use_fp16=self.fp16, resolution_level=self.level)
        depth = out['depth'].float().cpu().numpy()
        depth[~np.isfinite(depth)] = np.nan
        return depth, float(out['intrinsics'][0, 0]) * rgb.shape[1]     # intrinsics are normalised by the image size


@register('depth', 'depth_pro')
class DepthPro:
    """Depth Pro (Bochkovskii et al. 2025, ICLR), HF transformers. Needs more than 4 GB of VRAM."""

    def __init__(self, model: str = 'apple/DepthPro-hf', device: str = 'cuda', fp16: bool = True):
        import torch
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor
        self.torch, self.device = torch, device
        self.dtype = torch.float16 if fp16 and device != 'cpu' else torch.float32
        self.proc = DepthProImageProcessor.from_pretrained(model)
        self.model = DepthProForDepthEstimation.from_pretrained(model, dtype=self.dtype).to(device).eval()

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        with self.torch.inference_mode():
            x = self.proc(images=rgb, return_tensors='pt').to(self.device)
            out = self.model(pixel_values=x['pixel_values'].to(self.dtype))
            post = self.proc.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
        return post['predicted_depth'].float().cpu().numpy(), float(post['focal_length'])


@register('detect', 'grounding_dino')
class GroundingDINO:
    """Grounding DINO (Liu et al. 2024, ECCV), HF transformers. `prompts`: the object names to find."""

    def __init__(self, model: str, prompts: list[str], box_threshold: float = 0.35, text_threshold: float = 0.25,
                 device: str = 'cuda'):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch, self.device = torch, device
        self.thr, self.text_thr = box_threshold, text_threshold
        self.labels = list(prompts)
        self.text = '. '.join(self.labels) + '.'
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model).to(device).eval()

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        with self.torch.inference_mode():
            x = self.proc(images=rgb, text=self.text, return_tensors='pt').to(self.device)
            out = self.model(**x)
            r = self.proc.post_process_grounded_object_detection(out, x['input_ids'], threshold=self.thr,
                                                                 text_threshold=self.text_thr, target_sizes=[(h, w)])[0]
        names = r.get('text_labels', r.get('labels'))
        dets = []
        for box, score, name in zip(r['boxes'].cpu().numpy(), r['scores'].cpu().numpy(), names):
            label = next((l for l in self.labels if l in str(name) or str(name) in l), None) if name else None
            if label is not None:
                dets.append(dict(box=box.astype(np.float32), score=float(score), label=label))
        return dets
