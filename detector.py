"""
AI counting pipeline used by the game.

1. Grounding DINO  reads a word like "flower" and draws a box around everything that looks like one.
2. CLIP            double-checks each box ("is this really a flower?") and throws out mistakes.
3. SAM             traces the exact outline of every object that is left, for the coloured picture.

The final count is the number of boxes that survive steps 1 and 2.
"""
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# "base" is noticeably more accurate than "tiny" (slower, ~900 MB). Switch back to tiny if your laptop struggles.
DINO_ID = "IDEA-Research/grounding-dino-base"
SAM_ID = "facebook/sam-vit-base"
CLIP_ID = "openai/clip-vit-base-patch32"

MAX_SIDE = 1333          # images are shrunk to this size first (bigger = small objects are easier to find)
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 200, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
           (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128), (128, 128, 0)]


def shrink(image, max_side=MAX_SIDE):
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return image
    return image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def _font(size):
    try:
        return ImageFont.load_default(size=size)       # Pillow 10.1+
    except TypeError:
        return ImageFont.load_default()


def draw_overlay(image, boxes, masks=None):
    """Draw a thick box and a big numbered badge on every object the AI counted,
    plus a light colour fill from SAM, so people can check each one."""
    base = image.convert("RGBA")
    w, h = base.size
    if masks is not None and len(masks):
        layer = np.zeros((h, w, 4), dtype=np.uint8)
        for i, m in enumerate(masks):
            layer[m.astype(bool)] = (*PALETTE[i % len(PALETTE)], 70)
        base = Image.alpha_composite(base, Image.fromarray(layer, "RGBA"))
    out = base.convert("RGB")
    d = ImageDraw.Draw(out)
    line = max(3, round(min(w, h) / 180))
    font = _font(max(16, round(min(w, h) / 28)))
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = [float(v) for v in b]
        color = PALETTE[i % len(PALETTE)]
        d.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=line + 2)   # white edge so boxes show on any background
        d.rectangle([x0, y0, x1, y1], outline=color, width=line)
        label = str(i + 1)
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(4, line + 1)
        bx0, by0 = x0, max(0, y0 - th - 2 * pad)          # badge sits on top of the box
        if by0 == 0:
            by0 = y0                                       # no room above: put it inside
        d.rectangle([bx0, by0, bx0 + tw + 2 * pad, by0 + th + 2 * pad], fill=color, outline=(255, 255, 255), width=2)
        d.text((bx0 + pad - tb[0], by0 + pad - tb[1]), label, fill=(255, 255, 255), font=font,
               stroke_width=2, stroke_fill=(0, 0, 0))
    return out


class Counter:
    """Loads the three models once and counts objects in an image."""

    def __init__(self, use_clip=True, use_sam=True):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  CLIPModel, CLIPProcessor, SamModel, SamProcessor)
        self.torch = torch
        torch.set_grad_enabled(False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_clip, self.use_sam = use_clip, use_sam

        self.dino_proc = AutoProcessor.from_pretrained(DINO_ID)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(self.device).eval()
        if use_clip:
            self.clip_proc = CLIPProcessor.from_pretrained(CLIP_ID)
            self.clip = CLIPModel.from_pretrained(CLIP_ID).to(self.device).eval()
        if use_sam:
            self.sam_proc = SamProcessor.from_pretrained(SAM_ID)
            self.sam = SamModel.from_pretrained(SAM_ID).to(self.device).eval()

    # ---------- Step 1: Grounding DINO ----------
    def detect_raw(self, image, obj, box_threshold=0.30, text_threshold=0.25):
        """Run Grounding DINO once. Returns every box above box_threshold with its score."""
        text = obj.lower().strip().rstrip(".") + "."          # DINO wants lowercase text ending in "."
        inputs = self.dino_proc(images=image, text=text, return_tensors="pt").to(self.device)
        outputs = self.dino(**inputs)
        kw = dict(text_threshold=text_threshold, target_sizes=[image.size[::-1]])
        try:      # newer transformers
            res = self.dino_proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=box_threshold, **kw)[0]
        except TypeError:   # older transformers
            res = self.dino_proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=box_threshold, **kw)[0]
        boxes, scores = res["boxes"].cpu(), res["scores"].cpu()
        if len(boxes):
            # Drop boxes around almost the whole picture
            w, h = image.size
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep = area < 0.8 * w * h
            boxes, scores = boxes[keep], scores[keep]
        return boxes, scores

    def clean(self, boxes, scores):
        """Remove duplicates and "group" boxes. Returns the indices of the boxes to keep."""
        from torchvision.ops import nms
        torch = self.torch
        if not len(boxes):
            return torch.zeros(0, dtype=torch.long)
        k = nms(boxes, scores, iou_threshold=0.5)            # duplicate boxes on the same object
        b = boxes[k]
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        drop = torch.zeros(len(b), dtype=torch.bool)
        for i in range(len(b)):                              # a box wrapping 2+ other boxes is a group
            x0 = torch.maximum(b[i, 0], b[:, 0]); y0 = torch.maximum(b[i, 1], b[:, 1])
            x1 = torch.minimum(b[i, 2], b[:, 2]); y1 = torch.minimum(b[i, 3], b[:, 3])
            inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
            inside = inter / area.clamp(min=1) > 0.85
            inside[i] = False
            if inside.sum() >= 2:
                drop[i] = True
        return k[~drop]

    def detect(self, image, obj, box_threshold=0.30, text_threshold=0.25):
        boxes, scores = self.detect_raw(image, obj, box_threshold, text_threshold)
        idx = self.clean(boxes, scores)
        return boxes[idx], scores[idx]

    # ---------- Step 2: CLIP double-check ----------
    def verify(self, image, boxes, obj, min_prob=0.35):
        return self.clip_probs(image, boxes, obj) >= min_prob

    def clip_probs(self, image, boxes, obj):
        """How sure CLIP is that each box really is the object (0 to 1)."""
        torch = self.torch
        if not len(boxes):
            return torch.zeros(0)
        labels = [f"a photo of a {obj}", "a photo of something else", "a blurry background"]
        w, h = image.size
        crops = []
        for b in boxes.tolist():
            x0, y0, x1, y1 = b
            px, py = (x1 - x0) * 0.1, (y1 - y0) * 0.1      # a little context around each object
            crops.append(image.crop((max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))))
        probs = []
        for i in range(0, len(crops), 32):
            inp = self.clip_proc(text=labels, images=crops[i:i + 32], return_tensors="pt", padding=True).to(self.device)
            probs.append(self.clip(**inp).logits_per_image.softmax(-1)[:, 0].cpu())
        return torch.cat(probs)

    # ---------- Step 3: SAM outlines ----------
    def segment(self, image, boxes):
        if not len(boxes):
            return np.zeros((0, image.size[1], image.size[0]), dtype=bool)
        inp = self.sam_proc(image, return_tensors="pt").to(self.device)
        embeddings = self.sam.get_image_embeddings(inp["pixel_values"])   # look at the image once
        masks = []
        box_list = boxes.tolist()
        for i in range(0, len(box_list), 32):
            b = self.sam_proc(image, input_boxes=[box_list[i:i + 32]], return_tensors="pt").to(self.device)
            out = self.sam(image_embeddings=embeddings, input_boxes=b["input_boxes"], multimask_output=False)
            m = self.sam_proc.image_processor.post_process_masks(
                out.pred_masks.cpu(), b["original_sizes"].cpu(), b["reshaped_input_sizes"].cpu())[0]
            masks.append(m[:, 0].numpy())
        return np.concatenate(masks)

    def count(self, image, obj, box_threshold=0.30, clip_min=0.35, draw=True):
        t0 = time.time()
        image = shrink(image.convert("RGB"))
        boxes, _ = self.detect(image, obj, box_threshold=box_threshold)
        found = len(boxes)
        removed = 0
        if self.use_clip and clip_min > 0 and len(boxes):
            keep = self.verify(image, boxes, obj, min_prob=clip_min)
            removed = int((~keep).sum())
            boxes = boxes[keep]
        if not draw:     # quick mode for tuning: just the number
            return {"count": int(len(boxes)), "found_by_dino": found, "removed_by_clip": removed}
        masks = self.segment(image, boxes) if self.use_sam else None
        return {
            "count": int(len(boxes)),
            "boxes": [[round(v, 1) for v in b] for b in boxes.tolist()],
            "overlay": draw_overlay(image, boxes.tolist(), masks),
            "seconds": round(time.time() - t0, 1),
            "found_by_dino": found,
            "removed_by_clip": removed,
        }


    def count_best(self, image, obj, target, thresholds, clip_levels):
        """Find the sensitivity settings whose count is closest to `target`, FAST:
        DINO and CLIP each run only once; every setting is then just a filter on their scores."""
        torch = self.torch
        t0 = time.time()
        image = shrink(image.convert("RGB"))
        boxes, scores = self.detect_raw(image, obj, box_threshold=min(thresholds))
        if len(boxes) > 300:                                   # keep the 300 most confident
            top = scores.argsort(descending=True)[:300]
            boxes, scores = boxes[top], scores[top]
        probs = self.clip_probs(image, boxes, obj) if (self.use_clip and len(boxes)) else torch.ones(len(boxes))
        best, table = None, []
        for cm in clip_levels:
            for t in thresholds:
                sel = torch.nonzero(scores >= t).flatten()
                idx = sel[self.clean(boxes[sel], scores[sel])] if len(sel) else sel
                found = len(idx)
                if cm > 0:
                    idx = idx[probs[idx] >= cm]
                n = len(idx)
                table.append((t, cm, n))
                if best is None or abs(n - target) < abs(best[2] - target):
                    best = (t, cm, n, idx, found)
        t, cm, n, idx, found = best
        final = boxes[idx]
        masks = self.segment(image, final) if self.use_sam else None
        return {
            "count": int(n),
            "boxes": [[round(v, 1) for v in b] for b in final.tolist()],
            "overlay": draw_overlay(image, final.tolist(), masks),
            "seconds": round(time.time() - t0, 1),
            "found_by_dino": int(found),
            "removed_by_clip": int(found - n),
            "threshold": t, "clip_min": cm,
            "table": table,
        }


class DemoCounter:
    """Fake AI for trying the game without downloading models (set DEMO_MODE = True in app.py)."""

    def count(self, image, obj, box_threshold=0.30, true_count=None):
        import random
        time.sleep(3)
        image = shrink(image.convert("RGB"))
        n = max(0, (true_count or 10) + random.choice([-2, -1, 0, 0, 1]))
        return {"count": n, "overlay": image, "seconds": 3.0, "found_by_dino": n, "removed_by_clip": 0}
