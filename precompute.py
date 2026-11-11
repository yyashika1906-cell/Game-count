"""
Run the AI on every picture ahead of time and save the results in results/:
  results/<id>.png   the picture with a numbered box on every object the AI found
  results/<id>.json  the AI's count and box positions

  python precompute.py          AUTO: tries several sensitivity settings per picture and keeps
                                the one whose count matches "true_count" (or comes closest)
  python precompute.py --tune   just print a table of counts per setting, nothing saved
  python precompute.py --fixed  use each picture's "threshold" from images.json as is

Per-picture settings in images.json (all optional):
  "threshold": 0.30   lower = the AI finds MORE objects, higher = FEWER
  "clip_min":  0.35   how sure CLIP must be to keep a box; lower = keeps more, 0 = skip CLIP
  "prompt":    "red strawberry"   what the AI searches for, if different from "object"
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

from detector import Counter

ROOT = Path(__file__).parent
IMG_DIR, RES_DIR = ROOT / "images", ROOT / "results"
RES_DIR.mkdir(exist_ok=True)
MODE = "tune" if "--tune" in sys.argv else ("fixed" if "--fixed" in sys.argv else "auto")
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
CLIP_LEVELS = [0.35, 0.20, 0.0]          # tried in order if no threshold gives the exact count

items = json.loads((ROOT / "images.json").read_text(encoding="utf-8"))
print("Loading models (first time downloads about 1.5 GB)...")
counter = Counter()
summary = []

for item in items:
    path = IMG_DIR / item["file"]
    if not path.exists():
        print(f"SKIP  {item['file']}: file not found in images/")
        continue
    print(f"...   {item['id']}: looking for {item.get('prompt', item['object'])}", flush=True)
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    prompt = item.get("prompt", item["object"])
    truth = item.get("true_count")

    if MODE == "tune":
        cm = item.get("clip_min", 0.35)
        r = counter.count_best(image, prompt, truth or 0, THRESHOLDS, [cm])
        row = [f"{t:.2f}->{n:>3}" for t, _, n in r["table"]]
        print(f"{item['id']:<8} real {truth!s:>4} | " + "  ".join(row))
        continue

    if MODE == "fixed" or truth is None:
        t, cm = item.get("threshold", 0.30), item.get("clip_min", 0.35)
        r = counter.count(image, prompt, box_threshold=t, clip_min=cm)
    else:
        # DINO and CLIP run once; all the sensitivity settings are tried on their scores
        clip_levels = [item["clip_min"]] if "clip_min" in item else CLIP_LEVELS
        r = counter.count_best(image, prompt, truth, THRESHOLDS, clip_levels)
        t, cm = r["threshold"], r["clip_min"]

    r["overlay"].save(RES_DIR / f"{item['id']}.png")
    data = {k: v for k, v in r.items() if k not in ("overlay", "table")}
    data.update(threshold=t, clip_min=cm)
    (RES_DIR / f"{item['id']}.json").write_text(json.dumps(data))
    ok = truth is not None and r["count"] == truth
    off = "" if truth is None else ("  EXACT" if ok else f"  (real {truth}, off by {r['count'] - truth:+d})")
    print(f"DONE  {item['id']:<6} {prompt}: AI boxed {r['count']}{off}   "
          f"[threshold {t}, clip_min {cm}, {r['seconds']}s]", flush=True)
    summary.append((item["id"], ok))

if MODE != "tune":
    bad = [i for i, ok in summary if not ok]
    print(f"\n{len(summary) - len(bad)} of {len(summary)} pictures match the real count exactly.")
    if bad:
        print("Check these in results/ and try a more specific \"prompt\" in images.json: " + ", ".join(bad))
