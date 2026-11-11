"""
Review and fix the AI's boxes before the game uses them.

Run:  streamlit run review.py

For each picture you can:
  - remove wrong boxes (pick their numbers)
  - add a missed object: click its TOP-LEFT corner, then its BOTTOM-RIGHT corner
  - save: updates results/<id>.png and results/<id>.json, which the game shows
"""
import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

from detector import draw_overlay, shrink

ROOT = Path(__file__).parent
IMG_DIR, RES_DIR = ROOT / "images", ROOT / "results"
CFG = ROOT / "images.json"
SHOW_WIDTH = 900

st.set_page_config(page_title="Review AI boxes", layout="wide")
ss = st.session_state
ss.setdefault("boxes", {})
ss.setdefault("corner", None)
ss.setdefault("last_click", None)

items = json.loads(CFG.read_text(encoding="utf-8"))
ready = [it for it in items if (IMG_DIR / it["file"]).exists()]
if not ready:
    st.error("No pictures found in images/.")
    st.stop()

with st.sidebar:
    st.header("Pictures")
    labels = []
    for it in ready:
        j = RES_DIR / f"{it['id']}.json"
        data = json.loads(j.read_text()) if j.exists() else {}
        mark = "✅" if data.get("reviewed") else ("🤖" if data else "⬜")
        labels.append(f"{mark} {it['id']} · {it.get('title', '')}")
    pick = st.radio("Choose a picture", range(len(ready)), format_func=lambda i: labels[i])
    st.caption("⬜ no AI result yet · 🤖 AI result, not reviewed · ✅ reviewed")

item = ready[pick]
iid = item["id"]
res_json = RES_DIR / f"{iid}.json"
image = shrink(ImageOps.exif_transpose(Image.open(IMG_DIR / item["file"])).convert("RGB"))  # same size the AI used

if iid not in ss.boxes:
    data = json.loads(res_json.read_text()) if res_json.exists() else {}
    ss.boxes[iid] = [list(b) for b in data.get("boxes", [])]
    ss.corner = None
boxes = ss.boxes[iid]

st.title(f"Review: {item.get('title', iid)}")
c1, c2, c3 = st.columns(3)
c1.metric("Boxes now", len(boxes))
c2.metric("Real answer in images.json", item.get("true_count", "?"))
c3.metric("Looking for", item.get("prompt", item["object"]))

# Draw current boxes (and the first corner of a box being added)
shown = draw_overlay(image, boxes)
if ss.corner:
    d = ImageDraw.Draw(shown)
    x, y = ss.corner
    d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(255, 0, 0), outline=(255, 255, 255), width=3)

left, right = st.columns([3, 1])
with left:
    st.caption("➕ To add a missed object: click its top-left corner, then its bottom-right corner.")
    click = streamlit_image_coordinates(shown, width=SHOW_WIDTH, key=f"img_{iid}")
    if click and click != ss.last_click:
        ss.last_click = click
        scale = image.width / click["width"]
        pt = (click["x"] * scale, click["y"] * scale)
        if ss.corner is None:
            ss.corner = pt
        else:
            (x0, y0), (x1, y1) = ss.corner, pt
            boxes.append([round(min(x0, x1), 1), round(min(y0, y1), 1), round(max(x0, x1), 1), round(max(y0, y1), 1)])
            ss.corner = None
        st.rerun()

with right:
    st.subheader("Fix")
    if ss.corner:
        st.info("First corner set. Now click the bottom-right corner.")
        if st.button("Cancel this box"):
            ss.corner = None
            st.rerun()
    remove = st.multiselect("Remove wrong boxes (by number)", list(range(1, len(boxes) + 1)))
    if st.button("🗑️ Remove selected", disabled=not remove, use_container_width=True):
        ss.boxes[iid] = [b for i, b in enumerate(boxes, 1) if i not in remove]
        st.rerun()
    if st.button("↩️ Undo last added box", disabled=not boxes, use_container_width=True):
        boxes.pop()
        st.rerun()
    st.divider()
    if st.button("💾 Save for the game", type="primary", use_container_width=True):
        RES_DIR.mkdir(exist_ok=True)
        old = json.loads(res_json.read_text()) if res_json.exists() else {}
        old.update(count=len(boxes), boxes=boxes, reviewed=True)
        res_json.write_text(json.dumps(old))
        draw_overlay(image, boxes).save(RES_DIR / f"{iid}.png")
        st.success(f"Saved {len(boxes)} boxes.")
    if item.get("true_count") != len(boxes):
        st.warning(f"Boxes ({len(boxes)}) ≠ real answer ({item.get('true_count')}).")
        if st.button(f"Set real answer to {len(boxes)}", use_container_width=True):
            for it in items:
                if it["id"] == iid:
                    it["true_count"] = len(boxes)
            CFG.write_text("[\n" + ",\n".join("  " + json.dumps(it, ensure_ascii=False) for it in items) + "\n]\n",
                           encoding="utf-8")
            st.cache_data.clear()
            st.rerun()
    if st.button("Reload the AI’s original boxes", use_container_width=True):
        ss.boxes.pop(iid, None)
        st.rerun()
