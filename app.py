"""
Human vs AI: Count Challenge
1. Pick a setting: Urban, Indoor or Natural
2. Count the objects in 3 pictures, 20 seconds each
3. Compare your counts with the AI's answers

Run with:  streamlit run app.py
The AI's answers come from images.json ("true_count") and the coloured pictures
from results/ (made by precompute.py), so the app itself needs no AI libraries.
"""
import json
import math
import random
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

# ---------------- Settings ----------------
COUNT_SECONDS = 20                 # time per picture
IMAGES_PER_ROUND = 3
NICHES = {
    "urban":   {"label": "🏙️ Urban",               "blurb": "Streets, cars, buildings and city life"},
    "indoor":  {"label": "🏠 Indoor environment",  "blurb": "Classrooms, kitchens, shelves and rooms"},
    "natural": {"label": "🌿 Natural environment", "blurb": "Flowers, bees, birds and the outdoors"},
}
ROOT = Path(__file__).parent
IMG_DIR, RES_DIR = ROOT / "images", ROOT / "results"

st.set_page_config(page_title="Human vs AI: Count Challenge", page_icon="🔍",
                   layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
<style>
  .timer-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
  .timer-num { font-size: 2.2rem; font-weight: 800; }
  .timer-num.low { color: #d33; }
  .timer-track { height: 16px; background: rgba(128,128,128,0.2); border-radius: 8px; overflow: hidden; }
  .timer-fill { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #2e9e5b, #b07a3a);
                transition: width 0.5s linear; }
  .timer-fill.low { background: #d33; }
  .step { color: #888; font-weight: 600; margin-bottom: 0; }
  .result-card { border: 2px solid rgba(128,128,128,0.25); border-radius: 10px; padding: 12px 16px; margin-bottom: 14px; }
  div[data-testid="stMetricValue"] { font-size: 2.2rem; }
</style>""", unsafe_allow_html=True)


# ---------------- Data ----------------
@st.cache_data
def load_config():
    return json.loads((ROOT / "images.json").read_text(encoding="utf-8"))


def load_image(item):
    return ImageOps.exif_transpose(Image.open(IMG_DIR / item["file"])).convert("RGB")


def ai_picture(item):
    p = RES_DIR / f"{item['id']}.png"
    return Image.open(p) if p.exists() else None


def ai_count(item):
    """How many boxes the AI actually drew (from precompute.py), else the real answer."""
    j = RES_DIR / f"{item['id']}.json"
    if j.exists():
        return json.loads(j.read_text())["count"]
    return item["true_count"]


def plural(item):
    return item.get("plural") or item["object"] + "s"


def available(niche):
    return [it for it in load_config() if it.get("niche") == niche and (IMG_DIR / it["file"]).exists()]


def accuracy(guess, truth):
    """100% for exact, lower the further off (relative to the real answer)."""
    return max(0, round(100 * (1 - abs(guess - truth) / max(truth, 1))))


# ---------------- State ----------------
ss = st.session_state
ss.setdefault("screen", "home")


def start_game(niche):
    pool = available(niche)
    ss.niche = niche
    ss.round = random.sample(pool, min(IMAGES_PER_ROUND, len(pool)))
    ss.idx, ss.guesses = 0, []
    begin_image()


def begin_image():
    ss.screen, ss.start = "count", time.time()


def to_guess():
    ss.screen = "guess"


def go_home():
    ss.screen = "home"


@st.fragment(run_every=0.5)
def timer_bar():
    """Only this bar refreshes twice a second; when time runs out the page moves on."""
    left = COUNT_SECONDS - (time.time() - ss.start)
    if left <= 0:
        to_guess()
        st.rerun()
    low = "low" if left <= 5 else ""
    st.markdown(f"""
      <div class="timer-row"><span>⏱️ Time left</span><span class="timer-num {low}">{math.ceil(left)}s</span></div>
      <div class="timer-track"><div class="timer-fill {low}" style="width:{left / COUNT_SECONDS * 100:.1f}%"></div></div>
    """, unsafe_allow_html=True)


# One container for the page, so each screen fully replaces the last
with st.empty().container():

    # ---------- Screen 1: choose a setting ----------
    if ss.screen == "home":
        st.title("🔍 Human vs AI: Count Challenge")
        st.markdown(f"Can you count as well as an AI? Pick a setting, then count the objects in "
                    f"**{IMAGES_PER_ROUND} pictures**. You get **{COUNT_SECONDS} seconds** for each one.")
        keys = list(NICHES)
        choice = st.selectbox("Choose your setting", keys, index=None, placeholder="Select a setting…",
                              format_func=lambda k: f"{NICHES[k]['label']}  ·  {NICHES[k]['blurb']}")
        if choice:
            n = len(available(choice))
            if n == 0:
                st.warning(f"No pictures for {NICHES[choice]['label']} yet. Add some in images.json with \"niche\": \"{choice}\".")
            else:
                if n < IMAGES_PER_ROUND:
                    st.info(f"Only {n} picture(s) set up for this setting so far.")
                st.button("▶ Start", type="primary", on_click=start_game, args=(choice,), use_container_width=True)

    # ---------- Screen 2: count (timer bar on top) ----------
    elif ss.screen == "count":
        item = ss.round[ss.idx]
        st.markdown(f"<p class='step'>{NICHES[ss.niche]['label']} · Picture {ss.idx + 1} of {len(ss.round)}</p>",
                    unsafe_allow_html=True)
        st.subheader(f"Count the {plural(item)}! 👀")
        timer_bar()
        st.image(load_image(item), use_container_width=True)
        st.button("I’m done counting →", on_click=to_guess, type="primary", use_container_width=True)

    # ---------- Screen 3: enter the count (picture hidden) ----------
    elif ss.screen == "guess":
        item = ss.round[ss.idx]
        st.markdown(f"<p class='step'>{NICHES[ss.niche]['label']} · Picture {ss.idx + 1} of {len(ss.round)}</p>",
                    unsafe_allow_html=True)
        st.subheader("⏰ Time’s up!")
        with st.form(f"guess_{ss.idx}"):
            n = st.number_input(f"How many {plural(item)} did you count?", min_value=0, max_value=999, step=1, value=0)
            last = ss.idx == len(ss.round) - 1
            if st.form_submit_button("See the results 🏁" if last else "Next picture →", type="primary"):
                ss.guesses.append(int(n))
                if last:
                    ss.screen = "results"
                else:
                    ss.idx += 1
                    begin_image()
                st.rerun()

    # ---------- Screen 4: you vs AI ----------
    elif ss.screen == "results":
        st.title("🏁 You vs AI")
        rows = []
        for item, g in zip(ss.round, ss.guesses):
            truth = item["true_count"]
            rows.append((item, g, truth, accuracy(g, truth), ai_count(item)))
        matched = sum(1 for _, g, t, _, _ in rows if g == t)
        your_score = round(sum(r[3] for r in rows) / len(rows))
        ai_score = round(sum(accuracy(r[4], r[2]) for r in rows) / len(rows))

        c1, c2 = st.columns(2)
        c1.metric("🧑 Your accuracy", f"{your_score}%")
        c2.metric("🤖 AI accuracy", f"{ai_score}%")
        if matched == len(rows):
            st.success(f"🤝 Perfect! You matched the AI on all {len(rows)} pictures. It’s a tie!")
            st.balloons()
        elif your_score >= 80:
            st.info(f"🤖 The AI wins, but only just! You matched it on {matched} of {len(rows)} pictures.")
        else:
            st.warning(f"🤖 The AI wins this time. You matched it on {matched} of {len(rows)} pictures.")

        for i, (item, g, truth, acc, ai_n) in enumerate(rows, 1):
            st.markdown(f"#### {i}. {item.get('title', '')}: {plural(item)}")
            m1, m2, m3 = st.columns(3)
            m1.metric("🧑 You", g)
            m2.metric("🤖 AI", ai_n)
            m3.metric("✅ Real answer", truth)
            st.markdown("**You matched it!** ✅" if g == truth else f"You were off by **{abs(g - truth)}**.")
            pic = ai_picture(item)
            if pic is not None:
                # The proof: every object the AI counted has its own numbered box
                st.image(pic, use_container_width=True,
                         caption=f"🤖 The AI boxed and numbered {ai_n} {plural(item) if ai_n != 1 else item['object']}. "
                                 "Check them yourself!")
                with st.expander("Show the original picture"):
                    st.image(load_image(item), use_container_width=True)
            else:
                st.image(load_image(item), use_container_width=True, caption="Run precompute.py to see the AI’s boxes here.")
            st.divider()

        with st.expander("🧠 How does the AI count?"):
            st.markdown(
                "1. **Grounding DINO** reads a word like *“car”* and draws a box around everything that looks like one.\n"
                "2. **CLIP** double-checks every box and throws out mistakes.\n"
                "3. **SAM** traces the exact outline of each object (the colours).\n\n"
                "Counting fast is easy for a computer, but it had to *learn* what a car or a flower looks like first, "
                "from millions of example pictures.")

        b1, b2 = st.columns(2)
        b1.button("🔁 Play this setting again", on_click=start_game, args=(ss.niche,), use_container_width=True)
        b2.button("🏠 Choose another setting", on_click=go_home, type="primary", use_container_width=True)
