"""Visor dashboard:  streamlit run app/dashboard.py

Tabs:  Review (annotated video + jump to each violation) · Violations (table) ·
       Evidence (snapshot gallery) · Analytics.
"""
from __future__ import annotations

import base64
import html
import io
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

cfg = yaml.safe_load(open("config.yaml"))
DB_PATH = cfg["paths"]["db"]
VIDEO_DIR = Path(cfg["paths"].get("videos", "output/videos"))
# Demo mode (Streamlit Cloud): no trained models on the server, so show the saved results in demo/
DEMO_DIR = Path("demo")
DEMO = (DEMO_DIR / "violations.db").exists() and not Path(cfg["models"]["helmet"]).exists()
if DEMO:
    DB_PATH = str(DEMO_DIR / "violations.db")
    VIDEO_DIR = DEMO_DIR / "videos"
LEGACY_VIDEO = Path(DB_PATH).parent / "annotated.mp4"  # written by older versions of the pipeline

ASSETS = Path(__file__).parent / "assets"


def _load_icon():
    try:
        from PIL import Image
        return Image.open(ASSETS / "favicon.png")
    except Exception:
        return "🪖"


st.set_page_config(page_title="Visor", page_icon=_load_icon(), layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Figtree:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');
:root{
  --bg:#0C1013; --panel:#131A1F; --panel-2:#19222A; --line:#25313B; --line-soft:#1B252D;
  --text:#E9EFF3; --muted:#8B9AA5; --amber:#FFB224; --red:#F2545B; --ok:#3CCB9B;
  --display:'Bricolage Grotesque', 'Figtree', system-ui, sans-serif;
}
html, body, [class*="css"], .stApp { font-family:'Figtree', system-ui, sans-serif; }
.stApp { background:
   radial-gradient(900px 380px at 88% -8%, rgba(255,178,36,.07), transparent 62%),
   radial-gradient(700px 400px at -5% 105%, rgba(60,203,155,.04), transparent 60%), var(--bg); }
/* hi-vis hazard stripe along the top edge */
.stApp::before { content:""; position:fixed; top:0; left:0; right:0; height:4px; z-index:1000;
   background:repeating-linear-gradient(135deg, var(--amber) 0 10px, #0C1013 10px 20px); opacity:.85; }
header[data-testid="stHeader"] { background:transparent; }
.stDeployButton, [data-testid="stAppDeployButton"] { display:none !important; }
.block-container { padding-top:2rem; padding-bottom:3.5rem; max-width:1360px; }
section[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] h3 { font-family:var(--display); font-weight:700; font-size:1.15rem; letter-spacing:-.01em; }
::selection { background:rgba(255,178,36,.35); }
::-webkit-scrollbar { width:10px; height:10px; } ::-webkit-scrollbar-thumb { background:var(--line); border-radius:8px; }

/* header */
.hg-head { display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; margin-bottom:1.4rem; }
.hg-brand { display:flex; align-items:center; gap:14px; }
.hg-logo { width:48px; height:48px; border-radius:13px; overflow:hidden; background:#000;
           border:1px solid var(--line); box-shadow:0 8px 26px rgba(255,178,36,.12); }
.hg-logo img { width:100%; height:100%; display:block; }
.hg-title { font-family:var(--display); font-size:1.75rem; font-weight:700; letter-spacing:-.025em; line-height:1.05; }
.hg-sub { color:var(--muted); font-size:.9rem; margin-top:3px; }
.hg-pill { display:inline-flex; align-items:center; gap:9px; padding:8px 15px; border:1px solid var(--line);
           border-radius:999px; background:var(--panel); color:var(--muted); font-size:.85rem; }
.hg-dot { width:8px; height:8px; border-radius:50%; background:var(--ok); animation:hgpulse 2.4s ease-out infinite; }
@keyframes hgpulse { 0%{box-shadow:0 0 0 0 rgba(60,203,155,.55)} 70%,100%{box-shadow:0 0 0 9px rgba(60,203,155,0)} }
@media (prefers-reduced-motion: reduce){ .hg-dot{ animation:none; } }

/* summary strip: one calm panel instead of four boxes */
.hg-kpis { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); background:var(--panel);
           border:1px solid var(--line); border-radius:18px; margin-bottom:1.5rem; overflow:hidden; }
.hg-kpi { padding:20px 24px 18px; border-left:1px solid var(--line-soft); }
.hg-kpi:first-child { border-left:0; }
@media (max-width:900px){ .hg-kpis{ grid-template-columns:repeat(2,minmax(0,1fr)); }
  .hg-kpi:nth-child(3){ border-left:0; } .hg-kpi:nth-child(n+3){ border-top:1px solid var(--line-soft); } }
.hg-kpi-label { display:flex; align-items:center; gap:8px; color:var(--muted); font-size:.88rem; font-weight:500; }
.hg-kpi-label::before { content:""; width:8px; height:8px; border-radius:3px; background:var(--amber); }
.hg-kpi.danger .hg-kpi-label::before { background:var(--red); } .hg-kpi.ok .hg-kpi-label::before { background:var(--ok); }
.hg-kpi.cool .hg-kpi-label::before { background:#6C8CFF; }
.hg-kpi-value { font-family:var(--display); font-size:2.6rem; font-weight:700; letter-spacing:-.035em; margin-top:6px; line-height:1.05;
                font-variant-numeric:tabular-nums; white-space:nowrap; }
@media (max-width:600px){ .hg-kpi { padding:16px 16px 14px; } .hg-kpi-value { font-size:2.05rem; } }
.hg-kpi-note { color:var(--muted); font-size:.82rem; margin-top:4px; }

/* number plate badge (Indian style: IND strip on the left) */
.plate { display:inline-flex; align-items:stretch; font-family:'JetBrains Mono',monospace; font-weight:700; letter-spacing:.09em;
         border-radius:7px; background:#F6F3E6; color:#101214; border:2px solid #101214; font-size:.95rem; overflow:hidden; line-height:1; }
.plate i { font-style:normal; font-size:.5rem; letter-spacing:.04em; background:#1D3FA8; color:#fff; padding:0 5px; display:grid; place-items:center; }
.plate span { padding:6px 10px; }
.plate.bad { background:transparent; color:var(--muted); border:2px dashed var(--line); font-weight:500; }
.plate.bad span { padding:6px 10px; }

/* chips */
.chip { display:inline-flex; align-items:center; gap:6px; padding:3px 10px 3px 8px; border-radius:999px; font-size:.78rem; font-weight:600; }
.chip::before { content:""; width:6px; height:6px; border-radius:50%; background:currentColor; }
.chip.red { background:rgba(242,84,91,.13); color:var(--red); }
.chip.gray { background:var(--panel-2); color:var(--muted); }

/* evidence: camera stills */
.stills { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:20px; margin-top:.6rem; }
.still { min-width:0; }
.still-frame { position:relative; aspect-ratio:1/1; border-radius:14px; overflow:hidden; background:#000; border:1px solid var(--line); }
.still-frame img { width:100%; height:100%; object-fit:cover; object-position:50% 35%; display:block; }
.still-frame::before, .still-frame::after { content:""; position:absolute; width:20px; height:20px; z-index:2; pointer-events:none; border:0 solid rgba(255,255,255,.85); }
.still-frame::before { top:10px; left:10px; border-top-width:2px; border-left-width:2px; border-top-left-radius:4px; }
.still-frame::after { bottom:10px; right:10px; border-bottom-width:2px; border-right-width:2px; border-bottom-right-radius:4px; }
.still-id, .still-time { position:absolute; z-index:3; font-family:'JetBrains Mono',monospace; font-size:.72rem; font-weight:700;
        background:rgba(8,11,13,.72); color:#fff; padding:3px 8px; border-radius:6px; backdrop-filter:blur(4px); }
.still-id { top:10px; right:10px; } .still-time { bottom:10px; left:10px; }
.still-meta { padding:12px 2px 0; display:flex; flex-wrap:wrap; align-items:center; gap:8px; }
.still-when { flex-basis:100%; color:var(--muted); font-size:.82rem; }

/* detail panel + misc */
.hg-detail { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:16px 18px; }
.hg-row { display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid var(--line-soft); font-size:.92rem; }
.hg-row:last-child { border-bottom:0; } .hg-row span:first-child { color:var(--muted); }
.hg-empty { border:1px dashed var(--line); border-radius:16px; padding:46px 20px; text-align:center; color:var(--muted); background:var(--panel); }
.hg-section { display:flex; align-items:center; gap:9px; font-family:var(--display); font-size:1.02rem; font-weight:600; color:var(--text); margin:.2rem 0 .7rem; }
.hg-section::before { content:""; width:4px; height:15px; border-radius:2px; background:var(--amber); }

/* streamlit widgets */
[data-baseweb="tab-list"] { gap:12px !important; margin-bottom:.4rem; }
button[data-baseweb="tab"] { padding:14px 22px !important; margin:0 !important; background:transparent !important; }
button[data-baseweb="tab"], button[data-baseweb="tab"] p, button[data-baseweb="tab"] div {
        font-family:var(--display) !important; font-size:1.05rem !important; font-weight:600 !important; letter-spacing:-.005em; }
button[data-baseweb="tab"] { color:var(--muted) !important; transition:color .15s; }
button[data-baseweb="tab"]:hover { color:var(--text) !important; }
button[data-baseweb="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] p { color:var(--amber) !important; }
[data-baseweb="tab-highlight"] { background:var(--amber) !important; height:3px !important; border-radius:3px 3px 0 0; }
[data-baseweb="tab-border"] { background:var(--line) !important; height:1px !important; }
div[data-testid="stRadio"] div[role="radiogroup"] { gap:8px; }
div[data-testid="stRadio"] label[data-baseweb="radio"] { background:var(--panel); border:1px solid var(--line); padding:7px 16px;
        border-radius:999px; margin:0; transition:background .15s, border-color .15s; cursor:pointer; }
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display:none; }
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover { border-color:var(--muted); }
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) { background:var(--amber); border-color:var(--amber); }
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) * { color:#17120A !important; font-weight:600; }
div[data-baseweb="input"], div[data-baseweb="select"] > div { background:var(--panel-2) !important; border-radius:10px !important; border-color:var(--line) !important; }
div[data-testid="stVideo"], div[data-testid="stVideo"] video { border-radius:16px; border:1px solid var(--line); background:#000; }
div[data-testid="stVideo"] { box-shadow:0 18px 50px rgba(0,0,0,.45); }
div[data-testid="stImage"] img { border-radius:12px; }
button[kind="secondary"], .stDownloadButton button { border-radius:10px; border:1px solid var(--line); background:var(--panel); }
button[kind="secondary"]:hover, .stDownloadButton button:hover { border-color:var(--amber); color:var(--amber); }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- helpers
def fmt_time(seconds) -> str:
    if pd.isna(seconds):
        return "–"
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def short_name(path) -> str:
    return Path(str(path)).name if path else "–"


def stretch(fn, *args, **kwargs):
    """Full-width rendering that works across Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def plate_badge(plate, valid) -> str:
    p = html.escape(str(plate))
    if valid:
        return f'<span class="plate"><i>IND</i><span>{p}</span></span>'
    return f'<span class="plate bad"><span>{p}</span></span>'


def type_label(status) -> str:
    """Friendly type for the table (the detector's car/bus/truck guess is unreliable, e.g. auto-rickshaws)."""
    return "Helmet violation" if status == "NO_HELMET" else "Other vehicle"


def status_chip(status) -> str:
    if status == "NO_HELMET":
        return '<span class="chip red">No helmet</span>'
    return '<span class="chip gray">Other vehicle</span>'


@st.cache_data(show_spinner=False, max_entries=512)
def thumb_uri(path: str, mtime: float) -> str:
    """Small base64 JPEG so evidence stills can be laid out (and framed) in plain HTML."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((520, 520))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def still_card(r) -> str:
    try:
        uri = thumb_uri(str(r.snapshot_path), Path(str(r.snapshot_path)).stat().st_mtime)
    except Exception:
        return ""
    return (
        f'<div class="still"><div class="still-frame"><img src="{uri}" alt="Snapshot of detection {r.id}">'
        f'<span class="still-id">#{r.id}</span><span class="still-time">{fmt_time(r.video_time_s)}</span></div>'
        f'<div class="still-meta">{plate_badge(r.plate, r.plate_valid)} {status_chip(r.helmet_status)}'
        f'<div class="still-when">{r.detected_at:%d %b %Y, %H:%M}</div></div></div>'
    )


def kpi(label, value, note="", tone="") -> str:
    return (f'<div class="hg-kpi {tone}"><div class="hg-kpi-label">{label}</div>'
            f'<div class="hg-kpi-value">{value}</div><div class="hg-kpi-note">{note}</div></div>')


@st.cache_data(ttl=5)
def load_violations(db_path: str, mtime: float) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM violations ORDER BY id DESC", conn)
    if "video_path" not in df.columns:
        df["video_path"] = None
    df["detected_at"] = pd.to_datetime(df["detected_at"], errors="coerce")
    df["plate_valid"] = df["plate_valid"].fillna(0).astype(int)
    df["video_time_s"] = df["video_time_s"].fillna(0)
    return df


def list_videos() -> list[Path]:
    vids = sorted(VIDEO_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True) if VIDEO_DIR.exists() else []
    vids = [v for v in vids if not v.name.endswith(".raw.mp4")]
    if LEGACY_VIDEO.exists():
        vids.append(LEGACY_VIDEO)
    return vids


ROOT = Path(__file__).resolve().parent.parent          # project folder (where config.yaml lives)
UPLOAD_DIR = ROOT / "videos" / "uploaded"
SPEEDS = {
    "Fast: every 5th frame": 5,
    "Balanced: every 3rd frame": 3,
    "Thorough: every frame": 1,
}


def _same_path(a, b) -> bool:
    try:
        return Path(str(a)).resolve() == Path(str(b)).resolve()
    except OSError:
        return str(a) == str(b)


def show_result(last: dict) -> None:
    """Results of the most recent upload: summary, annotated video and evidence."""
    fresh = load_violations(DB_PATH, Path(DB_PATH).stat().st_mtime) if Path(DB_PATH).exists() else pd.DataFrame(columns=COLUMNS)
    rows = fresh[fresh["video_path"].map(lambda v: bool(v) and _same_path(v, last["video"]))] if len(fresh) else fresh
    n_v = int((rows["helmet_status"] == "NO_HELMET").sum()) if len(rows) else 0
    st.markdown(f'<div class="hg-section">Results for {html.escape(last["name"])}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hg-kpis" style="grid-template-columns:repeat(3,minmax(0,1fr))">'
        + kpi("Helmet violations", n_v, "riders without a helmet", "danger")
        + kpi("Other vehicle plates", len(rows) - n_v, "car, bus or truck")
        + kpi("Plates read OK", f"{int(rows['plate_valid'].sum()) if len(rows) else 0} / {len(rows)}", "valid plate format", "ok")
        + "</div>",
        unsafe_allow_html=True,
    )
    if last.get("video") and Path(last["video"]).exists():
        st.video(str(last["video"]), format="video/mp4")
    else:
        st.caption("The annotated video could not be found.")
    shown = rows[rows["snapshot_path"].map(lambda p: bool(p) and Path(str(p)).exists())] if len(rows) else rows
    if len(shown):
        st.markdown('<div class="stills">' + "".join(still_card(r) for r in shown.head(24).itertuples()) + "</div>",
                    unsafe_allow_html=True)
    elif not len(rows):
        st.markdown('<div class="hg-empty">No violations or plates were found in this video.</div>', unsafe_allow_html=True)
    if st.button("Clear and analyze another video"):
        st.session_state.pop("last_run", None)
        st.rerun()


def analyze_ui() -> None:
    """Upload a video, run the pipeline on it in the background and show the results."""
    if DEMO:
        st.markdown('<div class="hg-section">Analyze a video</div>', unsafe_allow_html=True)
        st.info("This is a read-only demo. The detection models run on the author's own computer, "
                "so uploads are turned off here. The other tabs show real results from a processed video. "
                "To analyze your own videos, run Visor locally (see the README).")
        return
    last = st.session_state.get("last_run")
    if last:
        show_result(last)
        return

    st.markdown('<div class="hg-section">Analyze a video</div>', unsafe_allow_html=True)
    st.caption("Upload a clip of a gate or road. Visor finds riders without helmets and reads number plates. "
               "Longer videos take longer: try \"Fast\" first.")
    up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv", "webm"])
    speed = st.radio("Speed", list(SPEEDS), index=1, horizontal=True)
    go = st.button("Analyze video", type="primary", disabled=up is None)
    if not (go and up is not None):
        return

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", up.name)
    dest = UPLOAD_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{safe}"
    dest.write_bytes(up.getbuffer())

    cmd = [sys.executable, "-m", "src.pipeline", "--source", str(dest), "--progress", "--frame-skip", str(SPEEDS[speed])]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    bar = st.progress(0.0, text="Loading models…")
    log = st.empty()
    lines, saved = [], None
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        m = re.match(r"\[PROGRESS\] (\d+)/(\d+)", line)
        if m:
            frac = min(int(m.group(1)) / max(int(m.group(2)), 1), 1.0)
            bar.progress(frac, text=f"Analyzing… {frac:.0%}")
            continue
        if line.startswith("Annotated video saved to "):
            saved = line.split("saved to ", 1)[1].strip()
        if line:
            lines.append(line)
            log.code("\n".join(lines[-8:]), language=None)
    code = proc.wait()

    if code != 0 or not saved:
        bar.empty()
        st.error("The video could not be analyzed. Details:")
        st.code("\n".join(lines[-12:]) or "No output", language=None)
        return
    bar.progress(1.0, text="Done")
    st.cache_data.clear()
    st.session_state["last_run"] = {"video": str((ROOT / saved) if not Path(saved).is_absolute() else saved), "name": up.name}
    st.rerun()


# --------------------------------------------------------------------------- data
COLUMNS = ["id", "detected_at", "source", "video_time_s", "track_id", "plate", "plate_valid",
           "helmet_status", "confidence", "snapshot_path", "plate_path", "video_path"]
if not Path(DB_PATH).exists():
    df_all = pd.DataFrame(columns=COLUMNS)
else:
    df_all = load_violations(DB_PATH, Path(DB_PATH).stat().st_mtime)

videos = list_videos()

try:
    _logo = base64.b64encode((ASSETS / "logo.png").read_bytes()).decode()
    logo_html = f'<img src="data:image/png;base64,{_logo}" alt="Visor logo">'
except OSError:
    logo_html = ""

last_seen = (df_all["detected_at"].max().strftime("%d %b %Y, %H:%M")
             if not df_all.empty and df_all["detected_at"].notna().any() else "no detections yet")
st.markdown(
    f"""
<div class="hg-head">
  <div class="hg-brand"><div class="hg-logo">{logo_html}</div>
    <div><div class="hg-title">Visor</div>
    <div class="hg-sub">Helmet compliance and number-plate evidence</div></div></div>
  <div class="hg-pill"><span class="hg-dot"></span>Last detection: {last_seen}</div>
</div>
""",
    unsafe_allow_html=True,
)

if df_all.empty and not videos:
    st.markdown(
        '<div class="hg-empty"><div style="font-size:2rem">🎥</div>'
        '<div style="font-size:1.1rem;color:#E6EDF3;margin:6px 0">Nothing to show yet</div>'
        "Upload a video below to get started.</div>",
        unsafe_allow_html=True,
    )
    analyze_ui()
    st.stop()

# --------------------------------------------------------------------------- sidebar filters
with st.sidebar:
    st.markdown("### Filters")
    query = st.text_input("Plate number", placeholder="e.g. MH12").strip().upper()
    only_readable = st.toggle("Only readable plates", value=False)
    min_conf = st.slider("Min. detection confidence", 0.0, 1.0, 0.0, 0.05)
    sources = sorted(df_all["source"].dropna().unique().tolist()) if not df_all.empty else []
    chosen_sources = st.multiselect("Video source", sources, format_func=short_name)
    st.divider()
    if stretch(st.button, "↻ Refresh data"):
        st.cache_data.clear()
        st.rerun()
    if DEMO:
        st.caption("Demo mode: showing saved results from a processed video.")
    else:
        st.caption("Tip: run the pipeline in another terminal, then press Refresh.")

view = df_all.copy()
if not view.empty:
    if query:
        view = view[view["plate"].fillna("").str.contains(query, regex=False)]
    if only_readable:
        view = view[view["plate_valid"] == 1]
    if min_conf > 0:
        view = view[view["confidence"].fillna(0) >= min_conf]
    if chosen_sources:
        view = view[view["source"].isin(chosen_sources)]

# --------------------------------------------------------------------------- KPIs
n_viol = int((view["helmet_status"] == "NO_HELMET").sum()) if len(view) else 0
n_other = len(view) - n_viol
unique = view.loc[view["plate"].fillna("UNREADABLE") != "UNREADABLE", "plate"].nunique() if len(view) else 0
readable = int(view["plate_valid"].sum()) if len(view) else 0
st.markdown(
    '<div class="hg-kpis">'
    + kpi("Helmet violations", n_viol, "riders without a helmet", "danger")
    + kpi("Other vehicle plates", n_other, "car, bus or truck")
    + kpi("Unique plates", unique, "distinct vehicles identified", "cool")
    + kpi("Plates read OK", f"{readable} / {len(view)}", "valid plate format", "ok")
    + "</div>",
    unsafe_allow_html=True,
)

kind = st.radio("Show", ["All", "Helmet violations", "Car / bus / truck plates"], horizontal=True)
if len(view):
    if kind == "Helmet violations":
        view = view[view["helmet_status"] == "NO_HELMET"]
    elif kind == "Car / bus / truck plates":
        view = view[view["helmet_status"] != "NO_HELMET"]

tab_analyze, tab_review, tab_table, tab_gallery, tab_stats = st.tabs(
    ["Analyze video", "Detection video", "Violations", "Evidence", "Analytics"]
)

with tab_analyze:
    analyze_ui()

# --------------------------------------------------------------------------- tab: detection video
with tab_review:
    if not videos:
        st.markdown(
            '<div class="hg-empty"><div style="font-size:2rem">🎞️</div>'
            '<div style="font-size:1.05rem;color:#E6EDF3;margin:6px 0">No annotated video found</div>'
            "Run the pipeline again (video saving is on by default):<br>"
            "<code>python -m src.pipeline --source videos/test.mp4</code></div>",
            unsafe_allow_html=True,
        )
    else:
        left, right = st.columns([2.1, 1], gap="large")

        with left:
            counts = df_all["video_path"].value_counts().to_dict() if not df_all.empty else {}
            vid = st.selectbox(
                "Processed video",
                videos,
                format_func=lambda p: f"{p.name}  ·  {counts.get(str(p), 0)} detection(s)",
            )
            run_df = view[view["video_path"] == str(vid)].sort_values("video_time_s") if not view.empty else view

            # jump-to selector: whole video, or the moment a specific violation was caught
            options = [None] + run_df["id"].tolist()
            labels = {r.id: f"#{r.id}  ·  {fmt_time(r.video_time_s)}  ·  {r.plate}" for r in run_df.itertuples()}
            pick = st.selectbox(
                "Jump to detection",
                options,
                format_func=lambda i: "▶  Play from start" if i is None else labels[i],
                key=f"jump_{vid.name}",
            )
            start = 0
            if pick is not None:
                start = max(0, int(run_df.loc[run_df["id"] == pick, "video_time_s"].iloc[0]) - 2)

            st.video(str(vid), format="video/mp4", start_time=start)
            st.caption("Boxes: 🟩 helmet · 🟥 no helmet · 🟨 number plate. The counter in the corner shows violations logged so far.")
            # only read the file into memory when asked (large videos made every page refresh slow)
            if st.checkbox("Prepare video for download", key=f"dl_{vid.name}"):
                try:
                    with open(vid, "rb") as f:
                        st.download_button("⬇ Download annotated video", f.read(), file_name=vid.name, mime="video/mp4")
                except OSError:
                    st.caption("Could not read the video file.")

        with right:
            st.markdown('<div class="hg-section">Selected detection</div>', unsafe_allow_html=True)
            if pick is None:
                if run_df.empty:
                    st.markdown('<div class="hg-empty">No detections in this video<br>(or hidden by your filters).</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="hg-empty">{len(run_df)} detection(s) in this video.<br>'
                        "Pick one under <b>Jump to detection</b> to see the evidence.</div>",
                        unsafe_allow_html=True,
                    )
            else:
                row = run_df[run_df["id"] == pick].iloc[0]
                snap, plate_img = row["snapshot_path"], row["plate_path"]
                if snap and Path(snap).exists():
                    st.image(snap)
                if plate_img and Path(plate_img).exists():
                    st.image(plate_img, width=220)
                conf = f"{row['confidence'] * 100:.0f}%" if pd.notna(row["confidence"]) else "–"
                st.markdown(
                    f"""<div class="hg-detail">
                    <div style="margin-bottom:8px">{plate_badge(row['plate'], row['plate_valid'])}
                    {status_chip(row['helmet_status'])}</div>
                    <div class="hg-row"><span>Video time</span><span>{fmt_time(row['video_time_s'])}</span></div>
                    <div class="hg-row"><span>Confidence</span><span>{conf}</span></div>
                    <div class="hg-row"><span>Type</span><span>{'Helmet violation' if row['helmet_status'] == 'NO_HELMET' else 'Other vehicle'}</span></div>
                    <div class="hg-row"><span>Track ID</span><span>{row['track_id']}</span></div>
                    <div class="hg-row"><span>Plate format</span><span>{'valid' if row['plate_valid'] else 'unverified'}</span></div>
                    <div class="hg-row"><span>Logged</span><span>{row['detected_at']:%d %b %Y, %H:%M}</span></div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            if not run_df.empty:
                st.markdown('<div class="hg-section" style="margin-top:1rem">In this video</div>', unsafe_allow_html=True)
                for r in run_df.head(30).itertuples():
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;'
                        f'border-bottom:1px solid var(--line)"><span style="color:var(--muted);font-family:\'JetBrains Mono\',monospace">'
                        f"{fmt_time(r.video_time_s)}</span>{plate_badge(r.plate, r.plate_valid)}</div>",
                        unsafe_allow_html=True,
                    )

# --------------------------------------------------------------------------- tab: table
with tab_table:
    if view.empty:
        st.markdown('<div class="hg-empty">No violations match the current filters.</div>', unsafe_allow_html=True)
    else:
        table = view.assign(
            time=view["video_time_s"].map(fmt_time),
            video=view["video_path"].map(short_name),
            source=view["source"].map(short_name),
        )[["id", "detected_at", "helmet_status", "plate", "plate_valid", "confidence", "time", "track_id", "source", "video"]]
        table["plate_valid"] = table["plate_valid"].astype(bool)
        table["helmet_status"] = table["helmet_status"].map(type_label)
        stretch(
            st.dataframe,
            table,
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "detected_at": st.column_config.DatetimeColumn("Detected", format="DD MMM YYYY, HH:mm"),
                "helmet_status": st.column_config.TextColumn("Type"),
                "plate": st.column_config.TextColumn("Plate"),
                "plate_valid": st.column_config.CheckboxColumn("Valid plate"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                "time": st.column_config.TextColumn("At (m:ss)", width="small"),
                "track_id": st.column_config.NumberColumn("Track", width="small"),
            },
        )
        st.download_button("⬇ Download CSV", view.drop(columns=["snapshot_path", "plate_path"]).to_csv(index=False),
                           "violations.csv", "text/csv")

# --------------------------------------------------------------------------- tab: evidence
with tab_gallery:
    shown = view[view["snapshot_path"].map(lambda p: bool(p) and Path(str(p)).exists())] if not view.empty else view
    if shown.empty:
        st.markdown('<div class="hg-empty">No snapshots to show.</div>', unsafe_allow_html=True)
    else:
        limit = st.select_slider("Show latest", options=[8, 12, 24, 48, 96], value=12)
        st.markdown('<div class="stills">' + "".join(still_card(r) for r in shown.head(limit).itertuples()) + "</div>",
                    unsafe_allow_html=True)

# --------------------------------------------------------------------------- tab: analytics
with tab_stats:
    if view.empty:
        st.markdown('<div class="hg-empty">Not enough data yet.</div>', unsafe_allow_html=True)
    else:
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown('<div class="hg-section">Detections by hour of day</div>', unsafe_allow_html=True)
            by_hour = view["detected_at"].dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()
            st.bar_chart(by_hour, color="#FFB020", height=280)
        with b:
            st.markdown('<div class="hg-section">Detections per day</div>', unsafe_allow_html=True)
            by_day = view.groupby(view["detected_at"].dt.date).size()
            st.bar_chart(by_day, color="#FF5A5F", height=280)

        st.markdown('<div class="hg-section" style="margin-top:1rem">Repeat offenders</div>', unsafe_allow_html=True)
        rep = (view[view["plate"].fillna("UNREADABLE") != "UNREADABLE"]
               .groupby("plate").agg(violations=("id", "count"), last_seen=("detected_at", "max"))
               .sort_values(["violations", "last_seen"], ascending=False).head(10).reset_index())
        if rep.empty:
            st.caption("No readable plates yet.")
        else:
            stretch(
                st.dataframe,
                rep,
                hide_index=True,
                column_config={
                    "plate": "Plate",
                    "violations": st.column_config.NumberColumn("Violations"),
                    "last_seen": st.column_config.DatetimeColumn("Last seen", format="DD MMM YYYY, HH:mm"),
                },
            )