"""Build the demo/ folder (for Streamlit Cloud) from your local results.

    python -m src.make_demo                      # uses the most recent processed video
    python -m src.make_demo --videos clip1.mp4 clip2.mp4

Copies the chosen annotated video(s), their snapshots and plate crops into demo/ and writes
demo/violations.db with clean relative paths (forward slashes) that work on the Linux cloud server.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
from pathlib import Path

import yaml

from src import db


def base(p) -> str:
    """File name from a Windows or Linux path string."""
    return re.split(r"[\\/]", str(p).strip())[-1]


def find(p) -> Path | None:
    if not p:
        return None
    for cand in (Path(str(p).replace("\\", "/")), Path(str(p))):
        if cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", help="annotated video file names to include (default: newest)")
    ap.add_argument("--out", default="demo")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    src_db = cfg["paths"]["db"]
    vid_dir = Path(cfg["paths"].get("videos", "output/videos"))
    if not Path(src_db).exists():
        raise SystemExit(f"No database at {src_db}. Run the pipeline on a video first.")

    vids = sorted((v for v in vid_dir.glob("*.mp4") if not v.name.endswith(".raw.mp4")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if args.videos:
        wanted = {base(v) for v in args.videos}
        vids = [v for v in vids if v.name in wanted]
    else:
        vids = vids[:1]
    if not vids:
        raise SystemExit(f"No annotated videos found in {vid_dir}. Run the pipeline with --save-video.")
    names = {v.name for v in vids}

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "videos").mkdir(parents=True)
    (out / "snapshots").mkdir()
    for v in vids:
        shutil.copy2(v, out / "videos" / v.name)

    conn = sqlite3.connect(src_db)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM violations ORDER BY id")]
    conn.close()

    new = db.connect(out / "violations.db")
    kept = 0
    for r in rows:
        if base(r.get("video_path")) not in names:
            continue
        for col in ("snapshot_path", "plate_path"):
            f = find(r.get(col))
            if f:
                shutil.copy2(f, out / "snapshots" / f.name)
                r[col] = f"{out.as_posix()}/snapshots/{f.name}"
            else:
                r[col] = None
        r["video_path"] = f"{out.as_posix()}/videos/{base(r['video_path'])}"
        r["source"] = base(r.get("source"))
        db.insert(new, r)
        kept += 1
    new.close()

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Demo built in {out}/ : {kept} rows, {len(vids)} video(s), {size:.1f} MB")
    if size > 80:
        print("WARNING: GitHub rejects files over 100 MB. Use a shorter clip for the demo.")


if __name__ == "__main__":
    main()