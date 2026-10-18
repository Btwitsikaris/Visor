"""Visor: detect motorcycles, check helmets, read plates, log violations.

    python -m src.pipeline --source videos/test.mp4
    python -m src.pipeline --source 0 --show            # webcam
    python -m src.pipeline --source rtsp://...          # CCTV stream
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from src import db, utils
from src.ocr import PlateReader

GREEN, RED, YELLOW, BLUE = (0, 200, 0), (0, 0, 230), (0, 220, 255), (255, 160, 0)


@dataclass
class TrackState:
    track_id: int
    first_frame: int
    last_frame: int
    helmet_frames: int = 0
    no_helmet_frames: int = 0
    best_nh_conf: float = 0.0
    snapshot: np.ndarray | None = None
    snapshot_time: float = 0.0
    plate_crops: list = field(default_factory=list)  # [(score, crop)], best first


@dataclass
class VehicleState:
    """A tracked car / bus / truck: we only care about its number plate."""
    track_id: int
    kind: str
    last_frame: int
    first_time: float = 0.0
    snapshot: np.ndarray | None = None
    plate_crops: list = field(default_factory=list)  # [(score, crop)], best first


def draw_box(img, box, color, label=None, thick=2):
    # scale line width and text with the frame size so 4K video stays readable
    k = max(1.0, img.shape[1] / 960)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, int(thick * k * 1.5)))
    if label:
        cv2.putText(img, label, (x1, max(y1 - int(8 * k), int(18 * k))), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * k, color, max(2, int(2 * k)))


def make_video_path(cfg, source) -> Path:
    """output/videos/<source-name>_<timestamp>.mp4  (one file per run, never overwritten)."""
    vid_dir = Path(cfg["paths"].get("videos", "output/videos"))
    vid_dir.mkdir(parents=True, exist_ok=True)
    name = "webcam" + str(source) if str(source).isdigit() else Path(str(source)).stem
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:40] or "video"
    return vid_dir / f"{name}_{dt.datetime.now():%Y%m%d_%H%M%S}.mp4"


def to_browser_mp4(raw: Path, final: Path) -> Path:
    """OpenCV's mp4v output can't play in browsers. Re-encode to H.264 so the dashboard can show it.
    Falls back to the raw file if no ffmpeg is available."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(final)]
        if subprocess.run(cmd).returncode == 0 and final.exists():
            raw.unlink(missing_ok=True)
            return final
    print("[WARN] ffmpeg not found or failed; the annotated video may not play in the browser. "
          "Install ffmpeg or `pip install imageio-ffmpeg`.")
    raw.replace(final)
    return final


def finalize(st: TrackState, cfg, reader, conn, snap_dir: Path, source: str, video_path=None) -> bool:
    """Decide if a finished track is a violation; if so OCR the plate and log it."""
    th = cfg["thresholds"]
    total = st.helmet_frames + st.no_helmet_frames
    if total == 0 or st.no_helmet_frames < th["min_violation_frames"]:
        return False
    if st.no_helmet_frames / total < th["violation_ratio"]:
        return False

    reads = []
    for _, plate_img in st.plate_crops:
        text, conf = reader.read(plate_img)
        if text and conf >= th["ocr_min_conf"]:
            reads.append((text, conf))
    plate, valid = utils.pick_plate(reads)

    now = dt.datetime.now()
    stem = f"{now:%Y%m%d_%H%M%S}_id{st.track_id}_{plate}"
    snap_path = plate_path = None
    if st.snapshot is not None:
        snap_path = str(snap_dir / f"{stem}.jpg")
        cv2.imwrite(snap_path, st.snapshot)
    if st.plate_crops:
        plate_path = str(snap_dir / f"{stem}_plate.jpg")
        cv2.imwrite(plate_path, st.plate_crops[0][1])

    db.insert(conn, {
        "detected_at": now.isoformat(timespec="seconds"),
        "source": source,
        "video_time_s": round(st.snapshot_time, 2),
        "track_id": st.track_id,
        "plate": plate,
        "plate_valid": int(valid),
        "helmet_status": "NO_HELMET",
        "confidence": round(st.best_nh_conf, 3),
        "snapshot_path": snap_path,
        "plate_path": plate_path,
        "video_path": str(video_path) if video_path else None,
    })
    print(f"[VIOLATION] track {st.track_id} | plate {plate} | conf {st.best_nh_conf:.2f}", flush=True)
    return True


def finalize_vehicle(vs: VehicleState, cfg, reader, conn, snap_dir: Path, source: str, video_path=None) -> bool:
    """Read and log the plate of a finished car/bus/truck track (no helmet logic)."""
    th = cfg["thresholds"]
    if not vs.plate_crops:
        return False
    reads = []
    for _, plate_img in vs.plate_crops:
        text, conf = reader.read(plate_img)
        if text and conf >= th["ocr_min_conf"]:
            reads.append((text, conf))
    plate, valid = utils.pick_plate(reads)

    now = dt.datetime.now()
    stem = f"{now:%Y%m%d_%H%M%S}_{vs.kind}_id{vs.track_id}_{plate}"
    snap_path = plate_path = None
    if vs.snapshot is not None:
        snap_path = str(snap_dir / f"{stem}.jpg")
        cv2.imwrite(snap_path, vs.snapshot)
    plate_path = str(snap_dir / f"{stem}_plate.jpg")
    cv2.imwrite(plate_path, vs.plate_crops[0][1])

    db.insert(conn, {
        "detected_at": now.isoformat(timespec="seconds"),
        "source": source,
        "video_time_s": round(vs.first_time, 2),
        "track_id": vs.track_id,
        "plate": plate,
        "plate_valid": int(valid),
        "helmet_status": f"PLATE_{vs.kind.upper()}",
        "confidence": round(float(vs.plate_crops[0][0]), 3),
        "snapshot_path": snap_path,
        "plate_path": plate_path,
        "video_path": str(video_path) if video_path else None,
    })
    print(f"[PLATE] {vs.kind} track {vs.track_id} | plate {plate}", flush=True)
    return True


def run(cfg, source, save_video=False, show=False, max_frames=None, progress=False):
    th, proc = cfg["thresholds"], cfg["processing"]
    helmet_names = {utils.norm_name(n) for n in cfg["classes"]["helmet_names"]}
    nohelmet_names = {utils.norm_name(n) for n in cfg["classes"]["no_helmet_names"]}

    vehicle_model = YOLO(cfg["models"]["vehicle"])
    helmet_model = YOLO(cfg["models"]["helmet"])
    plate_model = YOLO(cfg["models"]["plate"])
    reader = PlateReader(gpu=proc["gpu"])
    conn = db.connect(cfg["paths"]["db"])
    snap_dir = Path(cfg["paths"]["snapshots"])
    snap_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    skip = max(1, proc["frame_skip"])
    writer = None
    video_path = make_video_path(cfg, source) if save_video else None
    raw_path = video_path.with_suffix(".raw.mp4") if video_path else None
    tracks: dict[int, TrackState] = {}
    vcfg = cfg.get("vehicles", {})
    other_ids = {int(k): v for k, v in vcfg.get("classes", {}).items()} if vcfg.get("log_other_plates") else {}
    vtracks: dict[int, VehicleState] = {}
    moto_id = cfg["classes"]["motorcycle_id"]
    logged = 0
    plates_logged = 0
    frame_idx = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if progress and total_frames > 0 and frame_idx % 15 == 0:
                print(f"[PROGRESS] {frame_idx}/{total_frames}", flush=True)
            if max_frames and frame_idx >= max_frames:
                break
            if frame_idx % skip:
                continue

            # 1) motorcycles (tracked, so every bike keeps one ID)
            v = vehicle_model.track(frame, persist=True, classes=[moto_id] + list(other_ids),
                                    conf=th["vehicle_conf"], imgsz=proc["imgsz"],
                                    tracker="bytetrack.yaml", verbose=False)[0]
            bikes, others = [], []
            if v.boxes is not None and v.boxes.id is not None:
                for tid_, box_, cls_ in zip(v.boxes.id.int().cpu().tolist(), v.boxes.xyxy.cpu().numpy(),
                                            v.boxes.cls.int().cpu().tolist()):
                    if cls_ == moto_id:
                        bikes.append((tid_, box_))
                    elif cls_ in other_ids:
                        others.append((tid_, box_, other_ids[cls_]))

            # 2) helmet / no-helmet detections
            h = helmet_model.predict(frame, conf=th["helmet_conf"], imgsz=proc["imgsz"], verbose=False)[0]
            helmets, no_helmets = [], []
            for b, c, k in zip(h.boxes.xyxy.cpu().numpy(), h.boxes.conf.cpu().numpy(),
                               h.boxes.cls.int().cpu().tolist()):
                name = utils.norm_name(h.names[k])
                if name in helmet_names:
                    helmets.append((b, float(c)))
                elif name in nohelmet_names:
                    no_helmets.append((b, float(c)))

            # 3) number plates
            p = plate_model.predict(frame, conf=th["plate_conf"], imgsz=proc["imgsz"], verbose=False)[0]
            plates = [(b, float(c)) for b, c in zip(p.boxes.xyxy.cpu().numpy(), p.boxes.conf.cpu().numpy())]

            # 3b) each plate belongs to ONE vehicle: the smallest box that contains its centre
            cands = [("b", t_, bb_) for t_, bb_ in bikes] + [("v", t_, bb_) for t_, bb_, _ in others]
            plate_owner = []
            for pb_, _ in plates:
                ctr = utils.box_center(pb_)
                best = None
                for k_, t_, bb_ in cands:
                    if utils.point_in_box(ctr, bb_):
                        a_ = (bb_[2] - bb_[0]) * (bb_[3] - bb_[1])
                        if best is None or a_ < best[0]:
                            best = (a_, k_, t_)
                plate_owner.append((best[1], best[2]) if best else None)

            # 4) link everything to each tracked bike
            vis = frame.copy() if (save_video or show) else None
            for tid, bbox in bikes:
                st = tracks.setdefault(tid, TrackState(tid, frame_idx, frame_idx))
                st.last_frame = frame_idx
                zone = utils.rider_zone(bbox)
                nh = [(b, c) for b, c in no_helmets if utils.point_in_box(utils.box_center(b), zone)]
                hm = [(b, c) for b, c in helmets if utils.point_in_box(utils.box_center(b), zone)]

                # a frame only counts as "no helmet" if that detection is confident AND beats
                # any helmet detection on the same rider (avoids false alarms in crowded traffic)
                nh_best = max((c for _, c in nh), default=0.0)
                hm_best = max((c for _, c in hm), default=0.0)
                if nh_best >= th.get("no_helmet_conf", th["helmet_conf"]) and nh_best > hm_best:
                    st.no_helmet_frames += 1
                    if nh_best > st.best_nh_conf:
                        st.best_nh_conf = nh_best
                        st.snapshot = utils.crop(frame, zone, pad=10)
                        st.snapshot_time = frame_idx / fps
                elif hm:
                    st.helmet_frames += 1

                for (pb, pc), own in zip(plates, plate_owner):
                    if own == ("b", tid):
                        img = utils.crop(frame, pb, pad=6)
                        if img is not None:
                            area = (pb[2] - pb[0]) * (pb[3] - pb[1])
                            st.plate_crops.append((float(pc * area), img))
                            st.plate_crops.sort(key=lambda t: t[0], reverse=True)
                            del st.plate_crops[proc["plate_crops_per_track"]:]

                if vis is not None:
                    bad = st.no_helmet_frames > st.helmet_frames
                    draw_box(vis, bbox, RED if bad else GREEN, f"ID {tid}" + (" NO HELMET" if bad else ""))
                    for (pb, _), own in zip(plates, plate_owner):
                        if own == ("b", tid):
                            draw_box(vis, pb, YELLOW, "plate", 1)
            # 4b) cars / buses / trucks: only their plate matters
            for tid, bbox, kind in others:
                vs = vtracks.setdefault(tid, VehicleState(tid, kind, frame_idx, frame_idx / fps))
                vs.last_frame = frame_idx
                if vs.snapshot is None:
                    vs.snapshot = utils.crop(frame, bbox, pad=10)
                for (pb, pc), own in zip(plates, plate_owner):
                    if own == ("v", tid):
                        img = utils.crop(frame, pb, pad=6)
                        if img is not None:
                            area = (pb[2] - pb[0]) * (pb[3] - pb[1])
                            vs.plate_crops.append((float(pc * area), img))
                            vs.plate_crops.sort(key=lambda t: t[0], reverse=True)
                            del vs.plate_crops[proc["plate_crops_per_track"]:]
                if vis is not None:
                    draw_box(vis, bbox, BLUE, f"{kind} {tid}", 1)
                    for (pb, _), own in zip(plates, plate_owner):
                        if own == ("v", tid):
                            draw_box(vis, pb, YELLOW, "plate", 1)
            if vis is not None:
                for b, _ in helmets:
                    draw_box(vis, b, GREEN, None, 1)
                for b, _ in no_helmets:
                    draw_box(vis, b, RED, None, 1)

            # 5) close tracks that disappeared
            for tid in [t for t, s in tracks.items() if frame_idx - s.last_frame > th["lost_frames"]]:
                logged += finalize(tracks.pop(tid), cfg, reader, conn, snap_dir, str(source), video_path)

            for tid in [t for t, s_ in vtracks.items() if frame_idx - s_.last_frame > th["lost_frames"]]:
                plates_logged += finalize_vehicle(vtracks.pop(tid), cfg, reader, conn, snap_dir, str(source), video_path)

            if vis is not None:
                kk = max(1.0, vis.shape[1] / 960)
                cv2.putText(vis, f"Violations logged: {logged}" + (f" | Vehicle plates: {plates_logged}" if other_ids else ""),
                            (int(12 * kk), int(34 * kk)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * kk, BLUE, max(2, int(2 * kk)))
                if save_video:
                    if writer is None:
                        hh, ww = vis.shape[:2]
                        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / skip, (ww, hh))
                    writer.write(vis)
                if show:
                    disp = vis
                    if vis.shape[1] > 1280:  # shrink big (e.g. 4K) frames so the window fits the screen
                        sc = 1280 / vis.shape[1]
                        disp = cv2.resize(vis, (1280, int(vis.shape[0] * sc)))
                    cv2.imshow("Visor (q to quit)", disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        for st in tracks.values():
            logged += finalize(st, cfg, reader, conn, snap_dir, str(source), video_path)
        for vs in vtracks.values():
            plates_logged += finalize_vehicle(vs, cfg, reader, conn, snap_dir, str(source), video_path)
        cap.release()
        if writer is not None:
            writer.release()
            to_browser_mp4(raw_path, video_path)
            print(f"Annotated video saved to {video_path}")
        cv2.destroyAllWindows()
        conn.close()
    print(f"Done. {logged} violation(s) and {plates_logged} vehicle plate(s) logged to {cfg['paths']['db']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, webcam index (0) or RTSP url")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True,
                    help="save the annotated video to output/videos/ so the dashboard can play it "
                         "(on by default; use --no-save-video to skip)")
    ap.add_argument("--show", action="store_true", help="live preview window")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--frame-skip", type=int, default=None, help="override processing.frame_skip from the config")
    ap.add_argument("--progress", action="store_true", help="print [PROGRESS] lines (used by the dashboard)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.frame_skip:
        cfg["processing"]["frame_skip"] = args.frame_skip
    run(cfg, args.source, args.save_video, args.show, args.max_frames, args.progress)


if __name__ == "__main__":
    main()