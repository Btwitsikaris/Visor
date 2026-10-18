# 🛵 HelmetGuard: Smart Campus Entry & Helmet Safety System

Real-time computer-vision system that watches a gate camera, finds motorcycles and riders, checks
whether they wear helmets, reads the number plate, and logs every violation (snapshot + plate + time)
to a database with a web dashboard.

> **Demo:** _add a GIF or video link here_  
> **Results:** see the table at the bottom (fill it in with your own measured numbers).

## How it works

```
video / CCTV ─► YOLO + ByteTrack (motorcycles, one ID per bike)
              ├─► YOLO helmet model  (helmet / no_helmet)
              ├─► YOLO plate model   (license plate box)
              │
              ├─ link detections to each bike's "rider zone"
              ├─ vote over frames ─► violation if no-helmet in most frames
              ├─ EasyOCR on best plate crops ─► format fix + voting
              └─► SQLite + snapshots ─► Streamlit dashboard
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1: Get data (free)

1. Go to **Roboflow Universe** (universe.roboflow.com) and search _"helmet detection"_ and _"license plate"_.
   Pick datasets with a few thousand images and a license that allows your use.
2. Click **Download Dataset → YOLOv8 format** and unzip into `datasets/helmet/` and `datasets/plate/`.
3. Open each `data.yaml` and note the **class names**. Put them in `config.yaml` under
   `helmet_names` / `no_helmet_names`.
   Kaggle has similar datasets if you prefer it.

## Step 2: Train the two models

Free option: run these in a Google Colab notebook with a GPU (Runtime → Change runtime type).

```bash
python -m src.train --data datasets/helmet/data.yaml --name helmet --epochs 50
python -m src.train --data datasets/plate/data.yaml  --name plate  --epochs 50
```

Weights are saved to `models/helmet.pt` and `models/plate.pt`, and the final mAP is printed.
Copy those numbers into the results table below.

## Step 3: Run on a video

```bash
python -m src.pipeline --source videos/test.mp4 --save-video
```

Options: `--source 0` for a webcam, `--source rtsp://...` for CCTV, `--show` for a live window.
Output goes to `output/`: annotated video, violation snapshots, and `violations.db`.

## Step 4: Dashboard

```bash
streamlit run app/dashboard.py
```

## Tune it

All settings are in `config.yaml`: confidence thresholds, how many frames count as a violation,
frame skipping for speed, and GPU on/off.

## Tests

```bash
pip install pytest && pytest -q
```

## Results (fill in with your own numbers)

| Component | Metric | Value |
|---|---|---|
| Helmet model | mAP50 | _TBD_ |
| Plate model | mAP50 | _TBD_ |
| Plate OCR | exact-match accuracy on N test crops | _TBD_ |
| End-to-end | violations found vs. manually counted (N videos) | _TBD_ |

## Limitations

- Plate reading drops at night, with motion blur, or at steep camera angles.
- Helmet detection can fail when riders are occluded or far from the camera.
- Plates are validated against the Indian format; other formats fall back to raw OCR text.

## Privacy & ethics

This processes video of real people. Get permission from your institution before recording,
store snapshots securely, and blur faces and plates in any public demo.

## Tech stack

Python · Ultralytics YOLO · ByteTrack · OpenCV · EasyOCR · SQLite · Streamlit
