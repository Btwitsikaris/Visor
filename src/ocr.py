"""License-plate OCR using EasyOCR."""
from __future__ import annotations

import cv2
import numpy as np

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def preprocess(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 320:  # small plates read badly; upscale
        scale = min(320 / max(w, 1), 5.0)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)  # smooth noise but keep character edges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)  # unsharp mask: crisper characters
    return cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_REPLICATE)  # border helps EasyOCR


class PlateReader:
    def __init__(self, gpu: bool = False):
        import easyocr  # imported lazily: slow to load

        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read(self, crop):
        """Return (text, mean_confidence). Handles two-line motorcycle plates."""
        img = preprocess(crop)
        h = img.shape[0]
        results = self.reader.readtext(img, allowlist=ALLOW, detail=1, paragraph=False)
        if not results:
            return "", 0.0

        def key(r):
            ys = [p[1] for p in r[0]]
            xs = [p[0] for p in r[0]]
            row = 0 if np.mean(ys) < h / 2 else 1
            return (row, np.mean(xs))

        results.sort(key=key)
        text = "".join(r[1] for r in results)
        conf = float(np.mean([r[2] for r in results]))
        return text, conf