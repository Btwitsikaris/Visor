"""Pure helpers (no heavy dependencies) so they are easy to unit test."""
from __future__ import annotations

import re
from collections import defaultdict

# OCR confusion fixes, applied by position in an Indian plate (e.g. DL3CAB1234)
LETTER_FIX = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
DIGIT_FIX = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}


# Valid registration state / UT codes (first two letters of an Indian plate)
STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK",
    "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "TG", "UK", "UP", "WB",
}
# letters OCR commonly mistakes for another letter: misread -> likely real letter(s)
LETTER_CONFUSIONS = {"J": "D", "O": "D", "Q": "D", "I": "L", "T": "L", "H": "M", "N": "M",
                     "P": "R", "B": "R", "G": "C", "S": "C", "U": "V", "E": "F", "K": "H"}


def fix_state_code(code: str):
    """Return a valid 2-letter state code, correcting one commonly-confused letter if needed."""
    if code in STATE_CODES:
        return code
    cands = []
    for i in (0, 1):
        alt = LETTER_CONFUSIONS.get(code[i])
        if alt:
            fixed = code[:i] + alt + code[i + 1:]
            if fixed in STATE_CODES:
                cands.append(fixed)
    return cands[0] if len(cands) >= 1 else None


def norm_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_in_box(pt, box) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def rider_zone(box, up: float = 0.9, side: float = 0.1):
    """Expand a motorcycle box upward/sideways so it also covers the riders' heads."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - side * w, y1 - up * h, x2 + side * w, y2)


def crop(img, box, pad: int = 0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(int(x1 - pad), 0), max(int(y1 - pad), 0)
    x2, y2 = min(int(x2 + pad), w), min(int(y2 + pad), h)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def clean_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def fix_plate(text: str):
    """Return a corrected Indian-format plate (2 letters, 1-2 digits, 1-3 letters, 4 digits) or None."""
    text = clean_plate(text)
    best = None
    for d in (1, 2):
        s = len(text) - 2 - d - 4
        if not 1 <= s <= 3:
            continue
        template = "L" * 2 + "D" * d + "L" * s + "D" * 4
        out, changes, ok = [], 0, True
        for ch, kind in zip(text, template):
            if kind == "L":
                if ch.isalpha():
                    out.append(ch)
                elif ch in LETTER_FIX:
                    out.append(LETTER_FIX[ch]); changes += 1
                else:
                    ok = False; break
            else:
                if ch.isdigit():
                    out.append(ch)
                elif ch in DIGIT_FIX:
                    out.append(DIGIT_FIX[ch]); changes += 1
                else:
                    ok = False; break
        if ok:
            state = fix_state_code("".join(out[:2]))
            if state is None:  # not a real state code: do not present it as a valid plate
                continue
            plate = state + "".join(out[2:])
            changes += state != "".join(out[:2])
            if best is None or changes < best[1]:
                best = (plate, changes)
    return best[0] if best else None


def pick_plate(reads):
    """Vote across several OCR reads [(text, conf), ...]. Returns (plate, is_valid_format).

    Valid reads of the same length are combined character by character (weighted by OCR
    confidence), so one bad character in a single read gets out-voted by the others."""
    fixed = [(f, c) for raw, c in reads if (f := fix_plate(raw))]
    if fixed:
        by_len = defaultdict(float)
        for f, c in fixed:
            by_len[len(f)] += c
        length = max(by_len, key=by_len.get)
        group = [(f, c) for f, c in fixed if len(f) == length]
        chars = []
        for i in range(length):
            votes = defaultdict(float)
            for f, c in group:
                votes[f[i]] += c
            chars.append(max(votes, key=votes.get))
        voted = "".join(chars)
        # the per-character result must still be a valid plate; otherwise use the best whole read
        plate = voted if fix_plate(voted) == voted else max(group, key=lambda x: x[1])[0]
        return plate, True
    cleaned = [(clean_plate(r), c) for r, c in reads if len(clean_plate(r)) >= 4]
    if cleaned:
        return max(cleaned, key=lambda x: x[1])[0], False
    return "UNREADABLE", False