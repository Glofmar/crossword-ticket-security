"""Detect the hidden-letter panel ("draw"), cut it into cells and read it."""

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from .config import DrawCells, DrawDetect, GameConfig, HsvRange
from .ocr import CellClassifier
from .rules import DrawContext, RuleResult, all_passed, evaluate


@dataclass
class DrawReading:
    letters: list[str]
    probabilities: list[float]
    crop: np.ndarray
    detection_confidence: float
    rules: list[RuleResult]


class DrawDetector:
    def __init__(self, weights, class_name, cfg: DrawDetect):
        self.model = YOLO(str(weights))
        self.class_name = class_name
        self.cfg = cfg

    def detect(self, frame):
        """List of (confidence, crop) for each draw panel found in the frame."""
        result = self.model.predict(source=frame, conf=self.cfg.min_confidence, verbose=False)[0]
        if result.obb is None:
            return []
        h, w = frame.shape[:2]
        crops = []
        for cls, conf, box in zip(result.obb.cls.tolist(), result.obb.conf.tolist(), result.obb.xyxy.tolist()):
            if self.model.names[int(cls)] != self.class_name:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
            if x2 - x1 >= self.cfg.min_width and y2 - y1 >= self.cfg.min_height:
                crops.append((conf, frame[y1:y2, x1:x2]))
        return crops


def _row_cuts(ink, n, search):
    # with 2 rows and search=0.4 the cut is looked for between 30% and 70% of the height
    length = len(ink)
    cuts = [0]
    for i in range(1, n):
        lo, hi = int(length * (i - search) / n), int(length * (i + search) / n)
        cuts.append(lo + int(np.argmin(ink[lo:hi])) if lo < hi else int(i * length / n))
    cuts.append(length)
    return cuts


def _col_cuts(ink, n, search):
    length = len(ink)
    pitch = length / n
    margin = int(pitch * search)
    cuts = [0]
    for i in range(1, n):
        nominal = int(i * pitch)
        lo, hi = max(0, nominal - margin), min(length, nominal + margin)
        cuts.append(lo + int(np.argmin(ink[lo:hi])) if lo < hi else nominal)
    cuts.append(length)
    return cuts


def split_draw(crop, n_rows, n_cols, cfg: DrawCells):
    """Cut the panel into cells, row by row, along the lines with the least ink."""
    h, w = crop.shape[:2]
    ink = cv2.adaptiveThreshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, cfg.threshold_block, cfg.threshold_c)
    row_cuts = _row_cuts(ink.sum(axis=1), n_rows, cfg.row_search)

    cells, pad = [], cfg.pad
    for r in range(n_rows):
        y1, y2 = row_cuts[r], row_cuts[r + 1]
        col_ink = np.convolve(ink[y1:y2].sum(axis=0), np.ones(cfg.smoothing) / cfg.smoothing, mode="same")
        col_cuts = _col_cuts(col_ink, n_cols, cfg.col_search)
        for c in range(n_cols):
            cell = crop[max(0, y1 - pad):min(h, y2 + pad), max(0, col_cuts[c] - pad):min(w, col_cuts[c + 1] + pad)]
            if cell.size:
                cells.append(cell)
    return cells


def hidden_ratio(cell, hidden: HsvRange):
    """Share of the cell that still has the colour of an unscratched cell."""
    mask = cv2.inRange(cv2.cvtColor(cell, cv2.COLOR_BGR2HSV), hidden.low, hidden.high)
    return float(np.count_nonzero(mask) / (cell.shape[0] * cell.shape[1]))


def read_draw(detector: DrawDetector, classifier: CellClassifier, frame, game: GameConfig, cfg: DrawCells,
              grid_letters, words):
    """First panel of the frame that is fully scratched, readable and passes the draw rules."""
    for confidence, crop in detector.detect(frame):
        cells = split_draw(crop, game.draw_rows, game.draw_cols, cfg)
        if len(cells) != game.draw_size:
            continue
        # the classifier has no "hidden" class: an unscratched cell would come out as some letter
        if any(hidden_ratio(c, game.appearance.draw_hidden_hsv) > cfg.max_hidden_ratio for c in cells):
            continue
        predictions = classifier.predict(cells)
        if any(p.probability < cfg.min_cell_probability or p.label not in game.alphabet for p in predictions):
            continue

        letters = [p.label for p in predictions]
        results = evaluate(game.draw_rules, DrawContext(letters, grid_letters, words, game.vowels))
        if all_passed(results):
            return DrawReading(letters, [p.probability for p in predictions], crop, confidence, results)
    return None
