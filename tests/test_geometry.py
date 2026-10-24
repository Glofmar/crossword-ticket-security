from pathlib import Path

import numpy as np

from crosswordsec.config import CellBackground, HsvRange, load_pipeline
from crosswordsec.draw import hidden_ratio, split_draw
from crosswordsec.grid import order_corners, split_cells


def synthetic_grid(n=11, cell=20, line=2):
    """Green cells separated by light lines."""
    size = n * cell
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (40, 160, 40)
    for k in range(n + 1):
        p = min(k * cell, size - line)
        img[p:p + line, :] = (200, 230, 200)
        img[:, p:p + line] = (200, 230, 200)
    return img


def test_order_corners():
    pts = np.array([[10, 90], [90, 90], [10, 10], [90, 10]], np.float32)
    assert order_corners(pts).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


PIPELINE = load_pipeline(Path(__file__).parent.parent / "configs" / "pipeline.yaml")
GREEN = CellBackground(channel=1, min_value=60, dominance=1.15)


def test_split_cells_rectangular_grid():
    cells = split_cells(synthetic_grid(n=11)[:, :160], 11, 8, GREEN, PIPELINE.grid_cells)
    assert set(cells) == {(r, c) for r in range(11) for c in range(8)}


def test_split_cells_finds_every_cell():
    cells = split_cells(synthetic_grid(), 11, 11, GREEN, PIPELINE.grid_cells)
    assert len(cells) == 121
    assert set(cells) == {(r, c) for r in range(11) for c in range(11)}
    assert all(16 <= cell.shape[0] <= 26 and 16 <= cell.shape[1] <= 26 for cell in cells.values())


def test_split_cells_without_background_colour_finds_nothing():
    blue_grid = np.roll(synthetic_grid(), -1, axis=2)   # (B, G, R) -> (G, R, B): background is now blue
    assert split_cells(blue_grid, 11, 11, GREEN, PIPELINE.grid_cells) == {}


def test_split_draw_any_layout():
    panel = np.full((60, 240, 3), 170, np.uint8)
    for rows, cols in ((2, 7), (3, 6), (1, 20)):
        cells = split_draw(panel, rows, cols, PIPELINE.draw_cells)
        assert len(cells) == rows * cols
        assert all(c.size for c in cells)


def test_hidden_ratio_spots_unscratched_cells():
    green = HsvRange(np.array([35, 50, 50], np.uint8), np.array([90, 255, 255], np.uint8))
    scratched = np.full((30, 30, 3), 170, np.uint8)              # gray panel
    hidden = scratched.copy()
    hidden[5:25, 5:25] = (40, 180, 40)                           # green square with a "?"
    assert hidden_ratio(scratched, green) == 0
    assert hidden_ratio(hidden, green) > 0.25
