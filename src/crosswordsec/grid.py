"""Find the grid in a frame, straighten it and cut it into cells."""

import math

import cv2
import numpy as np
from scipy.optimize import minimize

from .config import Appearance, CellBackground, GridBorder, GridCells, GridLocate, GridMargins, GridSharpen, HsvRange


def _kernel(size):
    return np.ones((size, size), np.uint8)


def _in_range(hsv, colour: HsvRange):
    return cv2.inRange(hsv, colour.low, colour.high)


def locate_grid(frame, background: HsvRange, cfg: GridLocate):
    """Crop around the largest blob of grid colour. None if there is no grid big enough."""
    h, w = frame.shape[:2]
    mask = _in_range(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), background)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(cfg.open_kernel))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < w * h * cfg.min_area_ratio:
        return None

    x, y, cw, ch = cv2.boundingRect(largest)
    mx, my = int(cw * cfg.margin_ratio), int(ch * cfg.margin_ratio)
    return frame[max(0, y - my):min(h, y + ch + my), max(0, x - mx):min(w, x + cw + mx)]


def _border_mask(img, look: Appearance, cfg: GridBorder):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = _in_range(hsv, look.grid_border_hsv)
    if cv2.countNonZero(mask) < cfg.min_border_pixels:
        # no visible border, use the grid itself
        mask = _in_range(hsv, look.grid_background_hsv_loose)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(cfg.close_kernel))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(cfg.open_kernel))


def _intersection(seg1, seg2):
    # lines in homogeneous coordinates
    l1 = np.cross([seg1[0], seg1[1], 1.0], [seg1[2], seg1[3], 1.0])
    l2 = np.cross([seg2[0], seg2[1], 1.0], [seg2[2], seg2[3], 1.0])
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float32)


def _find_corners(mask, cfg: GridBorder):
    """Intersect the 4 longest distinct segments found along the border."""
    blurred = cv2.GaussianBlur(mask, (cfg.blur_kernel, cfg.blur_kernel), 0)
    edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)
    segments = cv2.HoughLinesP(edges, rho=cfg.hough_rho, theta=math.radians(cfg.hough_theta_deg),
                               threshold=cfg.hough_threshold, minLineLength=cfg.hough_min_line_length,
                               maxLineGap=cfg.hough_max_line_gap)
    if segments is None:
        return None

    # opencv 4 returns (N, 1, 4), opencv 5 returns (N, 4)
    candidates = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4).tolist():
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        candidates.append((math.hypot(x2 - x1, y2 - y1), angle, x1, y1, x2, y2))
    candidates.sort(reverse=True)   # longest first

    sides = []
    for _, angle, x1, y1, x2, y2 in candidates:
        mid = ((x1 + x2) / 2, (y1 + y2) / 2)
        same_side = any(abs(angle - a) < cfg.duplicate_angle_deg
                        and math.dist(mid, m) < cfg.duplicate_distance for a, m, _ in sides)
        if not same_side:
            sides.append((angle, mid, (x1, y1, x2, y2)))
        if len(sides) == 4:
            break
    if len(sides) != 4:
        return None

    # sort the sides around the centre so that neighbours in the list are adjacent sides
    centre = np.mean([m for _, m, _ in sides], axis=0)
    sides.sort(key=lambda s: math.atan2(s[1][1] - centre[1], s[1][0] - centre[0]))
    corners = [_intersection(sides[i][2], sides[(i + 1) % 4][2]) for i in range(4)]
    if any(c is None for c in corners):
        return None

    hull = cv2.convexHull(np.array(corners, dtype=np.float32)).reshape(-1, 2)
    quad_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(quad_mask, [hull.astype(np.int32)], 255)
    contours, _ = cv2.findContours(quad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, cfg.approx_epsilon_ratio * cv2.arcLength(contour, True), True)
    pts = approx.reshape(4, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(contour))
    return order_corners(pts)


def order_corners(pts):
    """Top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def _scale_quad(quad, margins):
    left, top, right, bottom = margins
    centre = quad.mean(axis=0)
    factors = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
    return (centre + (quad - centre) * factors).astype(np.float32)


def _warp(img, quad, side, flags=cv2.INTER_LINEAR):
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (side, side), flags=flags)


def _margin_cost(margins, img, quad, look: Appearance, cfg: GridMargins):
    # We want each edge of the warped image to sit on the border transition, with a bit
    # of grid colour still visible along it, and margins as small as possible.
    warped = _warp(img, _scale_quad(quad, margins), cfg.sample_size)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    border = _in_range(hsv, look.grid_border_hsv)
    k = cfg.sobel_kernel
    grad = cv2.magnitude(cv2.Sobel(border, cv2.CV_64F, 1, 0, ksize=k), cv2.Sobel(border, cv2.CV_64F, 0, 1, ksize=k))

    b = cfg.edge_band
    edges = [grad[:b, :].mean(), grad[-b:, :].mean(), grad[:, :b].mean(), grad[:, -b:].mean()]
    cost = sum(cfg.edge_weight / (e + cfg.edge_epsilon) for e in edges) + np.sum(margins) * cfg.margin_weight

    b = cfg.background_band
    for band in (hsv[:b, :], hsv[-b:, :], hsv[:, :b], hsv[:, -b:]):
        ratio = cv2.countNonZero(_in_range(band, look.grid_background_hsv_loose)) / (band.shape[0] * band.shape[1])
        if ratio < cfg.background_min_ratio:
            cost += (cfg.background_min_ratio - ratio) * cfg.background_penalty
    return cost


def rectify_grid(roi, look: Appearance, border: GridBorder, margins: GridMargins, sharpen: GridSharpen):
    """Square, perspective-corrected and sharpened image of the grid, or None."""
    quad = _find_corners(_border_mask(roi, look, border), border)
    if quad is None:
        return None

    res = minimize(_margin_cost, [1.0] * 4, args=(roi, quad, look, margins), method="SLSQP",
                   bounds=[margins.bounds] * 4, options={"maxiter": margins.max_iter, "ftol": margins.ftol})
    quad = _scale_quad(quad, res.x)

    side = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[0] - quad[3])))
    if side < 2:
        return None
    warped = _warp(roi, quad, side, flags=cv2.INTER_LANCZOS4)
    blurred = cv2.GaussianBlur(warped, (0, 0), sharpen.sigma)
    return cv2.addWeighted(warped, 1 + sharpen.amount, blurred, -sharpen.amount, 0)


def _cuts(edge_strength, length, n, search_ratio):
    # edge_strength[i] is the change between pixel i and i + 1
    search = int(length * search_ratio)
    cuts = [0]
    for i in range(1, n):
        nominal = int(i * length / n)
        lo, hi = max(0, nominal - search), min(length - 2, nominal + search)
        cuts.append(lo + int(np.argmax(edge_strength[lo:hi + 1])) if lo <= hi else nominal)
    cuts.append(length)
    return cuts


def _luminance(img):
    return img[:, :, :3].astype(np.float64).mean(axis=2)


def _grid_rows(column, colour: CellBackground, min_row_ratio):
    """Range of rows of a column that show the grid background, or None."""
    pixels = column[:, :, :3].astype(np.float64)
    main = pixels[:, :, colour.channel]
    others = [pixels[:, :, i] for i in range(3) if i != colour.channel]
    is_bg = (main >= colour.min_value) & np.logical_and.reduce([main >= o * colour.dominance for o in others])
    rows = np.flatnonzero(is_bg.mean(axis=1) >= min_row_ratio)
    if len(rows) == 0 or rows[0] >= rows[-1]:
        return None
    return int(rows[0]), int(rows[-1]) + 1


def split_cells(grid, n_rows, n_cols, colour: CellBackground, cfg: GridCells):
    """Cut the rectified grid into cells, keyed by (row, col).

    Boundaries go where the luminance changes the most, close to the regular positions.
    A column where no grid colour is found is skipped, so some cells can be missing.
    """
    height, width = grid.shape[:2]
    col_cuts = _cuts(np.abs(np.diff(_luminance(grid), axis=1)).sum(axis=0), width, n_cols, cfg.column_search_ratio)
    min_w = width * cfg.min_cell_ratio / n_cols
    min_h = height * cfg.min_cell_ratio / n_rows

    cells = {}
    for c in range(n_cols):
        x0, x1 = col_cuts[c], col_cuts[c + 1]
        if x1 - x0 < min_w:
            grow = (min_w - (x1 - x0)) / 2
            x0, x1 = x0 - grow, x1 + grow
        column = grid[:, max(0, int(x0 - cfg.pad)):min(width, int(x1 + cfg.pad))]

        rows = _grid_rows(column, colour, cfg.background_min_row_ratio)
        if rows is None:
            continue
        column = column[rows[0]:rows[1]]
        col_h = column.shape[0]
        row_cuts = _cuts(np.abs(np.diff(_luminance(column), axis=0)).sum(axis=1), col_h, n_rows, cfg.row_search_ratio)

        for r in range(n_rows):
            y0, y1 = row_cuts[r], row_cuts[r + 1]
            if y1 - y0 < min_h:
                grow = (min_h - (y1 - y0)) / 2
                y0, y1 = y0 - grow, y1 + grow
            cells[(r, c)] = column[max(0, int(y0 - cfg.pad)):min(col_h, int(y1 + cfg.pad))]
    return cells
