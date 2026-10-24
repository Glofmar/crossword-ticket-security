"""Load the YAML configs: game description, pipeline parameters and training settings."""

from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import yaml

from .rules import DRAW_RULES, GRID_RULES, Rule, parse_rules


def _build(cls, raw, section):
    # explicit errors instead of the TypeError a dataclass would raise
    if not isinstance(raw, dict):
        raise ValueError(f"{section}: expected a mapping, got {raw!r}")
    names = {f.name for f in fields(cls)}
    unknown, missing = set(raw) - names, names - set(raw)
    if unknown:
        raise ValueError(f"{section}: unknown key(s) {sorted(unknown)}; expected {sorted(names)}")
    if missing:
        raise ValueError(f"{section}: missing key(s) {sorted(missing)}")
    return cls(**raw)


def _repo_root(start):
    # paths in the configs are relative to the repository root
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class HsvRange:
    low: np.ndarray
    high: np.ndarray


def _hsv(raw, section):
    r = _build(HsvRange, raw, section)
    return HsvRange(np.array(r.low, dtype=np.uint8), np.array(r.high, dtype=np.uint8))


@dataclass(frozen=True)
class CellBackground:
    channel: int
    min_value: float
    dominance: float


@dataclass(frozen=True)
class Appearance:
    grid_background_hsv: HsvRange
    grid_background_hsv_loose: HsvRange
    grid_border_hsv: HsvRange
    draw_hidden_hsv: HsvRange
    cell_background: CellBackground


@dataclass(frozen=True)
class GameModels:
    detector: Path
    detector_draw_class: str
    cells: Path
    empty_class: str


@dataclass(frozen=True)
class GameConfig:
    name: str
    grid_rows: int
    grid_cols: int
    min_word_length: int
    draw_rows: int
    draw_cols: int
    alphabet: frozenset[str]
    vowels: frozenset[str]
    dictionary: Path
    payout: dict[int, float]
    grid_rules: tuple[Rule, ...]
    draw_rules: tuple[Rule, ...]
    appearance: Appearance
    models: GameModels

    @property
    def draw_size(self):
        return self.draw_rows * self.draw_cols


def load_game(path):
    path = Path(path)
    root = _repo_root(path.resolve().parent)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    grid, draw = raw["layout"]["grid"], raw["layout"]["draw"]
    app, models = raw["appearance"], _build(GameModels, raw["models"], "models")
    return GameConfig(
        name=path.stem,
        grid_rows=grid["rows"],
        grid_cols=grid["cols"],
        min_word_length=grid["min_word_length"],
        draw_rows=draw["rows"],
        draw_cols=draw["cols"],
        alphabet=frozenset(raw["alphabet"]),
        vowels=frozenset(raw["vowels"]),
        dictionary=root / raw["dictionary"],
        payout={int(k): float(v) for k, v in (raw.get("payout") or {}).items()},
        grid_rules=parse_rules(raw["rules"].get("grid"), GRID_RULES, "grid"),
        draw_rules=parse_rules(raw["rules"].get("draw"), DRAW_RULES, "draw"),
        appearance=Appearance(
            grid_background_hsv=_hsv(app["grid_background_hsv"], "appearance.grid_background_hsv"),
            grid_background_hsv_loose=_hsv(app["grid_background_hsv_loose"], "appearance.grid_background_hsv_loose"),
            grid_border_hsv=_hsv(app["grid_border_hsv"], "appearance.grid_border_hsv"),
            draw_hidden_hsv=_hsv(app["draw_hidden_hsv"], "appearance.draw_hidden_hsv"),
            cell_background=_build(CellBackground, app["cell_background"], "appearance.cell_background"),
        ),
        models=GameModels(root / models.detector, models.detector_draw_class, root / models.cells,
                          models.empty_class),
    )


@dataclass(frozen=True)
class Runtime:
    torch_threads: int


@dataclass(frozen=True)
class Scan:
    step_s: float
    draw_window_s: float
    duplicate_max_cells: int


@dataclass(frozen=True)
class GridLocate:
    open_kernel: int
    min_area_ratio: float
    margin_ratio: float


@dataclass(frozen=True)
class GridBorder:
    min_border_pixels: int
    close_kernel: int
    open_kernel: int
    blur_kernel: int
    canny_low: float
    canny_high: float
    hough_rho: float
    hough_theta_deg: float
    hough_threshold: int
    hough_min_line_length: float
    hough_max_line_gap: float
    duplicate_angle_deg: float
    duplicate_distance: float
    approx_epsilon_ratio: float


@dataclass(frozen=True)
class GridMargins:
    bounds: tuple
    max_iter: int
    ftol: float
    sample_size: int
    sobel_kernel: int
    edge_band: int
    edge_weight: float
    edge_epsilon: float
    margin_weight: float
    background_band: int
    background_min_ratio: float
    background_penalty: float


@dataclass(frozen=True)
class GridSharpen:
    sigma: float
    amount: float


@dataclass(frozen=True)
class GridCells:
    column_search_ratio: float
    row_search_ratio: float
    min_cell_ratio: float
    pad: int
    background_min_row_ratio: float


@dataclass(frozen=True)
class DrawDetect:
    min_confidence: float
    min_width: int
    min_height: int


@dataclass(frozen=True)
class DrawCells:
    threshold_block: int
    threshold_c: float
    row_search: float
    col_search: float
    smoothing: int
    pad: int
    min_cell_probability: float
    max_hidden_ratio: float


@dataclass(frozen=True)
class PipelineConfig:
    runtime: Runtime
    scan: Scan
    grid_locate: GridLocate
    grid_border: GridBorder
    grid_margins: GridMargins
    grid_sharpen: GridSharpen
    grid_cells: GridCells
    draw_detect: DrawDetect
    draw_cells: DrawCells


def _load_sections(path, config_cls, name):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sections = {f.name: f.type for f in fields(config_cls)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"{name}: unknown section(s) {sorted(unknown)}")
    return config_cls(**{key: _build(cls, raw.get(key), key) for key, cls in sections.items()})


def load_pipeline(path):
    cfg = _load_sections(path, PipelineConfig, "pipeline")
    bounds = tuple(cfg.grid_margins.bounds)   # scipy wants a tuple
    return replace(cfg, grid_margins=replace(cfg.grid_margins, bounds=bounds))


@dataclass(frozen=True)
class DetectorTraining:
    base_model: str
    epochs: int
    image_size: int
    batch: int
    device: str
    workers: int
    seed: int


@dataclass(frozen=True)
class CellsTraining:
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    val_ratio: float
    patience: int
    plateau_patience: int
    plateau_factor: float
    seed: int
    augment_rotation_deg: float
    augment_brightness: float
    augment_contrast: float


@dataclass(frozen=True)
class TrainingConfig:
    detector: DetectorTraining
    cells: CellsTraining


def load_training(path):
    return _load_sections(path, TrainingConfig, "training")
