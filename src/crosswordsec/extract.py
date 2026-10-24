"""Extract tickets (grid + draw panel) from videos of scratch cards being played.

    python -m crosswordsec.extract video.mp4 --game configs/games/example_crossword_9x9.yaml --out results/
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import GameConfig, PipelineConfig, load_game, load_pipeline
from .draw import DrawDetector, DrawReading, read_draw
from .game import BLOCK, Word, build_matrix, cell_difference, extract_words, grid_letters, load_dictionary, normalize
from .grid import locate_grid, rectify_grid, split_cells
from .ocr import CellClassifier
from .rules import GridContext, RuleResult, all_passed, completed_words, evaluate
from .video import Frame, VideoSampler

log = logging.getLogger(__name__)


@dataclass
class GridReading:
    matrix: tuple
    words: list[Word]
    letters: frozenset
    probabilities: dict
    rectified: np.ndarray
    rules: list[RuleResult]


@dataclass
class Ticket:
    video: str
    grid_frame: Frame
    grid: GridReading
    draw_frame: Frame
    draw: DrawReading

    @property
    def ticket_id(self):
        return f"{Path(self.video).stem}_{self.grid_frame.index:07d}"

    def to_record(self, game: GameConfig):
        done = completed_words(self.draw.letters, [w.text for w in self.grid.words])
        return {
            "ticket_id": self.ticket_id,
            "game": game.name,
            "video": self.video,
            "grid": {
                "frame_index": self.grid_frame.index,
                "time_s": round(self.grid_frame.time_s, 2),
                "matrix": ["".join(row) for row in self.grid.matrix],
                "words": [{"text": w.text, "direction": w.direction, "cells": [list(c) for c in w.cells]}
                          for w in self.grid.words],
                "min_cell_probability": round(min(self.grid.probabilities.values()), 4),
                "rules": [r.to_record() for r in self.grid.rules],
            },
            "draw": {
                "frame_index": self.draw_frame.index,
                "time_s": round(self.draw_frame.time_s, 2),
                "letters": "".join(self.draw.letters),
                "cell_probabilities": [round(p, 4) for p in self.draw.probabilities],
                "detection_confidence": round(self.draw.detection_confidence, 4),
                "rules": [r.to_record() for r in self.draw.rules],
            },
            "completed_words": done,
            "n_completed": len(done),
            "prize": game.payout.get(len(done), 0.0),
        }


class Extractor:
    def __init__(self, game: GameConfig, pipeline: PipelineConfig):
        self.game = game
        self.pipeline = pipeline
        torch.set_num_threads(pipeline.runtime.torch_threads)
        self.classifier = CellClassifier(game.models.cells)
        self.detector = DrawDetector(game.models.detector, game.models.detector_draw_class, pipeline.draw_detect)
        self.dictionary = load_dictionary(game.dictionary)

    def read_grid(self, frame):
        """Grid of the frame, or None if there is none or it breaks a grid rule."""
        game, p = self.game, self.pipeline
        roi = locate_grid(frame, game.appearance.grid_background_hsv, p.grid_locate)
        if roi is None:
            return None
        rectified = rectify_grid(roi, game.appearance, p.grid_border, p.grid_margins, p.grid_sharpen)
        if rectified is None:
            return None

        cells = split_cells(rectified, game.grid_rows, game.grid_cols, game.appearance.cell_background, p.grid_cells)
        positions = list(cells)
        predictions = self.classifier.predict([cells[pos] for pos in positions])
        empty = game.models.empty_class.upper()
        letters = {}
        for pos, pred in zip(positions, predictions):
            if pred.label == empty:
                letters[pos] = BLOCK
            elif pred.label in game.alphabet:
                letters[pos] = pred.label

        matrix = build_matrix(letters, game.grid_rows, game.grid_cols)
        words = extract_words(matrix, game.min_word_length)
        results = evaluate(game.grid_rules, GridContext([normalize(w.text) for w in words], self.dictionary))
        if not all_passed(results):
            return None
        probabilities = {pos: pred.probability for pos, pred in zip(positions, predictions)}
        return GridReading(matrix, words, grid_letters(matrix), probabilities, rectified, results)

    def find_draw(self, frames, grid_frame, grid):
        """Read the next frames until a valid draw shows up or the time window is over."""
        words = [w.text for w in grid.words]
        deadline = grid_frame.time_s + self.pipeline.scan.draw_window_s
        for frame in frames:
            reading = read_draw(self.detector, self.classifier, frame.image, self.game, self.pipeline.draw_cells,
                                grid.letters, words)
            if reading is not None:
                return frame, reading
            if frame.time_s >= deadline:
                break
        return None

    def process(self, video):
        tickets, seen = [], []
        max_diff = self.pipeline.scan.duplicate_max_cells
        with VideoSampler(video, self.pipeline.scan.step_s) as sampler:
            log.info("%s: %.0f s at %.2f fps, one frame every %d", video.name, sampler.duration_s,
                     sampler.fps, sampler.step)
            frames = iter(sampler)
            for frame in frames:
                grid = self.read_grid(frame.image)
                if grid is None:
                    continue
                # the same ticket is often shown again later, with some letters crossed out
                if any(cell_difference(grid.matrix, m) <= max_diff for m in seen):
                    continue
                log.info("grid at %.1f s", frame.time_s)
                found = self.find_draw(frames, frame, grid)
                if found is None:
                    log.info("  no valid draw within %.0f s", self.pipeline.scan.draw_window_s)
                    continue
                draw_frame, draw = found
                seen.append(grid.matrix)
                tickets.append(Ticket(video.name, frame, grid, draw_frame, draw))
                log.info("  draw %s at %.1f s", "".join(draw.letters), draw_frame.time_s)
        return tickets


def save_ticket(ticket, out_dir, game):
    folder = out_dir / ticket.ticket_id
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "frame_grid.jpg"), ticket.grid_frame.image)
    cv2.imwrite(str(folder / "grid.png"), ticket.grid.rectified)
    cv2.imwrite(str(folder / "frame_draw.jpg"), ticket.draw_frame.image)
    cv2.imwrite(str(folder / "draw.png"), ticket.draw.crop)
    record = ticket.to_record(game)
    (folder / "ticket.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract tickets from local videos.")
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--game", type=Path, required=True, help="game config, see configs/games/")
    parser.add_argument("--pipeline", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    game = load_game(args.game)
    extractor = Extractor(game, load_pipeline(args.pipeline))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "tickets.jsonl", "a", encoding="utf-8") as index:
        for video in args.videos:
            tickets = extractor.process(video)
            for ticket in tickets:
                record = save_ticket(ticket, args.out / video.stem, game)
                index.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.info("%s: %d ticket(s)", video.name, len(tickets))


if __name__ == "__main__":
    main()
