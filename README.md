# crossword-ticket-security

This repository studies the statistical security of crossword lottery tickets. The
game is presented [here](docs/game.md).

Unlike most lottery tickets, a crossword ticket shows something specific to each
ticket before it is scratched: the grid and its words. The letters that decide the
prize are hidden, but the grid is visible.

## Statistical security

Let

- $G$ be the visible grid,
- $R$ be the outcome of the ticket (losing, or the amount won).

For the grid to reveal nothing about the prize, we need

$$\boxed{P(R = r \mid G = g) = P(R = r)}$$

for every outcome $r$ and every visible grid $g$, which can be rewritten as

$$\boxed{I(G; R) = 0}$$

where $I(G; R)$ is the mutual information between the grid and the outcome.

We also propose an algorithm to generate tickets so that this property holds, see
[docs/secure_generation.md](docs/secure_generation.md).

The statistical analysis of real tickets, which tests whether the grid predicts the
prize, is in [docs/analysis.md](docs/analysis.md).

## Collecting the data

The study needs many (grid, outcome) pairs. The code finds every ticket shown in a
video and turns it into data, with a detection model, OCR and checks against the rules
of the game.

All the settings are in three `.yaml` files, so the code holds no value specific to a
game:

- `configs/games/<game>.yaml` describes a game and its rules, so the same code can be
  used for other crossword games. `example_crossword_9x9.yaml` describes a made-up game.
- `configs/pipeline.yaml` holds the parameters of the processing (frame step, image
  processing settings, thresholds).
- `configs/training.yaml` holds the settings used to train the CNN and the YOLO model.

### Pipeline

1. The input is a video containing tickets.
2. A frame is captured every `step_s` seconds (set in `configs/pipeline.yaml`).
3. The grid is located in the frame and straightened with OpenCV image processing
   (colour mask, Hough lines on the border, perspective correction).
4. Each cell of the grid is cut into a separate image, converted to grayscale and
   resized to 32x32.
5. A CNN trained beforehand gives a letter, or blank, for each cell.
6. Coherence is checked: the words formed by these letters must follow the rules of
   the game (for instance, every word must be in the dictionary). If the grid is
   coherent, it is kept.
7. In the following frames, a YOLO model trained beforehand looks for the panel of
   hidden letters of the ticket. Its cells are read by the same CNN, and the letters
   must also follow the rules of the game (number of letters, no duplicate, all
   present in the grid...). A panel that is not fully scratched yet is skipped.
8. The process starts again on the next frames. A grid that differs from an already
   kept grid by `duplicate_max_cells` cells or fewer is rejected, as it is most likely
   the same ticket seen again.

For each ticket, `results/<video>/<ticket_id>/` contains a `ticket.json` (grid, words,
hidden letters, completed words, prize, value of every rule) and the images used.
`results/tickets.jsonl` has one line per ticket.

### Rules

A game file gives the size of the grid and of the hidden panel, the alphabet, the
dictionary, the payout, the colours of the ticket, the models to use, and the rules.
Rules are written like this:

```yaml
rules:
  grid:
    - {rule: word_count, equals: 16}
    - {rule: words_in_dictionary, min: 1.0}
  draw:
    - {rule: letter_count, equals: 12}
    - {rule: distinct_letters, equals: 12}
    - {rule: vowels, min: 2}
    - {rule: letters_outside_grid, max: 0}
    - {rule: completed_words, min: 1}
```

The available rules are in `src/crosswordsec/rules.py`.

### Usage

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
python -m crosswordsec.extract video.mp4 --game configs/games/example_crossword_9x9.yaml --out results/
```

## Training

The repository also contains the code used to train the CNN and the YOLO model.

```bash
# one sub-folder of images per label: A/, B/, ..., EMPTY/
python -m crosswordsec.training.train_cells cells/ --out models/cells.pth

# YOLOv8-OBB export with data.yaml and train/, valid/, test/
python -m crosswordsec.training.train_detector annotations/ --out models/detector_obb.pt
```

Both scripts print an evaluation at the end. `models/README.md` describes how the
current models were trained and their scores.

## Tests

```bash
python -m pytest
```

## Notes

We do not distribute any model weights, dataset or dictionary, so as not to infringe
intellectual property rights. To run the code you need your own videos, annotations
and dictionary. `models/README.md` explains how the models were trained and
`data/README.md` what data is expected.

Parts of the code and documentation were written with the help of an AI coding agent.
All results were checked by running the code on real data.

## License

AGPL-3.0, see [LICENSE](LICENSE). The detector is trained and run with
[Ultralytics](https://github.com/ultralytics/ultralytics), which is itself released
under AGPL-3.0.
