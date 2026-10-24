"""Letter matrix of a grid and the words it contains."""

import unicodedata
from dataclasses import dataclass

BLOCK = "#"     # black cell
UNKNOWN = "?"   # cell that could not be read


@dataclass(frozen=True)
class Word:
    text: str
    direction: str   # "horizontal" or "vertical"
    cells: tuple     # (row, col) of each letter


def normalize(word):
    """Uppercase, without accents."""
    word = unicodedata.normalize("NFD", word.strip().upper())
    return "".join(c for c in word if unicodedata.category(c) != "Mn")


def load_dictionary(path):
    with open(path, encoding="utf-8") as f:
        return frozenset(normalize(line) for line in f if line.strip())


def build_matrix(letters, rows, cols):
    """letters: {(row, col): letter or BLOCK}. Missing cells become UNKNOWN."""
    return tuple(tuple(letters.get((r, c), UNKNOWN) for c in range(cols)) for r in range(rows))


def _is_letter(ch):
    return ch not in (BLOCK, UNKNOWN, "")


def extract_words(matrix, min_length=2):
    """Runs of at least min_length letters, rows first, then columns."""
    n_rows, n_cols = len(matrix), len(matrix[0]) if matrix else 0
    lines = [("horizontal", [(r, c) for c in range(n_cols)]) for r in range(n_rows)]
    lines += [("vertical", [(r, c) for r in range(n_rows)]) for c in range(n_cols)]

    words = []
    for direction, line in lines:
        run = []
        for pos in line + [None]:
            if pos is not None and _is_letter(matrix[pos[0]][pos[1]]):
                run.append(pos)
                continue
            if len(run) >= min_length:
                words.append(Word("".join(matrix[r][c] for r, c in run), direction, tuple(run)))
            run = []
    return words


def grid_letters(matrix):
    return frozenset(ch for row in matrix for ch in row if _is_letter(ch))


def cell_difference(a, b):
    """Number of cells that differ between two matrices."""
    return sum(x != y for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))
