from crosswordsec.game import BLOCK, UNKNOWN, build_matrix, cell_difference, extract_words, grid_letters, normalize


def matrix_from(rows):
    return build_matrix({(r, c): ch for r, row in enumerate(rows) for c, ch in enumerate(row)},
                        len(rows), len(rows[0]))


def test_normalize_strips_accents():
    assert normalize(" Été ") == "ETE"


def test_extract_words_rows_then_columns():
    words = extract_words(matrix_from(["CAT", "A#O", "RUN"]))
    assert [(w.text, w.direction) for w in words] == [
        ("CAT", "horizontal"), ("RUN", "horizontal"), ("CAR", "vertical"), ("TON", "vertical")]
    assert words[2].cells == ((0, 0), (1, 0), (2, 0))


def test_min_word_length():
    m = matrix_from(["AB#CDE"])
    assert [w.text for w in extract_words(m, min_length=2)] == ["AB", "CDE"]
    assert [w.text for w in extract_words(m, min_length=3)] == ["CDE"]


def test_rectangular_grid():
    m = matrix_from(["ABCD", "E##F"])
    assert [w.text for w in extract_words(m)] == ["ABCD", "AE", "DF"]


def test_unknown_cells_break_words():
    m = build_matrix({(0, 0): "A", (0, 1): "B", (0, 3): "C", (0, 4): "D"}, 1, 5)
    assert m[0][2] == UNKNOWN
    assert [w.text for w in extract_words(m)] == ["AB", "CD"]


def test_grid_letters_ignores_blocks():
    assert grid_letters(matrix_from(["AB", f"{BLOCK}C"])) == frozenset("ABC")


def test_cell_difference():
    a, b = matrix_from(["CAT", "A#O", "RUN"]), matrix_from(["CAT", "A##", "RUN"])
    assert cell_difference(a, a) == 0
    assert cell_difference(a, b) == 1
