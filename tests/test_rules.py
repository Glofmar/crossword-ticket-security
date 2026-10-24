from pathlib import Path

import pytest
import yaml

from crosswordsec.config import load_game, load_pipeline
from crosswordsec.rules import (DRAW_RULES, GRID_RULES, DrawContext, GridContext, all_passed, completed_words,
                                evaluate, parse_rules)

CONFIGS = Path(__file__).parent.parent / "configs"
GAME = CONFIGS / "games" / "example_crossword_9x9.yaml"
PIPELINE = CONFIGS / "pipeline.yaml"

DRAW_RULESET = parse_rules([
    {"rule": "letter_count", "equals": 12},
    {"rule": "distinct_letters", "equals": 12},
    {"rule": "vowels", "min": 2},
    {"rule": "letters_outside_grid", "max": 0},
    {"rule": "completed_words", "min": 1},
], DRAW_RULES, "draw")

GRID_LETTERS = frozenset("ABCDEFGHIJKLMNOPRSTU")
WORDS = ["BAIN", "CODE", "FLUTE"]
VOWELS = frozenset("AEIOU")


def draw_ok(letters):
    return all_passed(evaluate(DRAW_RULESET, DrawContext(list(letters), GRID_LETTERS, WORDS, VOWELS)))


def test_valid_draw():
    assert draw_ok("BAINCODEFLUT")
    assert completed_words(list("BAINCODEFLUT"), WORDS) == WORDS


@pytest.mark.parametrize("letters, broken", [
    ("BAINCODEFLUB", "distinct_letters"),       # repeated letter
    ("BAINCODEFLUZ", "letters_outside_grid"),
    ("BCDFGHJKLMNP", "vowels"),
    ("ABDEFGHIJKLM", "completed_words"),
    ("BAIN", "letter_count"),
])
def test_each_rule_can_reject(letters, broken):
    results = evaluate(DRAW_RULESET, DrawContext(list(letters), GRID_LETTERS, WORDS, VOWELS))
    assert broken in {r.rule for r in results if not r.passed}


def test_results_keep_measured_values():
    results = {r.rule: r for r in evaluate(DRAW_RULESET, DrawContext(list("BAINCODEFLUT"), GRID_LETTERS,
                                                                     WORDS, VOWELS))}
    assert results["vowels"].value == 5
    assert results["vowels"].bounds == "min=2"
    assert results["completed_words"].value == 3


def test_grid_rules():
    rules = parse_rules([{"rule": "word_count", "equals": 2}, {"rule": "words_in_dictionary", "min": 1.0}],
                        GRID_RULES, "grid")
    assert all_passed(evaluate(rules, GridContext(["CAT", "DOG"], frozenset({"CAT", "DOG"}))))
    assert not all_passed(evaluate(rules, GridContext(["CAT", "DOX"], frozenset({"CAT", "DOG"}))))
    assert not all_passed(evaluate(rules, GridContext(["CAT"], frozenset({"CAT"}))))


def test_unknown_rule_is_reported():
    with pytest.raises(ValueError, match="unknown rule 'voyels'"):
        parse_rules([{"rule": "voyels", "min": 3}], DRAW_RULES, "draw")


def test_unknown_bound_is_reported():
    with pytest.raises(ValueError, match="unknown bound"):
        parse_rules([{"rule": "vowels", "minimum": 3}], DRAW_RULES, "draw")


def test_rule_without_bound_is_reported():
    with pytest.raises(ValueError, match="at least one bound"):
        parse_rules([{"rule": "vowels"}], DRAW_RULES, "draw")


def test_shipped_configs_load():
    game = load_game(GAME)
    assert (game.grid_rows, game.grid_cols, game.draw_size) == (9, 9, 12)
    assert [r.name for r in game.draw_rules] == [
        "letter_count", "distinct_letters", "vowels", "letters_outside_grid", "completed_words"]
    assert game.payout[5] == 5
    assert game.appearance.grid_border_hsv.high.tolist() == [8, 255, 255]
    assert load_pipeline(PIPELINE).scan.duplicate_max_cells == 10


def _write(tmp_path, name, raw):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_game_rules_are_validated(tmp_path):
    raw = yaml.safe_load(GAME.read_text(encoding="utf-8"))
    raw["rules"]["draw"].append({"rule": "nonexistent", "min": 1})
    with pytest.raises(ValueError, match="nonexistent"):
        load_game(_write(tmp_path, "bad.yaml", raw))


def test_game_unknown_key_is_reported(tmp_path):
    raw = yaml.safe_load(GAME.read_text(encoding="utf-8"))
    raw["appearance"]["cell_background"]["colour"] = "green"
    with pytest.raises(ValueError, match="appearance.cell_background: unknown key"):
        load_game(_write(tmp_path, "bad.yaml", raw))


def test_pipeline_missing_key_is_reported(tmp_path):
    raw = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    del raw["grid_border"]["canny_low"]
    with pytest.raises(ValueError, match="grid_border: missing key.*canny_low"):
        load_pipeline(_write(tmp_path, "pipeline.yaml", raw))


def test_pipeline_unknown_section_is_reported(tmp_path):
    raw = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    raw["grid_cell"] = {}
    with pytest.raises(ValueError, match="unknown section"):
        load_pipeline(_write(tmp_path, "pipeline.yaml", raw))
