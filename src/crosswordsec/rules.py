"""Game rules from the config, e.g. `{rule: vowels, min: 3}`.

Each rule name maps to a function measuring something on the grid or the draw;
the config gives the bounds (equals, min, max). To add a rule, add a function
to GRID_RULES or DRAW_RULES.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GridContext:
    words: list
    dictionary: frozenset


@dataclass(frozen=True)
class DrawContext:
    letters: list
    grid_letters: frozenset
    words: list
    vowels: frozenset


def completed_words(letters, words):
    """Words that can be fully written with the draw letters."""
    available = set(letters)
    return [w for w in words if set(w) <= available]


def _dictionary_ratio(ctx):
    if not ctx.words:
        return 0.0
    return sum(w in ctx.dictionary for w in ctx.words) / len(ctx.words)


GRID_RULES = {
    "word_count": lambda ctx: len(ctx.words),
    "words_in_dictionary": _dictionary_ratio,
}

DRAW_RULES = {
    "letter_count": lambda ctx: len(ctx.letters),
    "distinct_letters": lambda ctx: len(set(ctx.letters)),
    "vowels": lambda ctx: sum(l in ctx.vowels for l in ctx.letters),
    "letters_outside_grid": lambda ctx: sum(l not in ctx.grid_letters for l in ctx.letters),
    "completed_words": lambda ctx: len(completed_words(ctx.letters, ctx.words)),
}

BOUNDS = ("equals", "min", "max")


@dataclass(frozen=True)
class Rule:
    name: str
    measure: object
    equals: float = None
    min: float = None
    max: float = None

    def holds(self, value):
        if self.equals is not None and value != self.equals:
            return False
        if self.min is not None and value < self.min:
            return False
        if self.max is not None and value > self.max:
            return False
        return True

    def bounds(self):
        return ", ".join(f"{b}={getattr(self, b)}" for b in BOUNDS if getattr(self, b) is not None)


@dataclass(frozen=True)
class RuleResult:
    rule: str
    value: float
    bounds: str
    passed: bool

    def to_record(self):
        return {"rule": self.rule, "value": self.value, "bounds": self.bounds, "passed": self.passed}


def parse_rules(entries, catalogue, section):
    rules = []
    for entry in entries or []:
        entry = dict(entry)
        name = entry.pop("rule", None)
        if name not in catalogue:
            raise ValueError(f"rules.{section}: unknown rule {name!r} (available: {', '.join(sorted(catalogue))})")
        unknown = set(entry) - set(BOUNDS)
        if unknown:
            raise ValueError(f"rules.{section}.{name}: unknown bound(s) {sorted(unknown)}, use {', '.join(BOUNDS)}")
        if not entry:
            raise ValueError(f"rules.{section}.{name}: needs at least one bound ({', '.join(BOUNDS)})")
        rules.append(Rule(name, catalogue[name], **entry))
    return tuple(rules)


def evaluate(rules, ctx):
    results = []
    for rule in rules:
        value = rule.measure(ctx)
        results.append(RuleResult(rule.name, value, rule.bounds(), rule.holds(value)))
    return results


def all_passed(results):
    return all(r.passed for r in results)
