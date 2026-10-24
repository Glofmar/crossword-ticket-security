# Generating statistically secure tickets

Goal: $P(R = r \mid G = g) = P(R = r)$ for every grid $g$ and outcome $r$, i.e.
$I(G; R) = 0$ (see the [README](../README.md)). The outcome is the number $k$ of
completed words, drawn from a target distribution $\pi$ set by the publisher (it fixes
the share of winning tickets and the average payout).

The grid and the outcome are made in two separate steps, so that the grid cannot say
anything about the outcome.

## 1. Generate a grid

1. Pick a layout. A layout says where the words go: their position, direction and
   length. Real grids all use the same number of words of each length, laid out
   differently, so we pick a layout following such a fixed pattern of lengths.
2. Fill the words one by one. For each slot, take at random a dictionary word of
   the right length that agrees with the letters already placed where it crosses other
   words. Starting with the longest words and the slots with the most crossings makes
   it easier.
3. Go back when stuck (backtracking). If no word fits a slot, the previous choice
   was bad: go back to the previous slot, try another word there, and continue. If no
   word works for a slot at all, the layout cannot be filled with these words, and we
   start again with another layout.
4. Check that every outcome is possible. Once the grid is full, list every valid
   draw (the sets of hidden letters allowed by the rules: number of letters, no repeated
   letter, minimum number of vowels, all taken from the grid) and count the words each
   one completes. The grid is kept only if every outcome $k$, from a losing ticket
   to the top prize, is produced by at least one draw. Otherwise, seeing the grid would
   rule out some prizes, so it is thrown away and a new grid is generated.

This listing is fast: a grid has about twenty distinct letters, which gives at most a
few million draws, checked in well under a second.

## 2. Make a ticket

1. Generate a grid $g$ as above, and keep the draws listed at step 1.4, grouped by the
   number of words they complete.
2. Draw the outcome $k \sim \pi$, without looking at $g$.
3. Pick at random one of the draws that complete exactly $k$ words. There is always one,
   since the grid passed the check.
4. Print $g$ in the open and the draw under the scratch coating, in random order.

## Why it is secure

The grid is built without knowing the outcome, the outcome is drawn without looking at
the grid, and every grid allows every outcome. Whatever the grid, the probability of
each outcome is the one set by the publisher:

$$P(R = r \mid G = g) = \pi(r) = P(R = r).$$

Two shortcuts must be avoided:

- Drawing $k$ first, then looking for a grid that can produce it: grids that cannot
  reach high prizes would then appear more often on losing tickets.
- Drawing the hidden letters at random and reading $k$ from them: grids with short
  words or common letters would then win more often.

In both cases the outcome depends on the grid.
