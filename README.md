[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Szeven/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Szeven/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast — Week One

Forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/) at
2026-11-26T00:00:00Z: five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## The program

`main.py` (Python 3.13, standard library only, no network) reads a case file
(`makefun/gta6-trophies-case/v1`) from `$MAKEFUN_INPUT_DIR` (else `/input/`) and
writes one forecast file per case to `$MAKEFUN_OUTPUT_DIR` (else `/output/`):

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/<case>.out : {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}
```

## The model

Exophase's rate is earned / tracked, and over a week both halves move: the
tracked player base explodes (dilution pushes every rate down) while players
progress through the story (progression pushes early and easy trophies up).
The forecast models both forces instead of assuming rates stay put:

- **Player growth.** The tracked count is extrapolated from the case history
  (log-linear fit when the history shows growth, otherwise a saturating prior
  that grows ~3x over 90 minutes and ~55x over 4.5 days).
- **Population rate per trophy.** Each trophy is classified from its own
  description (story chapter, story mission, grind/collectible, easy one-off,
  100% completion, platinum) and given a population earn probability `P` and a
  typical hours-to-earn `T`. New players earn it with probability
  `P * max(0, 1 - T / (0.3 * horizon_hours))`: nobody speed-runs Chapter 1 in
  90 minutes, but most active players get there in a week.
- **Mixing.** Earned counts at the target mix the players already tracked
  (who keep progressing, slowly) with the newcomers (who earn at the
  horizon-adjusted rate): `E' = E + N0 * q_exist + (N' - N0) * p_new`.
- **Spread.** Quantiles sit on the logit scale around the median with a spread
  that combines a random-walk diffusion (`0.5 * sqrt(days)`) with class
  uncertainty and growth uncertainty, so a refresh shock or a slow week stays
  inside the 90% band.

See [METHOD.md](METHOD.md) for the full method, the assumptions, and the
numbers it produces.

## Practice check

The program runs end to end on the kit's practice cases (`p00`, `p01`) and
writes valid forecast files for all 55 trophies (5 quantiles each,
non-decreasing, 0–100, 3,228 bytes per file). On the practice cases every
median moves down from the last read (55 of 55 down) — dilution over 90
minutes, matching the one refresh the house actually saw, which lowered
every rate but the 100% one. The trophy at 100% stays at 99.7%. Chapter 1
moves 1.25% → 1.11%; the platinum 0.63% → 0.19%.

On a synthetic final case (6,510-minute horizon, tracked base growing
638 → 45,000 in the history) the medians split 32 up / 23 down: Chapter 1
rises to ~44%, Chapter 2 to ~19%, while the platinum falls to ~0.01% and
the 250-stunt grind to ~0.02% — progression wins early, dilution wins late.

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
