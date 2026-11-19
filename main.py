#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target.

For every trophy in a case file, output five quantiles (5/25/50/75/95) of
its earn rate at the target instant.

The model
---------
An Exophase rate is earned / tracked, and over a week both halves move:

* Dilution. The tracked player base explodes after launch, and the players
  who arrive later have mostly not earned a given trophy yet, so every rate
  is pulled down toward the trophy's population rate.
* Progression. The players already tracked keep playing, so early-story and
  easy trophies climb as the week goes on.

The forecast mixes both forces per trophy:

1. Growth. The tracked count at the target is extrapolated from the case
   history. When the history shows the count moving, a log-linear fit of the
   count against time is projected forward with the growth rate halved (the
   flood slows). When the history is flat, a saturating prior is used:
   the count grows about 3x over 90 minutes and about 55x over 4.5 days.
2. Population rate. Each trophy is classified from its own description
   (story chapter, story mission, grind, easy one-off, completion,
   platinum) and given a population earn probability P and a typical
   hours-to-earn T. A newcomer who joins during the horizon earns it with
   probability P * max(0, 1 - T / (0.3 * horizon_hours)): nobody finishes
   Chapter 1 in 90 minutes, but most active players get there in a week.
3. Mixing. Earned counts at the target are the last earned count, plus the
   tracked players who earn it during the horizon (slowly, since the
   non-earners left are the slow ones), plus the newcomers who earn it at
   the horizon-adjusted rate.
4. Spread. Quantiles sit on the logit scale around the median. The spread
   combines a random-walk diffusion (0.5 * sqrt(days)) with a per-class
   uncertainty and a growth uncertainty (larger when the history is flat
   and no refresh has been seen yet), so a refresh shock stays inside the
   90% band instead of outside it the way a narrow band does.

Everything is deterministic: the same input always produces the same
output, byte for byte. Standard library only, no network, no randomness.
"""

import json
import math
import os
import re
import sys
from pathlib import Path

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

# Share of wall-clock time a typical player actually spends playing.
PLAY_SHARE = 0.3
# Random-walk diffusion of the logit rate, per sqrt(day).
DIFFUSION = 0.5


def logit(p):
    """Log-odds of a probability, clamped away from 0 and 1."""
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    """Inverse of logit, numerically safe for large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _chapter_number(text):
    """Chapter number from a description like 'Complete Chapter 3', else None."""
    m = re.search(r"chapter\s+(\d+)", text)
    return int(m.group(1)) if m else None


def _largest_number(text):
    """Largest integer mentioned in a description, else None."""
    nums = [int(n) for n in re.findall(r"\d+", text)]
    return max(nums) if nums else None


def classify(trophy, last_rate):
    """Classify a trophy from its tier, name and description.

    Returns (kind, population_rate, hours_to_earn). The population rate is
    the share of all week-one players who would eventually earn the trophy;
    the hours-to-earn is how long a typical player needs to get there.
    """
    tier = (trophy.get("tier") or "").lower()
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    text = name + " " + desc

    if tier == "platinum":
        return ("platinum", 0.001, 200.0)
    if "100% completion" in desc:
        return ("completion", 0.003, 150.0)
    if last_rate is not None and last_rate >= 99.5:
        # Everybody tracked already has it; newcomers get it too.
        return ("universal", 0.995, 0.5)

    chapter = _chapter_number(desc)
    if chapter is not None:
        # Early chapters are reached by many; later ones by fewer, later.
        pop = 0.58 * (0.72 ** (chapter - 1))
        return ("story", pop, 8.0 + 10.0 * (chapter - 1))
    if "complete the final mission" in desc:
        return ("story", 0.12, 60.0)
    if "mission strands" in desc:
        return ("story", 0.06, 35.0)
    if "final mission for" in desc:
        return ("story", 0.20, 25.0)

    biggest = _largest_number(desc)
    grind_words = ("stunt", "challenge", "location", "vehicle", "race",
                   "vault", "shop", "spot", "card set", "garage", "waink",
                   "meditate", "drone race", "gold medal", "bank vault",
                   "collectible", "explore all")
    if biggest is not None and biggest >= 3 and any(w in text for w in grind_words):
        # A counted grind: the bigger the count, the longer and rarer.
        if biggest >= 100:
            hours, pop = 90.0, 0.015
        elif biggest >= 20:
            hours, pop = 40.0, 0.030
        else:
            hours, pop = 25.0, 0.050
        if tier == "silver":
            pop *= 0.7
        return ("grind", pop, hours)
    if "earn $" in desc or "spend $" in desc:
        return ("grind", 0.040, 25.0)
    if any(w in desc for w in ("explore all", "every spot", "every location",
                               "each gang", "purchase all", "gold medal in all")):
        return ("grind", 0.030, 40.0)

    if desc.startswith("complete ") and "arcade" not in desc:
        # A named story mission with no count attached to it.
        return ("story", 0.20, 25.0)

    # A one-off feat: quick to do once a player tries it.
    pop = 0.18
    if tier == "silver":
        pop *= 0.6
    elif tier == "gold":
        pop *= 0.4
    return ("easy", pop, 3.0)


# Extra logit spread per class: how uncertain the population rate is.
CLASS_SPREAD = {
    "story": 0.35,
    "grind": 0.55,
    "easy": 0.45,
    "platinum": 0.70,
    "completion": 0.70,
    "universal": 0.15,
}


def _parse_time(value):
    """ISO 8601 UTC timestamp to epoch seconds (naive, Z-suffixed)."""
    if not value:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def growth_factor(history, horizon_hours):
    """Projected tracked-players multiplier over the horizon.

    Returns (factor, fitted): fitted is True when the history itself shows
    the count moving and the fit is used, else the saturating prior.
    """
    points = []
    for read in history or []:
        n = read.get("players_tracked")
        t = _parse_time(read.get("at"))
        if isinstance(n, (int, float)) and n and n > 0 and t is not None:
            points.append((t, float(n)))
    if len(points) >= 3:
        distinct = {n for _, n in points}
        if len(distinct) >= 2 and points[-1][1] > points[0][1]:
            t0 = points[0][0]
            xs = [(t - t0) / 3600.0 for t, _ in points]
            ys = [math.log(n) for _, n in points]
            n = len(xs)
            mx = sum(xs) / n
            my = sum(ys) / n
            denom = sum((x - mx) ** 2 for x in xs)
            if denom > 0:
                slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
                if slope > 0:
                    # Growth slows: project at half the fitted rate, capped.
                    factor = math.exp(slope * horizon_hours * 0.5)
                    return (min(factor, 80.0), True)
    # Flat history: no refresh has been seen yet, so use the prior curve.
    horizon_days = horizon_hours / 24.0
    return (1.0 + 55.0 * (1.0 - math.exp(-horizon_days / 1.5)), False)


def forecast_one(trophy, last_rate, players_now, factor, fitted, horizon_hours):
    """Five quantiles of one trophy's earn rate at the target, in percent."""
    kind, pop, hours = classify(trophy, last_rate)
    days = horizon_hours / 24.0

    if kind == "universal":
        median = 99.7
        sigma = CLASS_SPREAD["universal"]
    elif last_rate is None:
        # No read ever showed this trophy: fall back to the class prior,
        # reached slowly over the horizon, with a wide spread.
        median = 100.0 * pop * (1.0 - math.exp(-horizon_hours / (3.0 * hours)))
        median = min(max(median, 0.01), 99.9)
        sigma = math.sqrt((DIFFUSION * math.sqrt(days)) ** 2
                          + CLASS_SPREAD[kind] ** 2 + 0.3 ** 2)
    else:
        rate = last_rate / 100.0
        # Newcomers: only those with enough play time in the horizon earn it.
        p_new = pop * max(0.0, 1.0 - hours / (PLAY_SHARE * horizon_hours))
        p_new += 0.01 * pop
        # Players already tracked: the slow remainder keeps earning, slowly.
        # They have already had hours to earn it and have not, so their
        # hazard is low: the timescale is ten times the typical time-to-earn.
        q_exist = max(0.0, pop - rate) * (1.0 - math.exp(-horizon_hours / (10.0 * hours)))
        earned = rate * players_now
        target_players = players_now * factor
        earned += players_now * q_exist
        earned += (target_players - players_now) * p_new
        median = 100.0 * earned / target_players
        median = min(max(median, 0.01), 99.9)
        grow_unc = 0.20 if fitted else 0.45
        sigma = math.sqrt((DIFFUSION * math.sqrt(days)) ** 2
                          + CLASS_SPREAD[kind] ** 2 + grow_unc ** 2)

    centre = logit(median / 100.0)
    quantiles = []
    for z in Z_SCORES:
        q = 100.0 * sigmoid(centre + z * sigma)
        q = min(max(q, 0.01), 99.99)
        quantiles.append(round(q, 4))
    # Keep the row non-decreasing after rounding.
    for i in range(1, len(quantiles)):
        quantiles[i] = max(quantiles[i], quantiles[i - 1])
    return quantiles


def last_observed(history, index):
    """Last non-null rate and players count for one trophy position."""
    rate = None
    players = None
    for read in reversed(history or []):
        rates = read.get("rates") or []
        if rate is None and index < len(rates) and rates[index] is not None:
            rate = float(rates[index])
        if players is None and read.get("players_tracked"):
            players = float(read["players_tracked"])
        if rate is not None and players is not None:
            break
    return rate, players


def process_case(case):
    """Forecast every trophy of one case; returns the output object."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon = float(case.get("horizon_minutes") or 0) / 60.0
    if horizon <= 0:
        horizon = 1.0
    factor, fitted = growth_factor(history, horizon)

    rows = []
    for index, trophy in enumerate(trophies):
        rate, players = last_observed(history, index)
        if players is None:
            players = 1000.0
        q = forecast_one(trophy, rate, players, factor, fitted, horizon)
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted(in_dir.glob("*.in"))
    if not cases:
        print("no case files in %s" % in_dir, file=sys.stderr)
        return 1
    for case_path in cases:
        with open(case_path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = process_case(case)
        out_path = out_dir / (case_path.stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
            fh.write("\n")
        medians = [row["q"][2] for row in result["trophies"]]
        print("%s: %d trophies, median range %.2f-%.2f%% -> %s"
              % (case_path.name, len(result["trophies"]),
                 min(medians), max(medians), out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
