#!/usr/bin/env python3
"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

Model (see METHOD.md for the full write-up):
  * Earners are modelled per trophy, not rates: E_i(t) earners out of N(t)
    tracked players. Earners only grow; rates move because N grows
    (dilution) and because players keep earning (progression).
  * Each trophy has a constant per-elapsed-hour hazard h_i, calibrated so
    the old cohort's earn rate at the last read equals the observed rate:
    h_i = -ln(1 - r_i) / H0, H0 = hours from release to the last read.
    Where the history shows real movement, h_i is recalibrated by
    matching the last-24h earner growth (geometric mean with the prior).
  * A launch-night intensity factor: the hazard is calibrated over the
    span from release to the last read, but launch-night play is far
    more intense than later play, so future earning is scaled by
    g = clamp(0.25 + 0.75*H0/48, 0.25, 1.0), about 0.3 for a
    2.5-hour-old case and 1.0 once the history spans two days.
  * Tracked players N(t) follow a power law in (1 + hours since release),
    fitted by log-log regression on the history, shrunk toward a prior
    slope of 0.6 (the launch-night refresh taught us Exophase's counts
    jump in steps, so the fit is never trusted alone).
  * The forecast splits players into the old cohort (keeps earning with
    hazard h_i) and new arrivals (earn with the same hazard over the
    shorter time since they arrived), integrated in 24 log-spaced
    arrival buckets. Median rate = forecast earners / forecast players.
  * Quantile spread on the logit scale combines a random-walk term
    (0.5*sqrt(days)), player-growth uncertainty, refresh-step risk and
    model uncertainty; rare trophies get extra spread.

Deterministic: no randomness, no wall-clock reads. Standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

RELEASE_EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()
Z_VALUES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
TIER_PRIOR = {"platinum": 0.5, "gold": 5.0, "silver": 15.0, "bronze": 30.0}
BUCKETS = 24
ALPHA_PRIOR = 0.6


def parse_time(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def logit_pct(rate):
    p = clamp(rate, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(logit):
    if logit >= 0:
        return 100.0 / (1.0 + math.exp(-logit))
    exp_l = math.exp(logit)
    return 100.0 * exp_l / (1.0 + exp_l)


def hours_since_release(epoch):
    return max(0.0, (epoch - RELEASE_EPOCH) / 3600.0)


def intensity_factor(h0):
    """Future earning intensity relative to the calibration span.

    Launch-night play is far more intense than later play: a hazard
    calibrated over the first 2.5 hours overstates how fast players
    keep earning. The factor rises with the length of the calibration
    span and reaches 1.0 once the history spans two days.
    """
    return clamp(0.25 + 0.75 * h0 / 48.0, 0.25, 1.0)


def fit_player_growth(history):
    """Log-log slope of tracked players vs (1 + hours since release).

    Returns (alpha, from_prior). Falls back to the prior slope when the
    history shows fewer than three distinct player counts: on launch
    night Exophase's page sat unchanged for hours, so a flat history
    says nothing about growth.
    """
    points = []
    for read in history:
        players = read.get("players_tracked") or 0
        if players <= 0:
            continue
        h = hours_since_release(parse_time(read["at"]))
        points.append((math.log1p(h), math.log(players)))
    distinct = {p[1] for p in points}
    if len(points) < 3 or len(distinct) < 3:
        return ALPHA_PRIOR, True
    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    if sxx <= 1e-12:
        return ALPHA_PRIOR, True
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    slope = clamp(sxy / sxx, 0.05, 1.5)
    return 0.5 * slope + 0.5 * ALPHA_PRIOR, False


def calibrate_hazard(h_prior, window, idx, h_first, h_last):
    """Recalibrate one trophy's hazard from its last-24h earner growth.

    Solves (bisection) for the hazard that reproduces the observed
    earner growth from the window's first read to its last, then takes
    the geometric mean with the prior. Returns the prior unchanged when
    the window shows no player growth: a flat history is a stale page,
    not evidence of a zero hazard.
    """
    if len(window) < 3:
        return h_prior
    first = window[0]
    last = window[-1]
    n_first = first.get("players_tracked") or 0
    n_last = last.get("players_tracked") or 0
    if n_first <= 0 or n_last <= 1.05 * n_first:
        return h_prior
    r_first = first["rates"][idx]
    r_last = last["rates"][idx]
    if r_first is None or r_last is None:
        return h_prior
    e_first = float(r_first) * n_first / 100.0
    e_last = float(r_last) * n_last / 100.0
    if e_last <= e_first:
        return h_prior * 0.25
    d_h = max(0.0, h_last - h_first)
    if d_h <= 0:
        return h_prior

    def predicted(hazard):
        old_gain = (n_first - e_first) * (1.0 - math.exp(-hazard * d_h))
        new_gain = (n_last - n_first) * (1.0 - math.exp(-hazard * d_h / 2.0))
        return e_first + old_gain + new_gain

    low, high = 1e-9, 10.0
    if predicted(high) < e_last:
        h_fit = high
    else:
        for _ in range(60):
            mid = 0.5 * (low + high)
            if predicted(mid) < e_last:
                low = mid
            else:
                high = mid
        h_fit = 0.5 * (low + high)
    h_fit = clamp(h_fit, 0.05 * h_prior, 20.0 * h_prior)
    return math.sqrt(h_prior * h_fit)


def quantiles(median, spread):
    base = logit_pct(median)
    out = []
    for z in Z_VALUES:
        q = inv_logit_pct(base + z * spread)
        out.append(round(clamp(q, 0.01, 99.99), 4))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_trophy(rate, n0, h0, h_target, alpha, hazard, horizon_days,
                    from_prior):
    """Return the five quantiles for one trophy."""
    if rate >= 99.5:
        return [99.99, 99.99, 99.99, 99.99, 99.99]
    if n0 <= 0 or h_target <= h0:
        return quantiles(clamp(rate, 0.01, 99.99), 0.8)
    n_target = n0 * ((1.0 + h_target) / (1.0 + h0)) ** alpha
    # Future earning runs at the intensity factor: launch-night play is
    # far more intense than later play, so a hazard calibrated over the
    # first hours overstates how fast players keep earning.
    g = intensity_factor(h0)
    h_eff = h0 + g * (h_target - h0)
    # Old cohort: keeps earning with hazard h over its effective span.
    e_old = n0 * (1.0 - math.exp(-hazard * h_eff))
    e_old = clamp(e_old, rate * n0 / 100.0, float(n0))
    # New arrivals in log-spaced buckets; each earns only over the
    # (intensity-scaled) time since it arrived.
    def n_at(t):
        return n0 * ((1.0 + t) / (1.0 + h0)) ** alpha

    e_new = 0.0
    ratio = (h_target / h0) ** (1.0 / BUCKETS)
    boundary = h0
    for _ in range(BUCKETS):
        nxt = boundary * ratio
        arrived = n_at(nxt) - n_at(boundary)
        if arrived > 0:
            mid = math.sqrt(boundary * nxt)
            prob = 1.0 - math.exp(-hazard * g * (h_target - mid))
            e_new += arrived * prob
        boundary = nxt
    median = 100.0 * (e_old + e_new) / n_target
    median = clamp(median, 0.01, 99.99)
    spread = math.sqrt(
        (0.5 * math.sqrt(horizon_days)) ** 2
        + 0.60 ** 2
        + 0.50 ** 2
        + 0.35 ** 2
        + (0.30 if rate < 1.0 else 0.15 if rate < 5.0 else 0.0) ** 2
        + (0.30 if from_prior else 0.0) ** 2
    )
    return quantiles(median, spread)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(0.0, (target - at) / 86400.0)
    if history:
        last = history[-1]
        h0 = hours_since_release(parse_time(last["at"]))
        n0 = last.get("players_tracked") or 0
    else:
        h0 = hours_since_release(at)
        n0 = 0
    h0 = max(h0, 0.5)
    h_target = max(hours_since_release(target), h0)
    alpha, from_prior = fit_player_growth(history)
    # Calibration window: the last 24 hours of reads.
    if history:
        last_at = parse_time(history[-1]["at"])
        window = [r for r in history if parse_time(r["at"]) >= last_at - 86400]
        if len(window) < 3:
            window = history[-3:] if len(history) >= 3 else history
        h_first = hours_since_release(parse_time(window[0]["at"]))
        h_last = hours_since_release(parse_time(window[-1]["at"]))
    else:
        window = []
        h_first = h_last = h0
    rows = []
    for idx, trophy in enumerate(trophies):
        rate = None
        for read in reversed(history):
            value = read["rates"][idx]
            if value is not None:
                rate = float(value)
                break
        if rate is None:
            rate = TIER_PRIOR.get(trophy.get("tier"), 20.0)
        hazard = -math.log(1.0 - clamp(rate, 0.01, 99.99) / 100.0) / h0
        hazard = calibrate_hazard(hazard, window, idx, h_first, h_last)
        rows.append({
            "id": trophy["id"],
            "q": forecast_trophy(rate, n0, h0, h_target, alpha, hazard,
                                 horizon_days, from_prior),
        })
    return {"trophies": rows}


def main(argv):
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    case_files = sorted(input_dir.glob("*.in"))
    if not case_files:
        print("no case files in %s" % input_dir, file=sys.stderr)
        return 1
    for case_file in case_files:
        with open(case_file, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        out_path = output_dir / (case_file.stem + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
