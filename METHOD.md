# Method — GTA 6 trophy rarity forecast (round 2)

## The problem in one line

An earn rate is a ratio, `earners / players tracked`. Over a week both the
numerator and the denominator move a lot: new players arrive (dilution —
every rate falls) while existing players keep playing (progression — every
rate rises). Forecasting the rate directly, e.g. by extrapolating its recent
trend, confuses the two. This entry models **earners** and **players**
separately, then divides.

## The model

### 1. Per-trophy earning hazard

Each trophy `i` has a constant hazard `h_i` per elapsed hour: a player who
has not earned it yet earns it with probability `1 - exp(-h_i * t)` after
`t` hours. The hazard is calibrated from the last read alone:

```
h_i = -ln(1 - r_i) / H0
```

where `r_i` is the trophy's last observed rate (as a fraction) and `H0` is
the hours from release (2026-11-19T00:00Z) to the last read. A trophy at
1.1% after 2.5 hours gets a small hazard; a trophy at 80% gets a large one.
This is a prior, not a fit: it says "if the trophy's current rate was built
by a constant hazard since release, this is the hazard".

Where the history shows real movement, the hazard is recalibrated. Over the
last 24 hours of reads, the model solves (by bisection) for the hazard that
reproduces the observed earner growth from the window's first read to its
last, then takes the geometric mean of that fit and the prior. If the
window is flat (no player growth — the launch-night reads are all the same
stale page), the prior is kept untouched: a flat history is a data artefact,
not evidence of a zero hazard.

### 1b. Launch-night intensity

The hazard is calibrated over the span from release to the last read, but
launch-night play is far more intense than later play: in the first hours
everyone plays at once, and afterwards people sleep, work and play a few
hours a day. Future earning is therefore scaled by an intensity factor
`g = clamp(0.25 + 0.75*H0/48, 0.25, 1.0)`, about 0.3 for a case whose
history spans 2.5 hours and 1.0 once it spans two days. Without it, a
practice case's median rises above the last rate even though the known
answer is that every rate fell — the old cohort cannot keep earning at
launch-night pace. The factor applies to future earning only, never to
the calibration window.

### 2. Player growth

Tracked players `N(t)` follow a power law in hours since release,

```
N(t) = N0 * ((1 + t) / (1 + H0))^alpha
```

with `alpha` fitted by log-log least squares over the whole history and
shrunk halfway toward a prior of 0.6. If the history shows fewer than three
distinct player counts, the prior is used alone. The shrinkage matters:
Exophase refreshes its counts in steps, so a fit over two days should not
be trusted to extrapolate 4.5 days on its own.

### 3. Old cohort vs new arrivals

The forecast splits the target population:

- **Old cohort** (`N0` players): keeps earning with hazard `h_i` over the
  whole span to the target, so its earners are `N0 * (1 - exp(-h_i * H_target))`.
- **New arrivals** (`N_target - N0` players): they arrive between the last
  read and the target, so each has less time to earn. Arrivals are split
  into 24 log-spaced buckets; a bucket arriving at time `a` earns with
  probability `1 - exp(-h_i * (H_target - a))`.

Median rate = forecast earners / forecast players. A trophy whose hazard is
high keeps up with dilution; a trophy whose hazard is ~0 dilutes almost
purely. The trophy at 100% is special-cased to stay at 99.99%.

### 4. Spread (quantiles)

Quantiles are set on the logit scale:

```
q_tau = inv_logit(logit(median) + z_tau * s)
z = (-1.645, -0.674, 0, 0.674, 1.645)
s = sqrt((0.5*sqrt(days))^2 + 0.60^2 + 0.50^2 + 0.35^2 + rare^2 + flat^2)
```

- `0.5*sqrt(days)` — the house random-walk term.
- `0.60` — player-growth uncertainty (the target player count is a fit,
  and Exophase's counts jump in steps).
- `0.50` — refresh-step risk: the answer is the page as of its last refresh
  before the target, which may lag the target.
- `0.35` — model uncertainty (constant hazard is a simplification; story
  trophies really have spiky hazards).
- `rare` — `0.30` for trophies under 1%, `0.15` under 5%: a tiny rate pins
  the hazard weakly.
- `flat` — `0.30` when the player-growth slope came from the prior alone
  (no growth visible in the history).

For the final (4.52 days) this gives `s ≈ 1.36`; for the 90-minute
practice cases `s ≈ 0.96`. The practice cases straddle Exophase's first
refresh, which no model can foresee from flat history, so a wide spread
there is honest, not a bug.

## What it does not do

- No network, no model calls, no reading of Exophase at run time.
- No per-trophy description parsing: the hazard is calibrated from the
  observed rate, which already encodes how hard the trophy is.
- No trend extrapolation on the logit rate: over 4.5 days a straight line
  in logit space can run away for rare trophies; the hazard model is
  bounded by construction (rates stay in [0, 100]).
- Deterministic: no randomness, no wall-clock reads, no dict-order
  dependence — two runs on the same case write byte-identical output.

## Running it

```
python3 main.py          # reads $MAKEFUN_INPUT_DIR (default /input),
                         # writes $MAKEFUN_OUTPUT_DIR (default /output)
```

Reads every `*.in` file in the input directory (name order) and writes one
`<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.
The output directory is created if missing; the program never deletes it or
its contents, it only writes its own output files. Runtime is far under a
second per case.
