# GTA VI Trophy Rarity Forecast — Method

## What is being forecast

For every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on
Exophase at 2026-11-26T00:00:00Z: the percent of the players Exophase tracks
for the game who have earned that trophy. The program (`main.py`) outputs five
quantiles (5%, 25%, 50%, 75%, 95%) per trophy. Scoring is mean pinball loss on
the logit scale, so the quantiles should be the ones the forecaster really
believes, not a point guess with a decorative band.

## The two forces

An Exophase rate is a ratio, earned / tracked, and over launch week both
halves move hard:

- **Dilution.** The tracked base explodes. On launch night the house saw 638
  players tracked with every rate frozen for hours; the first refresh showed
  3,868 tracked and every rate but the one at 100% lower. Newly tracked
  players have mostly not earned a given trophy yet, so every rate is pulled
  down toward the trophy's population rate.
- **Progression.** The players already tracked keep playing. Early-story and
  easy trophies climb as the week goes on; the platinum and the grinds barely
  move, because almost nobody finishes them in a week.

A forecast that assumes rates stay put (the house baseline) is wrong in both
directions at once. The model here forecasts the ratio, not the rate.

## The model, per trophy

1. **Growth of the tracked base.** The tracked count at the target is
   extrapolated from the case history. When the history shows the count
   moving, a log-linear fit of count against time is projected forward at
   half the fitted rate (the flood slows), capped at 80x. When the history
   is flat — no refresh has been seen yet — a saturating prior is used:
   about 3x over 90 minutes, about 55x over 4.5 days.
2. **Population rate and time-to-earn.** Each trophy is classified from its
   own description and tier: story chapter (with the chapter number parsed
   out), final mission, story mission, mission strands, grind (a counted
   target: 250 stunts, 100 locations, 20 races, …), money grind, easy
   one-off, 100% completion, platinum, and the universal case (a trophy
   already at 100%). Each class gets a population earn probability `P`
   (the share of all week-one players who would eventually earn it) and a
   typical hours-to-earn `T`: Chapter 1 is `P = 0.58, T = 8h`; Chapter 4
   is `P = 0.22, T = 38h`; the final mission `P = 0.12, T = 60h`; a
   100-location grind `P = 0.015, T = 90h`; the platinum `P = 0.001,
   T = 200h`.
3. **Newcomers.** A player who joins during the horizon earns the trophy
   with probability `P * max(0, 1 - T / (0.3 * horizon_hours))`, plus a
   small tail for the fastest players. The `0.3` is the assumed share of
   wall-clock time a typical player actually spends playing. Nobody
   finishes Chapter 1 in 90 minutes; most active players get there in a
   week.
4. **Players already tracked.** They keep earning, slowly, since the
   non-earners left are the slow ones: they have already had hours to earn
   the trophy and have not, so their hazard is low. Their earn probability
   over the horizon is `(P - last_rate) * (1 - exp(-horizon / (10T)))`.
5. **Mixing.** `E' = E + N0 * q_exist + (N' - N0) * p_new`, and the median
   is `100 * E' / N'`. A trophy already at 100% stays there (`p = 0.995`).
6. **Spread.** Quantiles sit on the logit scale around the median with
   spread `sqrt((0.5 * sqrt(days))^2 + class^2 + growth^2)`: a random-walk
   diffusion plus per-class uncertainty (0.35 story, 0.55 grind, 0.45 easy,
   0.70 platinum/completion) plus growth uncertainty (0.45 when the history
   is flat and no refresh has been seen, 0.20 when the growth is fitted).
   The 90-minute practice horizon is dominated by whether a refresh lands
   at all, so the band is wide there on purpose; the 4.5-day final horizon
   is dominated by the diffusion term.

## Assumptions worth naming

- Exophase's tracked count keeps growing fast through week one (no
  saturation before the target) and refreshes keep arriving every few
  hours.
- The launch-night cohort is heavily selected (hardcore early players);
  the mass arriving later is slower, which is why the first refresh
  lowered every rate.
- Story progress follows chapter order and the chapter numbers in the
  descriptions; named missions sit mid-story.
- No trophy is removed or renamed before the target, and the trophy list
  at the target is the case's list.

## Verified runs

Practice cases (`p00`, `p01`, 90-minute horizons, flat history at 638
tracked): the program writes valid outputs for all 55 trophies (3,228 bytes
per file, 5 quantiles each, monotone, 0–100). Every median moves down from
the last read (55/55): Chapter 1 1.25% → 1.11%, the final mission 1.10% →
0.43%, the platinum 0.63% → 0.19%, the 100% trophy 100% → 99.7%. The two
practice cases produce identical outputs, as they should: their histories
are the same frozen page and their horizons are the same.

Synthetic final case (6,510-minute horizon, history extended with tracked
growing 638 → 3,868 → 12,000 → 45,000): the fitted growth projects the
tracked base up (capped at 80x) and the medians split 32 up / 23 down.
Chapter 1 0.78% → 44.3% (q05 10.9%, q95 83.8%), Chapter 2 0.68% → 19.1%,
Chapter 3 0.68% → 4.6%, Chapter 4 0.68% → 0.29%; the platinum 0.39% →
0.01%, 100% completion 0.39% → 0.01%, the 250-stunt grind 0.39% → 0.02%,
the 100-location grind 0.78% → 0.03%. Progression wins early, dilution
wins late — which is the whole bet of this entry against the house baseline
that holds every rate still.

## What the numbers look like

On the practice cases (`p00`, `p01`, 90-minute horizons, flat history at
638 tracked), the model forecasts dilution: medians move down from the last
read for almost every trophy, with the story chapters falling least and the
grinds and the platinum falling most. The one trophy at 100% stays at
~99.7%. That matches the one refresh the house actually saw, which lowered
every rate but the 100% one. The house baseline, which holds every rate
still, scores about 1.51 pinball loss on the practice; this model is built
to lose less than that on the final, where 4.5 days of dilution and
progression move the rates far from the last read in both directions.
