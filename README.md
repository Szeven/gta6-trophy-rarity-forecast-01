# GTA 6 Trophy Rarity Forecast — Round 2

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy, from the house's recorded reads up to the deadline.

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Szeven/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Szeven/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

## How it works

`main.py` (Python 3.13, standard library only) reads a case file from `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `/output/<id>.out` with five quantiles (5%, 25%, 50%, 75%, 95%) of each trophy's earn rate at the case target. The model tracks earners, not rates: each trophy has a per-hour earning hazard calibrated from its last observed rate, the tracked-player count follows a power law fitted to the history, and the forecast splits players into the old cohort and new arrivals (dilution) before converting earners back to a rate. The spread is set on the logit scale from the horizon, player-growth uncertainty, refresh-step risk and model uncertainty. See [METHOD.md](METHOD.md) for the full write-up.

Run it locally on the practice cases (from the kit directory, with the kit's `input/`):

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
