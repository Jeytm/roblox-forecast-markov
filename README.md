# roblox-forecast-markov

Markov rank-transition Monte Carlo forecaster for Roblox's Up-and-Coming chart.
Entered in make.fun's bounty "Predict Roblox's Up-and-Coming top 10".

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

## How it works
- Reads hourly chart snapshots from `/input/history.ndjson`
- Learns place→place transition distributions from consecutive reads (gap-bucketed)
- Monte-Carlo simulates 72h of chart evolution (600 runs, fixed seed — deterministic)
- Converts place-occupancy frequencies into per-place probabilities

Run: `python3 main.py` (Python 3.13, stdlib only, <1s)
