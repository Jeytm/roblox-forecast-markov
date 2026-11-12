"""Markov rank-transition Monte Carlo forecaster.

Predicts the Roblox "Up-and-Coming" top 10 from a history of chart
snapshots. Learns place-to-place transition distributions from
consecutive snapshot pairs (bucketed by the time gap between reads,
weighted by recency), then Monte-Carlo simulates HORIZON_H hours of
chart evolution with a fixed RNG seed so output is deterministic.

Stdlib only. Reads MAKEFUN_INPUT_DIR/history.ndjson (default /input)
and writes MAKEFUN_OUTPUT_DIR/forecast.json (default /output).
"""

import json
import os
import random
from datetime import datetime, timezone

SIMS = 600
HORIZON_H = 72.0
STEP_H = 6.0
SEED = 20261113
NAMED_PER_PLACE = 5
P_FLOOR = 0.04


def _parse_ts(s):
    """Parse an ISO timestamp; normalize naive stamps to UTC."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_history():
    """Read history.ndjson (one JSON object per line) from the input dir."""
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    path = os.path.join(in_dir, "history.ndjson")
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_transitions(hist):
    """Learn place->place transition distributions from consecutive pairs.

    Each consecutive pair (A, B) contributes one observation per game in
    A: its destination rank in B, or 0 if it dropped off the chart. Rows
    are keyed by (gap bucket, from-rank); observations are weighted by
    recency (later pairs count more).

    Returns (dest, tot): dest maps key -> {dest_rank: weight},
    tot maps key -> total weight.
    """
    dest = {}
    tot = {}
    num_pairs = len(hist) - 1
    for pair_index in range(num_pairs):
        a = hist[pair_index]
        b = hist[pair_index + 1]
        hours = (_parse_ts(b["at"]) - _parse_ts(a["at"])).total_seconds() / 3600.0
        if hours < 4:
            bucket = "b1"
        elif hours < 8:
            bucket = "b2"
        else:
            bucket = "b3"
        w = 1.0 + pair_index / num_pairs
        rank_b = {e["universe_id"]: e["rank"] for e in b["up_and_coming"]}
        for e in a["up_and_coming"]:
            i = e["rank"]
            j = rank_b.get(e["universe_id"], 0)
            key = (bucket, i)
            row = dest.setdefault(key, {})
            row[j] = row.get(j, 0.0) + w
            tot[key] = tot.get(key, 0.0) + w
    return dest, tot


def simulate(last, dest, tot, pool):
    """Monte-Carlo the chart forward HORIZON_H hours from `last`.

    Runs SIMS simulations with one seeded RNG shared across runs. Each
    step samples every surviving game's destination from the ('b2',
    current rank) transition row, drops games that fall off, renumbers
    the survivors 1..N (ties broken by previous rank), and records how
    often each game occupies each of the top 10 places.

    Returns occupancy: {place: {universe_id: count}} for places 1..10.
    """
    rng = random.Random(SEED)
    cur = {e["universe_id"]: e["rank"] for e in last["up_and_coming"]}
    order0 = sorted(pool, key=lambda u: cur[u])
    occupancy = {k: {} for k in range(1, 11)}
    for _ in range(SIMS):
        pos = dict(cur)
        inchart = set(pool)
        t = 0.0
        while t < HORIZON_H:
            newpos = {}
            for uid in order0:
                if uid not in inchart or uid not in pos:
                    continue
                key = ("b2", pos[uid])
                d = dest.get(key, {})
                tt = tot.get(key, 0)
                if tt == 0:
                    newpos[uid] = pos[uid]
                    continue
                r = rng.random()
                cum = 0.0
                keep = None
                for j in sorted(d):
                    cum += d[j] / tt
                    if r <= cum:
                        keep = j if j else None
                        break
                if keep:
                    newpos[uid] = keep
                else:
                    inchart.discard(uid)
            present = [u for u in order0 if u in newpos]
            present.sort(key=lambda u: (newpos[u], pos[u]))
            for rank, u in enumerate(present, 1):
                pos[u] = rank
            for u in list(pos):
                if u not in newpos:
                    del pos[u]
            t += STEP_H
        for u, r in pos.items():
            if r <= 10:
                row = occupancy[r]
                row[u] = row.get(u, 0) + 1
    return occupancy


def forecast_from(occupancy):
    """Convert occupancy counts into the forecast document.

    probabilities: for each place 1..10, the top NAMED_PER_PLACE games
    by occupancy whose p = count/SIMS clears P_FLOOR, as rows
    {universe_id, place, p}. top10: games ranked by total occupancy
    across places 1..10.
    """
    probabilities = []
    for k in range(1, 11):
        row = occupancy.get(k, {})
        items = sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))
        for uid, count in items[:NAMED_PER_PLACE]:
            p = count / SIMS
            if p >= P_FLOOR:
                probabilities.append(
                    {"universe_id": uid, "place": k, "p": round(p, 4)}
                )
    total_occ = {}
    for k in range(1, 11):
        for uid, count in occupancy.get(k, {}).items():
            total_occ[uid] = total_occ.get(uid, 0) + count
    top10 = [
        uid
        for uid, _ in sorted(total_occ.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    ]
    return {"probabilities": probabilities, "top10": top10}


def main():
    hist = load_history()
    hist.sort(key=lambda r: r["at"])
    hist = [r for r in hist if r.get("ok") == True]
    last = hist[-1]
    pool = [e["universe_id"] for e in last["up_and_coming"]]
    dest, tot = build_transitions(hist)
    occ = simulate(last, dest, tot, pool)
    out = forecast_from(occ)
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "forecast.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
