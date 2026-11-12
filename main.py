#!/usr/bin/env python3
"""Final Markov rank-transition Monte Carlo forecaster.

make.fun roblox-up-and-coming-forecast-01 entry.

Markov rank-transition simulation of the Up-and-Coming chart over the 72h window:
learns place-transition distributions from consecutive history reads (bucketed by
gap length), Monte-Carlo simulates the chart, and converts place-occupancy counts
into per-place probabilities. Deterministic (fixed seed, sorted iteration).
Stdlib only. Reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR
(default /output) /forecast.json.
"""
import json, os, random
from collections import defaultdict
from datetime import datetime, timedelta

IN = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUT = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
SIMS = 600
STEP_H = 6.0
HORIZON_H = 72.0
NAMED_PER_PLACE = 5
P_FLOOR = 0.04
SEED = 20261113


def parse(at):
    return datetime.fromisoformat(at.replace("Z", "+00:00"))


def bucket(dt_h):
    if dt_h < 4:
        return "b1"
    if dt_h < 8:
        return "b2"
    return "b3"


def build_transitions(hist):
    dest = defaultdict(lambda: defaultdict(float))
    tot = defaultdict(float)
    pairs = list(zip(hist, hist[1:]))
    n = len(pairs)
    for idx in range(n):
        w = 1.0 + idx / n  # recency weighting: newer transitions count more
        a, b = pairs[idx]
        dt = (parse(b["at"]) - parse(a["at"])).total_seconds() / 3600.0
        pa = {g["universe_id"]: g["rank"] for g in a["up_and_coming"]}
        pb = {g["universe_id"]: g["rank"] for g in b["up_and_coming"]}
        for uid in sorted(pa):
            i = pa[uid]
            j = pb.get(uid)
            key = (bucket(dt), i)
            dest[key][j if j else 0] += w  # 0 = left the chart
            tot[key] += w
    return dest, tot


def simulate(last, dest, tot, pool):
    rng = random.Random(SEED)
    cur = {g["universe_id"]: g["rank"] for g in last["up_and_coming"]}
    occupancy = defaultdict(lambda: defaultdict(float))
    order0 = [g["universe_id"] for g in last["up_and_coming"]]
    for _ in range(SIMS):
        pos = dict(cur)
        inchart = set(pool)
        t = 0.0
        while t < HORIZON_H:
            newpos = {}
            for uid in order0:
                if uid not in inchart:
                    continue
                i = pos.get(uid)
                if i is None:
                    continue
                key = ("b2", i)
                d = dest[key]
                tt = tot[key]
                if tt == 0:
                    newpos[uid] = i
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
            present.sort(key=lambda u: (newpos[u], pos.get(u, 999)))
            for rank, u in enumerate(present, start=1):
                pos[u] = rank
            for u in list(pos):
                if u not in newpos:
                    del pos[u]
            t += STEP_H
        for u, r in pos.items():
            if r <= 10:
                occupancy[r][u] += 1
    return occupancy


def forecast_from(occupancy, chart):
    probs = []
    for k in range(1, 11):
        occ = occupancy[k]
        items = sorted(occ.items(), key=lambda x: (-x[1], x[0]))
        for uid, c in items[:NAMED_PER_PLACE]:
            p = c / SIMS
            if p < P_FLOOR:
                continue
            probs.append({"universe_id": uid, "place": k, "p": round(p, 4)})
    total_occ = defaultdict(float)
    for k in range(1, 11):
        for uid, c in occupancy[k].items():
            total_occ[uid] += c
    top10 = [uid for uid, _ in sorted(total_occ.items(), key=lambda x: (-x[1], x[0]))[:10]]
    return {"probabilities": probs, "top10": top10}


def main():
    hist = []
    with open(os.path.join(IN, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                hist.append(json.loads(line))
    hist.sort(key=lambda h: h["at"])
    hist = [h for h in hist if h.get("ok")]
    last = hist[-1]
    pool = [g["universe_id"] for g in last["up_and_coming"]]
    dest, tot = build_transitions(hist)
    occ = simulate(last, dest, tot, pool)
    fc = forecast_from(occ, last["up_and_coming"])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "forecast.json"), "w") as f:
        json.dump(fc, f, indent=1, sort_keys=False)
        f.write("\n")


if __name__ == "__main__":
    main()
