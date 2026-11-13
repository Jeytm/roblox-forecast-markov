"""Markov rank-transition Monte Carlo forecaster for the Roblox Up-and-Coming chart.

Reads chart snapshots from $MAKEFUN_INPUT_DIR/history.ndjson, learns
gap-bucketed rank transitions from consecutive read pairs, Monte-Carlo
simulates 72h of chart evolution for each gap bucket (one ensemble member
per bucket), blends the members, and writes $MAKEFUN_OUTPUT_DIR/forecast.json.
"""

import json
import os
import random
from collections import defaultdict
from datetime import datetime

SIMS = 600            # Monte Carlo runs per ensemble member
HORIZON_H = 72.0      # forecast horizon in hours
STEP_H = 6.0          # hours per transition step
SEED = 20261113       # fixed RNG seed: every member/run is deterministic
NAMED_PER_PLACE = 12  # candidate games named per chart place
P_FLOOR = 0.01        # minimum probability to report

BUCKETS = ('b1', 'b2', 'b3')  # ensemble members, in blend order


def read_history(path):
    """Load NDJSON reads: keep truthy-ok entries, sorted by 'at' ascending."""
    reads = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get('ok', True):
                continue
            reads.append(rec)
    reads.sort(key=lambda r: datetime.fromisoformat(r['at']))
    return reads


def bucket_for(gap_h):
    """Map a read-gap in hours to its transition bucket."""
    if gap_h < 4:
        return 'b1'
    if gap_h < 8:
        return 'b2'
    return 'b3'


def build_transitions(hist):
    """Gap-bucketed rank-transition weights from consecutive read pairs.

    dest[(bucket, from_rank)][to_rank] -> accumulated weight; to_rank 0
    means the game dropped off the chart. Later (more recent) pairs weigh
    more: w = 1 + pair_index / num_pairs.
    """
    dest = {}
    tot = {}
    num_pairs = len(hist) - 1
    for idx in range(num_pairs):
        a, b = hist[idx], hist[idx + 1]
        gap_h = (datetime.fromisoformat(b['at']) - datetime.fromisoformat(a['at'])).total_seconds() / 3600.0
        bkt = bucket_for(gap_h)
        b_ranks = {e['universe_id']: e['rank'] for e in b['up_and_coming']}
        w = 1 + idx / num_pairs
        for e in a['up_and_coming']:
            key = (bkt, e['rank'])
            d = dest.setdefault(key, defaultdict(float))
            d[b_ranks.get(e['universe_id'], 0)] += w
            tot[key] = tot.get(key, 0.0) + w
    return dest, tot


def simulate(last, dest, tot, pool, bucket):
    """Monte-Carlo the chart HORIZON_H ahead for one bucket; return occupancy."""
    rng = random.Random(SEED)
    cur = {e['universe_id']: e['rank'] for e in last['up_and_coming']}
    order0 = list(pool)
    occupancy = {r: defaultdict(int) for r in range(1, 11)}
    for _ in range(SIMS):
        pos = dict(cur)
        inchart = set(pool)
        t = 0.0
        while t < HORIZON_H:
            newpos = {}
            for uid in order0:
                if uid not in inchart or uid not in pos:
                    continue
                key = (bucket, pos[uid])
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
                occupancy[r][u] += 1
    return occupancy


def forecast_from(occupancy):
    """Turn place-occupancy counts into named rows and a top10 ranking."""
    probabilities = []
    for place in range(1, 11):
        named = 0
        for uid, count in sorted(occupancy[place].items(), key=lambda kv: (-kv[1], kv[0])):
            if named >= NAMED_PER_PLACE:
                break
            p = count / SIMS
            if p >= P_FLOOR:
                probabilities.append({'universe_id': uid, 'place': place, 'p': round(p, 4)})
                named += 1
    totals = defaultdict(int)
    for place in range(1, 11):
        for uid, count in occupancy[place].items():
            totals[uid] += count
    top10 = [uid for uid, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))]
    return probabilities, top10


def main():
    indir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    hist = read_history(os.path.join(indir, 'history.ndjson'))
    last = hist[-1]
    pool = [e['universe_id'] for e in sorted(last['up_and_coming'], key=lambda e: e['rank'])]
    dest, tot = build_transitions(hist)

    member_probs = []
    member_top10 = []
    for bucket in BUCKETS:
        occupancy = simulate(last, dest, tot, pool, bucket)
        probs, top10 = forecast_from(occupancy)
        member_probs.append(probs)
        member_top10.append(top10)

    named = {}
    n_members = len(member_probs)
    for probs in member_probs:
        for row in probs:
            key = (row['place'], row['universe_id'])
            named[key] = named.get(key, 0.0) + row['p'] / n_members
    probabilities = []
    for (place, uid), p in named.items():
        if p >= P_FLOOR:
            probabilities.append({'universe_id': uid, 'place': place, 'p': round(p, 6)})
    probabilities.sort(key=lambda row: (row['place'], -row['p'], row['universe_id']))

    seen = set()
    top10 = []
    for member in member_top10:
        for uid in member:
            if uid not in seen:
                seen.add(uid)
                top10.append(uid)
    top10 = top10[:10]

    outdir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'forecast.json'), 'w') as f:
        json.dump({'probabilities': probabilities, 'top10': top10}, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
