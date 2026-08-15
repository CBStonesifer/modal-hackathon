#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "pyarrow"]
# ///
"""Find the keep threshold empirically, by asking whether curation helps a downstream policy.

    ./ablate.py                      # sweep keep fractions, compare against random subsets

Proxy task: predict the next end-effector delta from a short pose history — the poses ARE the
actions in this dataset, so this is the thing a behaviour-cloning policy actually learns. Ridge
regression, closed form, so a 12-point sweep with 5 seeds each is seconds rather than hours.

The evaluation set is FIXED across every condition and split BY OPERATOR, because a random
split leaks: near-duplicate retakes from one demonstrator would land on both sides.
"""

import argparse

import numpy as np
import pandas as pd

RIDGE = 1e-3
HARD_REASONS = ["too_short", "frame_count_mismatch", "no_visible_change", "mostly_dead_time"]


def build_xy(traj: np.ndarray, history: int, horizon: int):
    """(T, D) trajectory -> (obs, action). Action is displacement `horizon` steps ahead.

    Horizon matters enormously. At horizon=1 the task is linear extrapolation of smooth
    motion, a linear model saturates on ~300 episodes, and the ablation cannot discriminate
    anything. Longer horizons require actual task structure."""
    need = history + horizon + 1
    if len(traj) < need:
        return None, None
    end = len(traj) - horizon
    windows = [traj[i : end - history + i] for i in range(history)]
    obs = np.concatenate(windows, axis=1)
    anchor = traj[history - 1 : end - 1]
    action = traj[history - 1 + horizon : end - 1 + horizon] - anchor
    size = min(len(obs), len(action))
    return obs[:size], action[:size]


def fit_ridge(obs: np.ndarray, action: np.ndarray):
    obs = np.concatenate([obs, np.ones((len(obs), 1))], axis=1)
    gram = obs.T @ obs + RIDGE * len(obs) * np.eye(obs.shape[1])
    return np.linalg.solve(gram, obs.T @ action)


def evaluate(weights, obs: np.ndarray, action: np.ndarray) -> float:
    obs = np.concatenate([obs, np.ones((len(obs), 1))], axis=1)
    return float(((obs @ weights - action) ** 2).mean())


def pooled(pairs: dict, hashes) -> tuple:
    xs = [pairs[h][0] for h in hashes if h in pairs]
    ys = [pairs[h][1] for h in hashes if h in pairs]
    return np.concatenate(xs), np.concatenate(ys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", default="traj.npz")
    parser.add_argument("--manifest", default="keep_drop.csv")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--eval-operators", type=int, default=6)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=15, help="steps ahead to predict")
    args = parser.parse_args()

    data = np.load(args.traj, allow_pickle=True)
    manifest = pd.read_csv(args.manifest).set_index("episode_hash")

    pairs = {}
    for h in data["hashes"]:
        obs, action = build_xy(data[h].astype(np.float64), args.history, args.horizon)
        if obs is not None and len(obs) >= 8:
            pairs[h] = (obs, action)
    manifest = manifest.loc[[h for h in manifest.index if h in pairs]]
    print(f"{len(pairs):,} usable trajectories | {manifest.operator.nunique()} operators")
    print(f"proxy task: {args.history} frame(s) of pose -> displacement {args.horizon} steps "
          f"({args.horizon / 30:.2f}s) ahead")

    # --- fixed, operator-disjoint evaluation set -----------------------------------
    rng = np.random.default_rng(0)
    operators = manifest.operator.value_counts()
    held = list(operators.index[: args.eval_operators * 3 : 3][: args.eval_operators])
    eval_hashes = manifest.index[manifest.operator.isin(held)]
    pool = manifest.loc[~manifest.operator.isin(held)].copy()
    eval_obs, eval_action = pooled(pairs, eval_hashes)
    print(f"held-out: {len(held)} operators, {len(eval_hashes):,} episodes, {len(eval_obs):,} steps")
    print(f"pool:     {pool.operator.nunique()} operators, {len(pool):,} episodes\n")

    baseline_obs, baseline_action = pooled(pairs, pool.index)
    full = evaluate(fit_ridge(baseline_obs, baseline_action), eval_obs, eval_action)
    print(f"ALL DATA ({len(pool):,} episodes): held-out MSE {full:.6e}\n")

    # integrity failures are excluded from BOTH arms so the comparison isolates ranking rather
    # than filtering; embedding outliers stay in, since whether dropping them helps is the question
    hard = pool.reason.isin(HARD_REASONS)
    rankable = pool[~hard.fillna(False)].sort_values("score", ascending=False)
    print(f"excluded {int(hard.sum())} hard-drop episodes from both arms\n")

    print(f"{'keep':>6} {'n':>6} {'quality MSE':>14} {'random MSE':>14} {'random sd':>11} {'gain':>8}")
    rows = []
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        n = max(int(len(rankable) * frac), 20)
        quality = evaluate(fit_ridge(*pooled(pairs, rankable.index[:n])), eval_obs, eval_action)
        randoms = [
            evaluate(fit_ridge(*pooled(pairs, rng.permutation(rankable.index.to_numpy())[:n])),
                     eval_obs, eval_action)
            for _ in range(args.seeds)
        ]
        gain = (np.mean(randoms) - quality) / np.mean(randoms)
        print(f"{frac:>6.0%} {n:>6} {quality:>14.6e} {np.mean(randoms):>14.6e} "
              f"{np.std(randoms):>11.2e} {gain:>+7.1%}")
        rows.append({"keep_frac": frac, "n": n, "quality_mse": quality,
                     "random_mse": float(np.mean(randoms)), "random_sd": float(np.std(randoms)),
                     "gain": float(gain)})

    df = pd.DataFrame(rows)
    df.to_csv("ablation.csv", index=False)

    print("\n=== WHERE TO SET THE THRESHOLD ===")
    best = df.loc[df.gain.idxmax()]
    print(f"  largest gain over random:  keep {best.keep_frac:.0%} ({int(best.n)} episodes), "
          f"{best.gain:+.1%}")
    parity = df[df.quality_mse <= full]
    if len(parity):
        cheapest = parity.iloc[0]
        print(f"  smallest subset matching ALL data: keep {cheapest.keep_frac:.0%} "
              f"({int(cheapest.n)} episodes) at MSE {cheapest.quality_mse:.4e} vs {full:.4e}")
    else:
        print(f"  no subset matched all-data MSE ({full:.4e}) — curation does not pay here")
    print("  -> ablation.csv")


if __name__ == "__main__":
    main()
