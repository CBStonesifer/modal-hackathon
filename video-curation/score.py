#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]
# ///
"""Turn measured features into a keep/drop manifest with trim spans.

    ./score.py                       # uses joined.parquet (+ emb_all.npz if present)
    ./score.py --alpha 0.01 --repr raw

There is no keep quota. Episodes are dropped for two reasons only: a verifiable integrity
failure, or being a statistical outlier in the embedding space. Outliers are found by
clustering the episode descriptors and testing each episode's Mahalanobis distance to its own
cluster centre against chi-square, with Benjamini-Hochberg control across all episodes — so
the number kept is whatever the data supports, not a dial.

Feature selection for the ordering score is not hand-picked: any feature whose variance is
more than ETA2_LIMIT explained by `operator` is excluded, because scoring on it builds a
demonstrator detector rather than a quality metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ETA2_LIMIT = 0.35  # above this, a feature is mostly demonstrator identity

# feature -> +1 if higher is better, -1 if lower is better
DIRECTION = {
    "right_sparc": +1, "left_sparc": +1, "head_sparc": +1,      # SPARC: less negative = smoother
    "right_jerk_rms": -1, "left_jerk_rms": -1, "head_jerk_rms": -1,
    "right_idle_frac": -1, "left_idle_frac": -1, "head_idle_frac": -1,
    "low_detail_frac": -1, "faint_frac": -1, "size_cv": -1,
    "right_straightness": +1, "left_straightness": +1,
    "act_span": +1,
    "progress_dip": +1,      # a deeper dip means more visibly happened
    "progress_final": +1,    # ends closer to its own goal state
}


# Interpretable failure axes. Measured pairwise correlation between them is <= 0.14, and
# k-means over the drop population recovers exactly one cluster per axis — so these are
# distinct modes, not five names for the same thing.
AXES = {
    "poor_image":      {"low_detail_frac": +1, "faint_frac": +1, "size_cv": +1},
    "jittery":         {"right_sparc": -1, "left_sparc": -1, "head_sparc": -1},
    "unsteady_camera": {"head_jerk_rms": +1, "head_idle_frac": +1},
    "hesitant":        {"right_idle_frac": +1, "left_idle_frac": +1},
    "incomplete":      {"act_span": -1, "right_straightness": -1},
}
AXIS_FLAG = 1.0  # z above this counts as failing that axis
HARD_REASONS = ["too_short", "frame_count_mismatch", "no_visible_change", "mostly_dead_time"]

PCA_DIMS = 10
K_RANGE = range(2, 9)


def axis_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-axis badness, z-scored within demonstrator. Higher = worse."""
    out = {}
    for axis, features in AXES.items():
        parts = []
        for feature, sign in features.items():
            if feature not in df.columns:
                continue
            grouped = df.groupby("operator")[feature]
            spread = grouped.transform(lambda s: s.std() if s.std() and s.std() > 0 else 1.0)
            parts.append(sign * ((df[feature] - grouped.transform("median")) / spread).clip(-4, 4))
        if parts:
            out[axis] = pd.concat(parts, axis=1).mean(axis=1)
    return pd.DataFrame(out)


def failure_label(row: pd.Series) -> str:
    """Name *why* an episode was dropped, rather than 'it ranked low'."""
    failed = [axis for axis in row.index if row[axis] > AXIS_FLAG]
    if not failed:
        return "below_average"          # honest: ranked low, but no identifiable defect
    failed.sort(key=lambda a: -row[a])
    return failed[0] if len(failed) == 1 else "+".join(failed[:2])


def eta_squared(df: pd.DataFrame, column: str, group: str) -> float:
    total = df[column].var()
    if not np.isfinite(total) or total == 0:
        return 1.0
    within = df.groupby(group)[column].transform("mean")
    return float(1 - (df[column] - within).var() / total)


def descriptors(frames: np.ndarray, mode: str) -> np.ndarray:
    """(N, T, D) frame embeddings -> (N, K) episode descriptor.

    `shape` uses only within-episode comparisons (similarity to the episode's own final frame,
    and step magnitudes). That cancels the appearance offset carried by scene and demonstrator,
    which is 98% of the raw embedding's nearest-neighbour structure. `raw` is the mean frame
    embedding, kept so the confound can be measured rather than asserted."""
    if mode == "raw":
        mean = frames.mean(1)
        return mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-8)
    similarity = np.einsum("ijk,ik->ij", frames, frames[:, -1, :])
    steps = np.linalg.norm(np.diff(frames, axis=1), axis=2)
    return np.hstack([similarity, steps])


NULL_TRIALS = 20
NULL_QUANTILE = 0.99


def benjamini_hochberg(pvalue: np.ndarray, alpha: float) -> np.ndarray:
    order = np.argsort(pvalue)
    threshold = alpha * np.arange(1, len(pvalue) + 1) / len(pvalue)
    passed = np.flatnonzero(pvalue[order] <= threshold)
    flagged = np.zeros(len(pvalue), bool)
    if len(passed):
        flagged[order[: passed[-1] + 1]] = True
    return flagged


def robust_pvalues(block: np.ndarray, dims: int, seed: int):
    """MCD Mahalanobis distance -> p, with the chi-square null rescaled by simulation.

    MCD distances are only asymptotically chi-square; at n in the tens they run large, which is
    why the uncalibrated version flagged 18% of everything at alpha=0.01. Fitting the same
    estimator to Gaussian data of identical shape gives the scale factor to divide out.

    A plain covariance is estimated *from* the outliers and hides them (masking), which is why
    the estimator is MCD rather than np.cov."""
    measure = lambda b: MinCovDet(random_state=seed).fit(b).mahalanobis(b)
    rng = np.random.default_rng(seed)
    null = np.concatenate([measure(rng.standard_normal(block.shape)) for _ in range(NULL_TRIALS)])
    # match the tail, not the middle: every decision is made out at the 99th percentile, and a
    # median-matched scale leaves the tail heavy enough to flag 7.6% of pure Gaussian data
    scale = np.quantile(null, NULL_QUANTILE) / chi2.ppf(NULL_QUANTILE, dims)
    return chi2.sf(measure(block) / scale, df=dims), chi2.sf(null / scale, df=dims)


def cluster_outliers(x: np.ndarray, alpha: float, min_frac: float, seed: int):
    """Cluster, then reject episodes too far from their own cluster centre to be chance.

    'Outlier' is a calibrated p-value, not a percentile, and Benjamini-Hochberg holds the
    expected false-drop rate at alpha across all simultaneous tests. The same procedure is run
    on simulated Gaussian data so the false-positive rate is measured, not assumed."""
    whitened = StandardScaler().fit_transform(x)
    dims = min(PCA_DIMS, whitened.shape[1], len(whitened) - 1)
    points = PCA(n_components=dims, whiten=True, random_state=seed).fit_transform(whitened)

    fits = {k: KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(points)
            for k in K_RANGE if k < len(points)}
    labels = max(fits.values(), key=lambda l: silhouette_score(points, l))
    k = len(set(labels))

    sizes = {label: int((labels == label).sum()) for label in set(labels)}
    testable = [n for label, n in sizes.items() if n / len(points) >= min_frac]
    # one estimator everywhere, so the null floor is comparable: shrink the test basis until
    # even the smallest retained cluster can support an MCD fit
    dims = max(2, min(dims, min(testable) // 5)) if testable else dims
    points = points[:, :dims]

    pvalue = np.ones(len(points))
    nulls = []
    for label, n in sizes.items():
        member = labels == label
        if n / len(points) < min_frac:
            pvalue[member] = 0.0        # a mode too rare in the corpus to be a real behaviour
            continue
        pvalue[member], null = robust_pvalues(points[member], dims, seed)
        nulls.append(null)

    outlier = benjamini_hochberg(pvalue, alpha)
    null_pool = np.concatenate(nulls) if nulls else np.ones(1)
    return labels, outlier, pvalue, k, dims, null_pool


def add_embeddings(df: pd.DataFrame, npz: Path, mode: str) -> tuple[pd.DataFrame, np.ndarray]:
    if not npz.exists():
        return df.assign(progress_dip=np.nan, progress_final=np.nan, has_progress=False), None
    data = np.load(npz, allow_pickle=True)
    frames = data["frames"].astype(np.float32)
    frames = frames / np.maximum(np.linalg.norm(frames, axis=2, keepdims=True), 1e-8)
    similarity = np.einsum("ijk,ik->ij", frames, frames[:, -1, :])
    table = pd.DataFrame({
        "episode_hash": data["frame_hashes"],
        "progress_dip": similarity[:, 0] - similarity.min(1),   # positive = something happened
        "progress_final": similarity[:, -2],
        "_row": np.arange(len(frames)),
    })
    merged = df.merge(table, on="episode_hash", how="left")
    merged["has_progress"] = merged.progress_dip.notna()
    return merged, descriptors(frames, mode)


def zscore_within(df: pd.DataFrame, column: str, group: str) -> pd.Series:
    grouped = df.groupby(group)[column]
    spread = grouped.transform(lambda s: s.std() if s.std() and s.std() > 0 else 1.0)
    return ((df[column] - grouped.transform("median")) / spread).clip(-4, 4)


def hard_drops(df: pd.DataFrame) -> pd.Series:
    """Data-integrity failures, not matters of taste. Each is independently verifiable."""
    reasons = pd.Series("", index=df.index)
    rules = [
        (df.n_frames < 60, "too_short"),
        (df.n_frames != df.declared_frames.where(df.declared_frames > 0, df.n_frames),
         "frame_count_mismatch"),
        (df.has_progress & (df.progress_dip < 0.02), "no_visible_change"),
        (df.act_span < 0.35, "mostly_dead_time"),
    ]
    for mask, label in rules:
        mask = mask.fillna(False)
        reasons[mask & reasons.eq("")] = label
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined", default="joined.parquet")
    parser.add_argument("--frames", help="tier-1 output; with --kinematics, builds --joined")
    parser.add_argument("--kinematics", help="tier-2 output")
    parser.add_argument("--catalog", default="episodes.parquet")
    parser.add_argument("--embeddings", default="emb_all.npz")
    parser.add_argument("--repr", default="shape", choices=["shape", "raw"],
                        help="episode descriptor: within-episode dynamics, or mean embedding")
    parser.add_argument("--alpha", type=float, default=0.01, help="BH false-drop rate")
    parser.add_argument("--min-cluster-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="keep_drop.csv")
    args = parser.parse_args()

    # any slice can be scored: give it the two scan outputs and it assembles its own join
    if args.frames and args.kinematics:
        visual = pd.read_parquet(args.frames).drop(columns=["operator"], errors="ignore")
        motion = pd.read_parquet(args.kinematics)
        overlap = [c for c in visual.columns if c in motion.columns and c != "episode_hash"]
        df = motion.merge(visual.drop(columns=overlap), on="episode_hash").merge(
            pd.read_parquet(args.catalog)[["episode_hash", "operator"]], on="episode_hash")
        df.to_parquet(args.joined, index=False)
        print(f"built {args.joined}: {len(df):,} x {len(df.columns)}")
    else:
        df = pd.read_parquet(args.joined)
    df, descriptor = add_embeddings(df, Path(args.embeddings), args.repr)
    df["act_span"] = (df.right_act_end - df.right_act_start).clip(0, 1)
    print(f"{len(df):,} episodes | {df.operator.nunique()} operators | "
          f"progress available for {int(df.has_progress.sum()):,}")

    # --- 1. feature selection by measured confounding, not by taste ------------------
    candidates = [c for c in DIRECTION if c in df.columns and df[c].notna().any()]
    eta = {c: eta_squared(df.dropna(subset=[c]), c, "operator") for c in candidates}
    used = [c for c in candidates if eta[c] <= ETA2_LIMIT]
    dropped = [c for c in candidates if eta[c] > ETA2_LIMIT]

    print(f"\n=== FEATURE SELECTION (eta^2 vs operator, limit {ETA2_LIMIT:.0%}) ===")
    for c in sorted(candidates, key=lambda k: eta[k]):
        print(f"  {'USE ' if c in used else 'DROP'} {c:22s} {eta[c]:6.1%}")

    # --- 2. per-operator z-scores, then a composite ---------------------------------
    parts = []
    for c in used:
        filled = df[c].fillna(df[c].median())
        parts.append(DIRECTION[c] * zscore_within(df.assign(**{c: filled}), c, "operator"))
    df["score"] = pd.concat(parts, axis=1).mean(axis=1)

    # --- 3. decisions ---------------------------------------------------------------
    df["reason"] = hard_drops(df)
    df["decision"] = np.where(df.reason.ne(""), "drop", "")

    axes = axis_scores(df)
    df = pd.concat([df, axes.add_prefix("ax_")], axis=1)

    df["cluster"] = pd.NA
    df["outlier_p"] = np.nan
    if descriptor is not None:
        eligible = df.decision.eq("") & df._row.notna()
        rows = df.loc[eligible, "_row"].astype(int).to_numpy()
        labels, outlier, pvalue, k, dims, null_pool = cluster_outliers(
            descriptor[rows], args.alpha, args.min_cluster_frac, args.seed)
        df.loc[eligible, "cluster"] = labels
        df.loc[eligible, "outlier_p"] = pvalue
        drop = df.index[eligible][outlier]
        df.loc[drop, "decision"] = "drop"
        df.loc[drop, "reason"] = axes.loc[drop].apply(failure_label, axis=1)
        print(f"\n=== EMBEDDING CLUSTERING ({args.repr}, {descriptor.shape[1]} dims -> "
              f"{dims} PCs, k={k} by silhouette) ===")
        print(f"  {int(outlier.sum()):,} of {len(rows):,} tested episodes are outliers at "
              f"BH alpha={args.alpha} ({outlier.mean():.1%}); smallest p {pvalue.min():.2e}")
        print("  alpha   flagged   floor (same procedure on outlier-free Gaussian data)")
        for a in [0.01, 0.05, 0.10, 0.25]:
            print(f"  {a:>5.2f} {benjamini_hochberg(pvalue, a).mean():>8.1%} "
                  f"{benjamini_hochberg(null_pool, a).mean():>9.1%}")
        purity = normalized_mutual_info_score(df.loc[eligible, "operator"], labels)
        print(f"  cluster vs operator NMI: {purity:.3f}  "
              f"({'clusters track demonstrators' if purity > 0.3 else 'clusters are not people'})")
        table = df[eligible].assign(out=outlier).groupby("cluster").agg(
            n=("decision", "size"), outliers=("out", "sum"),
            top_operator=("operator", lambda s: s.value_counts(normalize=True).iloc[0]),
            mean_score=("score", "mean"))
        print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    df.loc[df.decision.eq(""), "decision"] = "keep"

    # a kept episode can still carry warnings — surface them rather than hiding them
    df["flags"] = axes.apply(lambda r: "+".join(a for a in r.index if r[a] > AXIS_FLAG), axis=1)

    # --- 4. trim spans (index range, applied to poses and frames alike) -------------
    df["trim_start"] = (df.right_act_start * (df.n_frames - 1)).round().astype("Int64")
    df["trim_end"] = (df.right_act_end * (df.n_frames - 1)).round().astype("Int64")
    df.loc[df.decision.eq("drop"), ["trim_start", "trim_end"]] = pd.NA

    columns = (["episode_hash", "operator", "decision", "score", "reason", "flags",
                "trim_start", "trim_end", "n_frames", "act_span", "progress_dip",
                "cluster", "outlier_p"]
               + [f"ax_{a}" for a in AXES])
    out = df[columns].sort_values(["decision", "score"], ascending=[True, False])
    out.to_csv(args.out, index=False, float_format="%.4f")

    # --- 5. report -------------------------------------------------------------------
    print(f"\n=== DECISIONS -> {args.out} ===")
    print(f"  keep {df.decision.eq('keep').sum():,}  ({df.decision.eq('keep').mean():.1%})")
    print(f"  drop {df.decision.eq('drop').sum():,}  ({df.decision.eq('drop').mean():.1%})")
    print(df[df.decision.eq("drop")].reason.value_counts().to_string())

    print("\n=== HOW MANY AXES DOES EACH EPISODE FAIL? ===")
    n_failed = (axes > AXIS_FLAG).sum(axis=1)
    print(pd.crosstab(n_failed, df.decision).to_string())
    soft = ~df.reason.isin(HARD_REASONS) & df.reason.ne("")
    if soft.any():
        print(f"\n  {(df[soft].reason == 'below_average').mean():.1%} of embedding outliers have "
              f"no identifiable defect on the five axes — statistically odd, not visibly broken")
    print(f"  {(axes[df.decision.eq('keep')] > AXIS_FLAG).any(axis=1).mean():.1%} of KEPT episodes "
          f"carry at least one warning flag")

    print("\n=== OPERATOR FAIRNESS (a curation engine must not delete people) ===")
    fair = df.groupby("operator").agg(
        n=("decision", "size"),
        keep_rate=("decision", lambda s: s.eq("keep").mean()),
        hard_drop=("reason", lambda s: s.isin(HARD_REASONS).mean()),
    ).sort_values("keep_rate")
    print(fair.to_string(float_format=lambda x: f"{x:.3f}"))
    # nothing enforces this any more — with the quota gone it is a measurement, not a guarantee
    print(f"\n  keep-rate spread: {fair.keep_rate.min():.1%} .. {fair.keep_rate.max():.1%} "
          f"(unconstrained; a wide spread means the outlier test is finding demonstrators)")
    if descriptor is not None:
        untested = int((~df.has_progress).sum())
        print(f"  {untested:,} episodes had no embedding and were never tested for outlierness")

    print("\n=== EXCLUDED FEATURES (would have been demonstrator detectors) ===")
    print("  " + ", ".join(f"{c} ({eta[c]:.0%})" for c in dropped) if dropped else "  none")


if __name__ == "__main__":
    main()
