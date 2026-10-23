"""Journey Selection and Journey Design modules for Behaviorally-Grounded Synthetic Personas.

Implements the two experimental stages owned by Vandan:
1. Journey Selection (`select_journeys`):
   - `cluster_k4`: Direct K=4 clustering over the 1,279 cleaned CMU-MOSEI recordings using
     centered sentiment shape + multi-emotion summaries.
   - `cluster_high_var`: Picks high-variability / multi-emotion recordings within each of the
     4 clusters instead of the flattest centroid-nearest curve.
   - `random_high_var`: Seeded random sampling of single recordings filtered for high sentiment
     swing and multi-emotion presence across the 6 discrete emotions.
2. Journey Design (`format_journey_conditioning`, `build_interview_prompts`):
   - `granularity`: `"sentiment_only"` vs. `"emotion_breakdown"` (sentiment + 6 emotions).
   - `representation`: `"verbal"` vs. `"quant"` (including sarcastic/guarded/skeptical tone cues).
   - `stochasticity`: `"fixed"` vs. `"range_or_alpha"` (value intervals `[low, high]` and alpha perturbation).
   - `placement`: `"system_whole_journey"` (entire arc in system prompt) vs. `"per_question_stage"`
     (stage-by-stage cue injected at each question turn).
   - `prompt_mode`: `"single_call"` vs. `"two_step_chain"` (Step A substantive answer -> Step B emotional modulation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

EMOTION_COLUMNS: List[str] = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
ALL_AFFECT_COLUMNS: List[str] = ["sentiment"] + EMOTION_COLUMNS

MANUAL_REVIEW_EXCLUSIONS: Dict[str, str] = {
    "34-ogj6rh6w": "explicit exchange involving multiple named people",
    "luQfWSfb1s0": "explicit colleague handover within source",
    "kh6eynD9OIk": "possible multiple accounts/speakers; unresolved from transcript alone",
}

SelectionMethod = Literal["provisional_4", "cluster_k4", "cluster_high_var", "random_high_var"]
GranularityMode = Literal["sentiment_only", "emotion_breakdown"]
RepresentationMode = Literal["verbal", "quant"]
StochasticityMode = Literal["fixed", "range_or_alpha"]
PlacementMode = Literal["system_whole_journey", "per_question_stage"]
PromptMode = Literal["single_call", "two_step_chain"]


@dataclass
class JourneyStage:
    """Single interview stage (1..n_stages) mapped from a CMU-MOSEI recording."""

    stage: int
    relative_time: float
    sentiment: float
    sentiment_delta: float
    sentiment_range: Tuple[float, float]
    emotions: Dict[str, float]
    emotion_ranges: Dict[str, Tuple[float, float]]
    dominant_emotions: List[Tuple[str, float]]
    snippet: str = ""


@dataclass
class SelectedJourney:
    """A selected CMU-MOSEI behavioral journey mapped onto N interview turns."""

    condition: str
    shape: str
    label: str
    selection_method: str
    video_id: str
    source_cluster: int
    n_utterances: int
    group_size: int
    distance_to_group_mean: float
    sentiment_span: float
    emotion_diversity: int
    short_summary: str
    curve_12: List[float]
    group_mean_12: List[float]
    stages: List[JourneyStage] = field(default_factory=list)
    snippets: List[str] = field(default_factory=list)


def _union_length(intervals: Sequence[Tuple[float, float]]) -> float:
    iv = sorted(intervals)
    total = 0.0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total


def resample_series(times: Sequence[float], values: Sequence[float], grid: np.ndarray) -> np.ndarray:
    """Interpolate a variable-length utterance sequence onto a relative-time grid in [0, 1]."""
    order = np.argsort(times, kind="stable")
    t = np.asarray(times, dtype=float)[order]
    v = np.asarray(values, dtype=float)[order]
    unique_t, inverse = np.unique(t, return_inverse=True)
    if len(unique_t) != len(t):
        v = np.array([v[inverse == i].mean() for i in range(len(unique_t))], dtype=float)
        t = unique_t
    if len(t) == 1 or t.max() == t.min():
        return np.full(len(grid), float(v.mean()), dtype=float)
    normalized = (t - t.min()) / (t.max() - t.min())
    return np.interp(grid, normalized, v)


def load_clean_mosei(
    transcript_csv: str | Path,
    assign_csv: Optional[str | Path] = None,
    min_utterances: int = 3,
    max_gap_seconds: float = 10.0,
    min_coverage: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load `segment_transcripts_with_labels.csv` and reproduce the 1,279 cleaned videos."""
    df = pd.read_csv(transcript_csv, dtype={"video_id": str}, keep_default_na=False)
    for col in ["interval_start", "interval_end"] + ALL_AFFECT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["mid"] = (df["interval_start"] + df["interval_end"]) / 2.0

    if assign_csv is not None and Path(assign_csv).exists():
        assignments = pd.read_csv(assign_csv, dtype={"video_id": str}, keep_default_na=False)
        retained_ids = set(assignments["video_id"].astype(str))
        clean = df[df["video_id"].isin(retained_ids)].copy()
        clean = clean.sort_values(["video_id", "mid"], kind="stable").reset_index(drop=True)
        return clean, assignments

    eps = 1e-6
    rows = []
    for vid, g in df.groupby("video_id"):
        iv = list(zip(g["interval_start"].astype(float), g["interval_end"].astype(float)))
        ivs = sorted(iv)
        span = max(e for _, e in ivs) - min(s for s, _ in ivs)
        gaps = [ivs[i + 1][0] - ivs[i][1] for i in range(len(ivs) - 1)]
        max_gap = max(gaps + [0.0])
        cov = _union_length(ivs) / span if span > 0 else 0.0
        keep = (
            len(iv) >= min_utterances
            and max_gap <= max_gap_seconds + eps
            and cov >= min_coverage - eps
            and str(vid) not in MANUAL_REVIEW_EXCLUSIONS
        )
        if keep:
            rows.append({"video_id": str(vid), "n_utterances": len(iv)})
    assignments = pd.DataFrame(rows)
    clean = df[df["video_id"].isin(set(assignments["video_id"]))].copy()
    clean = clean.sort_values(["video_id", "mid"], kind="stable").reset_index(drop=True)
    return clean, assignments


def _describe_shape(arc12: np.ndarray, emo_means: Dict[str, float]) -> Tuple[str, str, str]:
    """Return `(shape_slug, shape_label, verbal_summary)` from a 12-point curve and emotion means."""
    s0 = float(arc12[:4].mean())
    sm = float(arc12[4:8].mean())
    s1 = float(arc12[8:].mean())
    delta = s1 - s0
    curv = sm - (s0 + s1) / 2.0

    active_emos = sorted(
        [(e, v) for e, v in emo_means.items() if v >= 0.08],
        key=lambda item: item[1],
        reverse=True,
    )
    top_emo_str = (
        ", ".join(f"{e} ({v:.2f})" for e, v in active_emos[:2])
        if active_emos
        else "restrained affect"
    )

    if curv < -0.25 and s1 >= s0 - 0.15:
        slug = "positive_dip_recover"
        label = "Positive dip and recovery"
        summary = (
            f"Start constructive ({s0:+.2f}), dip into skepticism or friction mid-interview ({sm:+.2f}), "
            f"then recover to a measured close ({s1:+.2f}) with {top_emo_str}."
        )
    elif curv > 0.25 and abs(delta) < 0.55:
        slug = "mid_peak"
        label = "Middle peak"
        summary = (
            f"Start guarded ({s0:+.2f}), peak in engagement mid-interview ({sm:+.2f}), "
            f"then temper your enthusiasm by the close ({s1:+.2f}) with {top_emo_str}."
        )
    elif delta >= 0.30:
        slug = "warms_up"
        label = "Warms up"
        summary = (
            f"Start reserved or skeptical ({s0:+.2f}) and gradually warm up across the interview "
            f"to a positive close ({s1:+.2f}) with {top_emo_str}."
        )
    elif delta <= -0.30:
        slug = "cools_off"
        label = "Cools off"
        summary = (
            f"Start neutral or mildly open ({s0:+.2f}), then grow increasingly critical and skeptical "
            f"toward the end ({s1:+.2f}) with {top_emo_str}."
        )
    else:
        slug = "steady_critical"
        label = "Steady analytical"
        summary = (
            f"Maintain a steady, analytical temperament ({s0:+.2f} to {s1:+.2f}) with localized "
            f"reactions driven by {top_emo_str}."
        )
    return slug, label, summary


def _build_stages_for_video(
    g: pd.DataFrame,
    group_curves_5: Optional[np.ndarray],
    n_stages: int = 5,
    alpha: float = 0.35,
    rng: Optional[np.random.Generator] = None,
) -> List[JourneyStage]:
    """Interpolate all 7 affect channels onto `n_stages` points and compute alpha ranges."""
    grid_n = np.linspace(0.0, 1.0, n_stages)
    times = g["mid"].to_numpy(dtype=float)
    sent_n = resample_series(times, g["sentiment"].to_numpy(dtype=float), grid_n)

    # Compute per-stage residual std from group curves or within-video volatility
    if group_curves_5 is not None and len(group_curves_5) > 1:
        sent_std = np.std(group_curves_5, axis=0)
    else:
        raw_std = float(np.std(g["sentiment"].to_numpy(dtype=float)))
        sent_std = np.full(n_stages, max(0.35, raw_std))

    emo_stage_vals: Dict[str, np.ndarray] = {}
    emo_stage_stds: Dict[str, float] = {}
    for emo in EMOTION_COLUMNS:
        arr = g[emo].to_numpy(dtype=float)
        emo_stage_vals[emo] = np.clip(resample_series(times, arr, grid_n), 0.0, 3.0)
        emo_stage_stds[emo] = max(0.15, float(np.std(arr)))

    # Optionally perturb point values slightly if rng is provided
    if rng is not None and alpha > 0:
        noise = rng.normal(0.0, alpha * 0.25, size=n_stages)
        sent_n = np.clip(sent_n + noise, -3.0, 3.0)

    snippets_all = [str(s).strip() for s in g["transcript"].tolist() if str(s).strip()]
    stages: List[JourneyStage] = []
    for idx in range(n_stages):
        score = float(sent_n[idx])
        delta = 0.0 if idx == 0 else float(sent_n[idx] - sent_n[idx - 1])
        half_width = max(0.20, float(alpha * sent_std[idx]))
        s_low = float(np.clip(score - half_width, -3.0, 3.0))
        s_high = float(np.clip(score + half_width, -3.0, 3.0))

        emos: Dict[str, float] = {}
        emo_ranges: Dict[str, Tuple[float, float]] = {}
        for emo in EMOTION_COLUMNS:
            ev = float(emo_stage_vals[emo][idx])
            ehw = max(0.10, float(alpha * emo_stage_stds[emo]))
            emos[emo] = round(ev, 3)
            emo_ranges[emo] = (
                round(float(np.clip(ev - ehw, 0.0, 3.0)), 2),
                round(float(np.clip(ev + ehw, 0.0, 3.0)), 2),
            )

        dom = sorted(
            [(e, v) for e, v in emos.items() if v >= 0.05],
            key=lambda item: item[1],
            reverse=True,
        )
        snip_idx = min(len(snippets_all) - 1, int(round(idx * (len(snippets_all) - 1) / max(1, n_stages - 1))))
        snippet = snippets_all[snip_idx][:160] if snippets_all else ""

        stages.append(
            JourneyStage(
                stage=idx + 1,
                relative_time=float(grid_n[idx]),
                sentiment=round(score, 3),
                sentiment_delta=round(delta, 3),
                sentiment_range=(round(s_low, 2), round(s_high, 2)),
                emotions=emos,
                emotion_ranges=emo_ranges,
                dominant_emotions=dom[:3],
                snippet=snippet,
            )
        )
    return stages


def select_journeys(
    clean_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    selection_method: SelectionMethod = "cluster_k4",
    n_journeys: int = 4,
    n_stages: int = 5,
    alpha: float = 0.35,
    random_state: int = 42,
) -> List[SelectedJourney]:
    """Select `n_journeys` behavioral trajectories from the cleaned CMU-MOSEI dataset.

    Parameters
    ----------
    selection_method:
        - `"provisional_4"`: Original 6->4 cluster fold (C1=0, C2=1+4, C3=3, C4=2+5) picking
          the mean-nearest video (`etPDgRjrnJ8`, `TXiOSZdaLJ8`, `1olbQVO_C-8`, `v9gjg7TDlTA`).
        - `"cluster_k4"`: Direct K=4 clustering on centered 12-point sentiment shape + 6 emotion
          intensities, picking a clear high-signal exemplar per cluster.
        - `"cluster_high_var"`: Uses the 4 cluster groups but selects recordings with high
          within-video sentiment span (>= 1.5) and multi-emotion presence (>= 2 active emotions)
          so the emotional cycle is vivid rather than flat.
        - `"random_high_var"`: Seeded random sample of `n_journeys` distinct recordings from the
          high-variability, multi-emotion pool across the entire dataset.
    """
    grid12 = np.linspace(0.0, 1.0, 12)
    grid5 = np.linspace(0.0, 1.0, n_stages)

    video_rows: Dict[str, pd.DataFrame] = {
        str(vid): g.sort_values("mid", kind="stable") for vid, g in clean_df.groupby("video_id")
    }
    vids = sorted(video_rows.keys())
    arcs12 = {vid: resample_series(video_rows[vid]["mid"], video_rows[vid]["sentiment"], grid12) for vid in vids}
    arcs5 = {vid: resample_series(video_rows[vid]["mid"], video_rows[vid]["sentiment"], grid5) for vid in vids}

    # Per-video affect statistics for filtering & clustering
    sent_spans: Dict[str, float] = {}
    emo_means: Dict[str, Dict[str, float]] = {}
    emo_diversities: Dict[str, int] = {}
    for vid in vids:
        g = video_rows[vid]
        s = g["sentiment"].to_numpy(dtype=float)
        sent_spans[vid] = float(s.max() - s.min())
        em = {e: float(g[e].mean()) for e in EMOTION_COLUMNS}
        emo_means[vid] = em
        emo_diversities[vid] = int(sum(1 for e in EMOTION_COLUMNS if float(g[e].max()) >= 0.33))

    provisional_groups = [
        {"condition": "C1", "shape": "warms_up", "label": "Warms up", "clusters": [0]},
        {"condition": "C2", "shape": "positive_dip_recover", "label": "Positive dip and recovery", "clusters": [1, 4]},
        {"condition": "C3", "shape": "mid_peak", "label": "Middle peak", "clusters": [3]},
        {"condition": "C4", "shape": "cools_off", "label": "Cools off", "clusters": [2, 5]},
    ]

    selected: List[SelectedJourney] = []

    if selection_method in ("provisional_4", "cluster_high_var"):
        for idx, grp in enumerate(provisional_groups[:n_journeys]):
            member_ids = [
                str(v)
                for v in assignments_df.loc[assignments_df["cluster"].isin(grp["clusters"]), "video_id"].tolist()
                if str(v) in video_rows
            ]
            mat12 = np.vstack([arcs12[v] for v in member_ids])
            mat5 = np.vstack([arcs5[v] for v in member_ids])
            mean12 = mat12.mean(axis=0)
            dists = np.linalg.norm(mat12 - mean12, axis=1)

            if selection_method == "provisional_4":
                chosen_idx = int(np.argmin(dists))
                chosen_vid = member_ids[chosen_idx]
                dist_val = float(dists[chosen_idx])
            else:
                # Filter for rich emotional swing & multi-emotion presence while staying aligned with cluster shape
                candidates = []
                for m_i, v in enumerate(member_ids):
                    if sent_spans[v] >= 1.33 and emo_diversities[v] >= 3 and len(video_rows[v]) >= 4:
                        # Score balances closeness to cluster mean direction and rich emotional variance
                        score = float(dists[m_i]) - 0.18 * sent_spans[v] - 0.12 * emo_diversities[v]
                        candidates.append((score, m_i, v))
                if not candidates:
                    chosen_idx = int(np.argmin(dists))
                    chosen_vid = member_ids[chosen_idx]
                else:
                    candidates.sort(key=lambda item: item[0])
                    _, chosen_idx, chosen_vid = candidates[0]
                dist_val = float(dists[chosen_idx])

            g = video_rows[chosen_vid]
            _, _, auto_summary = _describe_shape(arcs12[chosen_vid], emo_means[chosen_vid])
            snippets = [str(x).strip()[:160] for x in g["transcript"].tolist() if str(x).strip()]
            pick_snips = list(dict.fromkeys([snippets[0], snippets[len(snippets) // 2], snippets[-1]]))[:3]
            stages = _build_stages_for_video(g, mat5, n_stages=n_stages, alpha=alpha)

            selected.append(
                SelectedJourney(
                    condition=grp["condition"],
                    shape=grp["shape"],
                    label=grp["label"],
                    selection_method=selection_method,
                    video_id=chosen_vid,
                    source_cluster=int(assignments_df.set_index("video_id").loc[chosen_vid, "cluster"]),
                    n_utterances=len(g),
                    group_size=len(member_ids),
                    distance_to_group_mean=dist_val,
                    sentiment_span=round(sent_spans[chosen_vid], 2),
                    emotion_diversity=emo_diversities[chosen_vid],
                    short_summary=auto_summary,
                    curve_12=[round(float(x), 3) for x in arcs12[chosen_vid]],
                    group_mean_12=[round(float(x), 3) for x in mean12],
                    stages=stages,
                    snippets=pick_snips,
                )
            )
        return selected

    if selection_method == "cluster_k4":
        # Direct K=4 clustering on centered 12-point sentiment + 6 emotion means + volatility
        feat_rows = []
        for v in vids:
            arc = arcs12[v]
            centered = arc - arc.mean()
            slope = float(np.polyfit(grid12, arc, 1)[0])
            curv = float(arc[4:8].mean() - (arc[:4].mean() + arc[8:].mean()) / 2.0)
            delta = float(arc[8:].mean() - arc[:4].mean())
            vol = float(np.abs(np.diff(arc)).mean())
            emo_vec = [emo_means[v][e] for e in EMOTION_COLUMNS]
            feat_rows.append(np.concatenate([centered, [slope, curv, delta, vol], emo_vec]))
        X = StandardScaler().fit_transform(np.vstack(feat_rows))
        km = KMeans(n_clusters=n_journeys, random_state=random_state, n_init=10).fit(X)
        labels = km.labels_

        # Order clusters from rising to falling delta for consistent C1..C4 numbering
        cluster_order = sorted(
            range(n_journeys),
            key=lambda c: float(np.mean([arcs12[vids[i]][8:].mean() - arcs12[vids[i]][:4].mean() for i in range(len(vids)) if labels[i] == c])),
            reverse=True,
        )
        used_slugs: set[str] = set()
        for rank, c_id in enumerate(cluster_order, 1):
            member_indices = [i for i, lab in enumerate(labels) if lab == c_id]
            member_ids = [vids[i] for i in member_indices]
            mat12 = np.vstack([arcs12[v] for v in member_ids])
            mat5 = np.vstack([arcs5[v] for v in member_ids])
            mean12 = mat12.mean(axis=0)
            dists = np.linalg.norm(X[member_indices] - km.cluster_centers_[c_id], axis=1)

            # Pick a recording near the K=4 cluster center that also has multi-emotion signal
            scored = []
            for local_i, v in enumerate(member_ids):
                if len(video_rows[v]) >= 4 and sent_spans[v] >= 1.0 and emo_diversities[v] >= 2:
                    scored.append((float(dists[local_i]) - 0.10 * emo_diversities[v], local_i, v))
            if scored:
                scored.sort(key=lambda item: item[0])
                _, chosen_local, chosen_vid = scored[0]
            else:
                chosen_local = int(np.argmin(dists))
                chosen_vid = member_ids[chosen_local]

            g = video_rows[chosen_vid]
            slug, label, summary = _describe_shape(mean12, emo_means[chosen_vid])
            if slug in used_slugs:
                slug = f"{slug}_{rank}"
                label = f"{label} ({rank})"
            used_slugs.add(slug)
            snippets = [str(x).strip()[:160] for x in g["transcript"].tolist() if str(x).strip()]
            pick_snips = list(dict.fromkeys([snippets[0], snippets[len(snippets) // 2], snippets[-1]]))[:3]
            stages = _build_stages_for_video(g, mat5, n_stages=n_stages, alpha=alpha)

            selected.append(
                SelectedJourney(
                    condition=f"C{rank}",
                    shape=slug,
                    label=label,
                    selection_method="cluster_k4",
                    video_id=chosen_vid,
                    source_cluster=int(c_id),
                    n_utterances=len(g),
                    group_size=len(member_ids),
                    distance_to_group_mean=float(dists[chosen_local]),
                    sentiment_span=round(sent_spans[chosen_vid], 2),
                    emotion_diversity=emo_diversities[chosen_vid],
                    short_summary=summary,
                    curve_12=[round(float(x), 3) for x in arcs12[chosen_vid]],
                    group_mean_12=[round(float(x), 3) for x in mean12],
                    stages=stages,
                    snippets=pick_snips,
                )
            )
        return selected

    if selection_method == "random_high_var":
        rng = np.random.default_rng(random_state)
        # Pool of high-variability recordings with >= 4 utterances, span >= 2.0, and >= 3 emotions active
        pool = [
            v
            for v in vids
            if len(video_rows[v]) >= 4 and sent_spans[v] >= 2.0 and emo_diversities[v] >= 3
        ]
        if len(pool) < n_journeys:
            pool = [v for v in vids if len(video_rows[v]) >= 4 and sent_spans[v] >= 1.33]

        # Stratify across quartiles of start-to-end sentiment change so the random sample covers distinct directions
        deltas = np.array([arcs12[v][8:].mean() - arcs12[v][:4].mean() for v in pool])
        order = np.argsort(deltas)
        chunks = np.array_split(order, n_journeys)
        chosen_vids = [pool[int(rng.choice(chunk))] for chunk in chunks]

        all_mat5 = np.vstack([arcs5[v] for v in pool])
        for rank, chosen_vid in enumerate(chosen_vids, 1):
            g = video_rows[chosen_vid]
            slug, label, summary = _describe_shape(arcs12[chosen_vid], emo_means[chosen_vid])
            snippets = [str(x).strip()[:160] for x in g["transcript"].tolist() if str(x).strip()]
            pick_snips = list(dict.fromkeys([snippets[0], snippets[len(snippets) // 2], snippets[-1]]))[:3]
            stages = _build_stages_for_video(g, all_mat5, n_stages=n_stages, alpha=alpha, rng=rng)
            orig_cluster = (
                int(assignments_df.set_index("video_id").loc[chosen_vid, "cluster"])
                if chosen_vid in set(assignments_df["video_id"])
                else -1
            )
            selected.append(
                SelectedJourney(
                    condition=f"C{rank}",
                    shape=f"random_{rank}_{slug}",
                    label=f"High-var sample {rank}: {label}",
                    selection_method="random_high_var",
                    video_id=chosen_vid,
                    source_cluster=orig_cluster,
                    n_utterances=len(g),
                    group_size=len(pool),
                    distance_to_group_mean=0.0,
                    sentiment_span=round(sent_spans[chosen_vid], 2),
                    emotion_diversity=emo_diversities[chosen_vid],
                    short_summary=summary,
                    curve_12=[round(float(x), 3) for x in arcs12[chosen_vid]],
                    group_mean_12=[round(float(x), 3) for x in arcs12[chosen_vid]],
                    stages=stages,
                    snippets=pick_snips,
                )
            )
        return selected

    raise ValueError(f"Unknown selection_method: {selection_method}")


def _verbal_tone_from_stage(st: JourneyStage, include_emotions: bool = True, allow_sarcasm: bool = True) -> str:
    """Translate a stage's sentiment and discrete emotions into a natural-language behavioral cue."""
    s = st.sentiment
    mag = abs(s)
    if mag < 0.12:
        sent_phrase = "an approximately neutral, observant tone"
    else:
        direction = "positive and receptive" if s > 0 else "critical and skeptical"
        strength = "mildly" if mag <= 0.85 else "moderately" if mag <= 1.75 else "strongly"
        sent_phrase = f"a {strength} {direction} tone"

    if st.stage == 1:
        shift_phrase = "This is your opening posture."
    else:
        d = st.sentiment_delta
        if abs(d) < 0.12:
            shift_phrase = "Keep a similar emotional temperature to your previous answer."
        elif d > 0:
            shift_phrase = f"Warm up relative to your last answer (shift {d:+.2f})."
        else:
            shift_phrase = f"Grow more guarded or critical than in your last answer (shift {d:+.2f})."

    if not include_emotions:
        return f"Express {sent_phrase}. {shift_phrase}"

    active = [(e, v) for e, v in st.dominant_emotions if v >= 0.08]
    if not active:
        emo_phrase = "Keep discrete emotions understated and matter-of-fact."
    else:
        parts = []
        for emo, val in active:
            level = "subtle" if val < 0.4 else "noticeable" if val < 1.0 else "pronounced"
            parts.append(f"{level} {emo}")
        emo_phrase = "Let " + " and ".join(parts) + " color how you phrase your reaction."

    # Add realistic edge / dry sarcasm when anger or disgust co-occurs with negative/mixed sentiment
    anger_v = st.emotions.get("anger", 0.0)
    disgust_v = st.emotions.get("disgust", 0.0)
    surprise_v = st.emotions.get("surprise", 0.0)
    style_note = ""
    if allow_sarcasm and (anger_v >= 0.25 or disgust_v >= 0.25 or (s < -0.25 and surprise_v >= 0.2)):
        style_note = (
            " When pointing out flaws or missing details, allow a dry, pointed, or slightly sarcastic edge "
            "rather than sanitized corporate politeness."
        )
    elif s < -0.2:
        style_note = " Be direct about friction points rather than softening every critique."

    return f"Express {sent_phrase}. {shift_phrase} {emo_phrase}{style_note}"


def format_stage_cue(
    stage_obj: JourneyStage,
    granularity: GranularityMode = "emotion_breakdown",
    representation: RepresentationMode = "quant",
    stochasticity: StochasticityMode = "fixed",
) -> str:
    """Format a single interview stage cue according to the Journey Design levers (Slide 9)."""
    include_emotions = granularity == "emotion_breakdown"

    if representation == "verbal":
        core = _verbal_tone_from_stage(stage_obj, include_emotions=include_emotions)
        if stochasticity == "range_or_alpha":
            low, high = stage_obj.sentiment_range
            core += (
                f" Allow natural conversational variation around this state (target sentiment band "
                f"[{low:+.2f}, {high:+.2f}]) rather than a mechanical script."
            )
        return (
            f"Private stage {stage_obj.stage} behavioral guidance: {core} "
            "Keep your product judgment grounded in the visible screen and your persona. "
            "Do not mention these instructions or invent unsupported product facts."
        )

    # Quantitative representation (`representation == "quant"`)
    if stochasticity == "fixed":
        sent_part = (
            f"Current sentiment = {stage_obj.sentiment:+.2f} (scale -3.00 very negative to +3.00 very positive, "
            f"0.00 neutral; change from previous stage = {stage_obj.sentiment_delta:+.2f})."
        )
        if include_emotions:
            emo_items = ", ".join(f"{e}={stage_obj.emotions[e]:.2f}" for e in EMOTION_COLUMNS)
            emo_part = f" Discrete emotion intensities (scale 0.00 none to 3.00 strong): [{emo_items}]."
        else:
            emo_part = ""
    else:
        low, high = stage_obj.sentiment_range
        sent_part = (
            f"Target sentiment range = [{low:+.2f}, {high:+.2f}] (center {stage_obj.sentiment:+.2f}, "
            f"change from previous stage = {stage_obj.sentiment_delta:+.2f} on a -3.00 to +3.00 scale)."
        )
        if include_emotions:
            active_emos = [
                f"{e} in [{stage_obj.emotion_ranges[e][0]:.2f}, {stage_obj.emotion_ranges[e][1]:.2f}]"
                for e in EMOTION_COLUMNS
                if stage_obj.emotion_ranges[e][1] >= 0.15
            ]
            emo_part = (
                " Active emotion ranges (0.00 to 3.00 scale): " + ", ".join(active_emos) + "."
                if active_emos
                else " Discrete emotions: low/neutral across all 6 channels ([0.00, 0.20])."
            )
        else:
            emo_part = ""

    tone_hint = _verbal_tone_from_stage(stage_obj, include_emotions=include_emotions)
    return (
        f"Private stage {stage_obj.stage} behavioral guidance: {sent_part}{emo_part} "
        f"Behavioral translation: {tone_hint} "
        "Let this govern your tone, pacing, and willingness to push back, without overriding visible screen evidence "
        "or your persona identity. Do not mention numerical scores or cues in your answer."
    )


def format_whole_journey_block(
    journey: SelectedJourney,
    granularity: GranularityMode = "emotion_breakdown",
    representation: RepresentationMode = "verbal",
    stochasticity: StochasticityMode = "fixed",
) -> str:
    """Format the entire N-stage emotional journey for upfront placement in the system prompt."""
    lines = [
        "### Behavioral & Emotional Journey Across This 5-Question Interview",
        f"Overall trajectory archetype: {journey.label} — {journey.short_summary}",
        "Your emotional state should evolve turn-by-turn across the 5 questions as follows:",
    ]
    for st in journey.stages:
        cue = format_stage_cue(
            st,
            granularity=granularity,
            representation=representation,
            stochasticity=stochasticity,
        )
        lines.append(f"- Question/Turn {st.stage}: {cue}")
    lines.append(
        "Follow this progression across the conversation so each answer connects naturally to your previous answer "
        "while reflecting the current turn's emotional state. Never state the numbers or announce the pattern."
    )
    return "\n".join(lines)


def build_interview_prompts(
    persona_text: str,
    shared_instruction: str,
    questions: Sequence[str],
    journey: Optional[SelectedJourney],
    granularity: GranularityMode = "emotion_breakdown",
    representation: RepresentationMode = "quant",
    stochasticity: StochasticityMode = "fixed",
    placement: PlacementMode = "per_question_stage",
) -> Tuple[str, List[str], List[str]]:
    """Build `(system_prompt, user_turn_prompts, stage_cues)` for a multi-turn interview.

    When `journey is None` (baseline C0), no emotional conditioning is added to either the
    system prompt or the per-question user prompts.
    """
    base_system = (
        "You are participating in a product research interview as the following persona. "
        "Stay in first person, remain true to your background, and connect your answers naturally "
        "across the conversation using memory of your earlier responses.\n\n"
        f"### Persona Identity\n{persona_text.strip()}\n\n"
        f"### Interview Task\n{shared_instruction.strip()}"
    )

    if journey is None:
        user_prompts = [f"Question {i}: {q.strip()}" for i, q in enumerate(questions, 1)]
        stage_cues = ["" for _ in questions]
        return base_system, user_prompts, stage_cues

    stage_cues = [
        format_stage_cue(
            st,
            granularity=granularity,
            representation=representation,
            stochasticity=stochasticity,
        )
        for st in journey.stages[: len(questions)]
    ]

    if placement == "system_whole_journey":
        whole_block = format_whole_journey_block(
            journey,
            granularity=granularity,
            representation=representation,
            stochasticity=stochasticity,
        )
        system_prompt = f"{base_system}\n\n{whole_block}"
        user_prompts = [f"Question {i} (Turn {i} of {len(questions)}): {q.strip()}" for i, q in enumerate(questions, 1)]
        return system_prompt, user_prompts, stage_cues

    if placement == "per_question_stage":
        system_prompt = base_system
        user_prompts = [
            f"{cue}\n\nQuestion {i} (Turn {i} of {len(questions)}): {q.strip()}"
            for i, (q, cue) in enumerate(zip(questions, stage_cues), 1)
        ]
        return system_prompt, user_prompts, stage_cues

    raise ValueError(f"Unknown placement: {placement}")


def build_two_step_modulation_prompt(
    persona_text: str,
    question: str,
    draft_answer: str,
    stage_cue: str,
) -> str:
    """Step B prompt for the two-step prompt chain (Adolfo & Andy Fano's fix for context squeezing).

    Step A generates the persona's substantive product critique from identity + conversation history.
    Step B modulates the delivery, tone, and emotional sharpness using the turn's affect cue while
    preserving every concrete screen observation and decision.
    """
    return (
        "You are refining a persona's draft interview answer so that its emotional delivery, pacing, "
        "and attitude faithfully reflect the current turn's behavioral cue, without losing any concrete "
        "product critique or inventing facts.\n\n"
        f"Persona Identity:\n{persona_text.strip()}\n\n"
        f"Interview Question:\n{question.strip()}\n\n"
        f"Substantive Draft Answer (from Step A):\n{draft_answer.strip()}\n\n"
        f"Target Behavioral & Emotional Cue for This Turn (Step B):\n{stage_cue.strip()}\n\n"
        "Rewrite the draft in first person (3 to 6 sentences) so the emotional state and tone come through "
        "clearly and naturally while keeping the exact same product decision, specific screen observations, "
        "and continuity with earlier turns. Return only the final spoken response."
    )
