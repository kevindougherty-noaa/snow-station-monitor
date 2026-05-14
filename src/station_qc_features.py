from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ============================================================
# BASIC PREP
# ============================================================

def prepare_station_observations(
    df: pd.DataFrame,
    *,
    station_col: str = "station_id",
    time_col: str = "valid_time",
    value_col: str = "obs_totalSnowDepth_mm",
) -> pd.DataFrame:
    """
    Standardize and sort observations before QC feature generation.

    Expected minimum columns
    ------------------------
    station_col
    time_col
    value_col

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe sorted by station/time.
    """
    required = [station_col, time_col, value_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[station_col, time_col, value_col])
    out = out.sort_values([station_col, time_col]).reset_index(drop=True)

    return out


# ============================================================
# HELPERS
# ============================================================

def _safe_divide(a: float, b: float) -> float:
    """Return a / b, or 0.0 if denominator is zero or missing."""
    if b == 0 or pd.isna(b):
        return 0.0
    return float(a) / float(b)


def _clip_0_1(x: float) -> float:
    """Clip a numeric value to [0, 1]."""
    if pd.isna(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _normalize_against_threshold(value: float, threshold: float) -> float:
    """
    Normalize a positive metric against a threshold to [0, 1].
    """
    if pd.isna(value) or pd.isna(threshold) or threshold <= 0:
        return 0.0
    if value <= 0:
        return 0.0
    return _clip_0_1(value / threshold)


def _normalize_negative_magnitude(value: float, threshold: float) -> float:
    """
    Normalize a negative-valued severity metric.

    Example
    -------
    value = -80, threshold = -60 -> high severity
    value = -20, threshold = -60 -> lower severity
    """
    if pd.isna(value) or pd.isna(threshold):
        return 0.0
    if threshold >= 0 or value >= 0:
        return 0.0
    return _clip_0_1(abs(value) / abs(threshold))


def _values_equal(a: float, b: float, tolerance: float) -> bool:
    """
    Compare values using an absolute tolerance.

    This is safer than exact equality when values may be floating point.
    """
    return abs(float(a) - float(b)) <= tolerance


# ============================================================
# STUCK VALUE FEATURES
# ============================================================

def compute_stuck_value_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    station_col: str = "station_id",
    time_col: str = "valid_time",
    value_col: str = "obs_totalSnowDepth_mm",
) -> pd.DataFrame:
    """
    Compute station-level stuck-value metrics.

    Returns one row per station with columns:
        - station_id
        - n_obs
        - max_run_len
        - max_run_days
        - pct_obs_in_repeats
        - n_repeat_events
        - stuck_score
        - flag_stuck
    """
    records: List[Dict[str, Any]] = []

    repeat_tolerance = config.get("repeat_value_tolerance", 0.0)

    for station_id, g in df.groupby(station_col, sort=False):
        g = g.sort_values(time_col).copy()

        if config.get("ignore_zero_values", True):
            g = g[g[value_col] >= config.get("min_nonzero_value_mm", 1.0)].copy()

        vals = g[value_col].to_numpy(dtype=float)
        times = pd.to_datetime(g[time_col]).to_numpy()
        n_obs = len(g)

        if n_obs < config.get("min_nonzero_obs", 20):
            records.append(
                {
                    "station_id": station_id,
                    "n_obs": n_obs,
                    "max_run_len": 0,
                    "max_run_days": 0.0,
                    "pct_obs_in_repeats": 0.0,
                    "n_repeat_events": 0,
                    "stuck_score": 0.0,
                    "flag_stuck": False,
                }
            )
            continue

        run_lengths: List[int] = []
        run_days: List[float] = []

        run_start = 0
        for i in range(1, n_obs + 1):
            end_run = False

            if i == n_obs:
                end_run = True
            else:
                same_value = _values_equal(vals[i], vals[i - 1], tolerance=repeat_tolerance)
                if not same_value:
                    end_run = True

            if end_run:
                run_len = i - run_start
                if run_len > 1:
                    start_time = pd.Timestamp(times[run_start])
                    end_time = pd.Timestamp(times[i - 1])
                    duration_days = (end_time - start_time).total_seconds() / 86400.0

                    run_lengths.append(run_len)
                    run_days.append(duration_days)

                run_start = i

        max_run_len = max(run_lengths) if run_lengths else 1
        max_run_days = max(run_days) if run_days else 0.0
        n_repeat_events = len(run_lengths)
        obs_in_repeats = int(sum(run_lengths)) if run_lengths else 0
        pct_obs_in_repeats = _safe_divide(obs_in_repeats, n_obs)

        score_run_len = _normalize_against_threshold(
            max_run_len,
            config["max_run_len_threshold"],
        )
        score_run_days = _normalize_against_threshold(
            max_run_days,
            config["max_run_days_threshold"],
        )
        score_repeat_frac = _normalize_against_threshold(
            pct_obs_in_repeats,
            config["pct_obs_in_repeats_threshold"],
        )

        stuck_score = np.mean([score_run_len, score_run_days, score_repeat_frac])
        stuck_score = _clip_0_1(stuck_score)

        flag_stuck = bool(
            (
                max_run_len >= config["max_run_len_threshold"]
                and max_run_days >= config["max_run_days_threshold"]
            )
            or (pct_obs_in_repeats >= config["pct_obs_in_repeats_threshold"])
        )

        records.append(
            {
                "station_id": station_id,
                "n_obs": n_obs,
                "max_run_len": int(max_run_len),
                "max_run_days": float(max_run_days),
                "pct_obs_in_repeats": float(pct_obs_in_repeats),
                "n_repeat_events": int(n_repeat_events),
                "stuck_score": float(stuck_score),
                "flag_stuck": flag_stuck,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# ANCHOR POINT FEATURES
# ============================================================

def compute_anchor_point_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    station_col: str = "station_id",
    time_col: str = "valid_time",
    value_col: str = "obs_totalSnowDepth_mm",
) -> pd.DataFrame:
    """
    Compute station-level anchor-point metrics.

    Current v1 logic:
    - bin values using bin_width_mm
    - identify bins with occupancy >= min_anchor_bin_fraction
    - measure the fraction of observations sitting in those anchor bins

    Returns one row per station with columns:
        - station_id
        - n_obs
        - anchor_fraction
        - n_anchor_bins
        - top_anchor_bin_fraction
        - anchor_event_count
        - anchor_score
        - flag_anchor
    """
    records: List[Dict[str, Any]] = []

    bin_width_mm = float(config.get("bin_width_mm", 1.0))
    min_anchor_bin_fraction = float(config.get("min_anchor_bin_fraction", 0.05))

    for station_id, g in df.groupby(station_col, sort=False):
        # time_col included for interface consistency even if not used yet
        _ = time_col
    
        if config.get("ignore_zero_values", True):
            g = g[g[value_col] >= config.get("min_nonzero_value_mm", 1.0)].copy()
    
        vals = g[value_col].dropna().to_numpy(dtype=float)
        n_obs = len(vals)
    
        if n_obs < config.get("min_nonzero_obs", 20):
            records.append(
                {
                    "station_id": station_id,
                    "n_obs": n_obs,
                    "anchor_fraction": 0.0,
                    "n_anchor_bins": 0,
                    "top_anchor_bin_fraction": 0.0,
                    "anchor_event_count": 0,
                    "anchor_score": 0.0,
                    "flag_anchor": False,
                }
            )
            continue

        binned = np.round(vals / bin_width_mm) * bin_width_mm
        bin_counts = pd.Series(binned).value_counts().sort_values(ascending=False)

        top_anchor_count = int(bin_counts.iloc[0]) if len(bin_counts) > 0 else 0
        top_anchor_bin_fraction = _safe_divide(top_anchor_count, n_obs)

        anchor_bins = bin_counts[bin_counts / n_obs >= min_anchor_bin_fraction]
        n_anchor_bins = int(len(anchor_bins))
        anchor_event_count = int(anchor_bins.sum()) if len(anchor_bins) > 0 else 0
        anchor_fraction = _safe_divide(anchor_event_count, n_obs)

        score_anchor_fraction = _normalize_against_threshold(
            anchor_fraction,
            config["anchor_fraction_threshold"],
        )
        score_top_bin = _normalize_against_threshold(
            top_anchor_bin_fraction,
            config["top_anchor_bin_fraction_threshold"],
        )
        score_n_bins = _normalize_against_threshold(
            n_anchor_bins,
            config["n_anchor_bins_threshold"],
        )

        anchor_score = np.mean([score_anchor_fraction, score_top_bin, score_n_bins])
        anchor_score = _clip_0_1(anchor_score)

        flag_anchor = bool(
            (
                top_anchor_bin_fraction >= config["top_anchor_bin_fraction_threshold"]
                and anchor_fraction >= config["anchor_fraction_threshold"]
            )
        )

        records.append(
            {
                "station_id": station_id,
                "n_obs": n_obs,
                "anchor_fraction": float(anchor_fraction),
                "n_anchor_bins": int(n_anchor_bins),
                "top_anchor_bin_fraction": float(top_anchor_bin_fraction),
                "anchor_event_count": int(anchor_event_count),
                "anchor_score": float(anchor_score),
                "flag_anchor": flag_anchor,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# STEP CHANGE FEATURES
# ============================================================

def compute_step_change_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    station_col: str = "station_id",
    time_col: str = "valid_time",
    value_col: str = "obs_totalSnowDepth_mm",
) -> pd.DataFrame:
    """
    Compute station-level step-change metrics.

    Returns one row per station with columns:
        - station_id
        - n_step_events
        - n_negative_step_events
        - pct_extreme_negative_steps
        - pct_extreme_negative_steps_of_all
        - neg_delta_mean_mm
        - neg_delta_std_mm
        - step_score
        - flag_step

    Notes
    -----
    In v1, "extreme" is based on a fixed threshold from config.
    Later, you can swap this for robust-sigma thresholding.
    """
    records: List[Dict[str, Any]] = []

    threshold_mm = float(config["neg_delta_mean_threshold_mm"])
    min_negative_events = int(config["min_negative_events"])

    for station_id, g in df.groupby(station_col, sort=False):
        g = g.sort_values(time_col).copy()
    
        vals = g[value_col].to_numpy(dtype=float)

        if len(vals) < 2:
            records.append(
                {
                    "station_id": station_id,
                    "n_step_events": 0,
                    "n_negative_step_events": 0,
                    "pct_extreme_negative_steps": 0.0,
                    "pct_extreme_negative_steps_of_all": 0.0,
                    "neg_delta_mean_mm": np.nan,
                    "neg_delta_std_mm": np.nan,
                    "step_score": 0.0,
                    "flag_step": False,
                }
            )
            continue

        deltas = np.diff(vals)
        n_step_events = len(deltas)

        neg_deltas = deltas[deltas < 0]
        n_negative_step_events = len(neg_deltas)

        if n_negative_step_events == 0:
            n_extreme = 0
            pct_extreme_negative_steps = 0.0
            pct_extreme_negative_steps_of_all = 0.0
            neg_delta_mean_mm = 0.0
            neg_delta_std_mm = 0.0
        else:
            n_extreme = int(np.sum(neg_deltas <= threshold_mm))
            pct_extreme_negative_steps = _safe_divide(n_extreme, n_negative_step_events)
            pct_extreme_negative_steps_of_all = _safe_divide(n_extreme, n_step_events)
            neg_delta_mean_mm = float(np.mean(neg_deltas))
            neg_delta_std_mm = float(np.std(neg_deltas))

        score_pct_extreme = _normalize_against_threshold(
            pct_extreme_negative_steps,
            config["pct_extreme_negative_steps_threshold"],
        )
        score_neg_mean = _normalize_negative_magnitude(
            neg_delta_mean_mm,
            threshold_mm,
        )

        step_score = np.mean([score_pct_extreme, score_neg_mean])
        step_score = _clip_0_1(step_score)

        enough_negative_events = n_negative_step_events >= min_negative_events
        flag_step = bool(
            enough_negative_events
            and (
                (pct_extreme_negative_steps >= config["pct_extreme_negative_steps_threshold"])
                or (neg_delta_mean_mm <= threshold_mm)
            )
        )

        records.append(
            {
                "station_id": station_id,
                "n_step_events": int(n_step_events),
                "n_negative_step_events": int(n_negative_step_events),
                "pct_extreme_negative_steps": float(pct_extreme_negative_steps),
                "pct_extreme_negative_steps_of_all": float(pct_extreme_negative_steps_of_all),
                "neg_delta_mean_mm": float(neg_delta_mean_mm),
                "neg_delta_std_mm": float(neg_delta_std_mm),
                "step_score": float(step_score),
                "flag_step": flag_step,
            }
        )

    return pd.DataFrame(records)

# ============================================================
# CADENCE FEATURES
# ============================================================
def compute_cadence_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    station_col: str = "station_id",
    time_col: str = "valid_time",
) -> pd.DataFrame:
    records = []

    long_gap_days = config.get("long_gap_days", 7)

    for station_id, g in df.groupby(station_col, sort=False):
        g = g.sort_values(time_col).copy()
        times = pd.to_datetime(g[time_col])

        n_obs = len(g)

        if n_obs < 2:
            records.append(
                {
                    "station_id": station_id,
                    "n_obs_total": n_obs,
                    "median_interval_hours": np.nan,
                    "median_interval_days": np.nan,
                    "mean_interval_hours": np.nan,
                    "max_interval_days": np.nan,
                    "pct_long_gaps": np.nan,
                    "reports_per_day": np.nan,
                    "low_cadence_flag": True,
                }
            )
            continue

        intervals_hours = times.diff().dt.total_seconds().dropna() / 3600.0
        intervals_days = intervals_hours / 24.0

        total_days = (times.max() - times.min()).total_seconds() / 86400.0
        reports_per_day = _safe_divide(n_obs, total_days) if total_days > 0 else n_obs

        median_interval_hours = float(intervals_hours.median())
        median_interval_days = median_interval_hours / 24.0
        mean_interval_hours = float(intervals_hours.mean())
        max_interval_days = float(intervals_days.max())

        pct_long_gaps = float((intervals_days > long_gap_days).mean())

        low_cadence_flag = bool(
            (median_interval_days > config["max_median_interval_days"])
            or (reports_per_day < config["min_reports_per_day"])
            or (pct_long_gaps > config["max_pct_long_gaps"])
            or (max_interval_days > config["max_allowed_gap_days"])
        )

        records.append(
            {
                "station_id": station_id,
                "n_obs_total": int(n_obs),
                "median_interval_hours": median_interval_hours,
                "median_interval_days": median_interval_days,
                "mean_interval_hours": mean_interval_hours,
                "max_interval_days": max_interval_days,
                "pct_long_gaps": pct_long_gaps,
                "reports_per_day": float(reports_per_day),
                "low_cadence_flag": low_cadence_flag,
            }
        )

    return pd.DataFrame(records)


def compute_station_cadence_metrics(df):
    """
    Compute reporting cadence metrics per station.
    """

    g = df.groupby("station_id")

    cadence_df = g.agg(
        n_obs=("valid_time", "count"),
        first_time=("valid_time", "min"),
        last_time=("valid_time", "max"),
        reporting_days=("valid_time", lambda x: x.dt.date.nunique()),
    ).reset_index()

    cadence_df["data_span_days"] = (
        cadence_df["last_time"] - cadence_df["first_time"]
    ).dt.days + 1

    cadence_df["obs_per_day"] = (
        cadence_df["n_obs"]
        / cadence_df["reporting_days"].clip(lower=1)
    )

    cadence_df["reporting_frequency"] = np.where(
        cadence_df["obs_per_day"] >= 24,
        "hourly",
        np.where(
            cadence_df["obs_per_day"] >= 4,
            "subdaily",
            np.where(
                cadence_df["obs_per_day"] >= 1,
                "daily",
                "sporadic",
            ),
        ),
    )

    return cadence_df