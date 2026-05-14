from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from station_qc_features import (
    compute_anchor_point_features,
    compute_step_change_features,
    compute_stuck_value_features,
    compute_cadence_features,
    prepare_station_observations,
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "min_obs_per_station": 100,
    "stuck": {
        "enabled": True,
        "max_run_len_threshold": 60,
        "max_run_days_threshold": 30,
        "pct_obs_in_repeats_threshold": 0.85,
        "repeat_value_tolerance": 0.0,
        "ignore_zero_values": True,
        "min_nonzero_value_mm": 1.0,
        "min_nonzero_obs": 20,
    },
    "anchor": {
        "enabled": True,
        "bin_width_mm": 5.0,
        "min_anchor_bin_fraction": 0.15,
        "anchor_fraction_threshold": 0.80,
        "top_anchor_bin_fraction_threshold": 0.60,
        "n_anchor_bins_threshold": 2,
        "ignore_zero_values": True,
        "min_nonzero_value_mm": 1.0,
        "min_nonzero_obs": 20,
    },
    "step": {
        "enabled": True,
        "pct_extreme_negative_steps_threshold": 0.08,
        "neg_delta_mean_threshold_mm": -60.0,
        "min_negative_events": 15,
    },
    "cadence": {
        "enabled": True,

        # --------------------------------------------------
        # Gap definition
        # --------------------------------------------------

        # A "long gap" is any interval longer than this
        # Typical snow networks report daily or more often
        "long_gap_days": 7,

        # --------------------------------------------------
        # Frequency thresholds
        # --------------------------------------------------

        # If the median reporting interval exceeds this,
        # the station is considered low cadence
        # (e.g., reporting less than every ~2 weeks)
        "max_median_interval_days": 14,

        # Minimum acceptable reporting frequency
        # 0.10 ≈ one report every 10 days
        "min_reports_per_day": 0.10,

        # --------------------------------------------------
        # Stability / reliability thresholds
        # --------------------------------------------------

        # If more than this fraction of intervals are long gaps,
        # the station is considered unreliable
        "max_pct_long_gaps": 0.30,

        # catches single large outages
        "max_allowed_gap_days": 21,
    },
    "final": {
        "min_methods_flagged": 2,
        "total_score_threshold": 0.85,
        "extreme_single_method_score_threshold": 0.98,
        "watchlist_score_threshold": 0.60,
    },
}


# ============================================================
# HELPERS
# ============================================================

def _require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


# ============================================================
# STANDARDIZERS
# These mostly protect the pipeline in case any method output drifts.
# ============================================================

def standardize_stuck_features(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "station_id",
        "n_obs",
        "max_run_len",
        "max_run_days",
        "pct_obs_in_repeats",
        "n_repeat_events",
        "stuck_score",
        "flag_stuck",
    ]
    _require_columns(df, required, "stuck_df")

    out = df[required].copy()
    out["flag_stuck"] = out["flag_stuck"].fillna(False).astype(bool)
    out["stuck_score"] = out["stuck_score"].fillna(0.0).astype(float)
    return out


def standardize_anchor_features(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "station_id",
        "n_obs",
        "anchor_fraction",
        "n_anchor_bins",
        "top_anchor_bin_fraction",
        "anchor_event_count",
        "anchor_score",
        "flag_anchor",
    ]
    _require_columns(df, required, "anchor_df")

    out = df[required].copy()
    out["flag_anchor"] = out["flag_anchor"].fillna(False).astype(bool)
    out["anchor_score"] = out["anchor_score"].fillna(0.0).astype(float)
    return out


def standardize_step_features(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "station_id",
        "n_step_events",
        "n_negative_step_events",
        "pct_extreme_negative_steps",
        "pct_extreme_negative_steps_of_all",
        "neg_delta_mean_mm",
        "neg_delta_std_mm",
        "step_score",
        "flag_step",
    ]
    _require_columns(df, required, "step_df")

    out = df[required].copy()
    out["flag_step"] = out["flag_step"].fillna(False).astype(bool)
    out["step_score"] = out["step_score"].fillna(0.0).astype(float)
    return out


# ============================================================
# MERGE LAYER
# ============================================================

def merge_station_qc_features(
    stuck_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    step_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge station-level QC feature tables into one summary table.
    """
    summary_df = stuck_df.merge(
        anchor_df,
        on="station_id",
        how="outer",
        suffixes=("", "_anchor"),
    )
    summary_df = summary_df.merge(
        step_df,
        on="station_id",
        how="outer",
        suffixes=("", "_step"),
    )

    flag_cols = [c for c in summary_df.columns if c.startswith("flag_")]
    score_cols = [c for c in summary_df.columns if c.endswith("_score")]

    for col in flag_cols:
        summary_df[col] = summary_df[col].fillna(False).astype(bool)

    for col in score_cols:
        summary_df[col] = summary_df[col].fillna(0.0).astype(float)

    return summary_df


# ============================================================
# FINAL RULES
# ============================================================

def build_reason_string(row: pd.Series) -> str:
    reasons = []

    if bool(row.get("flag_stuck", False)):
        reasons.append("stuck values")
    if bool(row.get("flag_anchor", False)):
        reasons.append("anchor points")
    if bool(row.get("flag_step", False)):
        reasons.append("step changes")

    if not reasons:
        return "no method thresholds exceeded"

    return "; ".join(reasons)


def apply_qc_decision_rules(
    summary_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Apply final station decision rules.

    Adds:
        - n_methods_flagged
        - qc_total_score
        - extreme_single_method
        - final_flag
        - watchlist_flag
        - reason
    """
    out = summary_df.copy()

    for col in ["flag_stuck", "flag_anchor", "flag_step"]:
        if col not in out.columns:
            out[col] = False

    for col in ["stuck_score", "anchor_score", "step_score"]:
        if col not in out.columns:
            out[col] = 0.0

    out["n_methods_flagged"] = (
        out[["flag_stuck", "flag_anchor", "flag_step"]]
        .sum(axis=1)
        .astype(int)
    )

    out["qc_total_score"] = out[["stuck_score", "anchor_score", "step_score"]].mean(axis=1)

    extreme_threshold = config["extreme_single_method_score_threshold"]

    out["extreme_single_method"] = (
        (out["stuck_score"] >= extreme_threshold)
        | (out["anchor_score"] >= extreme_threshold)
        | (out["step_score"] >= extreme_threshold)
    )

    # No method triggered, cannot be flagged
    out["final_flag"] = (
        (
            out["n_methods_flagged"] >= config["min_methods_flagged"]
        )
        | (
            (out["n_methods_flagged"] >= 1)
            & (out["qc_total_score"] >= config["total_score_threshold"])
        )
        | (
            (out["n_methods_flagged"] >= 1)
            & (out["extreme_single_method"])
        )
    )

    if "cadence_review_flag" in out.columns:
        out["final_flag"] = (
            out["final_flag"]
            & (~out["cadence_review_flag"])
        )

    out["watchlist_flag"] = (
        (~out["final_flag"])
        & (
            (out["n_methods_flagged"] == 1)
            | (out["qc_total_score"] >= config["watchlist_score_threshold"])
            | (out.get("cadence_review_flag", False))
        )
    )

    out["reason"] = out.apply(build_reason_string, axis=1)

    return out


# ============================================================
# PIPELINE
# ============================================================

def run_station_qc_pipeline(
    raw_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    *,
    station_col: str = "station_id",
    time_col: str = "valid_time",
    value_col: str = "obs_totalSnowDepth_mm",
) -> Dict[str, pd.DataFrame]:
    """
    Run the full consolidated station QC pipeline.

    Returns
    -------
    dict[str, pd.DataFrame]
        prepared_df
        eligible_df
        stuck_df
        anchor_df
        step_df
        summary_df
        bad_stations_df
        watchlist_df
    """
    config = config or DEFAULT_CONFIG

    prepared_df = prepare_station_observations(
        raw_df,
        station_col=station_col,
        time_col=time_col,
        value_col=value_col,
    )

    cadence_df = compute_cadence_features(
        prepared_df,
        config=config["cadence"],
        station_col=station_col,
        time_col=time_col,
    )

    station_counts = (
        prepared_df.groupby(station_col)
        .size()
        .rename("n_obs_total")
    )
    
    enough_obs_station_ids = station_counts[
        station_counts >= config["min_obs_per_station"]
    ].index
    
    if config["cadence"]["enabled"]:
        cadence_eligible_station_ids = cadence_df.loc[
            ~cadence_df["low_cadence_flag"],
            "station_id",
        ]
    
        eligible_station_ids = set(enough_obs_station_ids).intersection(
            set(cadence_eligible_station_ids)
        )
    else:
        eligible_station_ids = set(enough_obs_station_ids)
    
    eligible_df = prepared_df[
        prepared_df[station_col].isin(eligible_station_ids)
    ].copy()

    if config["stuck"]["enabled"]:
        stuck_df = compute_stuck_value_features(
            eligible_df,
            config=config["stuck"],
            station_col=station_col,
            time_col=time_col,
            value_col=value_col,
        )
        stuck_df = standardize_stuck_features(stuck_df)
    else:
        stuck_df = pd.DataFrame(
            columns=[
                "station_id",
                "n_obs",
                "max_run_len",
                "max_run_days",
                "pct_obs_in_repeats",
                "n_repeat_events",
                "stuck_score",
                "flag_stuck",
            ]
        )

    if config["anchor"]["enabled"]:
        anchor_df = compute_anchor_point_features(
            eligible_df,
            config=config["anchor"],
            station_col=station_col,
            time_col=time_col,
            value_col=value_col,
        )
        anchor_df = standardize_anchor_features(anchor_df)
    else:
        anchor_df = pd.DataFrame(
            columns=[
                "station_id",
                "n_obs",
                "anchor_fraction",
                "n_anchor_bins",
                "top_anchor_bin_fraction",
                "anchor_event_count",
                "anchor_score",
                "flag_anchor",
            ]
        )

    if config["step"]["enabled"]:
        step_df = compute_step_change_features(
            eligible_df,
            config=config["step"],
            station_col=station_col,
            time_col=time_col,
            value_col=value_col,
        )
        step_df = standardize_step_features(step_df)
    else:
        step_df = pd.DataFrame(
            columns=[
                "station_id",
                "n_step_events",
                "n_negative_step_events",
                "pct_extreme_negative_steps",
                "pct_extreme_negative_steps_of_all",
                "neg_delta_mean_mm",
                "neg_delta_std_mm",
                "step_score",
                "flag_step",
            ]
        )

    summary_df = merge_station_qc_features(stuck_df, anchor_df, step_df)

    summary_df = summary_df.merge(
        cadence_df,
        on="station_id",
        how="left",
    )

    summary_df["reporting_class"] = pd.cut(
        summary_df["reports_per_day"],
        bins=[-np.inf, 0.25, 1.0, 4.0, np.inf],
        labels=["sporadic", "daily_or_less", "subdaily", "high_frequency"],
    )

    summary_df["cadence_review_flag"] = (
        summary_df["low_cadence_flag"]
        | (summary_df["reports_per_day"] < 0.25)
        | (summary_df["pct_long_gaps"] > 0.50)
    )

    summary_df = apply_qc_decision_rules(summary_df, config["final"])

    bad_stations_df = (
        summary_df.loc[summary_df["final_flag"]]
        .sort_values(
            ["n_methods_flagged", "qc_total_score"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    watchlist_df = (
        summary_df.loc[summary_df["watchlist_flag"]]
        .sort_values(
            ["n_methods_flagged", "qc_total_score"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    return {
        "prepared_df": prepared_df,
        "eligible_df": eligible_df,
        "cadence_df": cadence_df,
        "stuck_df": stuck_df,
        "anchor_df": anchor_df,
        "step_df": step_df,
        "summary_df": summary_df,
        "bad_stations_df": bad_stations_df,
        "watchlist_df": watchlist_df,
    }


# ============================================================
# OUTPUT WRITER
# ============================================================

def write_station_qc_outputs(
    results: Dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    prefix: str = "",
) -> None:
    """
    Write key QC outputs to CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pfx = f"{prefix}_" if prefix else ""

    results["summary_df"].to_csv(output_dir / f"{pfx}station_qc_summary.csv", index=False)
    results["bad_stations_df"].to_csv(output_dir / f"{pfx}bad_stations.csv", index=False)
    results["watchlist_df"].to_csv(output_dir / f"{pfx}watchlist_stations.csv", index=False)
    results["stuck_df"].to_csv(output_dir / f"{pfx}stuck_features.csv", index=False)
    results["anchor_df"].to_csv(output_dir / f"{pfx}anchor_features.csv", index=False)
    results["step_df"].to_csv(output_dir / f"{pfx}step_features.csv", index=False)
    results["cadence_df"].to_csv(output_dir / f"{pfx}cadence_features.csv", index=False)


def run_and_write_station_qc(
    raw_df: pd.DataFrame,
    output_dir: str | Path,
    config: Optional[Dict[str, Any]] = None,
    *,
    prefix: str = "",
    station_col: str = "station_id",
    time_col: str = "valid_time",
    value_col: str = "obs_totalSnowDepth_mm",
) -> Dict[str, pd.DataFrame]:
    """
    Convenience wrapper: run pipeline and write outputs.
    """
    results = run_station_qc_pipeline(
        raw_df,
        config=config,
        station_col=station_col,
        time_col=time_col,
        value_col=value_col,
    )
    write_station_qc_outputs(results, output_dir=output_dir, prefix=prefix)
    return results
