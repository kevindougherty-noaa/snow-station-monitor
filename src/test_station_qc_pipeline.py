import pandas as pd
import numpy as np

from station_qc_features import (
    prepare_station_observations,
    compute_stuck_value_features,
    compute_anchor_point_features,
    compute_step_change_features,
)

from station_qc_pipeline import (
    DEFAULT_CONFIG,
    standardize_stuck_features,
    standardize_anchor_features,
    standardize_step_features,
    merge_station_qc_features,
    apply_qc_decision_rules,
    run_station_qc_pipeline,
)


def make_test_dataset() -> pd.DataFrame:
    """
    Build a small synthetic dataset with 4 stations:

    GOOD:
        Mostly varying values, should not strongly flag.

    STUCK:
        Long repeated runs.

    ANCHOR:
        Repeated returns to a few values.

    STEP:
        Frequent large negative steps.
    """
    rows = []

    base_time = pd.Timestamp("2025-01-01 00:00:00")

    # ---------------------------------------------------------
    # GOOD station: gradual, mostly varied values
    # ---------------------------------------------------------
    good_vals = [10, 11, 12, 12.5, 13, 12, 11.5, 11, 10.5, 10] * 12
    for i, v in enumerate(good_vals):
        rows.append(
            {
                "station_id": "GOOD",
                "valid_time": base_time + pd.Timedelta(hours=12 * i),
                "obs_totalSnowDepth_mm": v,
            }
        )

    # ---------------------------------------------------------
    # STUCK station: long flat runs
    # ---------------------------------------------------------
    stuck_vals = (
        [20] * 20
        + [19] * 18
        + [18] * 15
        + [17] * 14
        + [16] * 12
        + [15] * 10
        + [14] * 10
        + [13] * 10
        + [12] * 10
        + [11] * 10
    )
    for i, v in enumerate(stuck_vals):
        rows.append(
            {
                "station_id": "STUCK",
                "valid_time": base_time + pd.Timedelta(hours=12 * i),
                "obs_totalSnowDepth_mm": v,
            }
        )

    # ---------------------------------------------------------
    # ANCHOR station: repeatedly returns to same few bins
    # ---------------------------------------------------------
    anchor_vals = [5, 5, 5, 5, 5, 5, 5, 10, 12, 14] * 12
    for i, v in enumerate(anchor_vals):
        rows.append(
            {
                "station_id": "ANCHOR",
                "valid_time": base_time + pd.Timedelta(hours=12 * i),
                "obs_totalSnowDepth_mm": v,
            }
        )

    # ---------------------------------------------------------
    # STEP station: repeated sharp negative drops
    # ---------------------------------------------------------
    step_pattern = [200, 195, 190, 100, 98, 95, 30, 28, 25, 10]
    step_vals = step_pattern * 12
    for i, v in enumerate(step_vals):
        rows.append(
            {
                "station_id": "STEP",
                "valid_time": base_time + pd.Timedelta(hours=12 * i),
                "obs_totalSnowDepth_mm": v,
            }
        )

    return pd.DataFrame(rows)


def test_prepare_station_observations():
    df = pd.DataFrame(
        {
            "station_id": ["B", "A", "A"],
            "valid_time": ["2025-01-02", "2025-01-03", "2025-01-01"],
            "obs_totalSnowDepth_mm": [2, 3, 1],
        }
    )

    out = prepare_station_observations(df)

    assert list(out["station_id"]) == ["A", "A", "B"]
    assert pd.api.types.is_datetime64_any_dtype(out["valid_time"])
    assert len(out) == 3


def test_compute_stuck_value_features():
    df = make_test_dataset()
    prepared = prepare_station_observations(df)

    stuck_df = compute_stuck_value_features(
        prepared,
        config=DEFAULT_CONFIG["stuck"],
    )

    stuck_df = standardize_stuck_features(stuck_df)

    expected_cols = {
        "station_id",
        "n_obs",
        "max_run_len",
        "max_run_days",
        "pct_obs_in_repeats",
        "n_repeat_events",
        "stuck_score",
        "flag_stuck",
    }
    assert expected_cols.issubset(set(stuck_df.columns))

    stuck_row = stuck_df.loc[stuck_df["station_id"] == "STUCK"].iloc[0]
    good_row = stuck_df.loc[stuck_df["station_id"] == "GOOD"].iloc[0]

    assert bool(stuck_row["flag_stuck"]) is True
    assert stuck_row["stuck_score"] > good_row["stuck_score"]


def test_compute_anchor_point_features():
    df = make_test_dataset()
    prepared = prepare_station_observations(df)

    anchor_df = compute_anchor_point_features(
        prepared,
        config=DEFAULT_CONFIG["anchor"],
    )

    anchor_df = standardize_anchor_features(anchor_df)

    expected_cols = {
        "station_id",
        "n_obs",
        "anchor_fraction",
        "n_anchor_bins",
        "top_anchor_bin_fraction",
        "anchor_event_count",
        "anchor_score",
        "flag_anchor",
    }
    assert expected_cols.issubset(set(anchor_df.columns))

    anchor_row = anchor_df.loc[anchor_df["station_id"] == "ANCHOR"].iloc[0]

    assert bool(anchor_row["flag_anchor"]) is True
    assert anchor_row["top_anchor_bin_fraction"] >= 0.40
    assert anchor_row["n_anchor_bins"] >= 2


def test_compute_step_change_features():
    df = make_test_dataset()
    prepared = prepare_station_observations(df)

    step_df = compute_step_change_features(
        prepared,
        config=DEFAULT_CONFIG["step"],
    )

    step_df = standardize_step_features(step_df)

    expected_cols = {
        "station_id",
        "n_step_events",
        "n_negative_step_events",
        "pct_extreme_negative_steps",
        "pct_extreme_negative_steps_of_all",
        "neg_delta_mean_mm",
        "neg_delta_std_mm",
        "step_score",
        "flag_step",
    }
    assert expected_cols.issubset(set(step_df.columns))

    step_row = step_df.loc[step_df["station_id"] == "STEP"].iloc[0]
    good_row = step_df.loc[step_df["station_id"] == "GOOD"].iloc[0]

    assert bool(step_row["flag_step"]) is True
    assert step_row["step_score"] > good_row["step_score"]


def test_merge_station_qc_features():
    df = make_test_dataset()
    prepared = prepare_station_observations(df)

    stuck_df = standardize_stuck_features(
        compute_stuck_value_features(prepared, config=DEFAULT_CONFIG["stuck"])
    )
    anchor_df = standardize_anchor_features(
        compute_anchor_point_features(prepared, config=DEFAULT_CONFIG["anchor"])
    )
    step_df = standardize_step_features(
        compute_step_change_features(prepared, config=DEFAULT_CONFIG["step"])
    )

    summary_df = merge_station_qc_features(stuck_df, anchor_df, step_df)

    assert "station_id" in summary_df.columns
    assert "flag_stuck" in summary_df.columns
    assert "flag_anchor" in summary_df.columns
    assert "flag_step" in summary_df.columns
    assert "stuck_score" in summary_df.columns
    assert "anchor_score" in summary_df.columns
    assert "step_score" in summary_df.columns

    assert set(summary_df["station_id"]) == {"GOOD", "STUCK", "ANCHOR", "STEP"}


def test_apply_qc_decision_rules():
    summary_df = pd.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "flag_stuck": [True, False, False],
            "flag_anchor": [True, False, False],
            "flag_step": [True, True, False],
            "stuck_score": [0.95, 0.10, 0.05],
            "anchor_score": [0.90, 0.15, 0.05],
            "step_score": [0.95, 0.75, 0.05],
        }
    )

    out = apply_qc_decision_rules(summary_df, DEFAULT_CONFIG["final"])

    row_a = out.loc[out["station_id"] == "A"].iloc[0]
    row_b = out.loc[out["station_id"] == "B"].iloc[0]
    row_c = out.loc[out["station_id"] == "C"].iloc[0]

    assert row_a["n_methods_flagged"] == 3
    assert bool(row_a["final_flag"]) is True

    assert row_b["n_methods_flagged"] == 1
    assert bool(row_b["watchlist_flag"]) is True or bool(row_b["final_flag"]) is True

    assert row_c["n_methods_flagged"] == 0
    assert bool(row_c["final_flag"]) is False


def test_run_station_qc_pipeline_end_to_end():
    df = make_test_dataset()
    results = run_station_qc_pipeline(df, config=DEFAULT_CONFIG)

    expected_keys = {
        "prepared_df",
        "eligible_df",
        "stuck_df",
        "anchor_df",
        "step_df",
        "summary_df",
        "bad_stations_df",
        "watchlist_df",
    }
    assert expected_keys.issubset(set(results.keys()))

    summary_df = results["summary_df"]
    bad_df = results["bad_stations_df"]

    assert len(summary_df) > 0
    assert "final_flag" in summary_df.columns
    assert "reason" in summary_df.columns

    assert summary_df[["flag_stuck", "flag_anchor", "flag_step"]].any().any()


def test_pipeline_respects_min_obs_per_station():
    df = make_test_dataset()

    # Add a short-record station that should be excluded
    short_rows = pd.DataFrame(
        {
            "station_id": ["SHORT"] * 5,
            "valid_time": pd.date_range("2025-02-01", periods=5, freq="12h"),
            "obs_totalSnowDepth_mm": [1, 1, 1, 1, 1],
        }
    )

    df = pd.concat([df, short_rows], ignore_index=True)

    config = {
        **DEFAULT_CONFIG,
        "min_obs_per_station": 20,
    }

    results = run_station_qc_pipeline(df, config=config)
    summary_ids = set(results["summary_df"]["station_id"])

    assert "SHORT" not in summary_ids
