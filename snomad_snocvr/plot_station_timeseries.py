#!/usr/bin/env python3
"""
Plot snow-depth time series for one or more stations with a Cartopy inset map.

This script:
  - reads a CSV of station observations
  - optionally filters to a provided station list
  - otherwise can plot the top N stations by observation count
  - creates one PNG per station

Expected CSV columns:
  station_id
  valid_time
  lat
  lon
  obs_totalSnowDepth_mm

Optional usage examples:
  python plot_station_timeseries.py \
      --csv snocvr_obs_202409_202505.csv \
      --station-list station_ids.txt \
      --outdir station_plots

  python plot_station_timeseries.py \
      --csv snocvr_obs_202409_202505.csv \
      --top-n 10 \
      --outdir top10_plots

  python plot_station_timeseries.py \
      --csv snocvr_obs_202409_202505.csv \
      --station-id 72469 \
      --outdir single_station_plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# Column configuration
# =============================================================================
STATION_COL = "station_id"
TIME_COL = "valid_time"
LAT_COL = "lat"
LON_COL = "lon"
OBS_COL = "obs_totalSnowDepth_mm"


# =============================================================================
# Data loading / cleaning
# =============================================================================
def load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    """Load CSV and perform basic cleaning."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_cols = [STATION_COL, TIME_COL, LAT_COL, LON_COL, OBS_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df.copy()

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df[OBS_COL] = pd.to_numeric(df[OBS_COL], errors="coerce")
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")

    # Robustly remove extreme fill values if present
    df.loc[df[OBS_COL] < -1e30, OBS_COL] = np.nan
    df.loc[df[LAT_COL] < -1e30, LAT_COL] = np.nan
    df.loc[df[LON_COL] < -1e30, LON_COL] = np.nan

    df = df.dropna(subset=[STATION_COL, TIME_COL, LAT_COL, LON_COL, OBS_COL]).copy()

    # Normalize station IDs as strings
    df[STATION_COL] = df[STATION_COL].astype(str).str.strip()

    # Sort and deduplicate
    df = df.sort_values([STATION_COL, TIME_COL])
    df = df.drop_duplicates(subset=[STATION_COL, TIME_COL], keep="last")

    return df


def filter_flag_column(
    df: pd.DataFrame,
    flag_column: str | None = None,
    flag_value: int | float | str | None = None,
) -> pd.DataFrame:
    """
    Optionally filter dataframe on a flag column, e.g.
      flag_temporal_thinning == 0
    """
    if flag_column is None:
        return df

    if flag_column not in df.columns:
        raise ValueError(f"Requested flag column '{flag_column}' not found in CSV.")

    out = df.copy()
    if flag_value is None:
        out = out[out[flag_column].notna()].copy()
    else:
        out = out[out[flag_column] == flag_value].copy()

    return out


# =============================================================================
# Station selection
# =============================================================================
def read_station_list(path: str | Path) -> list[str]:
    """Read one station ID per line; ignore blank lines and comments."""
    path = Path(path)
    station_ids = []

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            station_ids.append(s)

    return station_ids


def top_n_stations(df: pd.DataFrame, n: int = 10) -> tuple[list[str], pd.Series]:
    """Return top N station IDs by number of rows."""
    station_counts = df.groupby(STATION_COL).size().sort_values(ascending=False)
    return station_counts.head(n).index.tolist(), station_counts


# =============================================================================
# Plotting
# =============================================================================
def plot_station_timeseries_with_cartopy_map(
    df: pd.DataFrame,
    station_id: str,
    out_png: str | Path | None = None,
    step_threshold: float = -50.0,
    map_lon_pad: float = 10.0,
    map_lat_pad: float = 8.0,
    figsize: tuple[float, float] = (14, 4),
):
    """
    Plot one station's time series with an inset map.

    Large negative changes are marked where:
        diff <= step_threshold
    """
    d = df[df[STATION_COL] == str(station_id)].copy()
    if d.empty:
        print(f"[WARN] No data for station_id='{station_id}'")
        return

    d = d.sort_values(TIME_COL).copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], utc=True, errors="coerce")
    d[OBS_COL] = pd.to_numeric(d[OBS_COL], errors="coerce")
    d = d.dropna(subset=[TIME_COL, OBS_COL, LAT_COL, LON_COL])

    if d.empty:
        print(f"[WARN] No valid plottable data for station_id='{station_id}'")
        return

    lat = float(d[LAT_COL].median())
    lon = float(d[LON_COL].median())

    d["diff"] = d[OBS_COL].diff()
    large_steps = d["diff"] <= step_threshold

    fig, ax = plt.subplots(figsize=figsize)

    # Main line
    ax.plot(
        d[TIME_COL],
        d[OBS_COL],
        lw=1.0,
        label="Snow depth (mm)",
        zorder=1,
    )

    # Points
    ax.scatter(
        d[TIME_COL],
        d[OBS_COL],
        s=18,
        alpha=0.7,
        linewidths=0,
        label="Observations",
        zorder=2,
    )

    # Large negative steps
    if large_steps.any():
        ax.scatter(
            d.loc[large_steps, TIME_COL],
            d.loc[large_steps, OBS_COL],
            s=40,
            marker="x",
            color="black",
            label=f"Δ ≤ {step_threshold} mm",
            zorder=3,
        )

    ax.set_title(f"{station_id} — Snow Depth Time Series")
    ax.set_ylabel("Snow depth (mm)")
    ax.set_xlabel("Time")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Inset map
    inset = fig.add_axes(
        [0.13, 0.30, 0.12, 0.40],
        projection=ccrs.PlateCarree(),
    )

    inset.set_extent(
        [lon - map_lon_pad, lon + map_lon_pad, lat - map_lat_pad, lat + map_lat_pad],
        crs=ccrs.PlateCarree(),
    )

    inset.add_feature(cfeature.LAND, facecolor="#f2f2df", zorder=0)
    inset.add_feature(cfeature.OCEAN, facecolor="#dbe7f2", zorder=0)
    inset.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
    inset.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=1)

    try:
        inset.add_feature(cfeature.STATES, linewidth=0.3, zorder=1)
    except Exception:
        pass

    inset.plot(
        lon,
        lat,
        marker="o",
        markersize=6,
        color="tab:blue",
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    inset.set_xticks([])
    inset.set_yticks([])

    inset.text(
        0.02,
        0.02,
        f"{lat:.2f}, {lon:.2f}",
        transform=inset.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", alpha=0.6),
    )

    if out_png:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved {out_png}")
    else:
        plt.show()


def plot_many_stations(
    df: pd.DataFrame,
    station_ids: list[str],
    out_dir: str | Path,
    step_threshold: float = -50.0,
):
    """Plot one PNG per station."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sid in station_ids:
        out_png = out_dir / f"{sid}_timeseries.png"
        plot_station_timeseries_with_cartopy_map(
            df=df,
            station_id=sid,
            out_png=out_png,
            step_threshold=step_threshold,
        )


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot station snow-depth time series with inset maps."
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Input CSV file with station observations.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory to save output PNGs.",
    )

    # Station selection options
    parser.add_argument(
        "--station-list",
        help="Text file with one station_id per line.",
    )
    parser.add_argument(
        "--station-id",
        help="Single station_id to plot.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Plot top N stations by row count.",
    )

    # Optional filtering
    parser.add_argument(
        "--flag-column",
        default=None,
        help="Optional flag column to filter on before plotting.",
    )
    parser.add_argument(
        "--flag-value",
        default=None,
        help="Value to keep in --flag-column, e.g. 0",
    )

    parser.add_argument(
        "--step-threshold",
        type=float,
        default=-50.0,
        help="Threshold for marking large negative steps.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = load_dataframe(args.csv)

    # Optional flag filtering
    flag_value = args.flag_value
    if flag_value is not None:
        # Try to interpret numerically when possible
        try:
            if "." in str(flag_value):
                flag_value = float(flag_value)
            else:
                flag_value = int(flag_value)
        except ValueError:
            pass

    df = filter_flag_column(
        df,
        flag_column=args.flag_column,
        flag_value=flag_value,
    )

    # Decide which stations to plot
    if args.station_id:
        station_ids = [str(args.station_id)]

    elif args.station_list:
        station_ids = read_station_list(args.station_list)

    elif args.top_n is not None:
        station_ids, station_counts = top_n_stations(df, n=args.top_n)
        print("Top stations by count:")
        print(station_counts.head(args.top_n))

    else:
        raise ValueError(
            "Must provide one of: --station-id, --station-list, or --top-n"
        )

    if not station_ids:
        print("[WARN] No station IDs selected. Exiting.")
        return

    print(f"Plotting {len(station_ids)} station(s)...")
    plot_many_stations(
        df=df,
        station_ids=station_ids,
        out_dir=args.outdir,
        step_threshold=args.step_threshold,
    )
    print("[done]")


if __name__ == "__main__":
    main()
