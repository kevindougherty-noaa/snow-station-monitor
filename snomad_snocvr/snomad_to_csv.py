from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# =============================================================================
# User settings
# =============================================================================
BASE_DIR = Path("/scratch3/NCEPDEV/stmp/Kevin.Dougherty/data/snow_data")
START_CYCLE = datetime.strptime("2024090100", "%Y%m%d%H")
END_CYCLE   = datetime.strptime("2025051418", "%Y%m%d%H")

OUT_CSV = BASE_DIR / "snocvr_obs_202409_202505.csv"
OUT_MISSING = BASE_DIR / "missing_snow_cycles.txt"

FILL_FLOAT = -3.368795e38
FILL_INT32 = -2147483643
FILL_INT64 = -9223372036854775801

# =============================================================================
# Helpers
# =============================================================================
def _mask_fill(a, fill):
    """Return numeric array with fill values replaced by NaN."""
    a = np.asarray(a)
    out = a.astype("float64", copy=False) if np.issubdtype(a.dtype, np.number) else a
    if np.issubdtype(a.dtype, np.number):
        out = out.copy()
        out[a == fill] = np.nan
    return out


def generate_cycles(start: datetime, end: datetime, step_hours: int = 6):
    """Yield datetime cycles from start to end inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(hours=step_hours)


def expected_file_path(base_dir: Path, cycle: datetime) -> Path:
    """Build expected diag file path for a given cycle."""
    ymd = cycle.strftime("%Y%m%d")
    hh = cycle.strftime("%H")
    fname = f"diag_snocvr_snomad_{ymd}{hh}.nc"

    return (
        base_dir
        / f"gdas.{ymd}"
        / hh
        / "analysis"
        / "snow"
        / fname
    )


def collect_cycle_records(base_dir: Path, start: datetime, end: datetime):
    """
    Build records for all existing files and track missing cycles.
    
    Returns
    -------
    records : list[dict]
        Each dict contains path, cycle, month
    missing : list[dict]
        Each dict contains cycle, path
    """
    records = []
    missing = []

    for cycle in generate_cycles(start, end, step_hours=6):
        fpath = expected_file_path(base_dir, cycle)
        month = cycle.strftime("%Y%m")

        if fpath.exists():
            records.append(
                {
                    "path": fpath,
                    "cycle": cycle.strftime("%Y%m%d%H"),
                    "month": month,
                }
            )
        else:
            missing.append(
                {
                    "cycle": cycle.strftime("%Y%m%d%H"),
                    "path": str(fpath),
                }
            )

    return records, missing


# =============================================================================
# NetCDF extraction
# =============================================================================
def extract_one_file(nc_path: str | Path) -> pd.DataFrame:
    nc_path = Path(nc_path)

    with Dataset(nc_path, mode="r") as ds:
        # ---- MetaData
        g_md = ds.groups["MetaData"]

        station_id = np.array(g_md.variables["stationIdentification"][:]).astype(str)
        dt_sec = np.array(g_md.variables["dateTime"][:])

        lat = _mask_fill(g_md.variables["latitude"][:], FILL_FLOAT)
        lon = _mask_fill(g_md.variables["longitude"][:], FILL_FLOAT)
        elev = _mask_fill(g_md.variables["stationElevation"][:], FILL_FLOAT)

        valid_time = pd.to_datetime(_mask_fill(dt_sec, FILL_INT64), unit="s", utc=True)

        # ---- ObsValue
        g_obs = ds.groups["ObsValue"]
        obs = _mask_fill(g_obs.variables["totalSnowDepth"][:], FILL_FLOAT)

        # ---- EffectiveQC0/1
        g_qc0 = ds.groups["EffectiveQC0"]
        g_qc1 = ds.groups["EffectiveQC1"]
        effqc0 = _mask_fill(g_qc0.variables["totalSnowDepth"][:], FILL_INT32)
        effqc1 = _mask_fill(g_qc1.variables["totalSnowDepth"][:], FILL_INT32)

        # ---- EffectiveError0/1
        g_err0 = ds.groups["EffectiveError0"]
        g_err1 = ds.groups["EffectiveError1"]
        efferr0 = _mask_fill(g_err0.variables["totalSnowDepth"][:], FILL_FLOAT)
        efferr1 = _mask_fill(g_err1.variables["totalSnowDepth"][:], FILL_FLOAT)

        # ---- DiagnosticFlags
        diag = {}
        g_df = ds.groups.get("DiagnosticFlags", None)
        if g_df is not None:
            for subname, subgrp in g_df.groups.items():
                v = subgrp.variables.get("totalSnowDepth", None)
                if v is None:
                    continue
                arr = np.array(v[:]).astype("uint8")
                diag[f"flag_{subname}"] = arr

        # ---- Build dataframe
        df = pd.DataFrame(
            {
                "station_id": station_id,
                "valid_time": valid_time,
                "lat": lat,
                "lon": lon,
                "elev_m": elev,
                "obs_totalSnowDepth_mm": obs,
                "effqc0": effqc0,
                "effqc1": effqc1,
                "efferr0": efferr0,
                "efferr1": efferr1,
                **diag,
            }
        )

        # Drop rows with missing valid_time
        df = df[df["valid_time"].notna()].copy()

        # provenance
        df["source_file"] = nc_path.name

        return df


def extract_many_to_csv(records, out_csv, overwrite=False, chunksize_files=25):
    """
    records: list of dicts with keys ['path', 'month', 'cycle']
    """
    out_csv = Path(out_csv)

    if overwrite and out_csv.exists():
        out_csv.unlink()

    wrote_header = out_csv.exists()
    batch = []

    for i, rec in enumerate(records, start=1):
        nc_path = rec["path"]
        month = rec["month"]
        cycle = rec["cycle"]

        try:
            df = extract_one_file(nc_path)
        except Exception as e:
            print(f"[ERROR] Failed reading {nc_path}: {e}")
            continue

        df["month"] = month
        df["cycle"] = cycle
        df["source_file"] = nc_path.name

        batch.append(df)

        if (i % chunksize_files) == 0:
            out = pd.concat(batch, ignore_index=True)
            out.to_csv(out_csv, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            print(f"[{i}/{len(records)}] appended {len(out):,} rows")
            batch = []

    if batch:
        out = pd.concat(batch, ignore_index=True)
        out.to_csv(out_csv, mode="a", header=not wrote_header, index=False)
        print(f"[done] appended {len(out):,} rows to {out_csv}")


def write_missing_cycles(missing, out_txt):
    out_txt = Path(out_txt)
    with open(out_txt, "w") as f:
        f.write("# Missing snow cycles\n")
        f.write("cycle,path\n")
        for rec in missing:
            f.write(f"{rec['cycle']},{rec['path']}\n")

    print(f"[done] wrote {len(missing)} missing cycles to {out_txt}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("Collecting cycle records...")
    records, missing = collect_cycle_records(BASE_DIR, START_CYCLE, END_CYCLE)

    print(f"Expected files : {len(records) + len(missing)}")
    print(f"Found files    : {len(records)}")
    print(f"Missing files  : {len(missing)}")

    write_missing_cycles(missing, OUT_MISSING)

    if not records:
        print("[WARN] No files found. Exiting without creating CSV.")
        return

    extract_many_to_csv(
        records,
        out_csv=OUT_CSV,
        overwrite=True,
        chunksize_files=25,
    )

if __name__ == "__main__":
    main()
