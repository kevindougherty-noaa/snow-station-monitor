from pathlib import Path
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
BASE_DIR = Path("/scratch3/NCEPDEV/stmp/Kevin.Dougherty/data/snow_data")
START_CYCLE = datetime.strptime("2024090100", "%Y%m%d%H")
END_CYCLE   = datetime.strptime("2025051418", "%Y%m%d%H")  # change if needed

OUT_MISSING_TXT = BASE_DIR / "missing_snow_cycles.txt"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def expected_file_path(base_dir: Path, cycle: datetime) -> Path:
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

def generate_cycles(start: datetime, end: datetime, step_hours: int = 6):
    current = start
    while current <= end:
        yield current
        current += timedelta(hours=step_hours)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    missing_cycles = []
    found_cycles = []
    total_expected = 0

    for cycle in generate_cycles(START_CYCLE, END_CYCLE, step_hours=6):
        total_expected += 1
        fpath = expected_file_path(BASE_DIR, cycle)

        if fpath.exists():
            found_cycles.append(cycle)
        else:
            missing_cycles.append((cycle, fpath))

    print("=" * 60)
    print(f"Base directory   : {BASE_DIR}")
    print(f"Start cycle      : {START_CYCLE:%Y-%m-%d %H:%M}")
    print(f"End cycle        : {END_CYCLE:%Y-%m-%d %H:%M}")
    print(f"Expected cycles  : {total_expected}")
    print(f"Found cycles     : {len(found_cycles)}")
    print(f"Missing cycles   : {len(missing_cycles)}")
    print("=" * 60)

    # Write missing cycles to text file
    with open(OUT_MISSING_TXT, "w") as f:
        f.write("# Missing snow diag cycles\n")
        f.write(f"# Base directory: {BASE_DIR}\n")
        f.write(f"# Start cycle: {START_CYCLE:%Y%m%d%H}\n")
        f.write(f"# End cycle:   {END_CYCLE:%Y%m%d%H}\n")
        f.write("# cycle,path\n")

        for cycle, path in missing_cycles:
            f.write(f"{cycle:%Y%m%d%H},{path}\n")

    print(f"Missing cycle list written to: {OUT_MISSING_TXT}")

    # Optional: print missing cycles to screen
    if missing_cycles:
        print("\nMissing cycles:")
        for cycle, _ in missing_cycles:
            print(cycle.strftime("%Y%m%d%H"))

if __name__ == "__main__":
    main()
