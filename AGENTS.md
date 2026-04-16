# AGENTS.md

## Project overview
This repository supports quality control analysis for snow station observations, with an emphasis on station-level behavior that may indicate bad data, unstable sensors, or stations that should be flagged for review.

The current repository includes:
- scripts for extracting snow observation data from NetCDF files into CSV
- scripts for plotting station time series
- exploratory notebooks for anchor-point logic, step-change logic, and stuck-value logic

The next development step is to move core QC logic from exploratory analysis into reusable Python code.

## Current feature focus
Implement reusable station-QC feature engineering for three behaviors:

1. Anchor-point behavior
   Stations whose observations repeatedly concentrate at one or a few preferred values or narrow value regions.

2. Stuck-value behavior
   Stations whose observations remain at exactly the same value, or nearly the same value, for unusually long runs that may indicate a sensor or reporting issue.

3. Step-change behavior
   Stations whose time series contains repeated large discontinuous changes, especially extreme negative or positive jumps that look operationally suspicious.

These features should first support analysis and ranking, and later be usable in a more operational rolling workflow.

## Domain definitions

### Anchor-point behavior
Anchor behavior does not require exact repeated duplicate values.
A station may repeatedly return to:
- one exact value
- a narrow value band
- one or a few preferred bins or local value regions

The implementation should be able to detect both:
- exact-value anchoring
- near-value or narrow-region anchoring

Useful outputs may include:
- anchor_bin_count
- largest_anchor_fraction
- total_anchor_fraction
- number_of_anchor_regions
- optional run / persistence summaries
- interpretable per-station summaries for plotting and review

### Stuck-value behavior
Stuck-value behavior refers to observations remaining unchanged, or nearly unchanged, across unusually long consecutive runs.

This should be treated separately from anchor behavior:
- anchor behavior is about repeated concentration in value space
- stuck behavior is about persistence through time

The implementation should be able to detect:
- exact repeated runs
- near-repeated runs within a small tolerance if appropriate
- unusually long persistence relative to station cadence or record length

Useful outputs may include:
- max_run_length
- max_run_duration
- repeated_run_count
- fraction_of_obs_in_runs
- fraction_of_time_in_runs
- stuck_value_count
- optional tolerance-based run metrics

### Step-change behavior
Step-change behavior refers to repeated discontinuous jumps in a station time series that are larger than expected from normal evolution.

The initial implementation should focus on interpretable station-level screening metrics, not a final classifier.

Useful outputs may include:
- observation-to-observation deltas
- large-step counts
- fraction_large_steps
- median_negative_delta
- robust_sigma_negative_delta
- extreme_step_rate
- station_score
- optional event-level outputs for flagged jumps

## Data assumptions and edge cases
The code should handle:
- irregular cadence
- missing values
- repeated timestamps
- short station records
- stations with naturally noisy but not broken behavior
- high-resolution data where anchor-region values may all differ slightly
- exact repeated values caused by legitimate snow persistence
- physically implausible fill values or placeholder values
- duplicate station-time rows

Cadence-aware logic is preferred when practical, but keep the first implementation simple and interpretable.

## Coding expectations
- Make minimal, high-confidence edits
- Do not refactor unrelated code
- Prefer extending existing repo patterns over inventing a large framework
- Keep metric computation separate from plotting and CLI code
- Prefer reusable functions over notebook-only logic
- Add docstrings for all public functions
- Use clear pandas/numpy-based implementations unless there is a strong reason otherwise
- Keep code readable for scientific users who may review or modify it later

## Repository evolution guidance
This repo is currently notebook-heavy. The preferred direction is to gradually move reusable logic into importable Python modules while keeping notebooks available for exploration and demonstration.

A good first step is something like:
- a reusable metrics module for anchor logic
- a reusable metrics module for stuck-value logic
- a reusable metrics module for step-change logic
- lightweight tests using synthetic station examples
- optional small driver/example scripts if needed

Do not overengineer packaging unless necessary. Start small.

## Testing expectations
For any new QC logic:
- add synthetic unit tests
- include at least one normal station example
- include at least one clearly anchored station example
- include at least one anchor-region example where values are similar but not identical
- include at least one station with exact repeated stuck runs
- include at least one station with long near-constant persistence if tolerance-based runs are implemented
- include at least one station with repeated extreme jumps
- validate edge cases such as missing values and duplicate timestamps when reasonable

## Preferred outputs
Prefer functions that return:
- station-level metrics DataFrame
- optional event-level DataFrame
- optional helper outputs needed for plotting/debugging

Do not couple metric calculation tightly to plotting.

## Prompt behavior guidance for agents
When asked to implement a feature:
1. inspect the repository structure first
2. identify the most natural location for the change
3. propose a minimal implementation plan
4. then implement
5. summarize files changed, assumptions, and follow-up ideas

If there is ambiguity, choose the smallest reasonable implementation and document the choice rather than expanding scope.

## Things to avoid
- broad repo refactors
- introducing heavy dependencies without strong justification
- turning exploratory thresholds into overly rigid production rules too early
- hiding important assumptions inside plotting code or notebooks
