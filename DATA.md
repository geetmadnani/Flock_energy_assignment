# DATA.md — Raw-data coverage, quality findings, and cleaning decisions

## Dataset

REFIT Electrical Load Measurements (University of Strathclyde). 20 UK households (House 14 is
skipped in the official numbering), each with 10 power channels: 1 whole-house **Aggregate**
clamp + 9 **Individual Appliance Monitors (IAMs)**, labelled `Appliance1`–`Appliance9`. Units are
**Watts (active power)**. Sensors were polled every 6–8 seconds but only logged on a change of
value, so native data is irregularly spaced. Two distributions are provided:

- **RAW**: as collected, no correction. Two files per house: `_Part1` (unavailable sensors
  recorded as `0`) and `_Part2` (unavailable sensors recorded as `NaN`). These are **not**
  duplicate encodings of the same period — in House 1 they are sequential collection phases
  (`Part1`: 2013-10-09 → 2014-10-01, `Part2`: 2014-06-27 → 2015-07-10, with a short overlap).
- **CLEAN** (official): one file per house. Per the dataset's own `CLEAN_READ_ME`, this
  corrects UK daylight-saving time, re-matches IAM columns to the right appliance when a unit
  was reset/moved, forward-fills `NaN`s, removes IAM spikes > 4000 W, and adds an `Issues` flag
  (`1` when `sum(IAMs) > Aggregate`, i.e. an internal inconsistency that "should be discarded").

Appliance identities differ per house (the 9 IAM slots were assigned per-household); the
washing-machine channel mapping used throughout this project (`src/data_utils.py:WASHING_MACHINE_CHANNEL`)
was taken directly from the official appliance list in `CLEAN_READ_ME_081116.txt`.

## Part 1 — Raw House 1 inspection and cleaning

**File used**: `RAW_House1_Part2.csv` (chosen over `Part1` because it marks unavailable sensors
as `NaN` rather than `0`, which is the honest representation needed to inspect real missingness).
This is a deliberate scoping choice, not the full raw history — `Part1` extends back to
2013-10-09 and could be concatenated for full coverage, but the assignment calls for inspecting
"its readings" (one house, essential handling skills), so a single, internally-consistent file
was used.

- **Date coverage**: 2014-06-27 21:57:50 → 2015-07-10 11:56:32 (377 days)
- **Rows**: 4,140,886
- **Sampling**: nominal 6–8 s (97.9% of successive gaps fall in {6,7,8}s); median gap 7 s

### Findings (from `src/inspect_house1.py`)

| Check | Result |
|---|---|
| Timestamps already sorted / out-of-order rows | Yes sorted; 0 out-of-order rows |
| Exact duplicate timestamps | 0 rows (defensive dedupe kept in the pipeline anyway, since this can't be assumed for every house/extract) |
| Gaps > 60 s | 3,603 |
| Gaps > 10 min | 59 |
| Gaps > 1 hr | 19 (largest ≈ 747,252 s ≈ 8.65 days) |
| Missing values | `Aggregate` 4.6% NaN; appliance channels 5.6%–46.2% NaN (sensors added/went offline at different times) |
| Appliance (IAM) max readings | up to 98,301 W — physically impossible for a single UK 13A socket (~3,000 W max) |
| Aggregate max reading | 29,159 W — high but plausible for a 4-bed detached house (shower + cooker + kettle simultaneously) |
| Rows where sum(appliances) > aggregate (+50 W tolerance) | 115,217 rows (2.78%) |

### Cleaning pipeline (`src/clean_house1.py`)

1. **Sort** by timestamp.
2. **Deduplicate**: group by exact timestamp and average numeric columns (no-op on this file — 0 duplicates — but included so the pipeline is safe to reuse on other houses/extracts).
3. **Null obvious impossible values**, using documented, non-arbitrary thresholds:
   - `Aggregate` outside [0, 20,000] W → `NaN` (74 rows, 0.002%). 20 kW was chosen empirically — it is above every physically-plausible simultaneous UK household load, while 29 kW peak readings elsewhere look like real coincident high-power events and are kept.
   - Any `ApplianceN` outside [0, 4,000] W → `NaN` (12–371 rows per channel, <0.01% each). This reuses the **same 4,000 W threshold the official REFIT cleaning pipeline uses** for IAM spike removal — a defensible, precedented choice rather than a guess.
   - Rows where `sum(appliances) > aggregate` are **flagged and counted, not corrected** — this mirrors the official `Issues` column's logic, but we do not attempt the official pipeline's appliance-channel re-matching, which is out of scope for this task.
4. **Resample to 1-minute bins** (mean of all readings falling in each minute).
5. **Fill short gaps, per channel**: a run of missing 1-minute bins up to **3 minutes** long is linearly time-interpolated, **computed independently for each column** rather than only where an entire row is missing. This matters because REFIT's dominant data-quality issue is exactly that: IAMs drop out independently of the aggregate meter and of each other (one appliance's plug monitor can go dark for hours while every other channel keeps reporting fine) — a row-wise missingness check would hide this entirely, since a row with 9 good columns and 1 `NaN` column still looks "fine" if you only check whether the *whole* row is empty.
6. **Flag longer outages per channel, do not fill them**: any remaining run of missing bins, computed per column, is left as `NaN` and logged to `data/processed/house1_outages.csv` (one row per channel-outage). Across all 10 channels there are **3,437 such outage runs totalling 770,272 channel-minutes**; per-channel missingness after gap-filling ranges from **9.76% (`Aggregate`) to 49.56% (`Appliance9`)** — see `data/processed/house1_per_channel_missingness.csv`. Most channels share the same handful of multi-day network/hardware outages (e.g. 2015-05-10 23:40 → 2015-05-19 15:09, ~8.65 days, present in 8 of 10 columns simultaneously), but `Appliance9` additionally has its own **~10-day outage invisible to the others** (2014-11-10 13:19 → 2014-11-20 15:26) — concrete evidence of channels failing independently, not just together.

### Outputs

- `data/processed/house1_clean_1min.parquet` — cleaned, 1-minute Aggregate + 9 appliance channels (with NaN gaps preserved for genuine outages)
- `data/processed/house1_outages.csv` — per-channel outage log (column, start, end, duration)
- `data/processed/house1_per_channel_missingness.csv` — % missing per channel after gap-filling
- `data/processed/house1_cleaning_summary.json` — all counts above, machine-readable
- `figures/house1_overview_before_after.png` — full-period daily-mean, before vs after
- `figures/house1_day_before_after.png` — one representative day (2014-12-12, 3 wash cycles visible) at 1-minute resolution, before vs after; the fridge trace shows a clear impossible ~7,600 W spike removed by cleaning

### Remaining concerns / not addressed

- We do not correct daylight-saving time (the official CLEAN pipeline does; raw data was used as-is for this section, as instructed).
- We do not re-match appliance channels after a device reset/move (REFIT notes this can happen; no evidence of it was specifically checked for House 1 beyond the sum-vs-aggregate flag).
- The 20,000 W aggregate and 4,000 W appliance thresholds are judgment calls grounded in UK domestic wiring limits and the official pipeline's own precedent, not hard physical certainties — a few genuine extreme-but-real readings could in principle be excluded, and a few real sensor faults under the threshold could pass through.

## Part 2 & 3 — Cleaned data used for washing-machine EDA and modelling

**Houses used**: 1, 2, 5, 6, 7, 10, 19, 20 — selected because each has a single, clearly-labelled
"Washing Machine" IAM (as opposed to a combined Washer-Dryer unit, which has a different
duty-cycle/power signature and would confound a washing-machine-only analysis).

**Files**: official `CLEAN_House{N}.csv` (single file per house, already DST-corrected,
NaN-forward-filled, IAM-spike-cleaned, with an `Issues` column). We additionally **drop rows
flagged `Issues == 1`** (set `Aggregate`/appliance values to `NaN` for that row), per the
dataset documentation's own recommendation that such rows "should be discarded." We then apply
the same 1-minute resampling and ≤3-minute short-gap interpolation described above, for
methodological consistency with Part 1.

| House | WM channel | Coverage start | Coverage end | Days | Available days | Missing % | Off-time % | Cycles detected | Cycles/week |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 2013-10-09 | 2015-07-10 | 639.0 | 557.9 | 12.68% | 98.29% | 411 | 5.16 |
| 2 | 2 | 2013-09-17 | 2015-05-28 | 617.4 | 466.3 | 24.47% | 96.24% | 339 | 5.09 |
| 5 | 3 | 2013-09-26 | 2015-07-06 | 648.3 | 572.4 | 11.71% | 93.36% | 949 | 11.61 |
| 6 | 2 | 2013-11-28 | 2015-06-28 | 577.4 | 491.0 | 14.97% | 98.40% | 139 | 1.98 |
| 7 | 5 | 2013-11-01 | 2015-07-08 | 613.2 | 535.2 | 12.71% | 92.70% | 859 | 11.23 |
| 10 | 5 | 2013-11-20 | 2015-06-30 | 587.0 | 524.5 | 10.65% | 92.04% | 678 | 9.05 |
| 19 | 2 | 2014-03-06 | 2015-06-20 | 470.5 | 440.4 | 6.39% | 96.36% | 270 | 4.29 |
| 20 | 4 | 2014-03-20 | 2015-06-23 | 460.3 | 432.8 | 5.97% | 97.62% | 198 | 3.20 |

**Cycles/week is normalised by available (non-missing) days, not calendar days** — a house with
more missing washing-machine data would otherwise look artificially under-used relative to a
house with better sensor coverage purely as a measurement artefact, not a real usage difference
(e.g. House 2, the highest-missing house at 24.47%, moves from 3.84/week under a naive
calendar-day normalisation to 5.09/week once missing time is excluded from the denominator).

(Full table: `data/processed/wm_availability.csv`; per-cycle table: `data/processed/wm_cycles.csv`.)

**Units/timezone caveat**: the `Time` column is DST-corrected local UK clock time (per the clean
readme); the `Unix` column is a true UTC epoch. All analysis here uses the `Time` column.

**Reproducing locally**: raw/clean CSVs are not committed to this repo (multi-GB). See
`README.md` for the download + extraction commands.
