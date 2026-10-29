# REFIT Data Cleaning and Washing-Machine Disaggregation

Take-home assignment: a reproducible pipeline that (1) inspects and cleans raw REFIT smart-meter
data for one household, (2) explores washing-machine usage across 8 households using the
official cleaned REFIT data, and (3) trains and evaluates a washing-machine load-disaggregation
model from whole-house aggregate power, with an honest discussion of where it does and doesn't
generalise.

- **What this is about / data source / cleaning decisions**: see [`DATA.md`](DATA.md)
- **EDA, model results, validation design, failure analysis**: see [`RESULTS.md`](RESULTS.md)
- **Utility-facing summary + Section 4 answers**: see [`Recommendation.md`](Recommendation.md)

## Repo layout

```
src/
  data_utils.py            shared loaders, 1-min resampling, gap-filling, appliance-channel map
  inspect_house1.py         Part 1: raw-data inspection (grounds the cleaning thresholds below)
  clean_house1.py           Part 1: cleaning pipeline + before/after plots for House 1
  washing_machine_eda.py    Part 2: cycle detection + cross-house EDA (8 households)
  disaggregation_model.py   Part 3: feature engineering, baselines, LightGBM model, 2 experiments
notebooks/
  walkthrough.ipynb         thin notebook that runs the three stages above end-to-end and
                            displays the resulting figures/tables inline
data/
  raw_extract/, clean_extract/   (gitignored — see "Getting the data" below)
  processed/                     small derived CSV/JSON outputs (committed)
figures/                          all PNG plots referenced by DATA.md / RESULTS.md (committed)
```

## Setup

```bash
conda create -n refit python=3.11 -y
conda activate refit
pip install -r requirements.txt
```

LightGBM on macOS needs the OpenMP runtime: `brew install libomp` (one-time, if you hit a
`libomp.dylib` load error).

## Getting the data

Raw/clean REFIT archives are too large to commit (~1.2 GB compressed, ~9 GB uncompressed for all
20 houses), so they're gitignored. Download and extract only what's needed:

```bash
# Raw (House 1 only, for Part 1)
curl -LO https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z
7z e -o./data/raw_extract REFIT_RAW_081116.7z RAW_House1_Part2.csv

# Official cleaned data (8 households used for Part 2 & 3 — see DATA.md for why these 8)
curl -LO https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z
7z e -o./data/clean_extract CLEAN_REFIT_081116.7z \
  CLEAN_House1.csv CLEAN_House2.csv CLEAN_House5.csv CLEAN_House6.csv \
  CLEAN_House7.csv CLEAN_House10.csv CLEAN_House19.csv CLEAN_House20.csv
```

(`brew install p7zip` for the `7z` command if you don't have it.)

## Running the pipeline end-to-end

```bash
python src/inspect_house1.py          # prints the raw-data findings summarised in DATA.md
python src/clean_house1.py            # Part 1: cleans House 1, writes figures/ + processed/
python src/washing_machine_eda.py     # Part 2: cycle detection + EDA across 8 households
python src/disaggregation_model.py    # Part 3: trains + evaluates the model (~25s on a laptop)
```

Each script is self-contained and re-runnable (outputs are overwritten, not appended). Total
runtime end-to-end is a couple of minutes, dominated by CSV parsing of the 8 ~350 MB clean files.
`notebooks/walkthrough.ipynb` runs the same three stages and inlines the resulting figures/tables
for a narrated read-through.

## Key assumptions and decisions (details + numbers in DATA.md / RESULTS.md)

- **Dedupe rule**: group by exact timestamp, average numeric columns (no duplicates were actually
  present in the House 1 file used, but the rule is included defensively).
- **Impossible-value thresholds**: `Aggregate` outside [0, 20,000] W, any appliance channel
  outside [0, 4,000] W (the second threshold matches the official REFIT cleaning pipeline's own
  IAM-spike threshold).
- **Gap handling**: ≤3-minute gaps are linearly interpolated; anything longer is left as `NaN`
  and logged as an outage, never silently filled.
- **Washing-machine cycle rule**: >10 W = "on", merge pauses ≤10 minutes, minimum cycle length 5
  minutes. Stated in plain language in `RESULTS.md`.
- **Train/val/test split**: by household (not by randomly-shuffled time windows), to directly
  test cross-household generalisation — the realistic deployment question for a utility. A
  second time-based split (within known households) is run as a diagnostic contrast, not as the
  primary claim.

## Trade-offs

- Gradient boosting (LightGBM) over a deep sequence model (e.g. Seq2Point): faster to train,
  iterate and explain within the assignment's time-box, at the cost of not learning convolutional
  appliance-signature shapes the way a CNN-based NILM model would. **This trade-off matters more
  than initially assumed**: published work (Murray et al. 2019, cited in `RESULTS.md`) achieves
  F1 = 0.79–0.86 doing the same unseen-REFIT-house washing-machine test we ran, using a
  windowed CNN/GRU plus a class-balanced evaluation protocol. So the gap to our F1 = 0.001 isn't
  purely a data-diversity limit — architecture (window-based shape recognition vs. our rolling
  tabular features) and evaluation protocol (balanced windows vs. our realistic 92–98%-off
  timeline) both contribute, per the literature comparison in `RESULTS.md`.
- Used 8 of 20 households (those with a dedicated, unambiguous washing-machine channel) rather
  than all 20, to keep runtime and reasoning manageable within the assignment's scope while still
  giving 4 training + 1 validation + 3 fully-held-out test households.

## Possible next steps, if continuing

- Per-home (or per-appliance-cluster) few-shot calibration instead of a single zero-shot model,
  directly motivated by the Experiment A vs B contrast in `RESULTS.md`.
- A proper Seq2Point/CNN baseline trained across the full 20 REFIT households (more appliance
  diversity) to see whether cross-house transfer improves with more varied training data, isolating
  whether the failure above is a data-diversity limit or a genuine architecture limit.
- Re-run Experiment A's evaluation under a class-balanced windowed protocol matching Murray et
  al. (2019) (see `RESULTS.md`), to isolate how much of our F1 gap vs. their published 0.79–0.86
  is architecture vs. evaluation-protocol difficulty.
- Address the 2.78% of House 1 rows where `sum(appliances) > aggregate` by re-deriving the
  official appliance-channel re-matching logic, instead of only flagging them.
