# RESULTS.md — Washing-machine EDA, model results, validation, failure analysis

## Part 2: Washing-machine EDA (official cleaned data, 8 households)

### Cycle-detection rule (plain language)

The washing machine is considered **"on"** in any 1-minute bin where its power exceeds **10 W**
(above typical standby / sensor-noise levels, well below its ~2,000 W heating/motor draw). A run
of "on" minutes is grouped into a single **wash cycle** together with any pause of **up to 10
minutes** of "off" time inside it — real wash cycles pause for filling, heating and between
spin stages, and we don't want to split one physical cycle into several. A candidate cycle must
last **at least 5 minutes** in total (on + internal pauses) to filter out noise blips. Code:
`src/washing_machine_eda.py:detect_cycles`.

Availability, missingness and cycle counts per house are in `DATA.md`'s table (source:
`data/processed/wm_availability.csv`). Two things stand out: **House 2** has the highest missing
rate (24.65%), and **House 5** has an unusually high cycle rate (10.4/week, ~1.5/day) — plausibly
genuine (4-occupant household) but also a reminder that a fixed 10 W/10 min/5 min rule is a
simplification; a slightly noisier channel or a second nearby appliance briefly tripping the
threshold would inflate this count too.

### Required plots

**(i) Representative 24h trace** — `figures/wm_24h_trace_house6.png`. House 6, 2013-12-28: three
clear washing-machine cycles are visible as sharp ~2,000 W spikes in the lower panel, each
lining up with a step in the aggregate trace above — confirming the channel is correctly picking
up the appliance's contribution to the whole-house signal.

**(ii) Detected cycle, zoomed** — `figures/wm_detected_cycle_house6.png`. A single detected
cycle (132 min, 2,069 W peak, 669 Wh) shows the characteristic two-phase shape: a sustained
~2,000 W heating/wash phase followed by a long, choppy, low-power (<400 W) spin/rinse tail —
exactly the kind of internal low-power period the 10-minute merge rule is designed to keep
inside one cycle rather than splitting it.

**(iii) Cross-house comparison** — `figures/wm_cross_house_comparison.png`. Duration, peak
power and energy-per-cycle all vary substantially by house: median peak power ranges from
~1,500 W (House 19) to ~2,300 W (House 1, 20), and House 5 shows an especially wide peak-power
spread (near 0 to 2,300 W) suggesting either a mixed-program machine or a mid-study appliance
change. **This cross-house variability is the central reason the disaggregation model below
struggles to generalise across households** — different machines have genuinely different power
signatures.

### Usage patterns — `figures/wm_usage_patterns.png`

Across all 8 houses combined, cycle starts peak sharply at **7–10am** and again at **18–19h**
(evening), with a clear dip overnight — consistent with people running a wash before/after work.
By day of week, **Friday–Sunday** show noticeably more cycles than Tuesday–Thursday, consistent
with weekend laundry catch-up.

## Part 3: Disaggregation model

### Setup

- **Input**: Aggregate power at 1-min resolution, plus engineered features: the raw aggregate
  value; rolling mean/std/min/max over trailing 5/15/30/60-minute windows (causal — only past +
  current data, as a real-time deployed model would have); lagged aggregate values and
  first-differences at 1/2/5/15-minute lags; sin/cos encodings of hour-of-day and day-of-week.
  (29 features total; see `data/processed/feature_importance.csv`.)
- **Target**: washing-machine power (W) at the same timestamp (regression).
- **Model**: LightGBM gradient-boosted trees (`regression_l1` objective), with early stopping on
  a validation house. Model complexity was intentionally kept modest per the assignment brief —
  the interesting question here is the validation design, not architecture.
- **Baselines**: (1) always-predict-0 ("always off"), (2) an hourly usage-profile baseline —
  mean washing-machine power for each hour-of-day, computed from the training households only.
- **Metrics**: MAE and RMSE (W) for magnitude accuracy; **SAE** (Signal Aggregate Error =
  `|Σpredicted − Σactual| / Σactual`, a standard NILM total-energy metric); and on/off-event
  **F1/precision/recall** (predicted vs actual minute > 10 W) for "did a cycle happen" accuracy.

### Experiment A — household split (the realistic deployment scenario)

Train = houses {1, 2, 5, 6}; Validation = house {7}; **Test = houses {10, 19, 20}, fully
held out**. Each house's entire timeline goes to exactly one split, so there is no leakage from
overlapping time windows — but also no exposure to those specific appliances during training.

| Model | MAE (W) | RMSE (W) | SAE | On/off F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| Always-off baseline | 16.10 | 148.44 | 1.000 | 0.000 | 0.000 | 0.000 |
| Hourly-profile baseline | 28.74 | 147.53 | 0.102 | 0.119 | 0.065 | 0.663 |
| **LightGBM** | **16.12** | **148.40** | **0.998** | **0.001** | 0.159 | 0.001 |

**The model does not beat the always-off baseline on unseen households** — it effectively
collapsed to predicting near-zero (early stopping triggered at boosting round 4). Per-house
breakdown (`data/processed/model_results.csv`) shows it is worst on House 10 (F1 0.0002) and
best, though still poor, on House 20 (F1 0.006). `figures/model_actual_vs_predicted_house10.png`
makes this visible directly: real 2,000 W+ wash cycles are completely missed by a flat
near-zero prediction line.

### Experiment B — time-based split within known households (contrast case)

To check this wasn't a broken pipeline, we ran a second experiment: each of the 8 houses' own
timeline is split chronologically 70% train / 15% validation / 15% test, then concatenated
across houses (train ends before validation starts, which ends before test starts, for every
house — no leakage, just a different, easier generalisation question: "same appliances,
later time period").

| Model | MAE (W) | RMSE (W) | SAE | On/off F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| Always-off baseline | 20.03 | 170.44 | 1.000 | 0.000 | 0.000 | 0.000 |
| Hourly-profile baseline | 34.25 | 168.86 | 0.162 | 0.132 | 0.072 | 0.790 |
| **LightGBM** | **19.63** | **166.53** | **0.932** | **0.444** | **0.661** | **0.334** |

Here the model clearly beats both baselines on F1 (0.44 vs 0.13 vs 0.00) and precision (0.66),
though recall is still only 0.33 — it misses roughly two-thirds of active washing-machine
minutes, and SAE of 0.93 means it still substantially under-estimates total washing energy, even
on households it has effectively seen before. `figures/model_actual_vs_predicted_timesplit_house1.png`
shows this concretely: the model picks up roughly when a cycle started, but predicts only
~100–400 W against an actual ~2,000–2,400 W peak.

### Why the contrast matters

Taken together, Experiments A and B isolate the real difficulty: **it's not that aggregate-only
disaggregation is hopeless — it's that a single model doesn't transfer its learned appliance
signature from one washing machine to a different make/model in a different house.** This is
exactly what the Part 2 cross-house comparison plot shows directly in the raw appliance data
(peak power ranging ~350 W–2,400 W across houses) — the model is being asked to recognise a
signature shape it has never seen.

### Feature importance (top 5, by gain — `data/processed/feature_importance.csv`)

1. `agg_roll_min_60` — the minimum aggregate value over the last 60 minutes
2. `agg_roll_mean_60`
3. `agg_roll_min_30`
4. `agg_roll_max_60`
5. `hour_sin`

The model leans heavily on **long (60-min) rolling statistics and time-of-day**, essentially
answering "has the household's baseline load been unusually low/stable, and is it a
plausible laundry hour?" rather than learning a fine-grained appliance power-shape signature —
consistent with it failing to transfer to unseen appliances.

### Failure cases

1. **Cross-household transfer (the dominant failure mode, Experiment A)**: the model never
   learns a transferable shape for "what a washing machine looks like" — it learns
   house-specific baseline/timing patterns from training homes that don't apply elsewhere.
2. **Magnitude under-estimation even within known households (Experiment B)**: the model
   correctly flags *when* a cycle is likely but predicts roughly 5–10x less power than the
   true peak, particularly during the low-power spin/pause phases that look similar to other
   background loads.
3. **Households with unstable appliance signatures** (e.g. House 5's wide peak-power spread in
   the Part 2 comparison, possibly a mid-study appliance change) are intrinsically harder for
   any single static model, seen or unseen.

### Validation design — why this avoids leakage

Both experiments split by **household or by time**, never by randomly shuffling overlapping
time windows across a household's series — which would let the model see a few minutes before
and after a given test window during training and trivially "predict" washing cycles it has
effectively already observed. Household-split (A) is the stronger test (a genuinely new home);
time-split (B) is a weaker but still valid test, useful here only as a diagnostic contrast.
