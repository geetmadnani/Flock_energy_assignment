# Recommendation.md — Utility-facing summary

## One-page summary: what this model is good for, and what it isn't

**What was built**: a washing-machine load-disaggregation model estimating appliance power from
whole-house aggregate smart-meter readings at 1-minute resolution, trained and evaluated on 8
REFIT households.

**Suitable uses today**:
- **Appliance-level usage analytics on homes with a short in-home calibration window.**
  Experiment B (time-split, same households) shows the model meaningfully beats a flat baseline
  (F1 0.44 vs 0.13) when it has previously seen that specific appliance — i.e. a model
  fine-tuned or trained per-home (or per a small cluster of similar homes/appliances) after a
  brief period of ground-truth sub-metering is a realistic near-term product.
- **Event-level laundry-usage insight from directly metered appliance data** (Part 2's EDA, not
  the model): cycle counts, timing, duration and peak power are reliably extractable wherever a
  washing-machine IAM already exists, with no disaggregation needed.

**Not yet suitable**:
- **Zero-shot deployment of one model across arbitrary new households.** Experiment A (fully
  held-out households) shows the model fails to beat an "always off" baseline (F1 0.000) — it
  does not transfer a learned appliance signature across different washing-machine makes/models.
  This should not be sold or deployed as a universal, instantly-generalising disaggregation
  product without per-home calibration data or a materially different modelling approach
  (e.g. appliance-signature libraries, few-shot calibration, or Seq2Point-style architectures
  trained on far larger and more diverse appliance libraries than the 4 training houses used here).

## Section 4 answers

### Practical energy-saving / customer-insight opportunity

The clearest, most defensible opportunity from this analysis is **not** the zero-shot
disaggregation model (see limitation above) but the **directly-metered cycle analysis from
Part 2**: for any household with washing-machine sub-metering (or, after a short calibration
period, model-assisted estimation per the point above), flag cycles with unusually high peak
power or long duration relative to that household's own history. A rising peak-power or
duration trend over time in a single home's own cycles is often an early signal of appliance
inefficiency (e.g. heating-element scale buildup, a failing bearing), and gives the utility a
concrete, individually-targeted nudge: "your washing machine is using X% more energy per cycle
than it did 6 months ago — consider a lower-temperature program or a service check," alongside a
simple off-peak timing nudge (shift the 7–10am/18–19h peaks seen in Part 2 to off-peak hours).

**How to measure whether it worked**: run a randomised controlled rollout — a treatment group
receiving the nudge/alert vs. a matched control group receiving nothing — and track (a) whether
treated households' cycle timing shifts away from peak hours (measured the same way as the
Part 2 hour-of-day histogram, before vs after), and (b) whether treated households' median
cycle energy (Wh/cycle, from Part 2's cycle table) decreases relative to control over the
following 2–3 months. Both are directly measurable from the same metering pipeline built here,
with no new instrumentation needed.

### Limitations

- **Overlapping appliance signatures**: other simultaneously-running high-power loads (kettle,
  shower, electric heater) look similar to a washing machine's heating phase in the aggregate
  signal alone; this is a known general NILM confound and is consistent with why long rolling
  statistics (not fine-grained shape) dominated the model's feature importance.
- **Missing data**: every house analysed here has 6–25% missing 1-minute bins (sensor/network
  outages); any production pipeline needs the same explicit short-gap-fill / long-gap-flag
  logic built in `src/data_utils.py`, since silently interpolating multi-day outages would
  fabricate cycles that never happened.
- **Household differences**: Part 2 and Part 3 both show this is the dominant source of error —
  different washing-machine makes/models have materially different peak power (350 W–2,400 W
  observed), duration and energy-per-cycle profiles, and a single trained model does not
  transfer between them (Experiment A).
- **Applying UK-home results to India**: REFIT households are UK detached/semi-detached homes
  with predominantly resistive-heating washing machines on a single-phase ~230 V supply and
  fairly uniform appliance ownership patterns. Indian households commonly have different
  appliance mixes (higher prevalence of air conditioners/coolers and inverter-based loads,
  different washing-machine heating behaviour — many don't heat water electrically at all,
  drawing much lower peak power), different voltage/frequency nominal values, and often more
  voltage instability — all of which mean the appliance "signatures" learned here would not
  transfer directly. Any India deployment would need its own labelled calibration data per
  appliance class, not a transplant of this model's weights.

### If only 15- or 30-minute smart-meter data were available

Almost everything in this analysis depends on 1-minute resolution capturing the washing
machine's actual power *shape* — the sharp step up to ~2,000 W, the multi-stage heat/spin
pattern, and cycle boundaries. At 15- or 30-minute resolution:
- **Cycle detection becomes far less reliable**: a 30–150-minute cycle might span only 1–5
  reading intervals, each reading itself an average (or a single instantaneous sample,
  depending on the meter) over a period mixing "machine on" and "machine off" time, destroying
  the sharp on/off transitions the 10 W threshold rule relies on.
- **Peak power becomes unobservable**: a 15/30-minute average will heavily smear a short ~2,000
  W heating phase into a much smaller apparent value, making peak-power-based insights (like the
  "aging appliance" signal proposed above) effectively unusable.
- **Disaggregation accuracy would collapse further**: the model's most useful features here
  (short-lag diffs, 5-minute rolling stats) have no equivalent at 15/30-minute resolution; only
  very coarse features (daily/hourly totals, time-of-day) would remain, which our hourly-profile
  baseline already shows gets a relatively low F1 (0.12–0.13) even at 1-minute resolution.
- **What would still work**: total daily/weekly energy consumption trends, and possibly
  very coarse "did laundry likely happen this half-hour" signals in *high-power* appliances
  that dominate a 15/30-minute average, but not washing-machine-specific disaggregation.
