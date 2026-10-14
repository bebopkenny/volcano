# Precursor Lab

**Does adding NASA satellite heat data to a seismic machine learning model make volcanic eruption forecasts better?**

This is a retrospective research project by Kenny Garcia (CSUF). It backtests two models on past volcanic activity and compares them:

- **Model A:** seismic features only
- **Model B:** the same model plus satellite thermal features (NASA VIIRS fire radiative power)

"Better" is measured three ways: earlier warning before real eruptions, fewer false alarms during unrest that did not lead to an eruption, and overall skill (ROC AUC, precision and recall). A result of "thermal data did not help" is a valid finding.

> **Retrospective study, not a live warning.** Nothing here forecasts future eruptions. For real hazard information, follow your local volcano observatory.

## Status

Work in progress. No results yet.

| Stage | State |
|---|---|
| Seismic download and daily features | Working for all 6 cases |
| Thermal (FIRMS) download and features | Working for all 6 cases |
| Labels and merged daily table | Working, with a no-leakage test |
| More cases (eruptions and unrest without eruption) | 4 eruptions and 2 unrest cases downloaded and built |
| Models and evaluation | Not started |
| Web app | Vite starter page only |

The full plan and milestones are in [PROJECT_BRIEF.md](PROJECT_BRIEF.md).

## How it fits together

```
Python pipeline (runs locally)                 Vite + React app (static site)
FDSN seismic  ─┐
NASA FIRMS     ├─► features ─► models ─► eval ─► public/data/*.json ─► charts
eruption dates ┘
```

All downloading, feature extraction and model training happens offline in Python. The pipeline writes small JSON files that the web app reads. There is no backend and no database.

## Running the pipeline

Requires Python 3.10 or newer. The commands below use [uv](https://docs.astral.sh/uv/); plain `python -m venv` and `pip` work too.

```bash
cd pipeline
uv venv .venv
uv pip install -p .venv/bin/python -r requirements.txt

.venv/bin/pytest                      # run the tests
.venv/bin/python -m scripts.fetch     # download seismic and FIRMS data, cache 10 minute features
.venv/bin/python -m scripts.build     # build the daily feature table and a sanity plot
```

Both scripts accept case ids (for example `mauna-loa-2022`) to run a single case. Cases are defined in `pipeline/config/cases.yaml`. Downloads and intermediate tables go to `pipeline/data/`, which is not committed.

Thermal data needs a free NASA FIRMS key. Request one at https://firms.modaps.eosdis.nasa.gov/api/map_key/ and put it in a `.env` file at the repo root:

```bash
cp .env.example .env   # then fill in FIRMS_MAP_KEY
```

## Running the web app

Requires Node 22 or newer.

```bash
npm install
npm run dev      # local dev server
npm run build    # production build into dist/
```

## Methods so far

**Seismic.** One station per volcano, vertical component, from public FDSN services (EarthScope). Ground velocity is band-passed and averaged over 10 minute windows, following Dempsey et al. 2020:

- RSAM: 2 to 5 Hz
- MF: 4.5 to 8 Hz
- HF: 8 to 16 Hz
- DSAR: ratio of MF to HF amplitude after integrating to displacement

Windows are aggregated to one row per UTC day. Because raw amplitude depends on the instrument and its distance from the vent, each value is expressed relative to that station's own typical level on earlier days, so features can be compared between volcanoes.

**Thermal.** VIIRS active fire detections (Suomi NPP and NOAA-20) from NASA FIRMS within 5 km of the vent: daily max and total fire radiative power, detection count, days since the last detection, and trailing means and trends. A day with no detection is recorded as "nothing seen", not "no heat", since cloud and wildfire-tuned thresholds can hide weak volcanic heat.

**Labels.** One row per volcano per day. The target is "eruption onset within the next N days" (N = 2 and 7). The onset day and everything after it are excluded, since the goal is to forecast the start of an eruption.

**No leakage.** Every feature on a given day uses data from that day and earlier only. A test changes the future and confirms the past does not move.

**Evaluation.** Leave-one-case-out: each case is scored by models trained on the other cases only. The alert threshold for a held-out case is the one with the best F1 on the training cases, using scores those cases got while held out themselves (an inner leave-one-case-out). Metrics: ROC AUC, PR AUC, eruptions detected, lead time, and false alarm days per year during unrest cases. 95% intervals come from resampling whole cases.

## Model runs

Every run is reported, including runs that did badly. A run's design is written here and committed before it is run.

**Run 1 (`rf`), 2026-10-14.** Random forest (300 trees, min 3 samples per leaf, balanced class weights, seed 0). Model A: all 42 station-relative seismic features. Model B: those plus all 23 thermal features. Baseline 0: today's RSAM relative to the station's normal. Result on the first 6 cases: neither model beat Baseline 0. At a 2 day horizon ROC AUC was 0.58 (Baseline 0), 0.57 (A) and 0.43 (B); false alarms during unrest were 3, 76 and 270 days per year. Model B raised an alarm on every day of the Kīlauea 2024 unrest, where the crater was still warm from the September 2023 eruption. With only 8 positive days, 66 features were too many.

**Run 2 (`logreg`), design fixed 2026-10-14, before running.** Fewer features and a simpler model, chosen because Run 1 overfit. Everything else (cases, labels, thresholds, metrics) is unchanged.
- Model A: `rsam_median_rel`, `rsam_median_rel_trend7`, `dsar_median_rel`, `dsar_median_rel_trend7`
- Model B: Model A plus `frp_max` (as log(1 + MW)) and `frp_max_trend30`
- Logistic regression with balanced class weights and default regularisation (C = 1). Missing values filled with the training median, then features standardised.

**Then:** add more eruption cases and re-run both Run 1 and Run 2 unchanged on the larger set.

## Cases

| Case | Type | Onset (UTC) | Station | Source |
|---|---|---|---|---|
| Mauna Loa, Hawaii, 2022 | Eruption | 2022-11-28 09:30 | HV.MOKD | USGS HVO |
| Kīlauea, Hawaii, 2020 | Eruption | 2020-12-21 07:30 | HV.UWE | USGS HVO |
| Shishaldin, Alaska, 2023 | Eruption | 2023-07-12, approximate | AV.SSLS | AVO |
| Great Sitkin, Alaska, 2021 | Eruption | 2021-05-26 05:04 | AV.GSTD | AVO |
| Mount Spurr, Alaska, Jan to Apr 2025 | Unrest, no eruption | none | AV.SPCP | AVO |
| Kīlauea, Hawaii, Nov 2023 to Feb 2024 (dike intrusions) | Unrest, no eruption | none | HV.UWE | USGS HVO |

Exact source links for every onset are in `pipeline/config/cases.yaml`.

## References

- Dempsey et al. 2020, *Automatic precursor recognition and real-time forecasting of sudden explosive volcanic eruptions at Whakaari, New Zealand*, Nature Communications. https://doi.org/10.1038/s41467-020-17375-2
- Ardid et al. 2025, *Ergodic seismic precursors and transfer learning for short term eruption forecasting at data scarce volcanoes*, Nature Communications. https://doi.org/10.1038/s41467-025-56689-x
- USGS Hawaiian Volcano Observatory, *November 27 to December 10, 2022 eruption of Mauna Loa*. https://www.usgs.gov/volcanoes/mauna-loa/science/november-27-december-10-2022-eruption-mauna-loa
- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov
- Smithsonian Global Volcanism Program: https://volcano.si.edu
- Seismic data: USGS Hawaiian Volcano Observatory network (HV), served by EarthScope.
