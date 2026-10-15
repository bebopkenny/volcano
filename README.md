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
| More cases (eruptions and unrest without eruption) | 10 eruptions and 2 unrest cases downloaded and built |
| Models and evaluation | Runs 1 and 2 done on all 12 cases; neither beats the RSAM baseline |
| Web app | Shell done: banner, routing, overview, case list; charts next |

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
.venv/bin/python -m scripts.evaluate  # train and score the models, leave one case out
.venv/bin/python -m scripts.export    # write public/data/*.json for the web app
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

Result on the first 6 cases: worse than Run 1. At a 2 day horizon ROC AUC was 0.09 (A) and 0.16 (B), well below chance, and both models alarmed on nearly every day. At 7 days: 0.46 (A) and 0.41 (B) against 0.66 for Baseline 0. The cause is in the data, not the code: in the 2 days before onset RSAM was about 7 times normal at Shishaldin but below normal at Kīlauea 2020 and Great Sitkin. A model trained without Shishaldin learns that quiet days come before eruptions and ranks Shishaldin backwards. Baseline 0 avoids this because "high RSAM means danger" is fixed, not learned from three examples.

**Runs 1 and 2 on 12 cases (10 eruptions), 2026-10-14.** Six eruptions were added (five at Kīlauea, plus Pavlof 2021) and both runs were repeated unchanged. Before running, Kīlauea 2021 was moved from station UWE to SDH: in that period UWE's 2 to 5 Hz amplitude was 0.02 to 26 times that of SDH and RIMD, which agreed with each other, and the UWE sensor epoch ends 2021-10-27. UWE matched SDH within a factor of about 2.5 in every other Kīlauea window.

| Run, horizon | ROC AUC: Baseline 0 / A / B | Eruptions caught: 0 / A / B | Unrest false alarms per year: 0 / A / B | B minus A ROC AUC, 95% CI |
|---|---|---|---|---|
| Run 1 (rf), 2 days | 0.60 / 0.41 / 0.46 | 1 / 4 / 2 of 10 | 3 / 177 / 182 | -0.02 to 0.14 |
| Run 1 (rf), 7 days | 0.61 / 0.41 / 0.43 | 4 / 9 / 7 of 10 | 5 / 330 / 351 | -0.03 to 0.06 |
| Run 2 (logreg), 2 days | 0.60 / 0.36 / 0.34 | 1 / 2 / 4 of 10 | 3 / 3 / 196 | -0.06 to 0.01 |
| Run 2 (logreg), 7 days | 0.61 / 0.52 / 0.50 | 4 / 4 / 4 of 10 | 5 / 18 / 18 | -0.05 to 0.01 |

Neither model beats Baseline 0, and Model B is never reliably better than Model A: every interval for the B minus A difference includes zero. Even Baseline 0 is only weakly better than chance (2 day ROC AUC 95% CI 0.41 to 0.77).

A likely reason is timing. Most Kīlauea eruptions here were preceded by hours, not days, of rising seismicity (HVO reports about an hour before September 2023), and Great Sitkin's RSAM rose for about 5 hours before its explosion. A forecast made once a day at midnight UTC cannot see those signals. Shishaldin and Pavlof, with RSAM 2 to 11 times normal in the 2 days before onset, are the exceptions.

**Leave-one-volcano-out (extra check, not a planned run).** Six of the ten eruptions are at Kīlauea, so leave-one-case-out lets a model learn Kīlauea from other Kīlauea cases. Holding out every case of a volcano together gives the same picture: Model A and B ROC AUC between 0.33 and 0.51, and B minus A intervals that include zero in all four run and horizon combinations. Run with `python -m scripts.evaluate --by-volcano`.

## Cases

| Case | Type | Onset (UTC) | Station | Source |
|---|---|---|---|---|
| Mauna Loa, Hawaii, 2022 | Eruption | 2022-11-28 09:30 | HV.MOKD | USGS HVO |
| Kīlauea, Hawaii, 2020 | Eruption | 2020-12-21 07:30 | HV.UWE | USGS HVO |
| Shishaldin, Alaska, 2023 | Eruption | 2023-07-12, approximate | AV.SSLS | AVO |
| Great Sitkin, Alaska, 2021 | Eruption | 2021-05-26 05:04 | AV.GSTD | AVO |
| Mount Spurr, Alaska, Jan to Apr 2025 | Unrest, no eruption | none | AV.SPCP | AVO |
| Kīlauea, Hawaii, Nov 2023 to Feb 2024 (dike intrusions) | Unrest, no eruption | none | HV.UWE | USGS HVO |
| Kīlauea, Hawaii, 2021 | Eruption | 2021-09-30 01:20 | HV.SDH | USGS |
| Kīlauea, Hawaii, Jun 2023 | Eruption | 2023-06-07 14:44 | HV.UWE | USGS HVO, NPS |
| Kīlauea, Hawaii, Sep 2023 | Eruption | 2023-09-11 01:15 | HV.UWE | USGS HVO |
| Kīlauea, Hawaii, Jun 2024 (southwest rift) | Eruption | 2024-06-03 10:30 | HV.UWE | USGS |
| Kīlauea, Hawaii, Dec 2024 | Eruption | 2024-12-23 12:30 | HV.UWE | USGS HVO |
| Pavlof, Alaska, 2021 | Eruption | 2021-08-05, before 17:55 | AV.PN7A | AVO |

Exact source links for every onset are in `pipeline/config/cases.yaml`.

## References

- Dempsey et al. 2020, *Automatic precursor recognition and real-time forecasting of sudden explosive volcanic eruptions at Whakaari, New Zealand*, Nature Communications. https://doi.org/10.1038/s41467-020-17375-2
- Ardid et al. 2025, *Ergodic seismic precursors and transfer learning for short term eruption forecasting at data scarce volcanoes*, Nature Communications. https://doi.org/10.1038/s41467-025-56689-x
- USGS Hawaiian Volcano Observatory, *November 27 to December 10, 2022 eruption of Mauna Loa*. https://www.usgs.gov/volcanoes/mauna-loa/science/november-27-december-10-2022-eruption-mauna-loa
- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov
- Smithsonian Global Volcanism Program: https://volcano.si.edu
- Seismic data: USGS Hawaiian Volcano Observatory network (HV), served by EarthScope.
