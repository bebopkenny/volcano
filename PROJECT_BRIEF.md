# Project brief: Precursor Lab (working title)

You are helping Kenny Garcia (CS student, CSUF) build a research project plus a web app. Read this whole file before writing code. The repo is already set up with Vite + React. Inspect it first and adapt this plan to what is there.

---

## 1. What we are building

### The research question
**Does adding NASA satellite heat data to a seismic machine learning model make volcanic eruption forecasts better?**

"Better" means three things, measured separately:
1. **Earlier warning** before real eruptions (lead time)
2. **Fewer false alarms** during volcanic unrest that did NOT lead to an eruption
3. **Better overall skill** (ROC AUC, precision/recall)

### How
1. Build a **baseline model** trained on seismic features only.
2. Build a **second model**, identical in every way except it also gets satellite thermal features (NASA VIIRS fire radiative power).
3. **Backtest both** on past eruptions AND on periods of unrest that did not end in eruption.
4. **Compare** them honestly. A result of "thermal data did not help" is a valid finding if shown rigorously.

### What this is NOT
- It is **not** a live eruption warning system. Everything is retrospective (backtesting on past events).
- The satellite is **not** used to detect eruptions directly (it would only see them once lava is already at the surface).
- No James Webb data. JWST does not observe Earth.

### Feedback from Kenny's professor (must be built in)
> "Test your models on more than one eruption and also include periods of volcanic unrest that didn't lead to eruptions, since avoiding false alarms is an important part of forecasting."

This is a core requirement, not an extra. The evaluation must include unrest-without-eruption cases ("failed eruptions") and report false alarm rates.

### Deliverables
1. A reproducible **Python research pipeline** (data download, features, models, evaluation).
2. A **static web app** (Vite + React) that presents the results: a volcano picker map, per-case stacked timelines, a replay slider, and a results summary.
3. A **write-up** in the README: objectives, assumptions, methods, results, limitations, lessons learned.

### Timeline reality
Kenny graduates in December 2026. Keep scope tight. Get one case working end to end first, then widen. A small, honest, finished study beats a big unfinished one.

---

## 2. Architecture

**Important stack correction:** Vite + React does not have "React API routes" (that is a Next.js feature). With Vite on Vercel, any backend goes in **Vercel Functions** (files in a top level `/api` folder). We likely don't need a backend at all.

```
┌──────────────────────────────┐        ┌───────────────────────────────┐
│ Python pipeline (runs locally)│        │ Vite + React app (Vercel)      │
│                              │        │                               │
│ FDSN seismic  ─┐             │        │ reads static JSON from         │
│ NASA FIRMS     ├─► features ─► models ─► public/data/*.json            │
│ eruption dates ┘             │  eval  │                               │
└──────────────────────────────┘        └───────────────────────────────┘
```

- **All heavy work (downloading, feature extraction, ML training, evaluation) happens offline in Python.** Never in the browser, never in a serverless function.
- The pipeline **exports small precomputed JSON files** into the web app's `public/data/` folder.
- The web app is a **static site**: fast, free to host, no API keys exposed, no quota issues.
- **No database.** The JSON files are the data layer.
- **Optional, later:** a single Vercel Function (`/api/recent-hotspots`) that proxies live FIRMS data so the MAP_KEY stays secret. Only build this if time allows. It is not part of the research.

---

## 3. Suggested repo structure

Adapt to the existing repo. If the Vite app is at the root, keep it there and add `pipeline/` beside it.

```
/
├── CLAUDE.md or PROJECT_BRIEF.md  (this file)
├── README.md                       (project write-up)
├── .env.example                    (FIRMS_MAP_KEY=, EARTHDATA_USERNAME=, ...)
├── .gitignore                      (.env, data/raw/, data/interim/, __pycache__, .venv, node_modules, dist)
├── src/                            (React app)
├── public/data/                    (exported JSON consumed by the app, committed)
├── api/                            (OPTIONAL Vercel Functions, later)
└── pipeline/
    ├── requirements.txt
    ├── config/cases.yaml           (every volcano case: station, dates, labels, sources)
    ├── volcano/                    (python package)
    │   ├── seismic.py              (download + feature extraction)
    │   ├── thermal.py              (FIRMS download + feature extraction)
    │   ├── labels.py               (eruption labels per day)
    │   ├── dataset.py              (merge into one daily feature table per case)
    │   ├── models.py               (baseline, model A, model B)
    │   ├── evaluate.py             (leave-one-out evaluation + metrics)
    │   └── export.py               (write JSON for the web app)
    ├── scripts/                    (CLI entry points: fetch, build, train, export)
    ├── notebooks/                  (exploration only, never the source of truth)
    ├── tests/                      (pytest)
    └── data/                       (raw/ and interim/ are gitignored)
```

---

## 4. Data sources

### 4a. Seismic data (not NASA)
- **Source:** FDSN web services. EarthScope (formerly IRIS) for US networks (e.g. HV Hawaii, AV Alaska, NC Northern California). INGV/EIDA for Italy, GeoNet for New Zealand if needed.
- **Library:** `obspy` (`obspy.clients.fdsn.Client`).
- **Approach:** one well-placed station per volcano, vertical component. Download only the windows we need (for example 90 days before eruption onset, plus unrest periods), process in chunks, and **store features, not raw waveforms**. Raw waveform cache goes in `pipeline/data/raw/` (gitignored).
- **Features** (standard in this literature, see Dempsey et al. 2020 and Ardid et al. 2025):
  - RSAM (real-time seismic amplitude), roughly 2 to 5 Hz band
  - MF (4.5 to 8 Hz) and HF (8 to 16 Hz) band amplitudes
  - DSAR (ratio of medium to high frequency displacement amplitude)
  - computed on 10 minute windows, then aggregated to daily statistics (mean, max, rolling trends over 2, 7 and 14 days)
- **Existing code worth evaluating first:** `github.com/martanto/eruption-forecast` (MIT license, forked from the Whakaari forecaster by Dempsey, active as of July 2026). Read its README and code. Either use it as a dependency, adapt parts with attribution, or write a simpler version. Tell Kenny which you chose and why.

### 4b. Satellite thermal data (NASA)
- **Primary: NASA FIRMS API** (Fire Information for Resource Management System).
  - Needs a free **MAP_KEY** (Kenny requests it at firms.modaps.eosdis.nasa.gov). Store as `FIRMS_MAP_KEY` in `.env`. Never commit it.
  - Area endpoint: `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{DAY_RANGE}/{DATE}`
  - `DAY_RANGE` is 1 to 5 days per request. Limit is 5,000 transactions per 10 minutes (longer ranges count as more transactions). Add rate limiting, retries and caching.
  - Sources: use standard processing (`VIIRS_SNPP_SP`, `VIIRS_NOAA20_SP`) for historical data. Check `/api/data_availability/` for date coverage before assuming.
  - For long histories, the FIRMS archive download tool may be easier than looping the API. Check and choose.
  - **Do not use MODIS.** Terra and Aqua are being retired (2026 to 2027). VIIRS is the continuity sensor.
- **Fallbacks:** `earthaccess` (raw VIIRS active fire products from NASA Earthdata, needs an Earthdata login) or Google Earth Engine dataset `NASA/VIIRS/002/VNP14A1` (band `MaxFRP`). Only switch if FIRMS fails.
- **Thermal features** (per day, inside a small radius around the vent, for example 5 km, to exclude wildfires):
  - max FRP, sum FRP, detection count
  - days since last detection
  - rolling mean and trend over 7, 14 and 30 days
  - a flag for "no detection" (no detection does not mean no heat; clouds and the wildfire tuned thresholds can hide weak volcanic heat)

### 4c. Eruption labels
- **Smithsonian Global Volcanism Program (GVP)** eruption database for onset dates.
- Cross check with observatory reports (USGS HVO, AVO, CalVO; INGV for Italy).
- Record the source and exact onset date/time of every label in `cases.yaml`.

---

## 5. Case selection

All cases must fall in the **VIIRS era (2012 onward)** and have **continuous seismic data available from a public FDSN service**. Verify both before committing a case. If data is missing, drop the case and tell Kenny.

**Candidates (verify each):**

| Type | Case | Notes |
|---|---|---|
| Eruption | Mauna Loa, Hawaii, Nov 2022 | Flagship case. A Pitt study found a thermal buildup about one month before. |
| Eruption | Kīlauea, Hawaii (2018, 2020, or 2023 events) | Good seismic coverage (HV network). |
| Eruption | Shishaldin, Alaska, 2023 | Used in recent ML tremor forecasting work. |
| Eruption | Pavlof or Great Sitkin, Alaska, 2021 | AV network. |
| Eruption | Etna, Italy | Lots of eruptions; check EIDA data access. |
| Unrest, no eruption | Mount Spurr, Alaska, 2024 to 2025 | Eruption looked likely, unrest faded. AVO took it off the alert list Aug 2025. |
| Unrest, no eruption | Long Valley Caldera, California | Long running unrest, no eruption. Pick specific windows. |
| Unrest, no eruption | Campi Flegrei, Italy | Ongoing unrest. Check data access. |

**MVP:** Mauna Loa 2022 end to end first. Then at least 3 eruptions and 2 unrest-without-eruption cases. Widen only if time allows.

---

## 6. Modeling

### Labels
- Unit of analysis: **one row per volcano per day.**
- Target: **"eruption onset within the next N days"** (binary). Primary N = 2, secondary N = 7. Make N configurable.
- Exclude days after an eruption has started from training (we forecast onset, not ongoing activity).
- Unrest-without-eruption cases have all negative labels. That is the point: they test false alarms.

### Models (identical setup for A and B)
- **Baseline 0:** simple RSAM threshold (the traditional method, so we can show ML adds value).
- **Model A:** seismic features only.
- **Model B:** seismic + thermal features.
- Optional **Model C:** thermal only, for context.
- Use simple, interpretable models: scikit-learn `RandomForestClassifier` or `HistGradientBoostingClassifier`. Same hyperparameters and same random seeds for A and B. Use class weights for imbalance.

### Evaluation (this is where the project lives or dies)
- **Leave-one-volcano-out** (or leave-one-case-out) cross validation. Never randomly shuffle days across train and test. That leaks the future.
- All rolling features must look **backwards only**. Write a test for this.
- Choose the alert threshold on training folds only, then apply it to the held-out case.
- Report per case and overall:
  - ROC AUC and precision/recall AUC
  - **lead time**: days of warning before onset at the chosen threshold
  - **false alarms**: alarm days per year during unrest-without-eruption cases
  - missed eruptions
- With so few eruptions, report uncertainty (bootstrap confidence intervals) and say clearly that the sample is small.
- Feature importance for Model B: which thermal features mattered, if any.

### Known limitations to state in the write-up
- Very few eruptions, so results are suggestive, not conclusive.
- VIIRS active fire thresholds are tuned for wildfires, so weak volcanic heat may be missed.
- Clouds create gaps.
- One seismic station per volcano.
- Thermal precursors may only matter for some eruption styles (open vent vs closed system). Discuss this.

---

## 7. Data contract: JSON exported for the web app

Keep files small (aim for under a few MB total). Suggested shapes:

`public/data/cases.json`
```json
[
  {
    "id": "mauna-loa-2022",
    "volcano": "Mauna Loa",
    "region": "Hawaii, USA",
    "lat": 19.475, "lon": -155.608,
    "type": "eruption",            // or "unrest"
    "onset": "2022-11-27T23:30:00Z", // null for unrest cases; verify exact time
    "windowStart": "2022-08-29",
    "windowEnd": "2022-12-10",
    "station": "HV.XXXX",          // fill in the real station
    "sources": ["GVP", "USGS HVO"]
  }
]
```

`public/data/timeseries/{caseId}.json`
```json
{
  "dates": ["2022-08-29", "..."],
  "rsam": [ ... ],
  "frpMax": [ ... ],
  "probSeismic": [ ... ],
  "probSeismicThermal": [ ... ],
  "threshold": 0.5
}
```

`public/data/metrics.json`: per case and overall metrics for Baseline 0, A, B (and C if built), including lead times and false alarm rates.

---

## 8. Web app (Vite + React)

### Pages / views
1. **Overview / home:** one sentence research question, the headline result, a persistent banner: **"Retrospective study, not a live warning."**
2. **Case explorer** (the main view):
   - Left: small map picker (MapLibre GL JS with a free basemap that needs no key, for example OpenFreeMap; verify) plus a list of cases. List items show volcano, year, and an "eruption" or "unrest" tag.
   - Right: **stacked timelines on one shared time axis**:
     1. seismic tremor (RSAM)
     2. satellite heat (VIIRS max FRP)
     3. eruption probability: Model A (dashed, neutral) vs Model B (solid, accent), with the alert threshold line
     4. eruption onset marker (red vertical line) when applicable
   - Score chips showing Model A vs Model B for that case.
   - **Replay slider** with play/pause: scrubbing hides everything after the "you are here" date so viewers see the run-up like a forecaster would have.
3. **Results:** comparison across all cases (lead time, false alarms, AUC) for Baseline vs A vs B. Make the false alarm comparison prominent.
4. **Methods:** data sources, features, validation, limitations, credits and citations.

### UI guidance
- Charting: Observable Plot or D3 (or another light library you justify). Timelines must share the x domain and update together.
- Responsive (works on a phone), accessible (labels, contrast, keyboard for the slider), light and dark mode.
- Plain language everywhere. Avoid hype words. Avoid em dashes in UI copy and the README (Kenny's style preference).
- Use real colors with meaning: one color for seismic, one for heat, neutral vs accent for Model A vs B, red only for eruption onset.
- Never present a probability as a live forecast.

---

## 9. Deployment (Vercel)
- Vercel builds the Vite app (`npm run build`, output `dist`). Static hosting, no server needed.
- The Python pipeline is NOT deployed. Its JSON output is committed into `public/data/`.
- If the optional `/api/recent-hotspots` function is built, it reads `FIRMS_MAP_KEY` from Vercel environment variables. Use `vercel dev` locally to run functions.

---

## 10. Milestones (in order, each with a definition of done)

- **M0: Recon and plan.** Inspect the repo. Confirm Node/Vite/React versions. Post a short plan to Kenny including any changes to this brief. Commit `.gitignore`, `.env.example`, this brief.
- **M1: Pipeline skeleton.** `pipeline/` package, requirements, `cases.yaml` with Mauna Loa 2022, pytest running.
- **M2: Seismic for one case.** Download and compute daily seismic features for Mauna Loa 2022. Save to `data/interim/`. Quick plot to sanity check RSAM rising before onset.
- **M3: Thermal for one case.** FIRMS download with caching and rate limiting, daily thermal features within the vent radius. Sanity plot.
- **M4: Labels + merged dataset** for one case. Leakage test passes.
- **M5: Add cases.** At least 3 eruptions and 2 unrest-without-eruption cases. Verify data availability for each.
- **M6: Models + evaluation.** Baseline 0, A, B with leave-one-out CV and all metrics. Results table printed and saved.
- **M7: Export JSON** for the web app.
- **M8: Web app shell.** Routing, layout, banner, loads `cases.json`.
- **M9: Case explorer.** Map picker, list, stacked synced timelines, score chips.
- **M10: Replay slider.**
- **M11: Results and Methods pages.**
- **M12: Deploy to Vercel.**
- **M13: README write-up** with objectives, assumptions, methods, results (with charts), limitations, lessons learned, citations.
- **Stretch:** Model C (thermal only); optional `/api/recent-hotspots`; a short section on what revisit time and resolution a satellite would need to catch the precursors we found (orbit tools: NASA GMAT or the `poliastro` Python library).

---

## 11. Git workflow: COMMIT FREQUENTLY

- **Commit early and often.** Make a commit every time something works, even small: a new function with its test, a config change, a component that renders. Do not batch hours of work into one commit.
- Each commit is one logical change, with a clear message in conventional commit style:
  - `feat(pipeline): compute daily RSAM features from FDSN waveforms`
  - `fix(thermal): handle empty FIRMS responses`
  - `test(dataset): add leakage test for rolling features`
  - `docs: add data sources section to README`
- Run tests / the build before committing when relevant. Do not commit broken code to main.
- **Never commit** `.env`, API keys, raw waveform data, or large intermediate files. Check `git status` before every commit.
- Ask Kenny before the first push to a remote, and before any force push or history rewrite.

---

## 12. Working agreement

- **Never fabricate data or results.** If a download fails or a source lacks data, stop and report it. Do not fill gaps with made up numbers. Placeholder data in the UI must be clearly marked as placeholder and removed before deploy.
- **Verify APIs by calling them.** Don't assume endpoint behavior from memory. Note what you verified.
- **Ask Kenny before big decisions:** dropping or adding cases, changing the research question, changing the label horizon, adding paid services.
- Keep the pipeline reproducible: one command per stage (`fetch`, `build`, `train`, `export`) with fixed random seeds.
- Write pytest tests for feature functions, labels, and the no-leakage rule.
- Explain your reasoning briefly when you make a modeling choice. Kenny is a CS student learning the geophysics side.
- Cite sources in the README and Methods page.

---

## 13. References

- Ardid et al. 2025, *Ergodic seismic precursors and transfer learning for short term eruption forecasting at data scarce volcanoes*, Nature Communications. https://doi.org/10.1038/s41467-025-56689-x
- Dempsey et al. 2020, *Automatic precursor recognition and real-time forecasting of sudden explosive volcanic eruptions at Whakaari, New Zealand*, Nature Communications.
- Rey-Devesa et al. 2024, *Universal machine learning approach to volcanic eruption forecasting using seismic features*, Frontiers in Earth Science. https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2024.1342468/full
- Fee et al. 2025, *A generalized deep learning model to detect and classify volcano seismicity* (VOISS-Net), Volcanica. https://doi.org/10.30909/vol/rjss1878
- Moran et al. 2011, *Failed magmatic eruptions: late-stage cessation of magma ascent*, Bulletin of Volcanology.
- Campus et al. 2022, *The transition from MODIS to VIIRS for global volcano thermal monitoring*, Sensors. https://www.mdpi.com/1424-8220/22/5/1713
- Flynn, Ramsey, Collins (Pitt), Mauna Loa 2022 thermal precursor study, Journal of Volcanology and Geothermal Research (2026).
- NASA FIRMS API: https://firms.modaps.eosdis.nasa.gov/api/area/
- eruption-forecast (MIT): https://github.com/martanto/eruption-forecast
- USGS AVO notice on Mount Spurr, Aug 2025: https://volcanoes.usgs.gov/hans-public/notice/DOI-USGS-AVO-2025-08-19T21:03:20+00:00
- Smithsonian Global Volcanism Program: https://volcano.si.edu
