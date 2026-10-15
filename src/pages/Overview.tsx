import { type Case, type Loaded, type Metrics, useJson } from '../data'

export function Overview({ cases }: { cases: Loaded<Case[]> }) {
  const metrics = useJson<Metrics>('metrics.json')
  const m = metrics.data
  const main = m && m.runs[`${m.primary.run}_${m.primary.horizonDays}d`].overall
  const eruptions = cases.data?.filter((c) => c.type === 'eruption').length
  const unrest = cases.data?.filter((c) => c.type === 'unrest').length

  return (
    <section className="page">
      <h1>Does adding NASA satellite heat data to a seismic model make eruption forecasts better?</h1>
      <p className="lede">
        We looked back at past volcanic activity and compared two forecasting models that are identical except
        for one thing: Model B also sees satellite heat measurements from NASA VIIRS.
      </p>

      <h2>Headline result</h2>
      {metrics.error && <p role="alert">Could not load results ({metrics.error}).</p>}
      {main && m && (
        <>
          <p>
            Across {eruptions} eruptions and {unrest} periods of unrest that did not end in an eruption, satellite
            heat did not make forecasts better. Neither model beat a simple tremor threshold.
          </p>
          <dl className="stats">
            <Stat label="Tremor threshold" auc={main['0'].roc_auc} alarms={main['0'].unrest_false_alarms_per_year} />
            <Stat label="Model A: seismic" auc={main.A.roc_auc} alarms={main.A.unrest_false_alarms_per_year} />
            <Stat label="Model B: seismic + heat" auc={main.B.roc_auc} alarms={main.B.unrest_false_alarms_per_year} />
          </dl>
          <p className="muted">
            Forecast skill is ROC AUC, where 0.5 is a coin flip and 1 is perfect. False alarms count alarm days per
            year during unrest. Main run: random forest, {m.primary.horizonDays} day warning window.
          </p>
        </>
      )}

      <p className="actions">
        <a className="button" href="#/cases">Explore the cases</a>
        <a href="#/results">See all results</a>
      </p>
    </section>
  )
}

function Stat({ label, auc, alarms }: { label: string; auc: number; alarms: number }) {
  return (
    <div className="stat">
      <dt>{label}</dt>
      <dd>
        <span className="stat-value">{auc.toFixed(2)}</span> forecast skill
      </dd>
      <dd>
        <span className="stat-value">{Math.round(alarms)}</span> false alarm days per year
      </dd>
    </div>
  )
}
