import { type Case, type Loaded, caseWhen } from '../data'

const UTC = new Intl.DateTimeFormat('en-US', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'UTC' })

export function Cases({ cases, selectedId }: { cases: Loaded<Case[]>; selectedId?: string }) {
  if (cases.error) return <section className="page"><p role="alert">Could not load cases ({cases.error}).</p></section>
  if (!cases.data) return <section className="page"><p className="muted">Loading cases…</p></section>

  const selected = cases.data.find((c) => c.id === selectedId) ?? cases.data[0]

  return (
    <div className="explorer">
      <aside className="case-list" aria-label="Cases">
        <h2>Cases</h2>
        <ul>
          {cases.data.map((c) => (
            <li key={c.id}>
              <a href={`#/cases/${c.id}`} aria-current={c === selected ? 'true' : undefined}>
                <span>{c.volcano}</span>
                <span className="muted">{caseWhen(c)}</span>
                <span className={`tag tag-${c.type}`}>{c.type}</span>
              </a>
            </li>
          ))}
        </ul>
      </aside>
      <section className="case-detail" aria-labelledby="case-title">
        <h1 id="case-title">
          {selected.volcano}, {caseWhen(selected)}
        </h1>
        <dl className="facts">
          <dt>Type</dt>
          <dd>{selected.type === 'eruption' ? 'Eruption' : 'Unrest that did not end in an eruption'}</dd>
          <dt>Region</dt>
          <dd>{selected.region}</dd>
          <dt>Onset</dt>
          <dd>{selected.onset ? `${UTC.format(new Date(selected.onset))} UTC` : 'None'}</dd>
          <dt>Window</dt>
          <dd>{selected.windowStart} to {selected.windowEnd}</dd>
          <dt>Seismic station</dt>
          <dd>{selected.station}</dd>
        </dl>
        <p className="muted">Timelines for this case are coming soon.</p>
        <h2>Sources</h2>
        <ul className="sources">
          {selected.sources.map((s) => <li key={s}><Source text={s} /></li>)}
        </ul>
      </section>
    </div>
  )
}

/** Sources are stored as "Label: https://..."; labels repeat, so the URL is shown too. */
function Source({ text }: { text: string }) {
  const m = text.match(/^(.*?):\s*(https?:\/\/\S+)$/)
  if (!m) return <>{text}</>
  return <>{m[1]}: <a href={m[2]} target="_blank" rel="noreferrer">{m[2].replace(/^https?:\/\//, '')}</a></>
}
