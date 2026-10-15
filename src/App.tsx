import { useEffect, useRef } from 'react'
import { type Case, useJson } from './data'
import { Cases } from './pages/Cases'
import { Overview } from './pages/Overview'
import { useRoute } from './route'

const PAGES = [
  { path: '', label: 'Overview' },
  { path: 'cases', label: 'Case explorer' },
  { path: 'results', label: 'Results' },
  { path: 'methods', label: 'Methods' },
]

export default function App() {
  const [page = '', id] = useRoute()
  const cases = useJson<Case[]>('cases.json')
  const main = useRef<HTMLElement>(null)
  const current = PAGES.find((p) => p.path === page)

  useEffect(() => {
    document.title = current && current.path ? `${current.label} | Precursor Lab` : 'Precursor Lab'
    window.scrollTo(0, 0)
  }, [current])

  let content
  if (page === '') content = <Overview cases={cases} />
  else if (page === 'cases') content = <Cases cases={cases} selectedId={id} />
  else if (current) content = <Placeholder title={current.label} />
  else content = <Placeholder title="Page not found" />

  return (
    <>
      {/* a plain #main link would change the hash route, so focus it directly */}
      <a className="skip" href="#main" onClick={(e) => { e.preventDefault(); main.current?.focus() }}>
        Skip to content
      </a>
      <p className="banner" role="note">
        <strong>Retrospective study, not a live warning.</strong> For real hazard information, follow your local volcano observatory.
      </p>
      <header className="site-header">
        <a className="brand" href="#/">Precursor Lab</a>
        <nav aria-label="Main">
          {PAGES.map((p) => (
            <a key={p.path} href={`#/${p.path}`} aria-current={p === current ? 'page' : undefined}>
              {p.label}
            </a>
          ))}
        </nav>
      </header>
      <main id="main" ref={main} tabIndex={-1}>
        {content}
      </main>
      <footer className="site-footer">
        <p>
          A research project by Kenny Garcia, CSUF. Seismic data from the USGS Hawaiian and Alaska volcano
          observatory networks via EarthScope. Thermal data from NASA FIRMS (VIIRS).
        </p>
      </footer>
    </>
  )
}

function Placeholder({ title }: { title: string }) {
  return (
    <section className="page">
      <h1>{title}</h1>
      <p className="muted">{title === 'Page not found' ? <a href="#/">Back to the overview</a> : 'This page is coming soon.'}</p>
    </section>
  )
}
