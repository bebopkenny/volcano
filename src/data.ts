import { useEffect, useState } from 'react'

// Shapes written by pipeline/scripts/export.py

export type Case = {
  id: string
  volcano: string
  region: string
  lat: number
  lon: number
  type: 'eruption' | 'unrest'
  onset: string | null
  windowStart: string
  windowEnd: string
  station: string
  sources: string[]
}

type ModelKey = '0' | 'A' | 'B'

export type Metrics = {
  primary: { run: string; horizonDays: number }
  runs: Record<string, {
    overall: Record<ModelKey, {
      roc_auc: number
      eruptions_detected: string
      unrest_false_alarms_per_year: number
    }>
  }>
}

export type Loaded<T> = { data?: T; error?: string }

/** Fetch one JSON file from public/data. */
export function useJson<T>(file: string): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({})
  useEffect(() => {
    let live = true
    fetch(`${import.meta.env.BASE_URL}data/${file}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`)
        return r.json() as Promise<T>
      })
      .then(
        (data) => live && setState({ data }),
        (e: Error) => live && setState({ error: e.message }),
      )
    return () => {
      live = false
    }
  }, [file])
  return state
}

const MONTH = new Intl.DateTimeFormat('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' })

/** "Jun 2023" for an eruption, "Nov 2023 to Feb 2024" for an unrest window. */
export function caseWhen(c: Case): string {
  if (c.onset) return MONTH.format(new Date(c.onset))
  return `${MONTH.format(new Date(c.windowStart))} to ${MONTH.format(new Date(c.windowEnd))}`
}
