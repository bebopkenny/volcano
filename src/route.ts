import { useSyncExternalStore } from 'react'

// Hash routes (#/cases/kilauea-2020) work on any static host without server rewrites.
function subscribe(onChange: () => void) {
  window.addEventListener('hashchange', onChange)
  return () => window.removeEventListener('hashchange', onChange)
}

const readHash = () => window.location.hash.replace(/^#\/?/, '')

/** Path segments of the current hash route, e.g. ['cases', 'kilauea-2020']. */
export function useRoute(): string[] {
  return useSyncExternalStore(subscribe, readHash).split('/').filter(Boolean)
}
