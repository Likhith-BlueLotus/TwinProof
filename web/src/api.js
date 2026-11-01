async function json(res) {
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`)
  return res.json()
}

export const listRuns = () => fetch('/api/runs').then(json)
export const getRun = (id) => fetch(`/api/runs/${id}`).then(json)
export const videoUrl = (id) => `/api/runs/${id}/video`
export const whatIf = (id, body) =>
  fetch(`/api/runs/${id}/whatif`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(json)
export const getBenchmark = () => fetch('/api/benchmark').then(json)
