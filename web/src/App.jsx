import { useEffect, useMemo, useRef, useState } from 'react'
import { getRun, listRuns, videoUrl, whatIf } from './api.js'
import { rowsFromEvents } from './events.js'
import VideoStage from './components/VideoStage.jsx'
import PhasePortrait from './components/PhasePortrait.jsx'
import VerdictCard from './components/VerdictCard.jsx'
import ClaimStream from './components/ClaimStream.jsx'
import WhatIf from './components/WhatIf.jsx'
import BenchmarkView from './components/BenchmarkView.jsx'

// recorded tracks -> the rig they came from (archived rigs live in sub-folders of logs/runs)
const RIGS = { s1_s1: 'S1 synthetic', take3_s1: 'S1 real · 31 Oct', 'nov01/take3_s1': 'S1 real · finger-held pin', 'nov01_clamped/take3_s1': 'S1 real · clamped pin',
  s2_incline_take1_s2: 'S2 real · 12° ramp, 31 Oct', 'nov01_s2/s2_incline_take2_s2': 'S2 real · 8° board, 1 Nov' }
const rigName = (r) => RIGS[r.track] || r.track || r.id
const runLabel = (r) => `${rigName(r)} · ${r.admitted}/${r.claims} admitted · ${r.id.slice(-15, -7)}`

const params = new URLSearchParams(location.search) // the landing page links to ?run=<id> and ?view=bench

export default function App() {
  const [runs, setRuns] = useState([])
  const [runId, setRunId] = useState(null)
  const [run, setRun] = useState(null)
  const [cursor, setCursor] = useState(Infinity) // rows shown; Infinity = whole run
  const [whatIfRecord, setWhatIfRecord] = useState(null)
  const [error, setError] = useState(null)
  const [view, setView] = useState(params.get('view') === 'bench' ? 'bench' : 'runs')
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme || 'dark')
  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : 'light'
    document.documentElement.dataset.theme = next
    try { localStorage.setItem('theme', next) } catch { /* private mode: the choice lasts this visit */ }
    setTheme(next)
  }
  const timeRef = useRef(0)
  const currentRun = useRef(runId)
  currentRun.current = runId

  useEffect(() => {
    listRuns().then((rs) => {
      setRuns(rs)
      const asked = rs.find((r) => r.id === params.get('run'))
      setRunId((asked || rs.find((r) => r.admitted) || rs[0])?.id ?? null)
    }).catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (!runId) return
    let live = true // a slow response for an earlier pick must not overwrite this one
    setRun(null); setWhatIfRecord(null); setCursor(Infinity); setError(null)
    getRun(runId).then((r) => live && setRun(r)).catch((e) => live && setError(e.message))
    return () => { live = false }
  }, [runId])

  const rows = useMemo(() => (run ? rowsFromEvents(run.events) : []), [run])
  const shown = rows.slice(0, cursor)
  const replaying = cursor < rows.length

  useEffect(() => {
    if (!replaying) return
    const id = setTimeout(() => setCursor((c) => c + 1), rows[cursor - 1]?.hold ?? 300)
    return () => clearTimeout(id)
  }, [cursor, replaying, rows])

  const latestVerdict = [...shown].reverse().find((r) => r.kind === 'verdict')?.record ?? null
  const provider = run?.summary.provider
  const isDev = provider?.provider === 'meta-dev-fallback'

  return (
    <div className="app">
      <header className="header">
        <div>
          <a className="wordmark" href="/">Twin<span>Proof</span></a>
          <div className="tagline">Film it. Twin it. Prove it.</div>
        </div>
        <div className="spacer" />
        {provider?.slug && (
          <span className={`badge${isDev ? ' dev' : ''}`} title={isDev ? 'Muse Spark Open keys pending: Muse Spark 1.3 stands in' : 'Muse Spark Open'}>
            <span className="dot" />{provider.slug}
          </span>
        )}
        <div className="tabs" role="group" aria-label="View">
          {[['runs', 'Agent runs'], ['bench', 'Benchmark']].map(([k, label]) => (
            <button key={k} aria-pressed={view === k} onClick={() => setView(k)}>{label}</button>
          ))}
        </div>
        <button className="theme-toggle" onClick={toggleTheme} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}>
          {theme === 'light' ? 'Dark' : 'Light'}
        </button>
        {view === 'runs' && (
          <label className="run-picker">
            <span>Run</span>
            <span className="select">
              <select value={runId ?? ''} onChange={(e) => setRunId(e.target.value)} aria-label="Agent run">
                {runs.map((r) => <option key={r.id} value={r.id}>{runLabel(r)}</option>)}
              </select>
            </span>
          </label>
        )}
      </header>

      {error && <p className="empty" role="alert">{error}. Is the API running on :8000?</p>}

      {view === 'bench' && <BenchmarkView />}

      {view === 'runs' && !run && runId && !error && <p className="empty">Loading run…</p>}

      {view === 'runs' && run && (
        <main className="main">
          <div className="col">
            <section className="panel">
              <h2 className="panel-title">Reality vs twin
                <span className="legend"><span><i style={{ background: 'var(--twin)' }} />twin (identified)</span><span>video: SAM 3.1-tracked</span></span>
              </h2>
              {run.video.available
                ? <VideoStage src={videoUrl(runId)} size={run.video.size} overlay={run.overlay} timeRef={timeRef} />
                : <p className="empty">Video not kept: this take was re-filmed. The fit, verdicts and phase portrait come from its logged track.</p>}
            </section>
            {run.overlay && (
              <section className="panel phase">
                <h2 className="panel-title">Phase portrait
                  <span className="legend">
                    <span><i style={{ background: 'var(--observed)' }} />observed</span>
                    <span><i style={{ background: 'var(--twin)' }} />twin</span>
                  </span>
                </h2>
                <PhasePortrait phase={run.overlay.phase} timeRef={timeRef} theme={theme} />
              </section>
            )}
          </div>
          <div className="col">
            <VerdictCard record={whatIfRecord || latestVerdict} label={whatIfRecord ? `What-if · ${JSON.stringify(whatIfRecord.what_if)}` : null} />
            <WhatIf key={runId} disabled={!run.summary.admitted || replaying}
              onSubmit={async (body) => {
                const id = runId
                const record = await whatIf(id, body)
                if (currentRun.current === id) setWhatIfRecord(record) // drop a result for a run the user left
              }} />
            <ClaimStream rows={shown} replaying={replaying}
              onReplay={() => { setWhatIfRecord(null); setCursor(0) }} />
          </div>
        </main>
      )}

      <footer className="footer">
        <span>Certificates are model-relative engineering checks, not safety certifications.</span>
        <span>SAM 3.1 · Muse Spark · MuJoCo</span>
      </footer>
    </div>
  )
}
