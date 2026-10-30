import { useEffect, useMemo, useRef, useState } from 'react'
import { getRun, listRuns, videoUrl, whatIf } from './api.js'
import { rowsFromEvents } from './events.js'
import VideoStage from './components/VideoStage.jsx'
import PhasePortrait from './components/PhasePortrait.jsx'
import VerdictCard from './components/VerdictCard.jsx'
import ClaimStream from './components/ClaimStream.jsx'
import WhatIf from './components/WhatIf.jsx'

export default function App() {
  const [runs, setRuns] = useState([])
  const [runId, setRunId] = useState(null)
  const [run, setRun] = useState(null)
  const [cursor, setCursor] = useState(Infinity) // rows shown; Infinity = whole run
  const [whatIfRecord, setWhatIfRecord] = useState(null)
  const [error, setError] = useState(null)
  const timeRef = useRef(0)

  useEffect(() => {
    listRuns().then((rs) => {
      setRuns(rs)
      setRunId((rs.find((r) => r.admitted) || rs[0])?.id ?? null)
    }).catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (!runId) return
    setRun(null); setWhatIfRecord(null); setCursor(Infinity)
    getRun(runId).then(setRun).catch((e) => setError(e.message))
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
          <div className="wordmark">Twin<span>Proof</span></div>
          <div className="tagline">Film it. Twin it. Prove it.</div>
        </div>
        <div className="spacer" />
        {provider && (
          <span className={`badge${isDev ? ' dev' : ''}`} title={isDev ? 'Muse Spark Open keys pending: Muse Spark 1.3 stands in' : 'Muse Spark Open'}>
            <span className="dot" />{provider.slug}{isDev ? ' · dev fallback' : ` · ${provider.provider}`}
          </span>
        )}
        <label className="tagline">Run{' '}
          <select value={runId ?? ''} onChange={(e) => setRunId(e.target.value)} aria-label="Agent run">
            {runs.map((r) => <option key={r.id} value={r.id}>{r.id.replace('agent_', '')} · {r.admitted}/{r.claims} admitted</option>)}
          </select>
        </label>
      </header>

      {error && <p className="empty" role="alert">{error}. Is the API running on :8000?</p>}

      {run && (
        <main className="main">
          <div className="col">
            <section className="panel">
              <h2 className="panel-title">Reality vs twin
                <span className="legend"><span><i style={{ background: 'var(--twin)' }} />twin (identified)</span><span>video: SAM 3.1-tracked</span></span>
              </h2>
              <VideoStage src={videoUrl(runId)} size={run.video.size} overlay={run.overlay} timeRef={timeRef} />
            </section>
            {run.overlay && (
              <section className="panel phase">
                <h2 className="panel-title">Phase portrait
                  <span className="legend">
                    <span><i style={{ background: 'var(--observed)' }} />observed</span>
                    <span><i style={{ background: 'var(--twin)' }} />twin</span>
                  </span>
                </h2>
                <PhasePortrait phase={run.overlay.phase} timeRef={timeRef} />
              </section>
            )}
          </div>
          <div className="col">
            <VerdictCard record={whatIfRecord || latestVerdict} label={whatIfRecord ? `What-if · ${JSON.stringify(whatIfRecord.what_if)}` : null} />
            <WhatIf disabled={!run.summary.admitted || replaying}
              onSubmit={async (body) => setWhatIfRecord(await whatIf(runId, body))} />
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
