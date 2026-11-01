import { useEffect, useRef, useState } from 'react'
import { parseCommand } from '../voice.js'

const Recognition = typeof window !== 'undefined' && (window.SpeechRecognition || window.webkitSpeechRecognition)

const FIELDS = [
  { key: 'mass_scale', label: 'Mass ×', min: 0.5, max: 2, step: 0.1 },
  { key: 'damping_scale', label: 'Damping ×', min: 0, max: 3, step: 0.1 },
  { key: 'torque_limit', label: 'Torque limit (N·m)', min: 0.1, max: 10, step: 0.1 },
]

// Operator what-if: re-verify the admitted controller under changed physics, typed or spoken.
export default function WhatIf({ disabled, onSubmit }) {
  const [values, setValues] = useState({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [heard, setHeard] = useState(null)
  const [listening, setListening] = useState(false)
  const rec = useRef(null)
  useEffect(() => () => rec.current?.abort(), []) // leaving the run or view turns the mic off
  const body = Object.fromEntries(Object.entries(values).filter(([, v]) => v !== '').map(([k, v]) => [k, Number(v)]))

  const run = async (b) => {
    setBusy(true); setError(null)
    try { await onSubmit(b) } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  const submit = (e) => { e.preventDefault(); run(body) }

  const listen = () => {
    if (listening) { rec.current?.stop(); return }
    const r = new Recognition()
    r.lang = 'en-US'; r.interimResults = false; r.maxAlternatives = 1
    r.onresult = (e) => {
      const text = e.results[0][0].transcript
      const spoken = parseCommand(text)
      setHeard({ text, ok: Object.keys(spoken).length > 0 })
      if (!Object.keys(spoken).length) return
      setValues(Object.fromEntries(Object.entries(spoken).map(([k, v]) => [k, String(v)])))
      run(spoken)
    }
    r.onerror = (e) => setError(`Microphone: ${e.error}`)
    r.onend = () => setListening(false)
    rec.current = r; setHeard(null)
    try { r.start(); setListening(true) } catch (err) { setError(`Microphone: ${err.message}`) }
  }
  return (
    <section className="panel">
      <h2 className="panel-title">What if…</h2>
      <form className="whatif" onSubmit={submit}>
        {FIELDS.map((f) => (
          <label key={f.key}>{f.label}
            <input type="number" inputMode="decimal" min={f.min} max={f.max} step={f.step} placeholder="—"
              value={values[f.key] ?? ''} onChange={(e) => setValues({ ...values, [f.key]: e.target.value })} />
          </label>
        ))}
        <button className="primary" type="submit" disabled={disabled || busy || !Object.keys(body).length}>
          {busy ? 'Verifying…' : 'Re-verify'}
        </button>
        {Recognition && (
          <button type="button" className={`mic${listening ? ' live' : ''}`} onClick={listen} disabled={disabled || busy}
            aria-pressed={listening} title='Say e.g. "torque limit 0.5" or "double the mass"'>
            {listening ? 'Listening…' : 'Speak'}
          </button>
        )}
      </form>
      {heard && <p className="footnote">Heard “{heard.text}”{heard.ok ? '' : ': no mass, damping or torque value recognised'}</p>}
      {error && <p className="footnote" role="alert">{error}</p>}
      {busy && <p className="footnote">Re-running G0–G5 on the changed twin takes about 30 s.</p>}
    </section>
  )
}
