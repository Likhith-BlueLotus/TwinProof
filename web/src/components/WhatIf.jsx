import { useState } from 'react'

const FIELDS = [
  { key: 'mass_scale', label: 'Mass ×', min: 0.5, max: 2, step: 0.1 },
  { key: 'damping_scale', label: 'Damping ×', min: 0, max: 3, step: 0.1 },
  { key: 'torque_limit', label: 'Torque limit (N·m)', min: 0.1, max: 10, step: 0.1 },
]

// Operator what-if: re-verify the admitted controller under changed physics.
export default function WhatIf({ disabled, onSubmit }) {
  const [values, setValues] = useState({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const body = Object.fromEntries(Object.entries(values).filter(([, v]) => v !== '').map(([k, v]) => [k, Number(v)]))

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError(null)
    try { await onSubmit(body) } catch (err) { setError(err.message) } finally { setBusy(false) }
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
      </form>
      {error && <p className="footnote" role="alert">{error}</p>}
      {busy && <p className="footnote">Re-running G0–G5 on the changed twin takes about 30 s.</p>}
    </section>
  )
}
