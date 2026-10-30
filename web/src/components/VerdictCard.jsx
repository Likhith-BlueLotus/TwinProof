import { useEffect, useRef, useState } from 'react'
import { GATE_ORDER, gateNote, witnessFacts } from '../events.js'

// Certificate or refusal: status, gate chips (G7 runs first: the patch guard), witness numbers.
export default function VerdictCard({ record, label }) {
  const [swapping, setSwapping] = useState(false)
  const prev = useRef(null)
  useEffect(() => {
    if (prev.current && record && prev.current !== record.claim_id + record.verdict) {
      setSwapping(true)
      const id = setTimeout(() => setSwapping(false), 200)
      return () => clearTimeout(id)
    }
    prev.current = record ? record.claim_id + record.verdict : null
  }, [record])

  if (!record) return <section className="panel verdict"><p className="empty">No verdict yet.</p></section>
  const status = record.verdict.toLowerCase()
  const gates = record.gates || {}
  return (
    <section className={`panel verdict ${status}${swapping ? ' swapping' : ''}`} aria-live="polite">
      <h2 className="panel-title">{label || 'Verifier verdict'}</h2>
      <div className="status">{record.verdict}</div>
      <div className="claim-id mono">{record.claim_id} · {record.type} · {record.proposer}</div>
      {record.reason && <p className="reason">{record.reason}</p>}
      {record.gates && (
        <div className="gates">
          {GATE_ORDER.map((g) => {
            const r = gates[g]
            const cls = r ? (r.pass ? 'pass' : 'fail') : ''
            return (
              <div key={g} className={`gate ${cls}`} title={r ? (r.pass ? 'passed' : 'failed') : 'not reached'}>
                <b>{g}</b><small className="mono">{r ? (gateNote(g, r) || (r.pass ? 'pass' : 'fail')) : '—'}</small>
              </div>
            )
          })}
        </div>
      )}
      {record.gates && (
        <dl className="witness">
          {witnessFacts(record).map(([k, v]) => <div key={k}><dt>{k}</dt><dd className="mono">{v}</dd></div>)}
        </dl>
      )}
    </section>
  )
}
