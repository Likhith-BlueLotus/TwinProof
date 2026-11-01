import { useEffect, useState } from 'react'
import { getBenchmark } from '../api.js'
import BetaReplication from './BetaReplication.jsx'

const pct = (v) => `${(100 * v).toFixed(1)}%`
const unit = (u) => u.replace(/^.*\//, '').replace(/_s1$/, '').replace('take', 'take ')
const ARM_NOTE = { A0: 'default physics', A1: 'agent, unverified', A2: 'TwinProof' }

// Three-arm benchmark from bench/results.json: headline tiles, held-out table, S2 replication.
export default function BenchmarkView() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => { getBenchmark().then(setData).catch((e) => setError(e.message)) }, [])

  if (error) return <p className="empty" role="alert">{error}</p>
  if (!data) return <p className="empty">Loading benchmark…</p>
  const { scenarios, seeded_bad: sb, h1 } = data.results
  const real = Object.entries(scenarios).filter(([n]) => !n.includes('synthetic'))
  const realCerts = real.reduce((n, [, sc]) => n + (sc.arms.A2.control ? 1 : 0), 0)
  const a0 = h1.A0

  return (
    <main className="bench">
      <section className="tiles">
        <Tile label="Verifier self-test" value={`${sb.caught}/${sb.n}`} good={sb.caught === sb.n && !sb.false_rejects.length}
          note={`seeded bugs caught · ${sb.false_rejects.length} false rejects`} />
        <Tile label="Held-out prediction" value={`${a0.a2_better}/${a0.pairs}`}
          note={`units where TwinProof beats default physics · Wilcoxon p = ${a0.p_one_sided.toFixed(3)}`} />
        <Tile label="Controllers certified on real footage" value={String(realCerts)} refused={realCerts === 0}
          note={realCerts === 0 ? 'every real twin refused at G1, with the failing number' : 'admitted with a certificate'} />
      </section>

      <section className="panel">
        <h2 className="panel-title">Held-out NRMSE by arm<span className="legend"><span>lower is better · best per scenario in bold</span></span></h2>
        <div className="table-wrap">
          <table className="arms">
            <thead><tr><th>Scenario</th><th>Arm</th><th>Twin</th><th>Held-out NRMSE</th><th>Control (N=256)</th></tr></thead>
            <tbody>
              {Object.entries(scenarios).map(([name, sc]) => {
                const best = Math.min(...Object.values(sc.arms).map((a) => mean(a.heldout_nrmse)))
                return Object.entries(sc.arms).map(([arm, a], i) => (
                  <tr key={name + arm} className={`${arm === 'A2' ? 'ours' : ''}${i === 0 ? ' first' : ''}`}>
                    {i === 0 && <th scope="rowgroup" rowSpan={Object.keys(sc.arms).length}>{name}</th>}
                    <td><b>{arm}</b> <span className="sub">{ARM_NOTE[arm]}</span></td>
                    <td className="sub">{a.structure}</td>
                    <td className={`mono${mean(a.heldout_nrmse) === best ? ' best' : ''}`}>
                      {Object.entries(a.heldout_nrmse).map(([u, v]) => <span key={u} className="cell-unit">{unit(u)} {pct(v)}</span>)}
                    </td>
                    <td>{a.control ? <span className="mono">{pct(a.control.success)} held</span>
                      : a.refusal ? <span className="refusal" title={a.refusal}>refused · {a.refusal.split(':')[0]}</span>
                        : <span className="sub">{a.control_note || '—'}</span>}</td>
                  </tr>
                ))
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <h2 className="panel-title">S2 replication across rigs</h2>
        <BetaReplication rigs={data.s2_beta} />
      </section>
    </main>
  )
}

function mean(o) {
  const v = Object.values(o)
  return v.reduce((a, b) => a + b, 0) / v.length
}

function Tile({ label, value, note, good, refused }) {
  return (
    <div className={`panel tile${good ? ' good' : ''}${refused ? ' refused' : ''}`}>
      <div className="panel-title">{label}</div>
      <div className="tile-value mono">{value}</div>
      <div className="sub">{note}</div>
    </div>
  )
}
