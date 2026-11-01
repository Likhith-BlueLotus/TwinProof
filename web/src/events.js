// Turn a run log into stream rows and verdict summaries. Pure functions, no React.

export const GATE_ORDER = ['G7', 'G0', 'G1', 'G2', 'G3', 'G4', 'G5']

const HOLD_MS = { verdict: 1400, fit: 900, default: 320 } // replay pacing: verdicts get time to be read

const fmt = (v, digits = 3) => (typeof v === 'number' ? Number(v.toPrecision(digits)).toString() : String(v))
// MJCF attributes are space-separated number strings ("1.00000094 0.005 1e-06")
const fmtAttr = (v) => String(v).split(' ').map((t) => (t !== '' && Number.isFinite(+t) ? fmt(+t) : t)).join(' ')

export function rowsFromEvents(events) {
  const rows = []
  for (const e of events) {
    const d = e.data
    if (e.kind === 'commentary') {
      const last = rows[rows.length - 1]
      if (last && last.kind === 'agent') last.text += d
      else rows.push({ kind: 'agent', label: 'Agent', text: d })
      continue
    }
    if (e.kind === 'provider') rows.push({ kind: 'provider', label: 'Model', text: `${d.slug} · ${d.provider}` })
    else if (e.kind === 'tool_call') rows.push({ kind: 'tool', label: 'Tool', text: d.name, detail: compactArgs(d.args) })
    else if (e.kind === 'fit') {
      const p = d.params
      rows.push({
        kind: 'fit', label: 'Fit',
        text: `${d.structure}: ${d.identifiable ? 'identifiable' : 'not identifiable'}`,
        detail: 'damping' in p ? `damping ${fmt(p.damping)} · frictionloss ${fmt(p.frictionloss)} · BIC ${fmt(d.bic, 5)}`
          : Object.entries(p).map(([k, v]) => `${k} ${fmt(v)}`).join(' · '),
      })
    } else if (e.kind === 'claim') {
      const s = d.spec
      rows.push({
        kind: 'claim', label: 'Claim',
        text: `${d.ref} · ${d.type}`,
        detail: s ? (s.gains ? `gains ${s.gains.map((g) => fmt(g)).join(', ')}` : `LQR Q ${s.q_diag.join('/')} R ${s.r_diag.join('/')}`) + ` · u_max ${s.u_max}`
          : Object.entries(d.patch).map(([k, v]) => `${k.split('@').pop()} ${fmtAttr(v)}`).join(' · '),
      })
    } else if (e.kind === 'verdict') {
      rows.push({ kind: 'verdict', label: d.verdict, status: d.verdict.toLowerCase(), text: d.claim_id, detail: d.reason || 'all gates pass', record: d })
    } else if (e.kind === 'final') {
      rows.push({ kind: 'final', label: 'Stop', text: d.error ? `error: ${d.error}` : d.text === 'verifier finished' ? d.text : 'agent finished' })
    }
  }
  return rows.map((r, i) => ({ ...r, id: i, hold: HOLD_MS[r.kind] ?? HOLD_MS.default }))
}

function compactArgs(args) {
  const s = JSON.stringify(args)
  return s.length > 90 ? `${s.slice(0, 87)}…` : s
}

// One short witness string per gate, for the chips.
export function gateNote(gate, w) {
  if (!w) return ''
  if (gate === 'G1' && 'nrmse' in w) return `${(w.nrmse * 100).toFixed(1)}%`
  if (gate === 'G2' && 'max_spectral_radius' in w) return `ρ ${w.max_spectral_radius.toFixed(3)}`
  if (gate === 'G3' && 'u_max_bound' in w) return `${w.u_max_bound.toFixed(2)} N·m`
  if (gate === 'G5' && 'p_fail_upper' in w) return `≤${(w.p_fail_upper * 100).toFixed(2)}%`
  return ''
}

// Headline numbers for the verdict card.
export function witnessFacts(record) {
  const g = record.gates || {}
  const facts = []
  if (g.G1?.nrmse !== undefined) facts.push(['Held-out NRMSE', `${(g.G1.nrmse * 100).toFixed(1)}%`])
  if (g.G2?.c_star) facts.push(['Certified c*', fmt(g.G2.c_star)])
  if (g.G3?.u_max_bound !== undefined) facts.push(['Torque bound', `${g.G3.u_max_bound.toFixed(2)} / ${g.G3.u_limit} N·m`])
  if (g.G5?.p_fail_upper !== undefined) facts.push(['Failure bound', `≤ ${(g.G5.p_fail_upper * 100).toFixed(2)}%`])
  if (g.G1?.ljung_box_p !== undefined && facts.length < 3) facts.push(['Whiteness p', fmt(g.G1.ljung_box_p)])
  return facts.slice(0, 3)
}
