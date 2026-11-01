// The can's inertia ratio beta = I / (m r^2), identified independently on two S2 rigs: dot + 95% interval on one axis.
const LO = 0.4, HI = 1.1, W = 520, ROW = 44, PAD_L = 132, PAD_R = 16, TOP = 22
const REFS = [[0.5, 'solid cylinder'], [1.0, 'thin shell']]

export default function BetaReplication({ rigs }) {
  const entries = Object.entries(rigs || {})
  if (!entries.length) return null
  const h = TOP + entries.length * ROW + 24
  const x = (v) => PAD_L + ((Math.min(Math.max(v, LO), HI) - LO) / (HI - LO)) * (W - PAD_L - PAD_R)
  return (
    <figure className="beta">
      <svg viewBox={`0 0 ${W} ${h}`} role="img" aria-label="Inertia ratio identified on each S2 rig, with 95% intervals">
        {REFS.map(([v, label]) => (
          <g key={v} className="ref">
            <line x1={x(v)} x2={x(v)} y1={TOP - 6} y2={h - 22} />
            <text x={x(v)} y={TOP - 10} textAnchor="middle">{label} · {v.toFixed(1)}</text>
          </g>
        ))}
        {entries.map(([name, r], i) => {
          const y = TOP + ROW * i + ROW / 2
          const half = 1.96 * r.beta * r.beta_rel_std
          const tip = `${name}: β = ${r.beta.toFixed(3)} ± ${(100 * r.beta_rel_std).toFixed(1)}% (1σ)`
          return (
            <g key={name}>
              <text className="row-label" x={PAD_L - 12} y={y + 4} textAnchor="end">{name}</text>
              <line className="ci" x1={x(r.beta - half)} x2={x(r.beta + half)} y1={y} y2={y} />
              <circle className="est" cx={x(r.beta)} cy={y} r={5}><title>{tip}</title></circle>
              <circle className="hit" cx={x(r.beta)} cy={y} r={14}><title>{tip}</title></circle>
              <text className="val mono" x={x(r.beta + half) + 8} y={y + 4}>{r.beta.toFixed(2)}</text>
            </g>
          )
        })}
        {[0.4, 0.6, 0.8, 1.0].map((v) => <text key={v} className="tick" x={x(v)} y={h - 6} textAnchor="middle">{v.toFixed(1)}</text>)}
      </svg>
      <figcaption className="footnote">β = I / (m r²) of the same can, fitted separately on each rig; whiskers are 95% intervals from the Laplace posterior.</figcaption>
    </figure>
  )
}
