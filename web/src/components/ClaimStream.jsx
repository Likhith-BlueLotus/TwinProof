import { useEffect, useRef } from 'react'

// Agent turns, tool calls, claims and verdicts, newest at the bottom.
export default function ClaimStream({ rows, onReplay, replaying }) {
  const list = useRef(null)
  useEffect(() => { list.current?.lastElementChild?.scrollIntoView({ block: 'nearest' }) }, [rows.length])
  return (
    <section className="panel stream">
      <h2 className="panel-title">
        Claim stream
        <span className="legend">
          <button onClick={onReplay} disabled={replaying}>{replaying ? 'Replaying…' : 'Replay run'}</button>
        </span>
      </h2>
      <ol className="stream-list" ref={list}>
        {rows.map((r) => (
          <li key={r.id} className={`row ${r.kind}${r.kind === 'verdict' ? ` verdict-row ${r.status}` : ''}`}>
            <span className="kind">{r.label}</span>
            <span className="body">
              {r.text}
              {r.detail && <><br /><span className="mono">{r.detail}</span></>}
            </span>
          </li>
        ))}
      </ol>
    </section>
  )
}
