import { useEffect, useRef } from 'react'

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

// Observed (dots) vs twin (line) in (angle, rate); a marker follows the video time.
export default function PhasePortrait({ phase, timeRef, theme }) {
  const canvas = useRef(null)

  useEffect(() => {
    if (!phase) return
    const c = canvas.current
    const ctx = c.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const colors = { twin: css('--twin'), obs: css('--observed'), grid: css('--border'), label: css('--fg-subtle') }
    const [obsQ, obsV] = phase.obs
    const [twQ, twV] = phase.twin
    const qMax = Math.max(...obsQ.map(Math.abs), ...twQ.map(Math.abs)) * 1.1
    const vMax = Math.max(...obsV.map(Math.abs), ...twV.map(Math.abs)) * 1.1
    let raf

    const frame = () => {
      const w = c.clientWidth, h = c.clientHeight
      if (c.width !== w * dpr) { c.width = w * dpr; c.height = h * dpr }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)
      const X = (q) => w / 2 + (q / qMax) * (w / 2 - 24)
      const Y = (v) => h / 2 - (v / vMax) * (h / 2 - 16)
      ctx.strokeStyle = colors.grid; ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.stroke()
      ctx.fillStyle = colors.label; ctx.font = '12px Geist Mono, monospace'
      const [xLabel, yLabel] = phase.labels || ['θ (rad)', 'θ̇ (rad/s)']
      ctx.fillText(xLabel, w - 58, h / 2 - 6); ctx.fillText(yLabel, w / 2 + 6, 14)

      ctx.strokeStyle = colors.twin; ctx.lineWidth = 2; ctx.globalAlpha = 0.9
      ctx.beginPath(); twQ.forEach((q, i) => (i ? ctx.lineTo(X(q), Y(twV[i])) : ctx.moveTo(X(q), Y(twV[i])))); ctx.stroke()
      ctx.globalAlpha = 0.75; ctx.fillStyle = colors.obs
      obsQ.forEach((q, i) => { ctx.beginPath(); ctx.arc(X(q), Y(obsV[i]), 2.2, 0, Math.PI * 2); ctx.fill() })
      ctx.globalAlpha = 1

      const t = timeRef.current
      let k = phase.t.findIndex((x) => x >= t)
      if (k < 0) k = phase.t.length - 1
      ctx.fillStyle = colors.obs; ctx.beginPath(); ctx.arc(X(obsQ[k]), Y(obsV[k]), 5, 0, Math.PI * 2); ctx.fill()
      ctx.strokeStyle = colors.twin; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(X(twQ[k]), Y(twV[k]), 7, 0, Math.PI * 2); ctx.stroke()
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [phase, timeRef, theme]) // theme: re-read the token colours

  return <canvas ref={canvas} role="img" aria-label="Phase portrait: observed states against the twin's prediction" />
}
