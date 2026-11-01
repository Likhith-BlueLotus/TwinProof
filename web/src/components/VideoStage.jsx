import { useEffect, useRef, useState } from 'react'

const twinColor = () => getComputedStyle(document.documentElement).getPropertyValue('--twin').trim() || '#4fd6e8'

// Video with the ghost twin drawn on the exact frame being shown; publishes media time via timeRef.
export default function VideoStage({ src, size, overlay, timeRef }) {
  const video = useRef(null)
  const canvas = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [t, setT] = useState(0)
  const [duration, setDuration] = useState(0)

  useEffect(() => {
    const v = video.current
    const ctx = canvas.current.getContext('2d')
    let handle
    const draw = (mediaTime) => {
      timeRef.current = mediaTime
      setT(mediaTime)
      ctx.clearRect(0, 0, size[0], size[1])
      const TWIN = twinColor() // read per frame so a theme switch recolours the ghost
      if (!overlay) return
      // the twin starts at t0 (decimation edge trim): hold its first pose before that
      const k = Math.max(0, Math.round((mediaTime - overlay.t0) * overlay.fps))
      if (overlay.kind === 's2') {  // ghost can: a ring on the twin's position along the roll
        const c = overlay.center_px[k]
        if (!c) return
        ctx.strokeStyle = TWIN; ctx.lineWidth = 4; ctx.globalAlpha = 0.9
        ctx.beginPath(); ctx.arc(c[0], c[1], overlay.radius_px, 0, Math.PI * 2); ctx.stroke()
        ctx.globalAlpha = 1; ctx.fillStyle = TWIN; ctx.font = '600 18px Geist, sans-serif'
        ctx.fillText('TWIN', c[0] - 22, c[1] - overlay.radius_px - 10)
        return
      }
      const tip = overlay.tip_px[k]
      if (!tip) return
      const [px, py] = overlay.pivot_px
      ctx.lineCap = 'round'
      ctx.strokeStyle = TWIN
      ctx.globalAlpha = 0.85
      ctx.lineWidth = 9
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(tip[0], tip[1]); ctx.stroke()
      ctx.globalAlpha = 1
      ctx.fillStyle = TWIN
      ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2); ctx.fill()
      ctx.font = '600 18px Geist, sans-serif'
      ctx.fillText('TWIN', tip[0] + 14, tip[1] + 6)
    }
    const loop = (_, meta) => {
      draw(meta ? meta.mediaTime : v.currentTime)
      handle = v.requestVideoFrameCallback ? v.requestVideoFrameCallback(loop) : requestAnimationFrame(() => loop())
    }
    handle = v.requestVideoFrameCallback ? v.requestVideoFrameCallback(loop) : requestAnimationFrame(() => loop())
    const onSeek = () => draw(v.currentTime)
    v.addEventListener('seeked', onSeek)
    return () => {
      v.removeEventListener('seeked', onSeek)
      if (v.cancelVideoFrameCallback) v.cancelVideoFrameCallback(handle)
      else cancelAnimationFrame(handle)
    }
  }, [overlay, size, timeRef])

  const toggle = () => (video.current.paused ? video.current.play() : video.current.pause())

  return (
    <div>
      <div className="stage">
        <video ref={video} src={src} muted playsInline loop
          onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
          onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)} />
        <canvas ref={canvas} width={size[0]} height={size[1]} aria-hidden="true" />
      </div>
      <div className="controls">
        <button onClick={toggle} aria-label={playing ? 'Pause video' : 'Play video'}>{playing ? 'Pause' : 'Play'}</button>
        <input type="range" min={0} max={duration || 0} step={1 / 60} value={t} aria-label="Video position"
          onChange={(e) => { video.current.currentTime = Number(e.target.value) }} />
        <span className="time mono">{t.toFixed(2)} s</span>
      </div>
    </div>
  )
}
