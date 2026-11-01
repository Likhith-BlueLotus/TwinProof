// TwinProof landing: Lenis smooth scroll, a tiny spring engine, clip-mask reveals, parallax, two carousels,
// the intro loader, the menu and the "film your own take" modal. Plain DOM, no framework.
import Lenis from 'lenis'
import './landing.css'

const $ = (s, root = document) => root.querySelector(s)
const $$ = (s, root = document) => [...root.querySelectorAll(s)]
const json = (el, key) => JSON.parse(el.dataset[key] || '{}')
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches
const canHover = () => matchMedia('(min-width: 769px) and (hover: hover)').matches // no hover springs on phones

// loader timing (ms)
const MIN_VISIBLE_MS = REDUCED ? 200 : 1400
const MAX_VISIBLE_MS = 2600
const EXIT_MS = REDUCED ? 0 : 850
const AUTOPLAY_MS = 3800

// ---------- adaptive rem: media queries scale down below 1920 px; this scales up above it ----------
const FONT_BASE = 16, BASE_W = 1920, COEF = 0.6666
function scaleUp() {
  const reduction = ((BASE_W - innerWidth) / BASE_W) * 100 * COEF
  const size = FONT_BASE - (FONT_BASE * reduction) / 100
  if (size > FONT_BASE) document.documentElement.style.fontSize = `${size}px`
  else document.documentElement.style.removeProperty('font-size')
}
scaleUp()
addEventListener('resize', scaleUp)

// ---------- springs: x'' = -tension (x - target) - friction x', the react-spring model in seconds ----------
const active = new Set()
const SUBSTEP = 1 / 240

function spring(el, values, apply = applyTransform) {
  const s = { el, x: { ...values }, v: {}, target: { ...values }, cfg: [170, 26], apply }
  for (const k in values) s.v[k] = 0
  apply(el, s.x)
  return s
}

function to(s, target, cfg = s.cfg) {
  for (const k in target) if (!(k in s.x)) { s.x[k] = target[k]; s.v[k] = 0 }
  Object.assign(s.target, target)
  s.cfg = cfg
  if (REDUCED) { Object.assign(s.x, s.target); s.apply(s.el, s.x); return }
  active.add(s)
}

function step(dt) {
  for (const s of active) {
    const [tension, friction] = s.cfg
    let moving = false
    for (const k in s.target) {
      let x = s.x[k], v = s.v[k]
      const target = s.target[k]
      for (let t = 0; t < dt; t += SUBSTEP) {
        v += (-tension * (x - target) - friction * v) * SUBSTEP
        x += v * SUBSTEP
      }
      if (Math.abs(x - target) < 1e-3 && Math.abs(v) < 1e-3) { x = target; v = 0 } else moving = true
      s.x[k] = x; s.v[k] = v
    }
    s.apply(s.el, s.x)
    if (!moving) active.delete(s)
  }
}

function applyTransform(el, { x = 0, y = 0, scale = 1, rotate = 0, opacity }) {
  el.style.transform = `translate3d(${x}px, ${y}px, 0) scale(${scale}) rotate(${rotate}deg)`
  if (opacity !== undefined) el.style.opacity = opacity
}

// ---------- parallax: element's section maps viewport travel (top at bottom -> 0, bottom at top -> 1) ----------
const parallax = $$('[data-parallax]').map((el) => ({ el, map: json(el, 'parallax'), root: el.closest('section') }))
function updateParallax() {
  const vh = innerHeight
  for (const { el, map, root } of parallax) {
    const r = root.getBoundingClientRect()
    const p = Math.min(1, Math.max(0, (vh - r.top) / (vh + r.height)))
    const [axis, [a, b]] = Object.entries(map)[0]
    el.style.transform = `translate${axis.toUpperCase()}(${a + (b - a) * p}%)`
  }
}

// ---------- Lenis + one rAF loop for scroll, springs and parallax ----------
const lenis = new Lenis({ smoothWheel: true })
lenis.on('scroll', () => checkPending())
addEventListener('resize', () => checkPending())
let last = null
function raf(t) {
  lenis.raf(t)
  step(Math.min(0.064, last === null ? 0 : (t - last) / 1000))
  last = t
  if (!REDUCED) updateParallax()
  requestAnimationFrame(raf)
}
requestAnimationFrame(raf)

let locks = 0 // loader, menu and modal can overlap
function lock() {
  if (locks++ === 0) { lenis.stop(); document.documentElement.classList.add('locked') }
}
function unlock() {
  if (--locks === 0) { lenis.start(); document.documentElement.classList.remove('locked') }
}

// anchors scroll smoothly through Lenis
document.addEventListener('click', (e) => {
  const a = e.target.closest('a[href^="#"]')
  if (!a || a.getAttribute('href') === '#') return
  const target = document.querySelector(a.getAttribute('href'))
  if (!target) return
  e.preventDefault()
  closeMenu()
  lenis.scrollTo(target, { offset: -8 })
})

// ---------- text reveals: words or lines slide up out of a clip box, staggered ----------
function prepareReveal(el) {
  const opt = { delay: 0, stagger: 120, duration: 950, ...json(el, 'reveal') }
  if (el.classList.contains('words') || el.classList.contains('fade-words')) {
    const words = el.textContent.trim().split(/\s+/)
    el.setAttribute('aria-label', el.textContent.trim())
    el.innerHTML = words.map((w) => `<span class="mask" aria-hidden="true"><span>${w}</span></span>`).join(' ')
  } else {
    el.innerHTML = $$(':scope > span', el).map((l) => `<span class="mask"><span>${l.innerHTML}</span></span>`).join('')
  }
  $$('.mask > span', el).forEach((s, i) => {
    s.style.transitionDuration = `${opt.duration}ms`
    s.style.transitionDelay = `${opt.delay + i * opt.stagger}ms`
  })
}

function replay(el) {
  el.classList.remove('revealed')
  void el.offsetWidth // restart the transition from the hidden state
  el.classList.add('revealed')
}

// ---------- in-view: spring from `from` to rest once, after an optional delay ----------
const REST = { opacity: 1, x: 0, y: 0, scale: 1, rotate: 0 }
const inviewSprings = new Map()
for (const el of $$('[data-inview]')) {
  const { from } = json(el, 'inview')
  inviewSprings.set(el, spring(el, from))
}
function playInview(el) {
  const { from, delay = 0, cfg } = json(el, 'inview')
  const target = Object.fromEntries(Object.keys(from).map((k) => [k, REST[k]]))
  setTimeout(() => to(inviewSprings.get(el), target, cfg), delay)
}

// Triggered by scroll position, not IntersectionObserver: a fast flick can carry an element from below the
// viewport to above it between two observer checks, and it would never reveal.
let ready = false
const pending = new Set()
function start(el) {
  if (el.hasAttribute('data-reveal')) el.classList.add('revealed')
  if (el.hasAttribute('data-inview')) playInview(el)
}
function checkPending() {
  const line = innerHeight * 0.88
  for (const el of pending) {
    if (el.getBoundingClientRect().top > line || (el.hasAttribute('data-gate') && !ready)) continue
    pending.delete(el)
    start(el)
  }
}

$$('[data-reveal]').forEach(prepareReveal)
$$('[data-reveal]:not([data-manual]), [data-inview]').forEach((el) => pending.add(el))

// ---------- hover springs (pointer devices wider than 768 px only) ----------
for (const el of $$('[data-hover]')) {
  const { from = {}, to: target, cfg } = json(el, 'hover')
  const rest = { ...Object.fromEntries(Object.keys(target).map((k) => [k, REST[k]])), ...from }
  const s = spring(el, rest)
  const root = el.closest('[data-hover-root]') || el
  root.addEventListener('pointerenter', () => canHover() && to(s, target, cfg))
  root.addEventListener('pointerleave', () => to(s, rest, cfg))
}

// ---------- hero rig slider: cross-fading glass cards, autoplay after the loader ----------
const RIGS = [
  { img: '/stills/s1-oct31.webp', brand: 'S1 pendulum', title: 'Held-out 0.8%', run: 'agent_take3_s1_20261031T194821', alt: 'Ruler pendulum mid-swing with its twin drawn on' },
  { img: '/stills/s2-ramp.webp', brand: 'S2 · 12° ramp', title: 'β 0.78 ± 4.5%', run: 'verify_s2_take1_20261101T170003', alt: 'Can rolling down a ramp with its twin drawn on' },
  { img: '/stills/s2-board.webp', brand: 'S2 · 8° board', title: 'β 0.74 ± 2.2%', run: 'verify_nov01_s2_take2_20261101T170008', alt: 'Can on a wooden board with its twin drawn on' },
]
const CARD_OUT = { opacity: 0, y: 16, scale: 0.96 }

function dots(container, n, onPick) {
  container.innerHTML = Array.from({ length: n }, (_, i) => `<button aria-label="Show ${i + 1} of ${n}"><span></span></button>`).join('')
  const buttons = $$('button', container)
  buttons.forEach((b, i) => b.addEventListener('click', () => onPick(i)))
  return (active) => buttons.forEach((b, i) => (i === active ? b.setAttribute('aria-current', 'true') : b.removeAttribute('aria-current')))
}

const rigStack = $('#rig-stack')
let rigIndex = 0, rigTimer = null
const setRigDot = dots($('#rig-dots'), RIGS.length, (i) => { showRig(i); restartRigTimer() })
function rigCard(r) {
  const card = document.createElement('a')
  card.className = 'glass rig-card-mini'
  card.href = `/console.html?run=${r.run}`
  card.innerHTML = `<img src="${r.img}" alt="${r.alt}" /><span><strong>${r.brand}</strong><small>${r.title}</small><em>Open run →</em></span>`
  return card
}
function showRig(i, first = false) {
  const old = rigStack.lastElementChild
  rigIndex = (i + RIGS.length) % RIGS.length
  const card = rigCard(RIGS[rigIndex])
  rigStack.append(card)
  setRigDot(rigIndex)
  if (first) return
  if (old) {
    to(spring(old, { opacity: 1, y: 0, scale: 1 }), CARD_OUT, [210, 24])
    setTimeout(() => old.remove(), 700)
  }
  to(spring(card, CARD_OUT), { opacity: 1, y: 0, scale: 1 }, [210, 24])
}
function restartRigTimer() {
  clearInterval(rigTimer)
  rigTimer = setInterval(() => showRig(rigIndex + 1), AUTOPLAY_MS)
}
showRig(0, true)

// ---------- verdict carousel: ghost words re-reveal, the rig photo cross-fades ----------
const VERDICTS = [
  { words: ['Says', 'No', 'With', 'Numbers'], img: '/stills/s1-oct31.webp', name: 'S1 pendulum · 31 Oct', role: 'Refused at G1 · Ljung–Box p = 7e-08', alt: 'Ruler pendulum on a pin, mid-swing' },
  { words: ['Twins', 'That', 'Know', 'Error'], img: '/stills/s1-clamped.webp', name: 'S1 clamped pin · 1 Nov', role: 'Held-out 3.1% / 3.4% · refused at G1', alt: 'Clamped-pin pendulum mid-swing' },
  { words: ['Same', 'Can,', 'Two', 'Ramps'], img: '/stills/s2-board.webp', name: 'S2 can · 8° board', role: 'β 0.74 ± 2.2% replicates 0.78 ± 4.5%', alt: 'Can rolling down a wooden board' },
]
const ghost = $('#verdicts-title')
const ghostSpans = $$('.ghost-clip > span', ghost)
const photoStack = $('#rig-photo-stack')
let verdictIndex = 0
const setVerdictDot = dots($('#verdict-dots'), VERDICTS.length, (i) => showVerdict(i))

function showVerdict(i, first = false) {
  verdictIndex = (i + VERDICTS.length) % VERDICTS.length
  const v = VERDICTS[verdictIndex]
  ghostSpans.forEach((s, k) => { s.textContent = v.words[k] })
  ghost.setAttribute('aria-label', v.words.join(' '))
  $('#rig-photo-name').textContent = v.name
  $('#rig-photo-role').textContent = v.role
  setVerdictDot(verdictIndex)
  const img = new Image()
  img.src = v.img; img.alt = v.alt; img.loading = 'lazy'
  const old = photoStack.lastElementChild
  photoStack.append(img)
  if (first) return
  replay(ghost)
  to(spring(img, { opacity: 0 }), { opacity: 1 }, [260, 26])
  if (old) setTimeout(() => old.remove(), 600)
}
showVerdict(0, true)
$('#verdict-prev').addEventListener('click', () => showVerdict(verdictIndex - 1))
$('#verdict-next').addEventListener('click', () => showVerdict(verdictIndex + 1))
ghost.setAttribute('data-reveal', '{}') // revealed by scroll like the rest; replayed on every slide change
pending.add(ghost)

// ---------- overlays: menu and the "film your own take" modal ----------
function overlay(root, backdropCfg, panelFrom, panelCfg, onOpen) {
  const backdrop = $('.menu-backdrop, .modal-backdrop', root)
  const panel = $('.menu-panel, .modal-panel', root)
  const sb = spring(backdrop, { opacity: 0 }, (el, { opacity }) => { el.style.opacity = opacity })
  const sp = spring(panel, panelFrom)
  let open = false, returnFocus = null
  const show = () => {
    if (open) return
    open = true; returnFocus = document.activeElement
    root.removeAttribute('inert'); root.classList.add('open'); lock()
    to(sb, { opacity: 1 }, backdropCfg)
    to(sp, Object.fromEntries(Object.keys(panelFrom).map((k) => [k, REST[k]])), panelCfg)
    onOpen?.()
  }
  const hide = () => {
    if (!open) return
    open = false
    root.setAttribute('inert', ''); root.classList.remove('open'); unlock()
    to(sb, { opacity: 0 }, backdropCfg); to(sp, panelFrom, panelCfg)
    returnFocus?.focus?.()
  }
  $$('[data-close]', root).forEach((b) => b.addEventListener('click', hide))
  document.addEventListener('keydown', (e) => e.key === 'Escape' && hide())
  return { show, hide }
}

const menuLinks = $$('.menu-links a')
const menuLinkSprings = menuLinks.map((a) => spring(a, { opacity: 0, y: 28 }))
const menu = overlay($('#menu'), [260, 30], { opacity: 0, y: -24 }, [220, 28], () => {
  menuLinkSprings.forEach((s, i) => { to(s, { opacity: 0, y: 28 }, [400, 40]); setTimeout(() => to(s, { opacity: 1, y: 0 }, [200, 26]), 120 + i * 70) })
  $('#menu .round-close').focus()
})
function closeMenu() { menu.hide() }
$('#menu-open').addEventListener('click', menu.show)

const takeTitle = $('#take-title')
const take = overlay($('#take'), [240, 30], { opacity: 0, y: 28, scale: 0.96 }, [240, 26], () => {
  replay(takeTitle)
  setTimeout(() => $('#take .round-close').focus(), 120)
})
$$('[data-open-take]').forEach((b) => b.addEventListener('click', () => { menu.hide(); take.show() }))
$('#take-copy').addEventListener('click', async (e) => {
  try { await navigator.clipboard.writeText($('#take-code').textContent); e.target.textContent = 'Copied' }
  catch { e.target.textContent = 'Select and copy' } // clipboard blocked (http, permissions)
  setTimeout(() => { e.target.textContent = 'Copy' }, 1600)
})

// ---------- loader: hold at least MIN_VISIBLE_MS after load, then lift the curtain and play the hero ----------
const loader = $('#loader')
lock()
history.scrollRestoration = 'manual' // the curtain lifts onto the hero, so every load starts at the top
scrollTo(0, 0)
to(spring($('#loader-mark'), { opacity: 0, y: 16 }), { opacity: 1, y: 0 }, [200, 22])
const fill = $('#loader-fill')
fill.style.transitionDuration = `${Math.max(0, MIN_VISIBLE_MS - 120)}ms`
setTimeout(() => fill.classList.add('full'), 120)

let countdown = null
function finishSoon() {
  if (countdown) return
  countdown = setTimeout(reveal, MIN_VISIBLE_MS)
}
function reveal() {
  ready = true
  unlock()
  loader.style.transitionDuration = `${EXIT_MS}ms`
  loader.classList.add('gone')
  setTimeout(() => loader.remove(), EXIT_MS)
  checkPending()
  restartRigTimer()
}
if (document.readyState === 'complete') finishSoon()
else addEventListener('load', finishSoon, { once: true })
setTimeout(finishSoon, MAX_VISIBLE_MS - MIN_VISIBLE_MS) // load never fired: still reveal by MAX_VISIBLE_MS
