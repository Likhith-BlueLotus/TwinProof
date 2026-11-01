// Spoken what-if -> the same bounded body the form sends. Deterministic: no model in the loop, so a
// mis-heard command can only produce a value the API's own bounds (WhatIf in api/main.py) accept or refuse.

const WORDS = { half: 0.5, double: 2, twice: 2, triple: 3, zero: 0, none: 0, one: 1, two: 2, three: 3, four: 4, five: 5 }
const NUMBER = String.raw`(\d+(?:\.\d+)?|${Object.keys(WORDS).join('|')})`

const RULES = [
  { key: 'mass_scale', noun: /\b(mass|weight|heavier|lighter)\b/ },
  { key: 'damping_scale', noun: /\b(damping|friction|drag)\b/ },
  { key: 'torque_limit', noun: /\b(torque|motor|limit)\b/ },
]

const toNumber = (w) => (w in WORDS ? WORDS[w] : Number(w))

/** @param {string} text  @returns {{mass_scale?: number, damping_scale?: number, torque_limit?: number}} */
export function parseCommand(text) {
  const t = ` ${text.toLowerCase().replace(/[,]/g, ' ')} `
  const body = {}
  for (const clause of t.split(/\band\b|;/)) {
    const rule = RULES.find((r) => r.noun.test(clause))
    if (!rule) continue
    if (/\b(double|twice)\b/.test(clause) && rule.key !== 'torque_limit') { body[rule.key] = 2; continue }
    if (/\b(half|halve)\b/.test(clause) && rule.key !== 'torque_limit') { body[rule.key] = 0.5; continue }
    if (/\bno (damping|friction|drag)\b/.test(clause)) { body[rule.key] = 0; continue }
    const m = clause.match(new RegExp(`\\b${NUMBER}\\s*(x|times|×|percent|%)?`))
    if (!m) continue
    let v = toNumber(m[1])
    if (m[2] === 'percent' || m[2] === '%') v /= 100
    if (Number.isFinite(v)) body[rule.key] = v
  }
  return body
}
