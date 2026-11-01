// Run: node --test web/src/voice.test.js
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { parseCommand } from './voice.js'

test('scales and absolute limits from plain speech', () => {
  assert.deepEqual(parseCommand('What if the mass is 1.5 times'), { mass_scale: 1.5 })
  assert.deepEqual(parseCommand('set the torque limit to 0.5'), { torque_limit: 0.5 })
  assert.deepEqual(parseCommand('double the damping and torque limit two'), { damping_scale: 2, torque_limit: 2 })
  assert.deepEqual(parseCommand('half the mass'), { mass_scale: 0.5 })
  assert.deepEqual(parseCommand('no friction'), { damping_scale: 0 })
  assert.deepEqual(parseCommand('mass 150 percent'), { mass_scale: 1.5 })
})

test('ignores speech with no known quantity', () => {
  assert.deepEqual(parseCommand('hello there'), {})
  assert.deepEqual(parseCommand('make it heavier'), {})
})
