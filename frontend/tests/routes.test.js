import test from 'node:test'
import assert from 'node:assert/strict'

import { routes } from '../src/router/routes.js'

test('router keeps public chat and about paths with lazy route components', () => {
  const paths = routes.map((route) => route.path)

  assert.deepEqual(paths, ['/', '/chat/:sessionId?', '/about'])
  assert.equal(typeof routes[1].component, 'function')
  assert.equal(typeof routes[2].component, 'function')
})
