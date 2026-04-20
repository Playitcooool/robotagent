import test from 'node:test'
import assert from 'node:assert/strict'

import {
  computeLandingMode,
  computeShowToolPanel,
  createWelcomeMessage,
  deriveResultRailData,
  normalizePlanningPayload,
  resolveAgentKey
} from '../src/lib/workbench.js'

test('createWelcomeMessage returns localized welcome copy', () => {
  assert.match(createWelcomeMessage('zh'), /欢迎使用 RobotAgent/)
  assert.match(createWelcomeMessage('en'), /Welcome to RobotAgent/)
})

test('computeLandingMode stays true until a user message appears', () => {
  assert.equal(computeLandingMode([]), true)
  assert.equal(computeLandingMode([{ role: 'assistant', text: createWelcomeMessage('zh') }]), true)
  assert.equal(
    computeLandingMode([
      { role: 'assistant', text: createWelcomeMessage('zh') },
      { role: 'user', text: '抓取这个物体' }
    ]),
    false
  )
})

test('computeShowToolPanel becomes true for planning, tool references, live frames, or tool agents', () => {
  const baseConversation = [{ role: 'assistant', text: 'hello' }]

  assert.equal(computeShowToolPanel({ liveFrame: null, planningState: null, conversation: baseConversation }), false)
  assert.equal(
    computeShowToolPanel({
      liveFrame: { image_url: 'frame.png' },
      planningState: null,
      conversation: baseConversation
    }),
    true
  )
  assert.equal(
    computeShowToolPanel({
      liveFrame: null,
      planningState: { steps: [{ id: '1', step: 'Plan', status: 'pending' }] },
      conversation: baseConversation
    }),
    true
  )
  assert.equal(
    computeShowToolPanel({
      liveFrame: null,
      planningState: null,
      conversation: [{ role: 'assistant', text: 'hello', webSearchResults: [{ url: 'https://example.com' }] }]
    }),
    true
  )
  assert.equal(
    computeShowToolPanel({
      liveFrame: null,
      planningState: null,
      conversation: [{ role: 'assistant', text: 'hello', agent: 'simulator' }]
    }),
    true
  )
})

test('normalizePlanningPayload normalizes alternate incoming event shapes', () => {
  assert.deepEqual(
    normalizePlanningPayload({
      plan: [
        { id: 1, title: 'Inspect scene', status: 'running' },
        { id: 2, task: 'Report outcome', state: 'done' }
      ],
      updated_at: 123
    }),
    {
      updatedAt: 123,
      steps: [
        { id: '1', step: 'Inspect scene', status: 'in_progress' },
        { id: '2', step: 'Report outcome', status: 'completed' }
      ]
    }
  )
})

test('resolveAgentKey collapses simulator and analysis aliases', () => {
  assert.equal(resolveAgentKey('simulator'), 'simulator')
  assert.equal(resolveAgentKey('analysis'), 'analysis')
  assert.equal(resolveAgentKey('data_analyzer'), 'analysis')
  assert.equal(resolveAgentKey('main'), 'main')
})

test('deriveResultRailData summarizes assistant activity for the right rail', () => {
  const summary = deriveResultRailData([
    { id: '1', role: 'assistant', agent: 'main', loading: true, loadingKind: 'search', text: 'Searching docs' },
    { id: '2', role: 'assistant', agent: 'analysis', text: 'Trajectory deviation is high.' },
    { id: '3', role: 'assistant', agent: 'main', webSearchResults: [{ title: 'Paper', url: 'https://example.com' }] },
    { id: '4', role: 'assistant', ragReferences: [{ title: 'Internal note', url: 'https://internal.local' }] }
  ])

  assert.equal(summary.activeTasks.length, 1)
  assert.equal(summary.activeTasks[0].agentKey, 'main')
  assert.equal(summary.latestSearchResults.length, 1)
  assert.equal(summary.latestRagReferences.length, 1)
  assert.equal(summary.toolOutputs.length, 1)
  assert.match(summary.toolOutputs[0].outputText, /Trajectory deviation/)
})
