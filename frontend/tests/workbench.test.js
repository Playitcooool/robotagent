import test from 'node:test'
import assert from 'node:assert/strict'

import {
  computeLandingMode,
  computeShowPlanningPanel,
  computeShowToolPanel,
  createWelcomeMessage,
  deriveResultRailData,
  hasRenderableAssistantContent,
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

test('computeShowToolPanel only shows live simulation frames', () => {
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
      planningState: { steps: [{ id: '1', step: 'Plan', status: 'pending' }], activeSource: 'main' },
      conversation: baseConversation
    }),
    false
  )
  assert.equal(
    computeShowToolPanel({
      liveFrame: null,
      planningState: { statusText: '仿真子代理执行中...', activeSource: 'simulator' },
      conversation: baseConversation
    }),
    false
  )
  assert.equal(
    computeShowToolPanel({
      liveFrame: null,
      planningState: null,
      conversation: [{ role: 'assistant', text: 'hello', webSearchResults: [{ url: 'https://example.com' }] }]
    }),
    false
  )
  assert.equal(
    computeShowToolPanel({
      liveFrame: null,
      planningState: null,
      conversation: [{ role: 'assistant', text: 'hello', agent: 'simulator' }]
    }),
    false
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
      statusText: '',
      activeSource: 'main',
      isActive: false,
      steps: [
        { id: '1', step: 'Inspect scene', status: 'in_progress' },
        { id: '2', step: 'Report outcome', status: 'completed' }
      ]
    }
  )
})

test('normalizePlanningPayload preserves structured status fields without steps', () => {
  assert.deepEqual(
    normalizePlanningPayload({
      status_text: '已转交 simulator 执行，正在处理中...',
      active_source: 'simulator',
      is_active: true,
      updated_at: 456
    }),
    {
      updatedAt: 456,
      statusText: '已转交 simulator 执行，正在处理中...',
      activeSource: 'simulator',
      isActive: true,
      steps: []
    }
  )
})

test('computeShowPlanningPanel shows steps or status text only', () => {
  assert.equal(computeShowPlanningPanel(null), false)
  assert.equal(computeShowPlanningPanel({ steps: [], statusText: '' }), false)
  assert.equal(computeShowPlanningPanel({ steps: [], statusText: '正在执行工具：task' }), true)
  assert.equal(computeShowPlanningPanel({ steps: [{ id: '1', step: 'Plan', status: 'pending' }] }), true)
})

test('hasRenderableAssistantContent ignores status-only placeholders', () => {
  assert.equal(hasRenderableAssistantContent({ text: '', thinking: '' }), false)
  assert.equal(hasRenderableAssistantContent({ text: '正在执行：task', statusOnly: '正在执行：task' }), false)
  assert.equal(hasRenderableAssistantContent({ text: '最终位置：[0, 0, 1]', statusOnly: '正在执行：task' }), true)
  assert.equal(hasRenderableAssistantContent({ thinking: 'reasoning' }), true)
  assert.equal(hasRenderableAssistantContent({ webSearchResults: [{ url: 'https://example.com' }] }), true)
  assert.equal(hasRenderableAssistantContent({ error: '处理请求失败' }), true)
})

test('resolveAgentKey collapses simulator and analysis aliases', () => {
  assert.equal(resolveAgentKey('simulator'), 'simulator')
  assert.equal(resolveAgentKey('analysis'), 'analysis')
  assert.equal(resolveAgentKey('data_analyzer'), 'analysis')
  assert.equal(resolveAgentKey('main'), 'main')
})

test('deriveResultRailData summarizes assistant activity for the right rail', () => {
  const summary = deriveResultRailData([
    { id: '1', role: 'assistant', agent: 'analysis', loading: true, loadingKind: 'search', text: 'Searching docs' },
    { id: '2', role: 'assistant', agent: 'analysis', loading: false, text: 'Trajectory deviation is high.' },
    { id: '3', role: 'assistant', agent: 'main', webSearchResults: [{ title: 'Paper', url: 'https://example.com' }] },
    { id: '4', role: 'assistant', ragReferences: [{ title: 'Internal note', url: 'https://internal.local' }] }
  ])

  assert.equal(summary.activeTasks.length, 1)
  assert.equal(summary.activeTasks[0].agentKey, 'analysis')
  assert.equal(summary.latestSearchResults.length, 1)
  assert.equal(summary.latestRagReferences.length, 1)
  assert.equal(summary.toolOutputs.length, 1)
  assert.match(summary.toolOutputs[0].outputText, /Trajectory deviation/)
})
