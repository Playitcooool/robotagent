export const WELCOME_MESSAGES = {
  zh: '欢迎使用 RobotAgent！请选择左侧会话或发起新对话。',
  en: 'Welcome to RobotAgent! Select a session on the left or start a new conversation.'
}

export function createWelcomeMessage(lang = 'zh') {
  return WELCOME_MESSAGES[lang] || WELCOME_MESSAGES.zh
}

export function createWelcomeConversation(lang = 'zh') {
  return [{ id: Date.now(), role: 'assistant', text: createWelcomeMessage(lang) }]
}

export function resolveAgentKey(agent) {
  const value = String(agent || '').trim().toLowerCase()
  if (value === 'simulator') return 'simulator'
  if (value === 'analysis' || value === 'data-analyzer' || value === 'data_analyzer') return 'analysis'
  return 'main'
}

export function computeLandingMode(conversation = []) {
  const messages = Array.isArray(conversation) ? conversation : []
  if (messages.length === 0) return true
  return !messages.some((message) => String(message?.role || '') === 'user')
}

export function computeShowToolPanel({ liveFrame = null } = {}) {
  return Boolean(liveFrame?.image_url)
}

export function computeShowPlanningPanel(planning = null) {
  const steps = Array.isArray(planning?.steps) ? planning.steps : []
  const statusText = String(planning?.statusText || '').trim()
  return steps.length > 0 || Boolean(statusText)
}

export function hasRenderableAssistantContent({
  text = '',
  thinking = '',
  webSearchResults = [],
  ragReferences = [],
  error = '',
  statusOnly = ''
} = {}) {
  const normalizedText = String(text || '').trim()
  const normalizedStatus = String(statusOnly || '').trim()
  const hasText = Boolean(normalizedText) && normalizedText !== normalizedStatus
  return Boolean(
    hasText ||
    String(thinking || '').trim() ||
    String(error || '').trim() ||
    (Array.isArray(webSearchResults) && webSearchResults.length) ||
    (Array.isArray(ragReferences) && ragReferences.length)
  )
}

export function normalizePlanningPayload(payload = {}) {
  const incoming = Array.isArray(payload?.plan)
    ? payload.plan
    : Array.isArray(payload?.steps)
      ? payload.steps
      : Array.isArray(payload?.plan?.steps)
        ? payload.plan.steps
        : []

  const statusText = String(payload?.status_text || payload?.statusText || '').trim()
  const activeSource = resolveAgentKey(payload?.active_source || payload?.activeSource || payload?.source || 'main')

  return {
    updatedAt: Number(payload?.updated_at || payload?.updatedAt || Date.now() / 1000),
    statusText,
    activeSource,
    isActive: Boolean(payload?.is_active ?? payload?.isActive ?? statusText),
    steps: incoming
      .map((item, index) => {
        const statusRaw = String(item?.status || item?.state || '').toLowerCase().replace('-', '_')
        let status = 'pending'
        if (statusRaw === 'in_progress' || statusRaw === 'running' || statusRaw === 'active') status = 'in_progress'
        if (statusRaw === 'completed' || statusRaw === 'done' || statusRaw === 'success') status = 'completed'
        const step = String(
          item?.step || item?.content || item?.title || item?.task || item?.text || ''
        ).trim()
        if (!step) return null
        return {
          id: String(item?.id || index + 1),
          step,
          status
        }
      })
      .filter(Boolean)
  }
}

export function createEmptyPlanningState() {
  return {
    steps: [],
    updatedAt: 0,
    statusText: '',
    activeSource: 'main',
    isActive: false
  }
}

export function deriveResultRailData(conversation = []) {
  const assistantMessages = (Array.isArray(conversation) ? conversation : [])
    .filter((item) => String(item?.role || '') === 'assistant')

  const activeTasks = assistantMessages
    .filter((item) => item?.loading && resolveAgentKey(item.agent) !== 'main')
    .slice(-3)
    .reverse()
    .map((item) => ({
      id: item.id,
      agentKey: resolveAgentKey(item.agent),
      agentLabel: resolveAgentLabel(item.agent),
      loadingKindLabel: item.loadingKind === 'search' ? '搜索中' : '执行中',
      statusText: String(item.text || '').trim() || '工具正在处理中...'
    }))

  const latestSearchResults = [...assistantMessages]
    .reverse()
    .find((item) => Array.isArray(item?.webSearchResults) && item.webSearchResults.length)?.webSearchResults?.slice(0, 6) || []

  const latestRagReferences = [...assistantMessages]
    .reverse()
    .find((item) => Array.isArray(item?.ragReferences) && item.ragReferences.length)?.ragReferences?.slice(0, 6) || []

  const toolOutputs = assistantMessages
    .filter((item) => {
      const key = resolveAgentKey(item.agent)
      return key !== 'main' && !item.loading && String(item.text || '').trim()
    })
    .slice(-4)
    .reverse()
    .map((item) => ({
      id: item.id,
      agentKey: resolveAgentKey(item.agent),
      agentLabel: resolveAgentLabel(item.agent),
      outputLabel: item.thinking ? '工具回复' : '最新输出',
      outputText: truncateText(item.text, 600)
    }))

  return {
    assistantMessages,
    activeTasks,
    latestSearchResults,
    latestRagReferences,
    toolOutputs
  }
}

export function resolveAgentLabel(agent) {
  const key = resolveAgentKey(agent)
  if (key === 'simulator') return 'Simulator'
  if (key === 'analysis') return 'Analysis'
  return 'Main'
}

export function truncateText(text, maxLen = 600) {
  const value = String(text || '').trim()
  if (value.length <= maxLen) return value
  return `${value.slice(0, maxLen)}...`
}
