import { computed, ref, watch } from 'vue'

import { useAuth } from './useAuth.js'
import { usePreferences } from './usePreferences.js'
import { useSSE } from './useSSE.js'
import {
  computeLandingMode,
  computeShowToolPanel,
  createEmptyPlanningState,
  createWelcomeConversation,
  createWelcomeMessage,
  normalizePlanningPayload,
  resolveAgentKey
} from '../lib/workbench.js'

const TYPEWRITER_THRESHOLD_THINKING = 50
const TYPEWRITER_THRESHOLD_TEXT = 100
const TYPEWRITER_CHARS_PER_FRAME = 50
const TYPEWRITER_FRAME_DELAY = 20
const CHECK_THINK_INTERVAL = 50

const auth = useAuth()
const preferences = usePreferences()
const { liveFrame, simStreamActive, startLiveFrameStream, stopLiveFrameStream } = useSSE(auth.authToken)

const conversation = ref(createWelcomeConversation(preferences.lang.value))
const currentSessionId = ref(`session_${Date.now()}`)
const sidebarReloadToken = ref(0)
const planningState = ref(createEmptyPlanningState())

const assistantStreams = {}
const assistantMessageIds = {}
const assistantTypewriters = {}
const typewriterAccum = {}
let sessionLoadController = null

const hasLiveFrame = computed(() => Boolean(liveFrame.value?.image_url))
const landingMode = computed(() => computeLandingMode(conversation.value))
const showToolPanel = computed(() => computeShowToolPanel({
  liveFrame: liveFrame.value,
  planningState: planningState.value,
  conversation: conversation.value
}))

watch(() => preferences.lang.value, (nextLang) => {
  const welcomeText = createWelcomeMessage(nextLang)
  if (conversation.value.length === 1 && conversation.value[0]?.role === 'assistant') {
    conversation.value[0].text = welcomeText
  }
})

function clearTypewriter(msgId) {
  if (assistantTypewriters[msgId]) {
    clearInterval(assistantTypewriters[msgId].interval)
    delete assistantTypewriters[msgId]
  }
  if (typewriterAccum[msgId]) {
    delete typewriterAccum[msgId]
  }
}

function stopAllTypewriters() {
  Object.keys(assistantTypewriters).forEach(clearTypewriter)
}

function resetConversation(newSessionId = null) {
  if (newSessionId) currentSessionId.value = newSessionId
  stopLiveFrameStream()
  liveFrame.value = null
  planningState.value = createEmptyPlanningState()
  conversation.value = createWelcomeConversation(preferences.lang.value)
}

function startTypewriter(msgId, idx, fullText, field = 'text') {
  const threshold = field === 'thinking' ? TYPEWRITER_THRESHOLD_THINKING : TYPEWRITER_THRESHOLD_TEXT
  if (fullText.length < threshold) {
    if (conversation.value[idx]) conversation.value[idx][field] = fullText
    return
  }

  const existing = typewriterAccum[msgId] || {
    textDisplayed: 0,
    textTotal: 0,
    thinkingDisplayed: 0,
    thinkingTotal: 0
  }
  typewriterAccum[msgId] = existing

  if (field === 'text') {
    existing.textTotal = fullText.length
    existing.textDisplayed = 0
  } else {
    existing.thinkingTotal = fullText.length
    existing.thinkingDisplayed = 0
  }

  if (assistantTypewriters[msgId]) {
    clearInterval(assistantTypewriters[msgId].interval)
    delete assistantTypewriters[msgId]
  }

  const interval = setInterval(() => {
    const currentIndex = conversation.value.findIndex((message) => message.id === msgId)
    if (currentIndex === -1) {
      clearTypewriter(msgId)
      return
    }

    const currentAccum = typewriterAccum[msgId]
    if (!currentAccum) return

    if (field === 'text') {
      currentAccum.textDisplayed = Math.min(currentAccum.textDisplayed + TYPEWRITER_CHARS_PER_FRAME, currentAccum.textTotal)
      conversation.value[currentIndex].text = fullText.slice(0, currentAccum.textDisplayed)
      if (currentAccum.textDisplayed >= currentAccum.textTotal) clearTypewriter(msgId)
    } else {
      currentAccum.thinkingDisplayed = Math.min(currentAccum.thinkingDisplayed + TYPEWRITER_CHARS_PER_FRAME, currentAccum.thinkingTotal)
      conversation.value[currentIndex].thinking = fullText.slice(0, currentAccum.thinkingDisplayed)
      if (currentAccum.thinkingDisplayed >= currentAccum.thinkingTotal) clearTypewriter(msgId)
    }
  }, TYPEWRITER_FRAME_DELAY)

  assistantTypewriters[msgId] = { interval }
}

function normalizeSource(source) {
  return resolveAgentKey(source)
}

function maybeStartSimStream(source) {
  if (normalizeSource(source) === 'simulator') startLiveFrameStream()
}

function ensureAgentMessage(assistantId, source) {
  const normalized = normalizeSource(source)
  const bucket = assistantMessageIds[assistantId] || (assistantMessageIds[assistantId] = {})
  if (bucket[normalized]) return bucket[normalized]

  const messageId = `${assistantId}:${normalized}`
  bucket[normalized] = messageId
  assistantStreams[messageId] = {
    text: '',
    thinking: '',
    thinkingTruncated: false,
    loadingKind: 'thinking',
    webSearchResults: [],
    ragReferences: []
  }

  const nextMessage = {
    id: messageId,
    role: 'assistant',
    agent: normalized,
    text: '',
    thinking: '',
    thinkingDone: false,
    thinkingTruncated: false,
    loading: true,
    loadingKind: 'thinking',
    webSearchResults: [],
    ragReferences: []
  }

  const mainMessageId = `${assistantId}:main`
  if (normalized !== 'main') {
    const mainIndex = conversation.value.findIndex((message) => message.id === mainMessageId)
    if (mainIndex !== -1 && conversation.value[mainIndex]?.loading) {
      conversation.value.splice(mainIndex, 0, nextMessage)
      return messageId
    }
  }

  conversation.value.push(nextMessage)
  return messageId
}

async function selectSession(session) {
  if (!session || session.__newConversation) {
    resetConversation(session?.session_id || `session_${Date.now()}`)
    return
  }

  currentSessionId.value = session.session_id
  if (sessionLoadController) sessionLoadController.abort()
  sessionLoadController = new AbortController()

  try {
    const res = await auth.apiFetch(`/api/messages?session_id=${encodeURIComponent(currentSessionId.value)}`, {
      signal: sessionLoadController.signal
    })

    if (res.ok) {
      const data = await res.json()
      conversation.value = Array.isArray(data) && data.length > 0
        ? data
        : createWelcomeConversation(preferences.lang.value)
    } else {
      conversation.value = createWelcomeConversation(preferences.lang.value)
    }
  } catch (error) {
    if (error?.name === 'AbortError') return
    conversation.value = createWelcomeConversation(preferences.lang.value)
  }

  planningState.value = createEmptyPlanningState()
}

function onSessionDeleted(sessionId) {
  const sid = String(sessionId || '')
  if (!sid) return
  if (sid === currentSessionId.value) resetConversation(`session_${Date.now()}`)
  sidebarReloadToken.value += 1
}

async function sendMessage(payload) {
  if (!auth.authUser.value) return

  const text = typeof payload === 'string' ? payload : String(payload?.text || '')
  const enabledTools = Array.isArray(payload?.enabledTools) ? payload.enabledTools : []
  if (!text.trim()) return

  stopLiveFrameStream()
  liveFrame.value = null
  planningState.value = createEmptyPlanningState()

  conversation.value.push({ id: Date.now(), role: 'user', text })

  const assistantId = Date.now() + 1
  const mainMessageId = `${assistantId}:main`
  conversation.value.push({
    id: mainMessageId,
    role: 'assistant',
    agent: 'main',
    text: '',
    thinking: '',
    thinkingDone: false,
    thinkingTruncated: false,
    loading: true,
    loadingKind: 'thinking'
  })

  assistantStreams[mainMessageId] = {
    text: '',
    thinking: '',
    thinkingTruncated: false,
    loadingKind: 'thinking',
    webSearchResults: [],
    ragReferences: []
  }
  assistantMessageIds[assistantId] = { main: mainMessageId }

  try {
    const res = await auth.apiFetch('/api/chat/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        session_id: currentSessionId.value,
        enabled_tools: enabledTools
      })
    })

    if (!res.ok) {
      const idx = conversation.value.findIndex((message) => message.id === mainMessageId)
      if (idx !== -1) {
        conversation.value[idx].loading = false
        conversation.value[idx].thinkingDone = true
        conversation.value[idx].text = `[请求错误] ${res.status} ${res.statusText}`
      }
      return
    }

    if (!res.body?.getReader) {
      const textBody = await res.text()
      const idx = conversation.value.findIndex((message) => message.id === mainMessageId)
      if (idx !== -1) {
        conversation.value[idx].loading = false
        conversation.value[idx].thinkingDone = true
        conversation.value[idx].text = textBody || '[后端返回空响应]'
      }
      return
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.trim()) continue

        let event = null
        try {
          event = JSON.parse(line)
        } catch (_) {
          continue
        }

        if (event.type === 'planning') {
          planningState.value = normalizePlanningPayload(event)
          continue
        }

        if (event.type === 'error') {
          const idx = conversation.value.findIndex((message) => message.id === mainMessageId)
          if (idx !== -1) {
            conversation.value[idx].loading = false
            conversation.value[idx].thinkingDone = true
            conversation.value[idx].text = `[后端错误] ${event.error}`
          }
          continue
        }

        if (event.type === 'done') {
          const bucket = assistantMessageIds[assistantId] || {}
          for (const messageId of Object.values(bucket)) {
            const idx = conversation.value.findIndex((message) => message.id === messageId)
            if (idx === -1) continue

            const streamState = assistantStreams[messageId]
            conversation.value[idx].loading = false

            if (streamState) {
              const finalText = streamState.text || conversation.value[idx].text || ''
              const finalThinking = streamState.thinking || conversation.value[idx].thinking || ''
              conversation.value[idx].thinkingTruncated = Boolean(streamState.thinkingTruncated || conversation.value[idx].thinkingTruncated)
              conversation.value[idx].loadingKind = streamState.loadingKind || 'thinking'
              conversation.value[idx].webSearchResults = streamState.webSearchResults || conversation.value[idx].webSearchResults || []
              conversation.value[idx].ragReferences = streamState.ragReferences || conversation.value[idx].ragReferences || []
              delete assistantStreams[messageId]

              if (finalThinking.length >= TYPEWRITER_THRESHOLD_THINKING) {
                conversation.value[idx].thinkingDone = false
                conversation.value[idx].thinking = ''
                startTypewriter(messageId, idx, finalThinking, 'thinking')
                const monitor = setInterval(() => {
                  const currentIndex = conversation.value.findIndex((message) => message.id === messageId)
                  if (currentIndex === -1) {
                    clearInterval(monitor)
                    return
                  }
                  if (!assistantTypewriters[messageId]) {
                    conversation.value[currentIndex].thinkingDone = true
                    clearInterval(monitor)
                  }
                }, CHECK_THINK_INTERVAL)
              } else {
                conversation.value[idx].thinking = finalThinking
                conversation.value[idx].thinkingDone = true
              }

              if (finalText.length >= TYPEWRITER_THRESHOLD_TEXT) {
                conversation.value[idx].text = ''
                startTypewriter(messageId, idx, finalText, 'text')
              } else {
                conversation.value[idx].text = finalText
              }
            }
          }
          delete assistantMessageIds[assistantId]
          continue
        }

        maybeStartSimStream(event.source)
        const messageId = ensureAgentMessage(assistantId, event.source)
        const idx = conversation.value.findIndex((message) => message.id === messageId)
        let streamState = assistantStreams[messageId]

        if (!streamState) {
          streamState = assistantStreams[messageId] = {
            text: '',
            thinking: '',
            thinkingTruncated: false,
            loadingKind: 'thinking',
            webSearchResults: [],
            ragReferences: []
          }
        }

        if (event.type === 'status') {
          const statusText = String(event.text || '').trim()
          if (!statusText) continue
          if (event.status_kind === 'search') streamState.loadingKind = 'search'
          if (idx !== -1) {
            conversation.value[idx].loading = true
            conversation.value[idx].loadingKind = streamState.loadingKind || 'thinking'
            if (!String(streamState.text || '').trim()) conversation.value[idx].text = statusText
          }
          continue
        }

        if (event.type === 'delta' || event.type === 'thinking') {
          const chunk = String(event.text || '')
          if (!chunk) continue
          clearTypewriter(messageId)

          if (event.type === 'delta') {
            streamState.text += chunk
            if (idx !== -1) {
              conversation.value[idx].loading = false
              conversation.value[idx].loadingKind = streamState.loadingKind || 'thinking'
              conversation.value[idx].text = streamState.text
            }
          } else {
            streamState.thinking += chunk
            if (idx !== -1) {
              conversation.value[idx].loading = false
              conversation.value[idx].loadingKind = streamState.loadingKind || 'thinking'
              conversation.value[idx].thinking = streamState.thinking
              conversation.value[idx].thinkingDone = false
              conversation.value[idx].thinkingTruncated = Boolean(streamState.thinkingTruncated)
            }
          }
          continue
        }

        if (event.type === 'thinking_done') {
          if (event.truncated) streamState.thinkingTruncated = true
          if (idx !== -1) {
            conversation.value[idx].thinkingDone = true
            if (event.truncated) conversation.value[idx].thinkingTruncated = true
          }
          continue
        }

        if (event.type === 'web_search_results') {
          streamState.webSearchResults = Array.isArray(event.results) ? event.results : []
          streamState.loadingKind = 'thinking'
          if (idx !== -1) {
            conversation.value[idx].webSearchResults = streamState.webSearchResults
            conversation.value[idx].loadingKind = 'thinking'
          }
          continue
        }

        if (event.type === 'rag_results') {
          streamState.ragReferences = Array.isArray(event.results) ? event.results : []
          if (idx !== -1) conversation.value[idx].ragReferences = streamState.ragReferences
        }
      }
    }
  } catch (error) {
    const idx = conversation.value.findIndex((message) => message.id === mainMessageId)
    if (idx !== -1) {
      conversation.value[idx].loading = false
      conversation.value[idx].thinkingDone = true
      conversation.value[idx].text = `[网络错误] ${String(error)}`
    }
  } finally {
    sidebarReloadToken.value += 1
  }
}

function teardown() {
  stopLiveFrameStream()
  if (sessionLoadController) sessionLoadController.abort()
  stopAllTypewriters()
}

export function useWorkbenchStore() {
  return {
    auth,
    preferences,
    conversation,
    currentSessionId,
    sidebarReloadToken,
    planningState,
    liveFrame,
    simStreamActive,
    hasLiveFrame,
    landingMode,
    showToolPanel,
    resetConversation,
    selectSession,
    onSessionDeleted,
    sendMessage,
    teardown
  }
}
