<template>
  <div v-if="authLoading" class="auth-shell">
    <div class="auth-card"><p>正在检查登录状态...</p></div>
  </div>

  <div v-else-if="!authUser" class="auth-shell">
    <AuthView @authed="onAuthed" />
  </div>

  <div v-else class="app-shell">
    <header class="app-topbar">
      <div class="topbar-left">
        <span class="brand">RobotAgent</span>
        <button
          :class="['top-nav-btn', activeTopTab === 'about' ? 'active' : '']"
          @click="activeTopTab = 'about'"
        >
          {{ t('about') }}
        </button>
        <button
          :class="['top-nav-btn', activeTopTab === 'chat' ? 'active' : '']"
          @click="activeTopTab = 'chat'"
        >
          {{ t('chat') }}
        </button>
      </div>
      <div class="topbar-right">
        <!-- Language toggle -->
        <button class="lang-toggle" @click="toggleLang" :title="t('switchLang')">
          {{ lang === 'zh' ? 'EN' : '中' }}
        </button>
        <!-- Theme toggle -->
        <button class="theme-toggle" @click="toggleTheme" :title="t('switchTheme')">
          {{ isDark ? '☀️' : '🌙' }}
        </button>
        <span class="whoami">{{ t('currentUser') }}：{{ authUser.username }}</span>
        <button class="logout" @click="onLogout">{{ t('logout') }}</button>
      </div>
    </header>

    <div v-if="activeTopTab === 'chat'" :class="['app-grid', showToolPanel ? 'has-right' : 'no-right']">
      <aside class="left">
        <Sidebar
          @selectSession="onSelectSession"
          @sessionDeleted="onSessionDeleted"
          @logout="onLogout"
          :reloadToken="sidebarReloadToken"
          :currentSessionId="currentSessionId"
          :authUser="authUser"
          :authToken="authToken"
          :lang="lang"
        />
      </aside>

      <main class="center">
        <ChatView
          :conversation="conversation"
          :planning="planningState"
          :landingMode="!showToolPanel"
          @sendMessage="onSendMessage"
        />
      </main>

      <Transition name="right-panel">
        <aside v-if="showToolPanel" class="right">
          <ToolResults :liveFrame="liveFrame" />
        </aside>
      </Transition>
    </div>

    <main v-else class="about-shell">
      <AboutView />
    </main>
  </div>
</template>

<script>
import { ref, onMounted, computed, onBeforeUnmount } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatView from './components/ChatView.vue'
import ToolResults from './components/ToolResults.vue'
import AuthView from './components/AuthView.vue'
import AboutView from './components/AboutView.vue'

const WELCOME_TEXT = '欢迎使用 RobotAgent！请选择左侧会话或发起新对话。'
const WELCOME_TEXT_EN = 'Welcome to RobotAgent! Select a session on the left or start a new conversation.'
const AUTH_TOKEN_KEY = 'robotagent_auth_token'
const THEME_KEY = 'robotagent_theme'
const LANG_KEY = 'robotagent_lang'

// Translations
const translations = {
  zh: {
    about: '项目介绍',
    chat: '对话',
    logout: '退出登录',
    currentUser: '当前用户',
    switchTheme: '切换主题',
    switchLang: '切换语言',
    welcome: '欢迎使用 RobotAgent！请选择左侧会话或发起新对话。',
  },
  en: {
    about: 'About',
    chat: 'Chat',
    logout: 'Logout',
    currentUser: 'Current User',
    switchTheme: 'Switch Theme',
    switchLang: 'Switch Language',
    welcome: 'Welcome to RobotAgent! Select a session on the left or start a new conversation.',
  }
}

export default {
  name: 'App',
  components: { Sidebar, ChatView, ToolResults, AuthView, AboutView },
  setup () {
    // Theme
    const isDark = ref(localStorage.getItem(THEME_KEY) !== 'light')
    const toggleTheme = () => {
      isDark.value = !isDark.value
      if (isDark.value) {
        document.documentElement.classList.remove('light')
        localStorage.setItem(THEME_KEY, 'dark')
      } else {
        document.documentElement.classList.add('light')
        localStorage.setItem(THEME_KEY, 'light')
      }
    }

    // Language
    const lang = ref(localStorage.getItem(LANG_KEY) || 'zh')
    const t = (key) => {
      return translations[lang.value]?.[key] || translations['zh'][key] || key
    }
    const toggleLang = () => {
      lang.value = lang.value === 'zh' ? 'en' : 'zh'
      localStorage.setItem(LANG_KEY, lang.value)
      // Update welcome text
      const welcomeText = lang.value === 'zh' ? WELCOME_TEXT : WELCOME_TEXT_EN
      if (conversation.value.length === 1 && conversation.value[0].id === 1) {
        conversation.value[0].text = welcomeText
      }
    }

    // Initialize theme
    if (!isDark.value) {
      document.documentElement.classList.add('light')
    }

    const initialSessionId = `session_${Date.now()}`
    const welcomeText = lang.value === 'zh' ? WELCOME_TEXT : WELCOME_TEXT_EN
    const conversation = ref([
      { id: 1, role: 'assistant', text: welcomeText }
    ])
    const currentSessionId = ref(initialSessionId)
    const sidebarReloadToken = ref(0)

    const assistantStreams = {}
    const assistantMessageIds = {}
    const assistantTypewriters = {} // msgId -> { interval, displayed, total, done }
    // msgId -> { textDisplayed, textTotal, thinkingDisplayed, thinkingTotal }
    const typewriterAccum = {}
    const liveFrame = ref(null)
    const planningState = ref({ steps: [], updatedAt: 0 })
    let liveFrameEventSource = null
    let liveFrameStartTimestamp = 0
    let sessionLoadController = null

    const authToken = ref(localStorage.getItem(AUTH_TOKEN_KEY) || '')
    const authUser = ref(null)
    const authLoading = ref(true)
    const activeTopTab = ref('chat')
    const simStreamActive = ref(false)
    const showToolPanel = computed(() => {
      if (simStreamActive.value) return true
      const msgs = Array.isArray(conversation.value) ? conversation.value : []
      if (msgs.length === 0) return false
      const hasUser = msgs.some(m => String(m?.role || '') === 'user')
      if (hasUser) return true
      // Existing sessions may start with assistant history only; if so, show tool panel too.
      return msgs.some((m) => {
        if (String(m?.role || '') !== 'assistant') return false
        const txt = String(m?.text || '').trim()
        return txt && txt !== WELCOME_TEXT
      })
    })

    function handleAuthExpired () {
      authToken.value = ''
      authUser.value = null
      localStorage.removeItem(AUTH_TOKEN_KEY)
      resetConversation(`session_${Date.now()}`)
    }

    function authHeaders (extra = {}) {
      const headers = { ...extra }
      if (authToken.value) headers.Authorization = `Bearer ${authToken.value}`
      return headers
    }

    async function apiFetch (url, options = {}) {
      const merged = {
        ...options,
        headers: authHeaders(options.headers || {})
      }
      const res = await fetch(url, merged)
      if (res.status === 401) handleAuthExpired()
      return res
    }

    async function checkAuth () {
      authLoading.value = true
      try {
        if (!authToken.value) {
          authUser.value = null
          return
        }
        const res = await apiFetch('/api/auth/me')
        if (!res.ok) {
          authUser.value = null
          return
        }
        const data = await res.json().catch(() => ({}))
        authUser.value = data?.user || null
      } finally {
        authLoading.value = false
      }
    }

    function onAuthed (payload) {
      authToken.value = payload.token || ''
      authUser.value = payload.user || null
      localStorage.setItem(AUTH_TOKEN_KEY, authToken.value)
      sidebarReloadToken.value += 1
    }

    async function onLogout () {
      try {
        await apiFetch('/api/auth/logout', { method: 'POST' })
      } catch (_) {}
      handleAuthExpired()
    }

    function resetConversation (newSessionId = null) {
      if (newSessionId) currentSessionId.value = newSessionId
      conversation.value = [{ id: Date.now(), role: 'assistant', text: WELCOME_TEXT }]
      liveFrame.value = null
      planningState.value = { steps: [], updatedAt: 0 }
      stopLiveFrameStream()
    }

    function startLiveFrameStream () {
      if (!authToken.value) return
      stopLiveFrameStream()

      liveFrameStartTimestamp = Date.now() / 1000
      const eventSource = new EventSource(`/api/sim/stream?since=${liveFrameStartTimestamp}`)

      eventSource.addEventListener('frame', (event) => {
        try {
          const payload = JSON.parse(event.data)
          if (
            payload.has_frame &&
            typeof payload.timestamp === 'number' &&
            payload.timestamp >= liveFrameStartTimestamp
          ) {
            simStreamActive.value = true
            liveFrame.value = payload
            if (payload.done) {
              // 完成后等待一下再关闭
              setTimeout(() => {
                if (liveFrameEventSource === eventSource) {
                  stopLiveFrameStream()
                }
              }, 800)
            }
          }
        } catch (e) {
          console.error('Failed to parse frame:', e)
        }
      })

      eventSource.onerror = () => {
        eventSource.close()
        // 只有在连接仍然处于活动状态时才重连
        if (liveFrameEventSource === eventSource && authToken.value && simStreamActive.value) {
          setTimeout(() => {
            // 再次检查，确保用户没有主动关闭
            if (liveFrameEventSource === null && authToken.value && simStreamActive.value) {
              startLiveFrameStream()
            }
          }, 1000)
        }
      }

      liveFrameEventSource = eventSource
    }

    function startTypewriter (msgId, idx, fullText, field = 'text') {
      const threshold = field === 'thinking' ? 50 : 100
      if (fullText.length < threshold) {
        conversation.value[idx][field] = fullText
        return
      }
      const ta = typewriterAccum[msgId] || { textDisplayed: 0, textTotal: 0, thinkingDisplayed: 0, thinkingTotal: 0 }
      typewriterAccum[msgId] = ta
      if (field === 'text') {
        ta.textTotal = fullText.length
        ta.textDisplayed = 0
      } else {
        ta.thinkingTotal = fullText.length
        ta.thinkingDisplayed = 0
      }

      if (assistantTypewriters[msgId]) {
        clearInterval(assistantTypewriters[msgId].interval)
      }

      const charsPerFrame = 50
      const frameDelay = 20
      const interval = setInterval(() => {
        const i = conversation.value.findIndex(m => m.id === msgId)
        if (i === -1) {
          clearInterval(interval)
          delete assistantTypewriters[msgId]
          delete typewriterAccum[msgId]
          return
        }
        if (field === 'text') {
          ta.textDisplayed = Math.min(ta.textDisplayed + charsPerFrame, ta.textTotal)
          conversation.value[i].text = fullText.slice(0, ta.textDisplayed)
          if (ta.textDisplayed >= ta.textTotal) {
            clearInterval(interval)
            delete assistantTypewriters[msgId]
          }
        } else {
          ta.thinkingDisplayed = Math.min(ta.thinkingDisplayed + charsPerFrame, ta.thinkingTotal)
          conversation.value[i].thinking = fullText.slice(0, ta.thinkingDisplayed)
          if (ta.thinkingDisplayed >= ta.thinkingTotal) {
            clearInterval(interval)
            delete assistantTypewriters[msgId]
          }
        }
      }, frameDelay)
      assistantTypewriters[msgId] = { interval }
    }

    function stopLiveFrameStream () {
      if (liveFrameEventSource) {
        liveFrameEventSource.close()
        liveFrameEventSource = null
      }
      simStreamActive.value = false
    }

    async function onSelectSession (session) {
      if (!session || session.__newConversation) {
        resetConversation(session?.session_id || `session_${Date.now()}`)
        return
      }
      const nextSessionId = session.session_id
      currentSessionId.value = nextSessionId
      if (sessionLoadController) sessionLoadController.abort()
      sessionLoadController = new AbortController()
      try {
        const res = await apiFetch(`/api/messages?session_id=${encodeURIComponent(nextSessionId)}`, {
          signal: sessionLoadController.signal
        })
        if (res.ok) {
          const data = await res.json()
          if (Array.isArray(data) && data.length > 0) {
            conversation.value = data
          } else {
            conversation.value = [{ id: Date.now(), role: 'assistant', text: WELCOME_TEXT }]
          }
        }
      } catch (err) {
        if (err?.name === 'AbortError') return
        conversation.value = [{ id: Date.now(), role: 'assistant', text: WELCOME_TEXT }]
      }
      planningState.value = { steps: [], updatedAt: 0 }
    }

    function onSessionDeleted (sessionId) {
      const sid = String(sessionId || '')
      if (!sid) return
      if (sid === currentSessionId.value) {
        resetConversation(`session_${Date.now()}`)
      }
      sidebarReloadToken.value += 1
    }

    async function onSendMessage (payload) {
      if (!authUser.value) return
      const text = typeof payload === 'string' ? payload : String(payload?.text || '')
      const enabledTools = Array.isArray(payload?.enabledTools) ? payload.enabledTools : []
      if (!text.trim()) return

      stopLiveFrameStream()
      liveFrame.value = null
      planningState.value = { steps: [], updatedAt: 0 }
      const userMsg = { id: Date.now(), role: 'user', text }
      conversation.value.push(userMsg)

      const assistantId = Date.now() + 1
      const mainMessageId = `${assistantId}:main`
      conversation.value.push({ id: mainMessageId, role: 'assistant', agent: 'main', text: '', thinking: '', thinkingDone: false, thinkingTruncated: false, loading: true })
      assistantStreams[mainMessageId] = {
        text: '',
        thinking: '',
        thinkingTruncated: false,
        loadingKind: 'thinking',
        webSearchResults: [],
        ragReferences: []
      }
      assistantMessageIds[assistantId] = { main: mainMessageId }

      function normalizeSource (raw) {
        const s = String(raw || '').trim().toLowerCase()
        if (s === 'simulator') return 'simulator'
        if (s === 'analysis' || s === 'data-analyzer' || s === 'data_analyzer') return 'analysis'
        return 'main'
      }

      function maybeStartSimStream (sourceRaw) {
        if (normalizeSource(sourceRaw) === 'simulator') {
          startLiveFrameStream()
        }
      }

      function ensureAgentMessage (sourceRaw) {
        const source = normalizeSource(sourceRaw)
        const bucket = assistantMessageIds[assistantId] || (assistantMessageIds[assistantId] = {})
        if (bucket[source]) return bucket[source]
        const msgId = `${assistantId}:${source}`
        bucket[source] = msgId
        assistantStreams[msgId] = { text: '', thinking: '', thinkingTruncated: false, loadingKind: 'thinking', webSearchResults: [], ragReferences: [] }
        const newMsg = {
          id: msgId,
          role: 'assistant',
          agent: source,
          text: '',
          thinking: '',
          thinkingDone: false,
          thinkingTruncated: false,
          loading: true,
          loadingKind: 'thinking',
          webSearchResults: [],
          ragReferences: []
        }
        if (source !== 'main') {
          const mainIdx = conversation.value.findIndex(m => m.id === mainMessageId)
          if (mainIdx !== -1 && conversation.value[mainIdx]?.loading) {
            conversation.value.splice(mainIdx, 0, newMsg)
            return msgId
          }
        }
        conversation.value.push(newMsg)
        return msgId
      }

      try {
        const res = await apiFetch('/api/chat/send', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            session_id: currentSessionId.value,
            enabled_tools: enabledTools
          })
        })

        if (!res.ok) {
          const idxErr = conversation.value.findIndex(m => m.id === mainMessageId)
          const errText = '无法连接后端（http ' + res.status + '）'
          if (idxErr !== -1) {
            conversation.value[idxErr].loading = false
            conversation.value[idxErr].text = errText
            conversation.value[idxErr].thinkingDone = true
          } else {
            conversation.value.push({ id: Date.now() + 2, role: 'assistant', text: errText })
          }
          return
        }

        if (!res.body || !res.body.getReader) {
          const textBody = await res.text()
          const idxNoStream = conversation.value.findIndex(m => m.id === mainMessageId)
          const txt = textBody || '[后端返回空响应]'
          if (idxNoStream !== -1) {
            conversation.value[idxNoStream].loading = false
            conversation.value[idxNoStream].text = txt
            conversation.value[idxNoStream].thinkingDone = true
          } else {
            conversation.value.push({ id: mainMessageId, role: 'assistant', agent: 'main', text: txt, thinkingDone: true })
          }
          return
        }

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buf = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })

          const lines = buf.split('\n')
          buf = lines.pop()

          for (const line of lines) {
            if (!line.trim()) continue
            let obj = null
            try {
              obj = JSON.parse(line)
            } catch (err) {
              continue
            }

            if (obj.type === 'delta') {
              maybeStartSimStream(obj.source)
              const msgId = ensureAgentMessage(obj.source)
              const idx = conversation.value.findIndex(m => m.id === msgId)
              const chunk = String(obj.text || '')
              if (!chunk) continue

              let st = assistantStreams[msgId]
              if (!st) {
                st = assistantStreams[msgId] = { text: '', thinking: '', thinkingTruncated: false, loadingKind: 'thinking', webSearchResults: [], ragReferences: [] }
              }

              st.text += chunk

              // Stop any existing typewriter — will be restarted on 'done' if needed
              if (assistantTypewriters[msgId]) {
                clearInterval(assistantTypewriters[msgId].interval)
                delete assistantTypewriters[msgId]
                delete typewriterAccum[msgId]
              }

              if (idx !== -1) {
                if (conversation.value[idx].loading) {
                  conversation.value[idx].loading = false
                }
                conversation.value[idx].loadingKind = st.loadingKind || 'thinking'
                conversation.value[idx].text = st.text
              } else {
                conversation.value.push({ id: msgId, role: 'assistant', agent: normalizeSource(obj.source), text: st.text, thinking: st.thinking || '', thinkingDone: false, thinkingTruncated: Boolean(st.thinkingTruncated), loadingKind: st.loadingKind || 'thinking', webSearchResults: st.webSearchResults || [], ragReferences: st.ragReferences || [] })
              }
            } else if (obj.type === 'thinking') {
              maybeStartSimStream(obj.source)
              const msgId = ensureAgentMessage(obj.source)
              const idx = conversation.value.findIndex(m => m.id === msgId)
              const chunk = String(obj.text || '')
              if (!chunk) continue
              let st = assistantStreams[msgId]
              if (!st) {
                st = assistantStreams[msgId] = { text: '', thinking: '', thinkingTruncated: false, loadingKind: 'thinking', webSearchResults: [], ragReferences: [] }
              }
              st.thinking += chunk

              // Stop any existing typewriter for this message
              if (assistantTypewriters[msgId]) {
                clearInterval(assistantTypewriters[msgId].interval)
                delete assistantTypewriters[msgId]
                delete typewriterAccum[msgId]
              }

              if (idx !== -1) {
                if (conversation.value[idx].loading) {
                  conversation.value[idx].loading = false
                }
                conversation.value[idx].loadingKind = st.loadingKind || 'thinking'
                conversation.value[idx].thinking = st.thinking
                conversation.value[idx].thinkingDone = false
                conversation.value[idx].thinkingTruncated = Boolean(st.thinkingTruncated)
              } else {
                conversation.value.push({
                  id: msgId,
                  role: 'assistant',
                  agent: normalizeSource(obj.source),
                  text: st.text || '',
                  thinking: st.thinking,
                  thinkingDone: false,
                  thinkingTruncated: Boolean(st.thinkingTruncated),
                  loading: false,
                  loadingKind: st.loadingKind || 'thinking',
                  webSearchResults: st.webSearchResults || [],
                  ragReferences: st.ragReferences || []
                })
              }
            } else if (obj.type === 'thinking_done') {
              maybeStartSimStream(obj.source)
              const msgId = ensureAgentMessage(obj.source)
              const idxThinkDone = conversation.value.findIndex(m => m.id === msgId)
              const st = assistantStreams[msgId]
              if (st && obj.truncated) {
                st.thinkingTruncated = true
              }
              if (idxThinkDone !== -1) {
                conversation.value[idxThinkDone].thinkingDone = true
                if (obj.truncated) conversation.value[idxThinkDone].thinkingTruncated = true
              }
            } else if (obj.type === 'error') {
              const errText = `[后端错误] ${obj.error}`
              const msgId = ensureAgentMessage('main')
              const idxErr2 = conversation.value.findIndex(m => m.id === msgId)
              if (idxErr2 !== -1) {
                conversation.value[idxErr2].loading = false
                conversation.value[idxErr2].text = errText
                conversation.value[idxErr2].thinkingDone = true
              } else {
                conversation.value.push({ id: Date.now() + 3, role: 'assistant', agent: 'main', text: errText })
              }
            } else if (obj.type === 'status') {
              maybeStartSimStream(obj.source)
              const statusText = String(obj.text || '').trim()
              if (!statusText) continue
              const msgId = ensureAgentMessage(obj.source)
              const idxStatus = conversation.value.findIndex(m => m.id === msgId)
              let st = assistantStreams[msgId]
              if (!st) {
                st = assistantStreams[msgId] = { text: '', thinking: '', thinkingTruncated: false, loadingKind: 'thinking', webSearchResults: [], ragReferences: [] }
              }
              if (obj.status_kind === 'search') {
                st.loadingKind = 'search'
              }
              if (idxStatus !== -1) {
                conversation.value[idxStatus].loading = true
                conversation.value[idxStatus].loadingKind = st.loadingKind || 'thinking'
                if (!String(st.text || '').trim()) {
                  conversation.value[idxStatus].text = statusText
                }
              } else {
                conversation.value.push({
                  id: msgId,
                  role: 'assistant',
                  agent: normalizeSource(obj.source),
                  text: statusText,
                  thinking: st.thinking || '',
                  thinkingDone: false,
                  thinkingTruncated: Boolean(st.thinkingTruncated),
                  loading: true,
                  loadingKind: st.loadingKind || 'thinking',
                  webSearchResults: st.webSearchResults || [],
                  ragReferences: st.ragReferences || []
                })
              }
            } else if (obj.type === 'web_search_results') {
              maybeStartSimStream(obj.source)
              const msgId = ensureAgentMessage(obj.source || 'main')
              const idxSrc = conversation.value.findIndex(m => m.id === msgId)
              let st = assistantStreams[msgId]
              if (!st) {
                st = assistantStreams[msgId] = { text: '', thinking: '', thinkingTruncated: false, loadingKind: 'thinking', webSearchResults: [], ragReferences: [] }
              }
              const refs = Array.isArray(obj.results) ? obj.results : []
              st.webSearchResults = refs
              st.loadingKind = 'thinking'
              if (idxSrc !== -1) {
                conversation.value[idxSrc].webSearchResults = refs
                conversation.value[idxSrc].loadingKind = 'thinking'
              }
            } else if (obj.type === 'rag_results') {
              const msgId = ensureAgentMessage(obj.source || 'main')
              const idxSrc = conversation.value.findIndex(m => m.id === msgId)
              let st = assistantStreams[msgId]
              if (!st) {
                st = assistantStreams[msgId] = { text: '', thinking: '', thinkingTruncated: false, loadingKind: 'thinking', webSearchResults: [], ragReferences: [] }
              }
              const refs = Array.isArray(obj.results) ? obj.results : []
              st.ragReferences = refs
              if (idxSrc !== -1) {
                conversation.value[idxSrc].ragReferences = refs
              }
            } else if (obj.type === 'planning') {
              const incoming = Array.isArray(obj.plan)
                ? obj.plan
                : (Array.isArray(obj.steps)
                    ? obj.steps
                    : (Array.isArray(obj.plan?.steps) ? obj.plan.steps : []))
              const normalized = incoming
                .map((item, idx) => {
                  const statusRaw = String(item?.status || item?.state || '').toLowerCase().replace('-', '_')
                  let status = 'pending'
                  if (statusRaw === 'in_progress' || statusRaw === 'running' || statusRaw === 'active') status = 'in_progress'
                  if (statusRaw === 'completed' || statusRaw === 'done' || statusRaw === 'success') status = 'completed'
                  const stepText = String(
                    item?.step || item?.content || item?.title || item?.task || item?.text || ''
                  ).trim()
                  if (!stepText) return null
                  return {
                    id: String(item?.id || idx + 1),
                    step: stepText,
                    status
                  }
                })
                .filter(Boolean)

              planningState.value = {
                steps: normalized,
                updatedAt: Number(obj.updated_at || Date.now() / 1000)
              }
            } else if (obj.type === 'done') {
              const bucket = assistantMessageIds[assistantId] || {}
              for (const msgId of Object.values(bucket)) {
                const idxDone = conversation.value.findIndex(m => m.id === msgId)
                if (idxDone === -1) continue
                conversation.value[idxDone].loading = false
                const stDone = assistantStreams[msgId]
                if (stDone) {
                  const finalText = stDone.text || conversation.value[idxDone].text || ''
                  const finalThinking = stDone.thinking || conversation.value[idxDone].thinking || ''
                  conversation.value[idxDone].thinkingTruncated = Boolean(stDone.thinkingTruncated || conversation.value[idxDone].thinkingTruncated)
                  conversation.value[idxDone].loadingKind = stDone.loadingKind || 'thinking'
                  conversation.value[idxDone].webSearchResults = stDone.webSearchResults || conversation.value[idxDone].webSearchResults || []
                  conversation.value[idxDone].ragReferences = stDone.ragReferences || conversation.value[idxDone].ragReferences || []
                  delete assistantStreams[msgId]

                  // Thinking typewriter: animate thinking if >= 50 chars
                  if (finalThinking.length >= 50) {
                    conversation.value[idxDone].thinkingDone = false
                    conversation.value[idxDone].thinking = ''
                    startTypewriter(msgId, idxDone, finalThinking, 'thinking')
                    // Mark thinking done when typewriter completes
                    const checkThinkDone = setInterval(() => {
                      const i = conversation.value.findIndex(m => m.id === msgId)
                      if (i === -1) { clearInterval(checkThinkDone); return }
                      if (!assistantTypewriters[msgId]) {
                        conversation.value[i].thinkingDone = true
                        clearInterval(checkThinkDone)
                      }
                    }, 50)
                  } else {
                    conversation.value[idxDone].thinking = finalThinking
                    conversation.value[idxDone].thinkingDone = true
                  }

                  // Text typewriter: animate text if >= 100 chars
                  if (finalText.length >= 100) {
                    conversation.value[idxDone].text = ''
                    startTypewriter(msgId, idxDone, finalText, 'text')
                  } else {
                    conversation.value[idxDone].text = finalText
                  }
                }
              }
              delete assistantMessageIds[assistantId]
            }
          }
        }

        if (buf.trim()) {
          try {
            const obj = JSON.parse(buf)
            if (obj && obj.type === 'delta') {
              const chunk = String(obj.text || '')
              const msgId = ensureAgentMessage(obj.source)
              const idxRem = conversation.value.findIndex(m => m.id === msgId)
              let st = assistantStreams[msgId]
              if (!st) {
                st = assistantStreams[msgId] = { text: '', thinking: '', thinkingTruncated: false, webSearchResults: [], ragReferences: [] }
              }
              st.text += chunk
              if (assistantTypewriters[msgId]) {
                clearInterval(assistantTypewriters[msgId].interval)
                delete assistantTypewriters[msgId]
                delete typewriterAccum[msgId]
              }
              if (idxRem !== -1) {
                conversation.value[idxRem].loading = false
                conversation.value[idxRem].text = st.text
              } else {
                conversation.value.push({ id: msgId, role: 'assistant', agent: normalizeSource(obj.source), text: st.text, thinking: st.thinking || '', thinkingDone: false, thinkingTruncated: Boolean(st.thinkingTruncated), webSearchResults: st.webSearchResults || [], ragReferences: st.ragReferences || [] })
              }
            }
          } catch (err) {
            // ignore invalid trailing line
          }
        }
      } catch (e) {
        const idxNet = conversation.value.findIndex(m => m.id === mainMessageId)
        const errMsg = '[网络错误] ' + String(e)
        if (idxNet !== -1) {
          conversation.value[idxNet].loading = false
          conversation.value[idxNet].text = errMsg
          conversation.value[idxNet].thinkingDone = true
        } else {
          conversation.value.push({ id: Date.now() + 2, role: 'assistant', text: errMsg })
        }
      } finally {
        sidebarReloadToken.value += 1
      }
    }

    onMounted(checkAuth)
    onBeforeUnmount(() => {
      stopLiveFrameStream()
      if (sessionLoadController) sessionLoadController.abort()
      for (const tw of Object.values(assistantTypewriters)) {
        clearInterval(tw.interval)
      }
      for (const msgId in assistantTypewriters) {
        delete assistantTypewriters[msgId]
      }
      for (const msgId in typewriterAccum) {
        delete typewriterAccum[msgId]
      }
    })

    return {
      authLoading,
      activeTopTab,
      authToken,
      authUser,
      conversation,
      liveFrame,
      planningState,
      currentSessionId,
      showToolPanel,
      sidebarReloadToken,
      isDark,
      lang,
      t,
      toggleTheme,
      toggleLang,
      onAuthed,
      onLogout,
      onSelectSession,
      onSessionDeleted,
      onSendMessage
    }
  }
}
</script>
