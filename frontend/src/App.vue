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
          项目介绍
        </button>
        <button
          :class="['top-nav-btn', activeTopTab === 'chat' ? 'active' : '']"
          @click="activeTopTab = 'chat'"
        >
          对话
        </button>
      </div>
      <div class="topbar-right">
        <span class="whoami">当前用户：{{ authUser.username }}</span>
        <button class="logout" @click="onLogout">退出登录</button>
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
import { ref, onMounted, computed } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatView from './components/ChatView.vue'
import ToolResults from './components/ToolResults.vue'
import AuthView from './components/AuthView.vue'
import AboutView from './components/AboutView.vue'

const WELCOME_TEXT = '欢迎使用 RobotAgent！请选择左侧会话或发起新对话。'
const AUTH_TOKEN_KEY = 'robotagent_auth_token'

export default {
  name: 'App',
  components: { Sidebar, ChatView, ToolResults, AuthView, AboutView },
  setup () {
    const initialSessionId = `session_${Date.now()}`
    const conversation = ref([
      { id: 1, role: 'assistant', text: WELCOME_TEXT }
    ])
    const currentSessionId = ref(initialSessionId)
    const sidebarReloadToken = ref(0)

    const assistantStreams = {}
    const assistantMessageIds = {}
    const liveFrame = ref(null)
    const planningState = ref({ steps: [], updatedAt: 0 })
    let liveFrameSource = null
    let liveFramePollStart = 0

    const authToken = ref(localStorage.getItem(AUTH_TOKEN_KEY) || '')
    const authUser = ref(null)
    const authLoading = ref(true)
    const activeTopTab = ref('chat')
    const showToolPanel = computed(() => {
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
      if (liveFrameSource || !authToken.value) return
      liveFramePollStart = Date.now() / 1000
      const url = `/api/sim/stream?since=${encodeURIComponent(liveFramePollStart)}`
      liveFrameSource = new EventSource(url)

      liveFrameSource.addEventListener('frame', (evt) => {
        try {
          const payload = JSON.parse(evt.data || '{}')
          if (!payload || !payload.has_frame) return
          liveFrame.value = payload
        } catch (_) {
          // ignore parse errors
        }
      })

      liveFrameSource.addEventListener('error', () => {
        stopLiveFrameStream()
      })
    }

    function stopLiveFrameStream () {
      if (!liveFrameSource) return
      liveFrameSource.close()
      liveFrameSource = null
    }

    async function onSelectSession (session) {
      if (!session || session.__newConversation) {
        resetConversation(session?.session_id || `session_${Date.now()}`)
        return
      }
      const nextSessionId = session.session_id
      currentSessionId.value = nextSessionId
      try {
        const res = await apiFetch(`/api/messages?session_id=${encodeURIComponent(nextSessionId)}`)
        if (res.ok) {
          const data = await res.json()
          if (Array.isArray(data) && data.length > 0) {
            conversation.value = data
          } else {
            conversation.value = [{ id: Date.now(), role: 'assistant', text: WELCOME_TEXT }]
          }
        }
      } catch (_) {
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

    async function onSendMessage (text) {
      if (!authUser.value) return

      liveFrame.value = null
      planningState.value = { steps: [], updatedAt: 0 }
      startLiveFrameStream()
      const userMsg = { id: Date.now(), role: 'user', text }
      conversation.value.push(userMsg)

      const assistantId = Date.now() + 1
      const mainMessageId = `${assistantId}:main`
      conversation.value.push({ id: mainMessageId, role: 'assistant', agent: 'main', text: '', thinking: '', thinkingDone: false, thinkingTruncated: false, loading: true })
      assistantStreams[mainMessageId] = {
        text: '',
        thinking: '',
        thinkingTruncated: false
      }
      assistantMessageIds[assistantId] = { main: mainMessageId }

      function normalizeSource (raw) {
        const s = String(raw || '').trim().toLowerCase()
        if (s === 'simulator') return 'simulator'
        if (s === 'analysis' || s === 'data-analyzer' || s === 'data_analyzer') return 'analysis'
        return 'main'
      }

      function ensureAgentMessage (sourceRaw) {
        const source = normalizeSource(sourceRaw)
        const bucket = assistantMessageIds[assistantId] || (assistantMessageIds[assistantId] = {})
        if (bucket[source]) return bucket[source]
        const msgId = `${assistantId}:${source}`
        bucket[source] = msgId
        assistantStreams[msgId] = { text: '', thinking: '', thinkingTruncated: false }
        conversation.value.push({
          id: msgId,
          role: 'assistant',
          agent: source,
          text: '',
          thinking: '',
          thinkingDone: false,
          thinkingTruncated: false,
          loading: true
        })
        return msgId
      }

      try {
        const res = await apiFetch('/api/chat/send', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, session_id: currentSessionId.value })
        })

        if (!res.ok) {
          const idxErr = conversation.value.findIndex(m => m.id === assistantId)
          const errText = '无法连接后端（http ' + res.status + '）'
          if (idxErr !== -1) {
            conversation.value[idxErr].loading = false
            conversation.value[idxErr].text = errText
          } else {
            conversation.value.push({ id: Date.now() + 2, role: 'assistant', text: errText })
          }
          return
        }

        if (!res.body || !res.body.getReader) {
          const textBody = await res.text()
          const idxNoStream = conversation.value.findIndex(m => m.id === assistantId)
          const txt = textBody || '[后端返回空响应]'
          if (idxNoStream !== -1) {
            conversation.value[idxNoStream].loading = false
            conversation.value[idxNoStream].text = txt
          } else {
            conversation.value.push({ id: assistantId, role: 'assistant', text: txt })
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
              const msgId = ensureAgentMessage(obj.source)
              const idx = conversation.value.findIndex(m => m.id === msgId)
              const chunk = String(obj.text || '')
              if (!chunk) continue

              let st = assistantStreams[msgId]
              if (!st) {
                st = assistantStreams[msgId] = { text: '', thinking: '', thinkingTruncated: false }
              }

              if (idx !== -1) {
                if (conversation.value[idx].loading) {
                  conversation.value[idx].loading = false
                }
                st.text += chunk
                conversation.value[idx].text = st.text
              } else {
                st.text += chunk
                conversation.value.push({ id: msgId, role: 'assistant', agent: normalizeSource(obj.source), text: st.text, thinking: st.thinking || '', thinkingDone: false, thinkingTruncated: Boolean(st.thinkingTruncated) })
              }
            } else if (obj.type === 'thinking') {
              const msgId = ensureAgentMessage(obj.source)
              const idx = conversation.value.findIndex(m => m.id === msgId)
              const chunk = String(obj.text || '')
              if (!chunk) continue
              let st = assistantStreams[msgId]
              if (!st) {
                st = assistantStreams[msgId] = { text: '', thinking: '', thinkingTruncated: false }
              }
              st.thinking += chunk
              if (idx !== -1) {
                if (conversation.value[idx].loading) {
                  conversation.value[idx].loading = false
                }
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
                  loading: false
                })
              }
            } else if (obj.type === 'thinking_done') {
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
              const statusText = String(obj.text || '').trim()
              if (!statusText) continue
              const msgId = ensureAgentMessage(obj.source)
              const idxStatus = conversation.value.findIndex(m => m.id === msgId)
              let st = assistantStreams[msgId]
              if (!st) {
                st = assistantStreams[msgId] = { text: '', thinking: '', thinkingTruncated: false }
              }
              if (idxStatus !== -1) {
                if (conversation.value[idxStatus].loading) {
                  conversation.value[idxStatus].loading = false
                }
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
                  loading: false
                })
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
                  conversation.value[idxDone].text = stDone.text || conversation.value[idxDone].text
                  conversation.value[idxDone].thinking = stDone.thinking || conversation.value[idxDone].thinking || ''
                  conversation.value[idxDone].thinkingDone = true
                  conversation.value[idxDone].thinkingTruncated = Boolean(stDone.thinkingTruncated || conversation.value[idxDone].thinkingTruncated)
                  delete assistantStreams[msgId]
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
                st = assistantStreams[msgId] = { text: '', thinking: '', thinkingTruncated: false }
              }
              st.text += chunk
              if (idxRem !== -1) {
                conversation.value[idxRem].loading = false
                conversation.value[idxRem].text = st.text
              } else {
                conversation.value.push({ id: msgId, role: 'assistant', agent: normalizeSource(obj.source), text: st.text, thinking: st.thinking || '', thinkingDone: false, thinkingTruncated: Boolean(st.thinkingTruncated) })
              }
            }
          } catch (err) {
            // ignore invalid trailing line
          }
        }
      } catch (e) {
        const idxNet = conversation.value.findIndex(m => m.role === 'assistant' && m.loading)
        const errMsg = '[网络错误] ' + String(e)
        if (idxNet !== -1) {
          conversation.value[idxNet].loading = false
          conversation.value[idxNet].text = errMsg
        } else {
          conversation.value.push({ id: Date.now() + 2, role: 'assistant', text: errMsg })
        }
      } finally {
        sidebarReloadToken.value += 1
      }
    }

    onMounted(checkAuth)

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
      onAuthed,
      onLogout,
      onSelectSession,
      onSessionDeleted,
      onSendMessage
    }
  }
}
</script>
