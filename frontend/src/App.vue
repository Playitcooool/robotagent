<template>
  <div v-if="authLoading" class="auth-shell">
    <div class="auth-card"><p>正在检查登录状态...</p></div>
  </div>

  <div v-else-if="!authUser" class="auth-shell">
    <AuthView @authed="onAuthed" />
  </div>

  <div v-else class="app-shell">
    <header class="app-topbar">
      <div class="topbar-left">RobotAgent</div>
      <div class="topbar-right">
        <span class="whoami">当前用户：{{ authUser.username }}</span>
        <button class="logout" @click="onLogout">退出登录</button>
      </div>
    </header>

    <div class="app-grid">
      <aside class="left">
        <Sidebar
          @selectSession="onSelectSession"
          @logout="onLogout"
          :reloadToken="sidebarReloadToken"
          :currentSessionId="currentSessionId"
          :authUser="authUser"
          :authToken="authToken"
        />
      </aside>

      <main class="center">
        <ChatView :conversation="conversation" @sendMessage="onSendMessage" />
      </main>

      <aside class="right">
        <ToolResults
          :result="toolResult"
          :liveFrame="liveFrame"
          :planning="planningState"
          :timeline="timelineState"
        />
      </aside>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatView from './components/ChatView.vue'
import ToolResults from './components/ToolResults.vue'
import AuthView from './components/AuthView.vue'

const WELCOME_TEXT = '欢迎使用 RobotAgent！请选择左侧会话或发起新对话。'
const AUTH_TOKEN_KEY = 'robotagent_auth_token'

export default {
  name: 'App',
  components: { Sidebar, ChatView, ToolResults, AuthView },
  setup () {
    const initialSessionId = `session_${Date.now()}`
    const conversation = ref([
      { id: 1, role: 'assistant', text: WELCOME_TEXT }
    ])
    const currentSessionId = ref(initialSessionId)
    const sidebarReloadToken = ref(0)

    const assistantStreams = {}
    const toolResult = ref(null)
    const liveFrame = ref(null)
    const planningState = ref({ steps: [], updatedAt: 0 })
    const timelineState = ref({ items: [], updatedAt: 0 })
    let liveFrameSource = null
    let liveFramePollStart = 0

    const authToken = ref(localStorage.getItem(AUTH_TOKEN_KEY) || '')
    const authUser = ref(null)
    const authLoading = ref(true)

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
      toolResult.value = null
      liveFrame.value = null
      planningState.value = { steps: [], updatedAt: 0 }
      timelineState.value = { items: [], updatedAt: 0 }
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
          const ts = Number(payload.timestamp || 0)
          if (ts > 0 && ts < liveFramePollStart) return
          liveFrame.value = payload
          if (payload.done) {
            setTimeout(stopLiveFrameStream, 1000)
          }
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
      toolResult.value = null
      planningState.value = { steps: [], updatedAt: 0 }
      timelineState.value = { items: [], updatedAt: 0 }
    }

    async function onSendMessage (text) {
      if (!authUser.value) return

      liveFrame.value = null
      planningState.value = { steps: [], updatedAt: 0 }
      timelineState.value = { items: [], updatedAt: 0 }
      startLiveFrameStream()
      const userMsg = { id: Date.now(), role: 'user', text }
      conversation.value.push(userMsg)

      const assistantId = Date.now() + 1
      conversation.value.push({ id: assistantId, role: 'assistant', text: '', loading: true })
      assistantStreams[assistantId] = {
        buffer: '',
        displayed: '',
        interval: null
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
        const originalUserText = text || ''

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
              const idx = conversation.value.findIndex(m => m.id === assistantId)
              let incoming = obj.text || ''
              if (idx !== -1) {
                const currentlyShown = conversation.value[idx].text || ''
                if ((!currentlyShown || currentlyShown === '') && originalUserText && incoming.startsWith(originalUserText)) {
                  incoming = incoming.slice(originalUserText.length)
                  incoming = incoming.replace(/^\s*[:：\-–—]?\s*/, '')
                }

                if (conversation.value[idx].loading) {
                  conversation.value[idx].loading = false
                }

                let st = assistantStreams[assistantId]
                if (!st) {
                  st = assistantStreams[assistantId] = { buffer: '', displayed: '', interval: null }
                }

                const prev = st.buffer || ''
                const growth = incoming.length - prev.length
                if (incoming.startsWith(prev) && growth > 0 && growth <= 50) {
                  st.buffer = incoming
                  st.displayed = incoming
                  const idx2 = conversation.value.findIndex(m => m.id === assistantId)
                  if (idx2 !== -1) conversation.value[idx2].text = st.displayed
                } else if (incoming.length > prev.length && incoming.startsWith(prev) && growth > 50) {
                  st.buffer = incoming
                  if (!st.interval) {
                    st.interval = setInterval(() => {
                      try {
                        const nextLen = Math.min(st.displayed.length + 6, st.buffer.length)
                        if (nextLen > st.displayed.length) {
                          st.displayed = st.buffer.slice(0, nextLen)
                          const idx2 = conversation.value.findIndex(m => m.id === assistantId)
                          if (idx2 !== -1) conversation.value[idx2].text = st.displayed
                        }
                      } catch (e) {
                        // ignore interval update errors
                      }
                    }, 25)
                  }
                } else {
                  st.buffer = incoming
                  st.displayed = incoming
                  const idx2 = conversation.value.findIndex(m => m.id === assistantId)
                  if (idx2 !== -1) conversation.value[idx2].text = st.displayed
                }
              } else {
                if (originalUserText && incoming.startsWith(originalUserText)) {
                  incoming = incoming.slice(originalUserText.length).replace(/^\s*[:：\-–—]?\s*/, '')
                }
                conversation.value.push({ id: assistantId, role: 'assistant', text: incoming })
                assistantStreams[assistantId] = { buffer: incoming, displayed: incoming, interval: null }
              }
            } else if (obj.type === 'error') {
              const errText = `[后端错误] ${obj.error}`
              const idxErr2 = conversation.value.findIndex(m => m.id === assistantId)
              if (idxErr2 !== -1) {
                conversation.value[idxErr2].loading = false
                conversation.value[idxErr2].text = errText
              } else {
                conversation.value.push({ id: Date.now() + 3, role: 'assistant', text: errText })
              }
            } else if (obj.type === 'tool') {
              toolResult.value = obj.result ?? obj.data ?? obj.payload ?? null
            } else if (obj.type === 'planning') {
              const incoming = Array.isArray(obj.plan) ? obj.plan : []
              const normalized = incoming
                .map((item, idx) => {
                  const statusRaw = String(item?.status || '').toLowerCase()
                  let status = 'pending'
                  if (statusRaw === 'in_progress') status = 'in_progress'
                  if (statusRaw === 'completed') status = 'completed'
                  const stepText = String(item?.step || item?.content || '').trim()
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
            } else if (obj.type === 'timeline') {
              const item = obj.item || {}
              const next = {
                kind: String(item.kind || 'event'),
                title: String(item.title || 'event'),
                detail: String(item.detail || ''),
                timestamp: Number(obj.updated_at || Date.now() / 1000)
              }
              const prevItems = Array.isArray(timelineState.value.items) ? timelineState.value.items : []
              const merged = [...prevItems, next]
              if (merged.length > 80) merged.splice(0, merged.length - 80)
              timelineState.value = {
                items: merged,
                updatedAt: next.timestamp
              }
            } else if (obj.type === 'done') {
              const idxDone = conversation.value.findIndex(m => m.id === assistantId)
              if (idxDone !== -1) {
                conversation.value[idxDone].loading = false
                const stDone = assistantStreams[assistantId]
                if (stDone) {
                  conversation.value[idxDone].text = stDone.buffer || stDone.displayed || conversation.value[idxDone].text
                  if (stDone.interval) {
                    clearInterval(stDone.interval)
                    stDone.interval = null
                  }
                  delete assistantStreams[assistantId]
                }
              }
            }
          }
        }

        if (buf.trim()) {
          try {
            const obj = JSON.parse(buf)
            if (obj && obj.type === 'delta') {
              const idxRem = conversation.value.findIndex(m => m.id === assistantId)
              if (idxRem !== -1) {
                conversation.value[idxRem].loading = false
                conversation.value[idxRem].text = obj.text
              } else {
                conversation.value.push({ id: assistantId, role: 'assistant', text: obj.text })
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
      authToken,
      authUser,
      conversation,
      toolResult,
      liveFrame,
      planningState,
      timelineState,
      currentSessionId,
      sidebarReloadToken,
      onAuthed,
      onLogout,
      onSelectSession,
      onSendMessage
    }
  }
}
</script>
