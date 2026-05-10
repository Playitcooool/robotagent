<template>
  <div class="rail-shell">
    <header class="rail-header">
      <div>
        <p class="eyebrow">Sessions</p>
        <h2>{{ t('history') }}</h2>
      </div>
      <span class="session-count">{{ sessions.length }} {{ t('sessions') }}</span>
    </header>

    <button class="new-session-btn" type="button" @click="startNew">+ {{ t('newChat') }}</button>

    <div v-if="sessions.length === 0" class="empty-state">
      <div class="empty-icon">◎</div>
      <strong>{{ lang === 'zh' ? '还没有历史会话' : 'No sessions yet' }}</strong>
      <p>{{ lang === 'zh' ? '创建一个新任务，对话与执行记录会持续沉淀在这里。' : 'Start a mission and the session history will accumulate here.' }}</p>
    </div>

    <div v-else class="session-list">
      <article
        v-for="session in filteredSessions"
        :key="session.session_id"
        :class="['session-row', { active: session.session_id === currentSessionId }]"
        @click="select(session)"
      >
        <input
          v-if="editingSessionId === session.session_id"
          ref="titleInput"
          v-model="editingTitle"
          class="title-edit"
          @blur="saveTitle(session)"
          @keyup.enter="saveTitle(session)"
          @keyup.escape="cancelEdit"
          @click.stop
        />
        <button v-else class="session-title" type="button">{{ sessionTitle(session) }}</button>
        <span class="session-time">{{ formatTime(session.updated_at) }}</span>
        <div class="session-actions" @click.stop>
          <button class="icon-action" type="button" :title="t('rename')" @click="startEdit(session)">✎</button>
          <button class="delete-action" type="button" :title="t('delete')" :disabled="deletingSessionId === session.session_id" @click="removeSession(session)">×</button>
        </div>
      </article>
    </div>

    <div v-if="toast" class="toast">{{ toast }}</div>
  </div>
</template>

<script>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const TOAST_DURATION = 2000
const TITLE_TRUNCATE_LENGTH = 34

export default {
  name: 'Sidebar',
  props: {
    reloadToken: { type: Number, default: 0 },
    currentSessionId: { type: String, default: '' },
    authToken: { type: String, default: '' },
    authUser: { type: Object, default: null },
    lang: { type: String, default: 'zh' }
  },
  emits: ['selectSession', 'logout', 'sessionDeleted'],
  setup (props, { emit }) {
    const sessions = ref([])
    const deletingSessionId = ref('')
    const editingSessionId = ref('')
    const editingTitle = ref('')
    const titleInput = ref(null)
    const toast = ref('')
    let loadController = null
    let loadSeq = 0

    const translations = {
      zh: {
        history: '任务会话',
        sessions: '个会话',
        emptySession: '空会话',
        newChat: '新建任务',
        rename: '重命名',
        delete: '删除会话',
        renameSuccess: '会话已重命名',
        confirmDelete: (title) => `确认删除会话「${title}」？`
      },
      en: {
        history: 'Mission Sessions',
        sessions: 'sessions',
        emptySession: 'Empty session',
        newChat: 'New Session',
        rename: 'Rename',
        delete: 'Delete session',
        renameSuccess: 'Session renamed',
        confirmDelete: (title) => `Delete session "${title}"?`
      }
    }

    function t(key) {
      return translations[props.lang]?.[key] || translations.zh[key] || key
    }

    async function load() {
      const seq = ++loadSeq
      if (loadController) loadController.abort()
      loadController = new AbortController()

      if (!props.authToken) {
        sessions.value = []
        return
      }

      try {
        const response = await fetch('/api/sessions', {
          headers: { Authorization: `Bearer ${props.authToken}` },
          signal: loadController.signal
        })

        if (seq !== loadSeq) return
        sessions.value = response.ok ? await fetchSessions(response) : []
      } catch (error) {
        if (error?.name === 'AbortError') return
        sessions.value = []
      }
    }

    async function fetchSessions(response) {
      const data = await response.json().catch(() => [])
      return Array.isArray(data) ? data.filter(Boolean) : []
    }

    function select(session) {
      emit('selectSession', session)
    }

    function startNew() {
      emit('selectSession', { __newConversation: true, session_id: `session_${Date.now()}` })
    }

    async function removeSession(session) {
      const sessionId = String(session?.session_id || '')
      if (!sessionId || !props.authToken) return
      const ok = window.confirm(t('confirmDelete')(sessionTitle(session)))
      if (!ok) return
      deletingSessionId.value = sessionId

      try {
        const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${props.authToken}` }
        })
        if (!response.ok) return
        sessions.value = sessions.value.filter((item) => item.session_id !== sessionId)
        emit('sessionDeleted', sessionId)
      } finally {
        deletingSessionId.value = ''
      }
    }

    function startEdit(session) {
      editingSessionId.value = session.session_id
      editingTitle.value = session.title || session.preview || ''
      nextTick(() => {
        if (titleInput.value) {
          titleInput.value.focus()
          titleInput.value.select()
        }
      })
    }

    function cancelEdit() {
      editingSessionId.value = ''
      editingTitle.value = ''
    }

    function saveTitle(session) {
      if (!editingSessionId.value) return
      const nextTitle = editingTitle.value.trim()
      if (!nextTitle) {
        cancelEdit()
        return
      }
      session.title = nextTitle
      showToast(t('renameSuccess'))
      cancelEdit()
    }

    function showToast(message) {
      toast.value = message
      setTimeout(() => { toast.value = '' }, TOAST_DURATION)
    }

    function sessionTitle(session) {
      const value = (session?.preview || session?.title || '').trim()
      if (!value) return props.lang === 'zh' ? '新任务' : 'New Session'
      return value.length > TITLE_TRUNCATE_LENGTH ? `${value.slice(0, TITLE_TRUNCATE_LENGTH)}...` : value
    }

    function formatTime(timestamp) {
      if (!timestamp) return ''
      const date = new Date(timestamp * 1000)
      const now = new Date()
      const diffDays = Math.floor((now - date) / 86400000)
      if (diffDays === 0) return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      if (diffDays === 1) return props.lang === 'zh' ? '昨天' : 'Yesterday'
      if (diffDays < 7) return props.lang === 'zh' ? `${diffDays}天前` : `${diffDays}d ago`
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' })
    }

    onMounted(load)
    watch(() => props.reloadToken, load)
    watch(() => props.authToken, load)
    onBeforeUnmount(() => {
      if (loadController) loadController.abort()
    })

    return {
      sessions,
      deletingSessionId,
      editingSessionId,
      editingTitle,
      titleInput,
      toast,
      filteredSessions: sessions,
      t,
      select,
      startNew,
      removeSession,
      startEdit,
      cancelEdit,
      saveTitle,
      sessionTitle,
      formatTime
    }
  }
}
</script>

<style scoped>
.rail-shell {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.rail-header,
.new-session-btn {
  border-radius: 14px;
}

.rail-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.eyebrow {
  margin: 0 0 6px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #8fb7ff;
}

.rail-header h2 {
  margin: 0;
  font-size: 18px;
}

.session-count {
  color: var(--muted);
  font-size: 12px;
}

.new-session-btn,
.icon-action,
.delete-action {
  cursor: pointer;
}

.new-session-btn {
  width: 100%;
  margin-bottom: 10px;
  padding: 11px 14px;
  border: 1px solid rgba(95, 156, 255, 0.28);
  background:
    linear-gradient(180deg, rgba(47, 125, 255, 0.18), rgba(47, 125, 255, 0.08)),
    rgba(9, 15, 25, 0.88);
  color: #e9f2ff;
  font-weight: 700;
}

.icon-action,
.delete-action {
  border: none;
  background: transparent;
  color: var(--text);
  border-radius: 999px;
}

.session-list {
  flex: 1;
  min-height: 0;
  overflow: auto;
  display: grid;
  align-content: start;
  gap: 2px;
  padding-right: 4px;
}

.session-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 8px;
  min-height: 34px;
  padding: 4px 6px 4px 10px;
  border-radius: 10px;
  cursor: pointer;
  color: var(--muted);
  transition: background 0.16s ease, color 0.16s ease;
}

.session-row:hover {
  background: rgba(255, 255, 255, 0.04);
  color: var(--text);
}

.session-row.active {
  background: rgba(47, 125, 255, 0.14);
  color: #dce8ff;
}

.session-title {
  min-width: 0;
  border: none;
  background: transparent;
  color: inherit;
  padding: 0;
  text-align: left;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  cursor: pointer;
}

.session-actions {
  display: flex;
  gap: 2px;
  opacity: 0;
  transition: opacity 0.16s ease;
}

.session-row:hover .session-actions,
.session-row.active .session-actions {
  opacity: 1;
}

.icon-action,
.delete-action {
  width: 24px;
  height: 24px;
  display: inline-grid;
  place-items: center;
  color: inherit;
}

.delete-action {
  color: #ff8d8d;
}

.session-time,
.empty-state p {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.55;
}

.empty-state strong {
  font-size: 14px;
}

.title-edit {
  width: 100%;
  border: 1px solid rgba(95, 156, 255, 0.4);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.06);
  color: var(--text);
  padding: 6px 9px;
  font: inherit;
}

.empty-state {
  display: grid;
  gap: 8px;
  place-items: start;
  padding: 18px;
  border: 1px dashed rgba(255, 255, 255, 0.12);
  border-radius: 18px;
}

.empty-icon {
  display: inline-grid;
  place-items: center;
  width: 36px;
  height: 36px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.05);
  color: #8fb7ff;
}

.toast {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  padding: 10px 16px;
  border-radius: 999px;
  background: rgba(6, 10, 16, 0.92);
  color: white;
  box-shadow: 0 18px 34px rgba(0, 0, 0, 0.28);
  z-index: 999;
}
</style>
