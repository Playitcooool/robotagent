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

    <div v-if="authUser" class="operator-card">
      <div class="operator-copy">
        <small>{{ lang === 'zh' ? '当前操作者' : 'Current Operator' }}</small>
        <strong>{{ authUser.username }}</strong>
      </div>
      <button class="logout-mini" type="button" @click="$emit('logout')">{{ t('logout') }}</button>
    </div>

    <div class="search-block">
      <input
        v-model="searchQuery"
        class="session-search"
        type="search"
        :placeholder="lang === 'zh' ? '搜索会话、摘要或标题' : 'Search sessions, titles, or previews'"
      />
    </div>

    <div v-if="sessions.length === 0 || (searchQuery && filteredSessions.length === 0)" class="empty-state">
      <div class="empty-icon">{{ sessions.length === 0 ? '◎' : '⌕' }}</div>
      <strong>{{ sessions.length === 0 ? (lang === 'zh' ? '还没有历史会话' : 'No sessions yet') : (lang === 'zh' ? '没有匹配项' : 'No matches') }}</strong>
      <p>{{ sessions.length === 0 ? (lang === 'zh' ? '创建一个新任务，对话与执行记录会持续沉淀在这里。' : 'Start a mission and the session history will accumulate here.') : (lang === 'zh' ? '尝试更短的关键词或切换到其他会话。' : 'Try a shorter keyword or switch to another session.') }}</p>
    </div>

    <div v-else class="session-list">
      <article
        v-for="session in filteredSessions"
        :key="session.session_id"
        :class="['session-card', { active: session.session_id === currentSessionId }]"
        @click="select(session)"
      >
        <div class="session-top">
          <div class="session-main">
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
            <strong v-else>{{ sessionTitle(session) }}</strong>
            <span class="snippet">{{ snippet(session.preview) || t('emptySession') }}</span>
          </div>

          <div class="session-actions" @click.stop>
            <button class="icon-action" type="button" :title="t('rename')" @click="startEdit(session)">✎</button>
            <button class="icon-action" type="button" :title="t('export')" @click="exportSession(session)">⤓</button>
            <button class="icon-action" type="button" :title="t('share')" @click="shareSession(session)">⤴</button>
            <button class="delete-action" type="button" :title="t('delete')" :disabled="deletingSessionId === session.session_id" @click="removeSession(session)">×</button>
          </div>
        </div>

        <div class="session-meta">
          <span v-if="session.message_count != null">{{ session.message_count }} {{ lang === 'zh' ? '条消息' : 'messages' }}</span>
          <span v-if="session.updated_at">{{ formatTime(session.updated_at) }}</span>
        </div>
      </article>
    </div>

    <div v-if="toast" class="toast">{{ toast }}</div>
  </div>
</template>

<script>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const TOAST_DURATION = 2000
const SNIPPET_MAX_LENGTH = 80
const TITLE_TRUNCATE_LENGTH = 18

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
    const searchQuery = ref('')
    let loadController = null
    let loadSeq = 0

    const filteredSessions = computed(() => {
      if (!searchQuery.value.trim()) return sessions.value
      const query = searchQuery.value.toLowerCase()
      return sessions.value.filter((session) => {
        const title = (session.title || '').toLowerCase()
        const preview = (session.preview || '').toLowerCase()
        return title.includes(query) || preview.includes(query)
      })
    })

    const translations = {
      zh: {
        history: '任务会话',
        sessions: '个会话',
        logout: '退出',
        emptySession: '空会话',
        newChat: '新建任务',
        rename: '重命名',
        export: '导出',
        share: '分享',
        delete: '删除会话',
        exportSuccess: '会话已导出',
        shareSuccess: '链接已复制到剪贴板',
        renameSuccess: '会话已重命名',
        confirmDelete: (title) => `确认删除会话「${title}」？`
      },
      en: {
        history: 'Mission Sessions',
        sessions: 'sessions',
        logout: 'Logout',
        emptySession: 'Empty session',
        newChat: 'New Session',
        rename: 'Rename',
        export: 'Export',
        share: 'Share',
        delete: 'Delete session',
        exportSuccess: 'Session exported',
        shareSuccess: 'Link copied to clipboard',
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

    async function exportSession(session) {
      if (!props.authToken) return
      const response = await fetch(`/api/messages?session_id=${encodeURIComponent(session.session_id)}`, {
        headers: { Authorization: `Bearer ${props.authToken}` }
      })
      if (!response.ok) return

      const messages = await response.json()
      const blob = new Blob([JSON.stringify({
        title: session.title || session.preview || 'RobotAgent Session',
        exportedAt: new Date().toISOString(),
        messages: (Array.isArray(messages) ? messages : []).map((item) => ({
          role: item.role,
          content: item.text || item.content || ''
        }))
      }, null, 2)], { type: 'application/json' })

      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `robotagent-${session.session_id}.json`
      anchor.click()
      URL.revokeObjectURL(url)
      showToast(t('exportSuccess'))
    }

    async function shareSession(session) {
      const shareUrl = `${window.location.origin}/share/${session.session_id}`
      try {
        await navigator.clipboard.writeText(shareUrl)
        showToast(t('shareSuccess'))
      } catch (_) {
        prompt(t('share'), shareUrl)
      }
    }

    function showToast(message) {
      toast.value = message
      setTimeout(() => { toast.value = '' }, TOAST_DURATION)
    }

    function snippet(value) {
      return (value || '').slice(0, SNIPPET_MAX_LENGTH) + ((value || '').length > SNIPPET_MAX_LENGTH ? '…' : '')
    }

    function sessionTitle(session) {
      const value = (session?.title || session?.preview || '').trim()
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
      searchQuery,
      filteredSessions,
      t,
      select,
      startNew,
      removeSession,
      startEdit,
      cancelEdit,
      saveTitle,
      exportSession,
      shareSession,
      snippet,
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
.operator-card,
.session-card,
.search-block,
.new-session-btn {
  border-radius: 18px;
}

.rail-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
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
.logout-mini,
.icon-action,
.delete-action {
  cursor: pointer;
}

.new-session-btn {
  width: 100%;
  margin-bottom: 12px;
  padding: 14px 16px;
  border: 1px solid rgba(95, 156, 255, 0.28);
  background:
    linear-gradient(180deg, rgba(47, 125, 255, 0.18), rgba(47, 125, 255, 0.08)),
    rgba(9, 15, 25, 0.88);
  color: #e9f2ff;
  font-weight: 700;
}

.operator-card,
.search-block,
.session-card {
  border: 1px solid rgba(255, 255, 255, 0.08);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.015)),
    rgba(8, 13, 21, 0.88);
}

.operator-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px;
  margin-bottom: 12px;
}

.operator-copy {
  display: grid;
  gap: 3px;
}

.operator-copy small {
  color: var(--muted);
}

.logout-mini,
.icon-action,
.delete-action {
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.04);
  color: var(--text);
  border-radius: 999px;
}

.logout-mini {
  padding: 8px 10px;
}

.search-block {
  padding: 10px 12px;
}

.session-search {
  width: 100%;
  border: none;
  background: transparent;
  color: var(--text);
  outline: none;
  font: inherit;
}

.session-list {
  flex: 1;
  min-height: 0;
  overflow: auto;
  display: grid;
  gap: 10px;
  padding-right: 4px;
}

.session-card {
  padding: 14px;
  cursor: pointer;
  transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
}

.session-card:hover {
  transform: translateY(-2px);
}

.session-card.active {
  border-color: rgba(95, 156, 255, 0.34);
  background:
    linear-gradient(180deg, rgba(47, 125, 255, 0.14), rgba(47, 125, 255, 0.05)),
    rgba(10, 16, 27, 0.92);
}

.session-top {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.session-main {
  flex: 1;
  min-width: 0;
  display: grid;
  gap: 6px;
}

.session-main strong,
.empty-state strong {
  font-size: 14px;
}

.snippet,
.session-meta,
.empty-state p {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.55;
}

.session-actions {
  display: flex;
  gap: 6px;
}

.icon-action,
.delete-action {
  width: 28px;
  height: 28px;
  display: inline-grid;
  place-items: center;
}

.delete-action {
  color: #ff8d8d;
}

.session-meta {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 12px;
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
