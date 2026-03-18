<template>
  <div class="sidebar">
    <div class="header">
      <h2>{{ t('history') }}</h2>
      <span>{{ sessions.length }} {{ t('sessions') }}</span>
    </div>

    <div class="userline" v-if="authUser">
      <span class="user">{{ authUser.username }}</span>
      <button class="logout-mini" @click="$emit('logout')">{{ t('logout') }}</button>
    </div>

    <div class="list">
      <div
        v-for="s in sessions"
        :key="s.session_id"
        :class="['item', { active: s.session_id === currentSessionId }]"
        @click="select(s)"
      >
        <div class="item-row">
          <div class="meta">
            <!-- Edit mode -->
            <input
              v-if="editingSessionId === s.session_id"
              v-model="editingTitle"
              class="title-edit"
              @blur="saveTitle(s)"
              @keyup.enter="saveTitle(s)"
              @keyup.escape="cancelEdit"
              ref="titleInput"
              @click.stop
            />
            <!-- Display mode -->
            <strong v-else>{{ sessionTitle(s) }}</strong>
            <span class="snippet">{{ snippet(s.preview) || t('emptySession') }}</span>
          </div>
          <!-- Action buttons -->
          <div class="item-actions" @click.stop>
            <!-- Rename -->
            <button
              class="action-btn"
              :title="t('rename')"
              @click="startEdit(s)"
            >
              ✏️
            </button>
            <!-- Export -->
            <button
              class="action-btn"
              :title="t('export')"
              @click="exportSession(s)"
            >
              📥
            </button>
            <!-- Share -->
            <button
              class="action-btn"
              :title="t('share')"
              @click="shareSession(s)"
            >
              🔗
            </button>
            <!-- Delete -->
            <button
              class="delete-mini"
              :title="t('delete')"
              :disabled="deletingSessionId === s.session_id"
              @click="removeSession(s)"
            >
              ×
            </button>
          </div>
        </div>
      </div>
    </div>
    <div class="footer">
      <button @click="startNew">+ {{ t('newChat') }}</button>
    </div>

    <!-- Toast notification -->
    <div v-if="toast" class="toast">{{ toast }}</div>
  </div>
</template>

<script>
import { ref, onMounted, watch, onBeforeUnmount, nextTick } from 'vue'

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

    // Translations
    const t = (key) => {
      const translations = {
        zh: {
          history: '对话历史',
          sessions: '个会话',
          logout: '退出',
          emptySession: '空会话',
          newChat: '新对话',
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
          history: 'History',
          sessions: 'sessions',
          logout: 'Logout',
          emptySession: 'Empty session',
          newChat: 'New Chat',
          rename: 'Rename',
          export: 'Export',
          share: 'Share',
          delete: 'Delete',
          exportSuccess: 'Session exported',
          shareSuccess: 'Link copied to clipboard',
          renameSuccess: 'Session renamed',
          confirmDelete: (title) => `Delete session "${title}"?`
        }
      }
      return translations[props.lang]?.[key] || translations['zh'][key] || key
    }

    async function load () {
      const seq = ++loadSeq
      if (loadController) loadController.abort()
      loadController = new AbortController()
      if (!props.authToken) {
        sessions.value = []
        return
      }
      try {
        const res = await fetch('/api/sessions', {
          headers: { Authorization: `Bearer ${props.authToken}` },
          signal: loadController.signal
        })
        if (seq !== loadSeq) return
        if (res.ok) {
          const data = await res.json()
          sessions.value = Array.isArray(data) ? data.filter(Boolean) : []
        } else {
          sessions.value = []
        }
      } catch (e) {
        if (e?.name === 'AbortError') return
        sessions.value = []
      }
    }

    function select (s) { emit('selectSession', s) }
    function startNew () {
      emit('selectSession', { __newConversation: true, session_id: `session_${Date.now()}` })
    }

    async function removeSession (s) {
      const sid = String(s?.session_id || '')
      if (!sid || !props.authToken) return
      const ok = window.confirm(t('confirmDelete')(sessionTitle(s)))
      if (!ok) return
      deletingSessionId.value = sid
      try {
        const res = await fetch(`/api/sessions/${encodeURIComponent(sid)}`, {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${props.authToken}` }
        })
        if (!res.ok) return
        sessions.value = sessions.value.filter(item => item.session_id !== sid)
        emit('sessionDeleted', sid)
      } catch (_) {
      } finally {
        deletingSessionId.value = ''
      }
    }

    // Rename functions
    function startEdit (s) {
      editingSessionId.value = s.session_id
      editingTitle.value = s.title || s.preview || ''
      nextTick(() => {
        if (titleInput.value) {
          titleInput.value.focus()
          titleInput.value.select()
        }
      })
    }

    function cancelEdit () {
      editingSessionId.value = ''
      editingTitle.value = ''
    }

    async function saveTitle (s) {
      if (!editingSessionId.value) return
      const newTitle = editingTitle.value.trim()
      if (!newTitle) {
        cancelEdit()
        return
      }
      // Update locally
      s.title = newTitle
      // Show toast
      showToast(t('renameSuccess'))
      cancelEdit()
    }

    // Export session
    async function exportSession (s) {
      if (!props.authToken) return
      try {
        const res = await fetch(`/api/messages?session_id=${encodeURIComponent(s.session_id)}`, {
          headers: { Authorization: `Bearer ${props.authToken}` }
        })
        if (!res.ok) return
        const messages = await res.json()

        // Create export data
        const exportData = {
          title: s.title || s.preview || 'RobotAgent Session',
          exportedAt: new Date().toISOString(),
          messages: messages.map(m => ({
            role: m.role,
            content: m.text || m.content || ''
          }))
        }

        // Download as JSON
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `robotagent-${s.session_id}.json`
        a.click()
        URL.revokeObjectURL(url)

        showToast(t('exportSuccess'))
      } catch (e) {
        console.error('Export failed:', e)
      }
    }

    // Share session
    async function shareSession (s) {
      // Create a shareable link (using session ID)
      const shareUrl = `${window.location.origin}/share/${s.session_id}`
      try {
        await navigator.clipboard.writeText(shareUrl)
        showToast(t('shareSuccess'))
      } catch (e) {
        // Fallback: show URL in prompt
        prompt(t('share'), shareUrl)
      }
    }

    // Toast
    function showToast (msg) {
      toast.value = msg
      setTimeout(() => { toast.value = '' }, 2000)
    }

    onMounted(load)
    watch(() => props.reloadToken, () => { load() })
    watch(() => props.authToken, () => { load() })
    onBeforeUnmount(() => {
      if (loadController) loadController.abort()
    })

    function snippet (t) { return (t || '').slice(0, 80) + (t && t.length > 80 ? '…' : '') }
    function sessionTitle (s) {
      const t = (s?.title || s?.preview || '').trim()
      if (!t) return props.lang === 'zh' ? '新对话' : 'New Chat'
      return t.length > 18 ? `${t.slice(0, 18)}...` : t
    }

    return {
      sessions,
      deletingSessionId,
      editingSessionId,
      editingTitle,
      titleInput,
      toast,
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
      sessionTitle
    }
  }
}
</script>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
}

.header h2 {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
}

.header span {
  color: var(--muted);
  font-size: 12px;
}

.userline {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 6px 8px;
  background: #111521;
}

.userline .user {
  font-size: 12px;
  color: var(--muted);
}

.logout-mini {
  border: 1px solid var(--line);
  background: transparent;
  color: var(--text);
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 12px;
  cursor: pointer;
}

.list {
  flex: 1;
  overflow-y: auto;
  padding-right: 3px;
  scrollbar-width: thin;
  scrollbar-color: rgba(154, 164, 178, 0.45) #111521;
}

.list::-webkit-scrollbar {
  width: 9px;
}

.list::-webkit-scrollbar-track {
  background: #111521;
  border-radius: 999px;
}

.list::-webkit-scrollbar-thumb {
  background: rgba(154, 164, 178, 0.45);
  border-radius: 999px;
  border: 2px solid #111521;
}

.item {
  margin-bottom: 8px;
  padding: 10px 11px;
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  background: linear-gradient(180deg, rgba(18, 22, 34, 0.96), rgba(16, 20, 31, 0.96));
  cursor: pointer;
  transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s ease;
}

.item:hover {
  border-color: rgba(255, 255, 255, 0.12);
  transform: translateY(-1px);
}

.item.active {
  border-color: rgba(47, 125, 255, 0.5);
  background: #121a2c;
}

.item-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
}

.meta {
  display: flex;
  flex-direction: column;
  min-width: 0;
  flex: 1;
}

.meta strong {
  font-size: 13px;
  color: var(--text);
  display: block;
  max-width: 100%;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.title-edit {
  font-size: 13px;
  font-weight: 600;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid var(--accent);
  border-radius: 4px;
  padding: 2px 6px;
  color: var(--text);
  width: 100%;
  max-width: 140px;
}

.snippet {
  margin-top: 6px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.35;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
}

.item-actions {
  display: flex;
  gap: 4px;
  opacity: 0;
  transition: opacity 0.2s ease;
}

.item:hover .item-actions {
  opacity: 1;
}

.action-btn {
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 12px;
  padding: 2px 4px;
  border-radius: 4px;
  transition: background 0.2s ease;
}

.action-btn:hover {
  background: rgba(255, 255, 255, 0.1);
}

.delete-mini {
  border: 1px solid rgba(255, 107, 107, 0.35);
  background: transparent;
  color: var(--danger);
  border-radius: 999px;
  width: 22px;
  height: 22px;
  padding: 0;
  font-size: 14px;
  line-height: 1;
  cursor: pointer;
  flex: 0 0 auto;
}

.delete-mini:hover {
  background: rgba(255, 107, 107, 0.1);
}

.delete-mini:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.footer {
  padding-top: 8px;
}

.footer button {
  width: 100%;
  padding: 10px 14px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: transparent;
  color: var(--text);
  font-weight: 600;
  cursor: pointer;
}

/* Toast */
.toast {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.8);
  color: #fff;
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 14px;
  z-index: 1000;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateX(-50%) translateY(10px); }
  to { opacity: 1; transform: translateX(-50%) translateY(0); }
}

/* Mobile adjustments */
@media (max-width: 900px) {
  .item-actions {
    opacity: 1;
  }

  .item {
    padding: 8px;
  }

  .snippet {
    font-size: 11px;
  }
}
</style>
