<template>
  <div class="sidebar">
    <div class="header">
      <h2>对话历史</h2>
      <span>{{ sessions.length }} 个会话</span>
    </div>
    <div class="list">
      <div
        v-for="s in sessions"
        :key="s.session_id"
        :class="['item', { active: s.session_id === currentSessionId }]"
        @click="select(s)"
      >
        <div class="meta">
          <strong>{{ s.session_id }}</strong>
          <span class="snippet">{{ snippet(s.preview) || '空会话' }}</span>
        </div>
      </div>
    </div>
    <div class="footer">
      <button @click="startNew">+ 新对话</button>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, watch } from 'vue'

export default {
  name: 'Sidebar',
  props: {
    reloadToken: { type: Number, default: 0 },
    currentSessionId: { type: String, default: '' }
  },
  emits: ['selectSession'],
  setup (props, { emit }) {
    const sessions = ref([])

    async function load () {
      try {
        const res = await fetch('/api/sessions')
        if (res.ok) {
          const data = await res.json()
          sessions.value = Array.isArray(data) ? data.filter(Boolean) : []
        }
        else sessions.value = []
      } catch (e) {
        sessions.value = []
      }
    }

    function select (s) { emit('selectSession', s) }
    function startNew () {
      emit('selectSession', { __newConversation: true, session_id: `session_${Date.now()}` })
    }

    onMounted(load)
    watch(() => props.reloadToken, () => { load() })

    function snippet (t) { return (t || '').slice(0, 80) + (t && t.length > 80 ? '…' : '') }

    return { sessions, select, startNew, snippet }
  }
}
</script>
