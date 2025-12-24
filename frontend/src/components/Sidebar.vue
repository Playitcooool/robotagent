<template>
  <div class="sidebar">
    <div class="header">对话历史</div>
    <div class="list">
      <div v-for="m in messages" :key="m.id" class="item" @click="select(m)">
        <div class="meta">
          <strong>{{ m.role === 'user' ? '你' : '模型' }}</strong>
          <span class="snippet">{{ snippet(m.text) }}</span>
        </div>
      </div>
    </div>
    <div class="footer">
      <button @click="startNew">+ 新对话</button>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'

export default {
  name: 'Sidebar',
  emits: ['selectMessage'],
  setup (_, { emit }) {
    const messages = ref([])

    async function load() {
      try {
        const res = await fetch('/api/messages')
        if (res.ok) messages.value = await res.json()
        else messages.value = sample()
      } catch (e) {
        messages.value = sample()
      }
    }

    function sample () {
      return [
        { id: 1, role: 'assistant', text: '示例对话：你好，我是 RobotAgent。' },
        { id: 2, role: 'user', text: '请帮我查一下最新的论文。' }
      ]
    }

    function select (m) { emit('selectMessage', m) }
    function startNew () { emit('selectMessage', { id: Date.now(), role: 'user', text: '' }) }

    onMounted(load)

    function snippet (t) { return (t || '').slice(0, 80) + (t && t.length > 80 ? '…' : '') }

    return { messages, select, startNew, snippet }
  }
}
</script>
