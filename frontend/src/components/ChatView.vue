<template>
  <div class="chatview">
    <div class="messages">
      <div v-for="m in conversation" :key="m.id" :class="['message', m.role]">
        <div v-if="m.loading" class="bubble typing">
          <span class="dot" aria-hidden="true"></span>
          <span class="dot" aria-hidden="true"></span>
          <span class="dot" aria-hidden="true"></span>
        </div>
        <div v-else class="bubble">{{ m.text }}</div>
      </div>
    </div>

      <form class="composer" @submit.prevent="send">
        <input v-model="text" placeholder="向模型提问，按 Enter 发送" autocomplete="off" />
        <button type="submit">发送</button>
      </form>
  </div>
</template>

<script>
import { ref, watch } from 'vue'
export default {
  name: 'ChatView',
  props: {
    conversation: { type: Array, default: () => [] },
  },
  emits: ['sendMessage'],
  setup (props, { emit }) {
    const text = ref('')

    function send () {
      if (!text.value.trim()) return
      emit('sendMessage', text.value.trim())
      text.value = ''
    }

    // auto-scroll to bottom when conversation updates
    watch(() => props.conversation.length, () => {
      setTimeout(() => {
        const el = document.querySelector('.messages')
        if (el) el.scrollTop = el.scrollHeight
      }, 50)
    })

    return { text, send }
  }
}
</script>

<style scoped>
.typing {
  display: inline-flex;
  gap: 6px;
  align-items: center;
}
.dot {
  width: 8px;
  height: 8px;
  background: currentColor;
  border-radius: 50%;
  opacity: 0.25;
  animation: blink 1s infinite;
}
.dot:nth-child(2) { animation-delay: 0.15s }
.dot:nth-child(3) { animation-delay: 0.3s }
@keyframes blink {
  0% { opacity: 0.25 }
  50% { opacity: 1 }
  100% { opacity: 0.25 }
}
</style>

<style scoped>
.assistant-draft { margin: 12px 0; }
.ai-draft { min-height: 40px; display: inline-flex; align-items: center; }
.ai-draft { color: #111; }
</style>
