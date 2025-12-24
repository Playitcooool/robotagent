<template>
  <div class="chatview">
    <div class="messages">
      <div v-for="m in conversation" :key="m.id" :class="['message', m.role]">
        <div class="bubble">{{ m.text }}</div>
      </div>
    </div>

    <form class="composer" @submit.prevent="send">
      <input v-model="text" placeholder="向模型提问，按 Enter 发送" />
      <button type="submit">发送</button>
    </form>
  </div>
</template>

<script>
import { ref, watch } from 'vue'
export default {
  name: 'ChatView',
  props: { conversation: { type: Array, default: () => [] } },
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
