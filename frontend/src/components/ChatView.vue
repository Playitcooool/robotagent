<template>
  <div class="chatview">
    <div class="messages">
      <div
        v-for="m in conversation"
        :key="m.id"
        :class="['message', m.role]"
      >
        <!-- typing indicator -->
        <div v-if="m.loading" class="bubble typing">
          <span class="dot" aria-hidden="true"></span>
          <span class="dot" aria-hidden="true"></span>
          <span class="dot" aria-hidden="true"></span>
        </div>

        <!-- markdown-rendered message -->
        <div
          v-else
          class="bubble markdown"
          v-html="renderMarkdown(m.text)"
        ></div>
      </div>
    </div>

    <form class="composer" @submit.prevent="send">
      <input
        v-model="text"
        placeholder="向模型提问，按 Enter 发送"
        autocomplete="off"
      />
      <button type="submit">发送</button>
    </form>
  </div>
</template>

<script>
import { ref, watch } from 'vue'
import MarkdownIt from 'markdown-it'

// Markdown renderer (LLM-safe)
const md = new MarkdownIt({
  html: false,     // 禁止 HTML，防 XSS
  breaks: true,    // 支持换行
  linkify: true    // 自动识别链接
})

export default {
  name: 'ChatView',
  props: {
    conversation: {
      type: Array,
      default: () => []
    }
  },
  emits: ['sendMessage'],
  setup (props, { emit }) {
    const text = ref('')

    function send () {
      if (!text.value.trim()) return
      emit('sendMessage', text.value.trim())
      text.value = ''
    }

    function renderMarkdown (content) {
      return md.render(content || '')
    }

    // auto-scroll when new message arrives
    watch(
      () => props.conversation.length,
      () => {
        setTimeout(() => {
          const el = document.querySelector('.messages')
          if (el) el.scrollTop = el.scrollHeight
        }, 50)
      }
    )

    return {
      text,
      send,
      renderMarkdown
    }
  }
}
</script>

<style scoped>
.chatview {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}

.message {
  margin-bottom: 12px;
}

.message.user {
  text-align: right;
}

.bubble {
  display: inline-block;
  max-width: 80%;
  padding: 10px 14px;
  border-radius: 8px;
  line-height: 1.6;
}

.message.user .bubble {
  background: #daf1ff;
}

.message.assistant .bubble {
  background: #f3f3f3;
}

/* typing indicator */
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

/* markdown content */
.markdown :deep(pre) {
  background: #1e1e1e;
  color: #f8f8f2;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;
}

.markdown :deep(code) {
  font-family: Menlo, Monaco, Consolas, monospace;
  font-size: 0.9em;
}

.markdown :deep(p) {
  margin: 0.5em 0;
}

.composer {
  display: flex;
  gap: 8px;
  padding: 10px;
  border-top: 1px solid #ddd;
}

.composer input {
  flex: 1;
  padding: 8px;
}

.composer button {
  padding: 8px 16px;
}
</style>
