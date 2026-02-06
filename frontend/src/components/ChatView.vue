<template>
  <div class="chatview">
    <div class="messages" ref="messagesRef">
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
      <textarea
        v-model="text"
        placeholder="向模型提问：Enter 发送，Shift+Enter 换行"
        @keydown="onKeydown"
        rows="2"
      ></textarea>
      <button type="submit">发送</button>
    </form>
  </div>
</template>

<script>
import { ref, watch, onMounted } from 'vue'
import MarkdownIt from 'markdown-it'
import markdownItGfm from 'markdown-it-multimd-table'
import markdownItKatex from 'markdown-it-katex'
import markdownItHighlightjs from 'markdown-it-highlightjs'
import hljs from 'highlight.js'

// Markdown renderer with GFM (tables), KaTeX (math) and syntax highlighting
const md = new MarkdownIt({
  html: false,     // 禁止 HTML，防 XSS
  breaks: true,    // 支持换行
  linkify: true,   // 自动识别链接
  typographer: true
})

md.use(markdownItGfm)
md.use(markdownItKatex)
md.use(markdownItHighlightjs, { auto: true, hljs })

// ensure links open safely in new tab
const defaultLinkRender = md.renderer.rules.link_open || function (tokens, idx, options, env, self) {
  return self.renderToken(tokens, idx, options)
}
md.renderer.rules.link_open = function (tokens, idx, options, env, self) {
  const aIndex = tokens[idx].attrIndex('target')
  if (aIndex < 0) tokens[idx].attrPush(['target', '_blank'])
  else tokens[idx].attrs[aIndex][1] = '_blank'

  // add rel for security
  const relIndex = tokens[idx].attrIndex('rel')
  if (relIndex < 0) tokens[idx].attrPush(['rel', 'noopener noreferrer'])
  else tokens[idx].attrs[relIndex][1] = 'noopener noreferrer'

  return defaultLinkRender(tokens, idx, options, env, self)
}

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
    const messagesRef = ref(null)

    function send () {
      if (!text.value.trim()) return
      emit('sendMessage', text.value.trim())
      text.value = ''
    }

    function renderMarkdown (content) {
      return md.render(content || '')
    }

    function onKeydown (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        send()
      }
    }

    // auto-scroll when new message arrives
    watch(
      () => props.conversation.length,
      () => {
        setTimeout(() => {
          const el = messagesRef.value
          if (el) el.scrollTop = el.scrollHeight
        }, 50)
      }
    )

    return {
      text,
      send,
      renderMarkdown,
      onKeydown,
      messagesRef
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
  background: linear-gradient(180deg,#0f4b7a,#145f9a);
  color: #e6eef6;
}

.message.assistant .bubble {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  color: #e6eef6;
  border: 1px solid rgba(255,255,255,0.03);
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
.markdown ::v-deep pre {
  background: #1e1e1e;
  color: #f8f8f2;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;
}
.markdown ::v-deep code {
  font-family: Menlo, Monaco, Consolas, monospace;
  font-size: 0.9em;
}
.markdown ::v-deep p {
  margin: 0.5em 0;
}

.composer {
  display: flex;
  gap: 8px;
  padding: 10px;
  border-top: 1px solid rgba(255,255,255,0.03);
}

.composer textarea {
  flex: 1;
  padding: 8px;
  border-radius: 6px;
  border: 1px solid rgba(255,255,255,0.04);
  background: transparent;
  color: #e6eef6;
  resize: vertical;
}

.composer button {
  padding: 8px 16px;
}
</style>
