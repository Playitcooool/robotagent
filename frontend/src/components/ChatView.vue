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
        ref="textareaRef"
        v-model="text"
        placeholder="向模型提问：Enter 发送，Shift+Enter 换行"
        @keydown="onKeydown"
        @input="onInput"
        rows="2"
      ></textarea>
      <button type="submit" :disabled="!canSend">发送</button>
    </form>
  </div>
</template>

<script>
import { ref, watch, nextTick, onMounted } from 'vue'
import MarkdownIt from 'markdown-it'
import markdownItGfm from 'markdown-it-multimd-table'
import markdownItKatex from 'markdown-it-katex'
import markdownItHighlightjs from 'markdown-it-highlightjs'
import hljs from 'highlight.js'

const md = new MarkdownIt({
  html: false,
  breaks: true,
  linkify: true,
  typographer: true
})

md.use(markdownItGfm)
md.use(markdownItKatex)
md.use(markdownItHighlightjs, { auto: true, hljs })

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
    const textareaRef = ref(null)
    const canSend = ref(false)

    function send () {
      const payload = text.value.trim()
      if (!payload) return
      emit('sendMessage', payload)
      text.value = ''
      canSend.value = false
      nextTick(autoResizeTextarea)
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

    function onInput () {
      canSend.value = text.value.trim().length > 0
      autoResizeTextarea()
    }

    function autoResizeTextarea () {
      const el = textareaRef.value
      if (!el) return
      el.style.height = 'auto'
      const nextHeight = Math.min(el.scrollHeight, 220)
      el.style.height = `${nextHeight}px`
    }

    function scrollToBottom () {
      const el = messagesRef.value
      if (!el) return
      el.scrollTop = el.scrollHeight
    }

    watch(
      () => props.conversation.length,
      () => {
        nextTick(scrollToBottom)
      }
    )

    watch(
      () => props.conversation.map(item => `${item.id}:${item.text}:${item.loading}`).join('|'),
      () => {
        nextTick(scrollToBottom)
      }
    )

    onMounted(() => {
      autoResizeTextarea()
      scrollToBottom()
    })

    return {
      text,
      canSend,
      send,
      renderMarkdown,
      onKeydown,
      onInput,
      messagesRef,
      textareaRef
    }
  }
}
</script>
