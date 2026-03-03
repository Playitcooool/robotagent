<template>
  <div :class="['chatview', landingMode ? 'landing-mode' : '']">
    <Transition name="landing-fade" mode="out-in">
      <div v-if="!landingMode" class="messages" ref="messagesRef">
        <div
          v-for="m in conversation"
          :key="m.id"
          :class="['message', m.role]"
        >
          <!-- typing indicator -->
          <div v-if="m.loading" class="bubble typing-card">
            <div class="typing-head">
              <span class="typing-label">正在思考并调用工具</span>
              <span class="typing-dots" aria-hidden="true">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
              </span>
            </div>
            <div class="typing-skeleton">
              <span class="sk-line w-90" aria-hidden="true"></span>
              <span class="sk-line w-72" aria-hidden="true"></span>
              <span class="sk-line w-56" aria-hidden="true"></span>
            </div>
          </div>

          <!-- markdown-rendered message -->
          <div v-else class="bubble">
            <ThinkingTrace
              v-if="m.role === 'assistant' && m.thinking"
              :done="Boolean(m.thinkingDone)"
            />
            <div
              class="markdown answer"
              v-html="renderMarkdown(m.text)"
            ></div>
          </div>
        </div>
      </div>
      <div v-else class="landing-greet">
        <h2 class="landing-title">{{ landingGreeting }}</h2>
      </div>
    </Transition>

    <PlanningPanel v-if="!landingMode" :planning="planning" />

    <form :class="['composer', landingMode ? 'composer-landing' : '']" @submit.prevent="send">
      <textarea
        ref="textareaRef"
        v-model="text"
        placeholder="向模型提问：Enter 发送，Shift+Enter 换行"
        @keydown="onKeydown"
        @compositionstart="onCompositionStart"
        @compositionend="onCompositionEnd"
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
import ThinkingTrace from './ThinkingTrace.vue'
import PlanningPanel from './PlanningPanel.vue'

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
  components: { ThinkingTrace, PlanningPanel },
  props: {
    conversation: {
      type: Array,
      default: () => []
    },
    planning: {
      type: [Object, null],
      default: null
    },
    landingMode: {
      type: Boolean,
      default: false
    }
  },
  emits: ['sendMessage'],
  setup (props, { emit }) {
    const text = ref('')
    const messagesRef = ref(null)
    const textareaRef = ref(null)
    const canSend = ref(false)
    const composing = ref(false)
    const landingGreeting = ref('你好，我是 RobotAgent。有什么可以帮你？')

    function send () {
      const payload = text.value.trim()
      if (!payload) return
      emit('sendMessage', payload)
      text.value = ''
      canSend.value = false
      nextTick(autoResizeTextarea)
    }

    function renderMarkdown (content) {
      const raw = String(content || '')
      // Some upstream responses contain escaped newlines as literal "\n".
      const normalized = raw.includes('\\n') && !raw.includes('\n')
        ? raw.replace(/\\n/g, '\n').replace(/\\t/g, '\t')
        : raw
      return md.render(normalized)
    }

    function onKeydown (e) {
      if (composing.value || e.isComposing || e.keyCode === 229) return
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        send()
      }
    }

    function onCompositionStart () {
      composing.value = true
    }

    function onCompositionEnd () {
      composing.value = false
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
      () => props.conversation.map(item => `${item.id}:${item.text}:${item.thinking || ''}:${item.loading}:${item.thinkingDone ? 1 : 0}:${item.thinkingTruncated ? 1 : 0}`).join('|'),
      () => {
        nextTick(scrollToBottom)
      }
    )

    onMounted(() => {
      autoResizeTextarea()
      scrollToBottom()
    })

    watch(
      () => props.conversation,
      (next) => {
        if (!Array.isArray(next)) return
        const firstAssistant = next.find(item => String(item?.role || '') === 'assistant' && String(item?.text || '').trim())
        if (firstAssistant) {
          landingGreeting.value = String(firstAssistant.text)
        }
      },
      { immediate: true, deep: true }
    )

    return {
      text,
      canSend,
      landingGreeting,
      send,
      renderMarkdown,
      onKeydown,
      onCompositionStart,
      onCompositionEnd,
      onInput,
      messagesRef,
      textareaRef
    }
  }
}
</script>
