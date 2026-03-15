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
          <div v-if="m.loading" :class="['bubble', 'typing-card', isSubagent(m) ? 'subagent-bubble' : '']">
            <div class="typing-head">
              <span class="typing-label">
                {{ m.loadingKind === 'search' ? '搜索中' : '正在思考并调用工具' }}
              </span>
              <span v-if="m.loadingKind === 'search'" class="search-spinner" aria-hidden="true">🔍</span>
              <span v-else class="typing-dots" aria-hidden="true">
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
          <div v-else :class="['bubble', isSubagent(m) ? 'subagent-bubble' : '']">
            <template v-if="isSubagent(m)">
              <details class="subagent-details">
                <summary class="subagent-summary">
                  <div :class="['agent-chip', `agent-${agentKey(m.agent)}`]">
                    <span class="agent-icon">{{ agentIcon(m.agent) }}</span>
                    <span class="agent-name">{{ agentName(m.agent) }}</span>
                  </div>
                  <span class="subagent-hint">点击展开</span>
                </summary>
                <div class="subagent-body">
                  <ThinkingTrace
                    v-if="m.role === 'assistant' && m.thinking"
                    :done="Boolean(m.thinkingDone)"
                  />
                  <div
                    class="markdown answer"
                    v-html="renderMarkdown(m.text)"
                  ></div>
                  <div v-if="m.role === 'assistant' && Array.isArray(m.webSearchResults) && m.webSearchResults.length" class="web-sources">
                    <div class="web-sources-title">搜索结果与出处</div>
                    <ul>
                      <li v-for="(r, idx) in m.webSearchResults" :key="`${m.id}-src-${idx}`">
                        <a :href="r.url" target="_blank" rel="noopener noreferrer">{{ r.title || r.url }}</a>
                        <span v-if="r.snippet" class="snippet">{{ r.snippet }}</span>
                      </li>
                    </ul>
                  </div>
                  <div v-if="m.role === 'assistant' && Array.isArray(m.ragReferences) && m.ragReferences.length" class="rag-sources">
                    <div class="rag-sources-title">参考资料</div>
                    <ul>
                      <li v-for="(r, idx) in m.ragReferences" :key="`${m.id}-rag-${idx}`">
                        <a :href="r.url" target="_blank" rel="noopener noreferrer">{{ r.title || r.url }}</a>
                      </li>
                    </ul>
                  </div>
                </div>
              </details>
            </template>
            <template v-else>
              <div v-if="m.role === 'assistant'" :class="['agent-chip', `agent-${agentKey(m.agent)}`]">
                <span class="agent-icon">{{ agentIcon(m.agent) }}</span>
                <span class="agent-name">{{ agentName(m.agent) }}</span>
              </div>
              <ThinkingTrace
                v-if="m.role === 'assistant' && m.thinking"
                :done="Boolean(m.thinkingDone)"
              />
              <div
                class="markdown answer"
                v-html="renderMarkdown(m.text)"
              ></div>
              <div v-if="m.role === 'assistant' && Array.isArray(m.webSearchResults) && m.webSearchResults.length" class="web-sources">
                <div class="web-sources-title">搜索结果与出处</div>
                <ul>
                  <li v-for="(r, idx) in m.webSearchResults" :key="`${m.id}-src-${idx}`">
                    <a :href="r.url" target="_blank" rel="noopener noreferrer">{{ r.title || r.url }}</a>
                    <span v-if="r.snippet" class="snippet">{{ r.snippet }}</span>
                  </li>
                </ul>
              </div>
              <div v-if="m.role === 'assistant' && Array.isArray(m.ragReferences) && m.ragReferences.length" class="rag-sources">
                <div class="rag-sources-title">参考资料</div>
                <ul>
                  <li v-for="(r, idx) in m.ragReferences" :key="`${m.id}-rag-${idx}`">
                    <a :href="r.url" target="_blank" rel="noopener noreferrer">{{ r.title || r.url }}</a>
                  </li>
                </ul>
              </div>
            </template>
          </div>
        </div>
      </div>
      <div v-else class="landing-greet">
        <h2 class="landing-title">{{ landingGreeting }}</h2>
      </div>
    </Transition>

    <PlanningPanel v-if="!landingMode" :planning="planning" />

    <form :class="['composer', landingMode ? 'composer-landing' : '']" @submit.prevent="send">
      <div class="tool-picker" ref="toolPickerRef">
        <button
          type="button"
          class="tool-plus-btn"
          @click="toggleToolMenu"
          :aria-expanded="toolMenuOpen ? 'true' : 'false'"
          aria-label="选择工具"
        >
          <span class="tool-plus-symbol">+</span>
        </button>
        <div v-if="toolMenuOpen" class="tool-menu">
          <button
            type="button"
            :class="['tool-item', webSearchEnabled ? 'active' : '']"
            @click="toggleWebSearch"
            aria-label="联网搜索"
          >
            <span class="tool-item-icon" aria-hidden="true">🌐</span>
            <span class="tool-item-main">
              <span class="tool-item-title">联网搜索</span>
              <span class="tool-item-sub">获取外部网页信息</span>
            </span>
            <span class="tool-item-check">{{ webSearchEnabled ? '已启用' : '未启用' }}</span>
          </button>
        </div>
      </div>
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
      <button type="submit" class="send-btn" :disabled="!canSend">发送</button>
    </form>
  </div>
</template>

<script>
import { ref, watch, nextTick, onMounted, onBeforeUnmount } from 'vue'
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
    const markdownCache = new Map()
    const text = ref('')
    const messagesRef = ref(null)
    const textareaRef = ref(null)
    const toolPickerRef = ref(null)
    const canSend = ref(false)
    const composing = ref(false)
    const landingGreeting = ref('你好，我是 RobotAgent。有什么可以帮你？')
    const toolMenuOpen = ref(false)
    const enabledTools = ref([])
    const webSearchEnabled = ref(false)

    function send () {
      const payload = text.value.trim()
      if (!payload) return
      emit('sendMessage', {
        text: payload,
        enabledTools: [...enabledTools.value]
      })
      text.value = ''
      canSend.value = false
      nextTick(autoResizeTextarea)
    }

    function renderMarkdown (content) {
      const raw = String(content || '')
      if (markdownCache.has(raw)) return markdownCache.get(raw)
      const rendered = md.render(preprocessMarkdown(raw))
      markdownCache.set(raw, rendered)
      if (markdownCache.size > 200) {
        const oldestKey = markdownCache.keys().next().value
        markdownCache.delete(oldestKey)
      }
      return rendered
    }

    function preprocessMarkdown (raw) {
      let text = String(raw || '')
      // Some upstream responses contain escaped newlines as literal "\n".
      if (text.includes('\\n') && !text.includes('\n')) {
        text = text.replace(/\\n/g, '\n').replace(/\\t/g, '\t')
      }

      // Normalize common math delimiters outside fenced code blocks:
      // \( ... \) -> $...$,  \[ ... \] -> $$...$$
      const parts = text.split(/(```[\s\S]*?```)/g)
      const normalized = parts.map((part) => {
        if (part.startsWith('```')) return part
        return part
          .replace(/\\\(([\s\S]*?)\\\)/g, (_m, expr) => `$${String(expr).trim()}$`)
          .replace(/\\\[([\s\S]*?)\\\]/g, (_m, expr) => `$$\n${String(expr).trim()}\n$$`)
      })
      return normalized.join('')
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

    function toggleToolMenu () {
      toolMenuOpen.value = !toolMenuOpen.value
    }

    function toggleWebSearch () {
      webSearchEnabled.value = !webSearchEnabled.value
      if (webSearchEnabled.value) {
        if (!enabledTools.value.includes('web_search')) {
          enabledTools.value.push('web_search')
        }
      } else {
        enabledTools.value = enabledTools.value.filter((t) => t !== 'web_search')
      }
    }

    function handleGlobalClick (evt) {
      if (!toolMenuOpen.value) return
      const root = toolPickerRef.value
      if (!root) return
      const target = evt.target
      if (target && !root.contains(target)) {
        toolMenuOpen.value = false
      }
    }

    function handleGlobalKeydown (evt) {
      if (evt.key === 'Escape') {
        toolMenuOpen.value = false
      }
    }

    function agentKey (agent) {
      const a = String(agent || '').toLowerCase()
      if (a === 'simulator') return 'simulator'
      if (a === 'analysis' || a === 'data-analyzer' || a === 'data_analyzer') return 'analysis'
      return 'main'
    }

    function agentName (agent) {
      const key = agentKey(agent)
      if (key === 'simulator') return 'Simulator Agent'
      if (key === 'analysis') return 'Analysis Agent'
      return 'Main Agent'
    }

    function agentIcon (agent) {
      const key = agentKey(agent)
      if (key === 'simulator') return '🛠'
      if (key === 'analysis') return '📊'
      return '🧠'
    }

    function isSubagent (msg) {
      if (!msg || msg.role !== 'assistant') return false
      return agentKey(msg.agent) !== 'main'
    }

    function autoResizeTextarea () {
      const el = textareaRef.value
      if (!el) return
      el.style.height = 'auto'
      const nextHeight = Math.min(el.scrollHeight, 220)
      el.style.height = `${nextHeight}px`
    }

    let scrollRaf = 0

    function scrollToBottom () {
      const el = messagesRef.value
      if (!el) return
      el.scrollTop = el.scrollHeight
    }

    function scheduleScrollToBottom () {
      if (scrollRaf) cancelAnimationFrame(scrollRaf)
      scrollRaf = requestAnimationFrame(() => {
        nextTick(scrollToBottom)
      })
    }

    watch(
      () => props.conversation.length,
      () => {
        scheduleScrollToBottom()
      }
    )

    watch(
      () => props.conversation.slice(-3).map(item => {
        const textLen = String(item?.text || '').length
        const thinkLen = String(item?.thinking || '').length
        return `${item?.id}:${textLen}:${thinkLen}:${item?.loading ? 1 : 0}:${item?.thinkingDone ? 1 : 0}`
      }).join('|'),
      () => {
        scheduleScrollToBottom()
      }
    )

    onMounted(() => {
      autoResizeTextarea()
      scrollToBottom()
      document.addEventListener('click', handleGlobalClick)
      document.addEventListener('keydown', handleGlobalKeydown)
    })

    onBeforeUnmount(() => {
      if (scrollRaf) cancelAnimationFrame(scrollRaf)
      document.removeEventListener('click', handleGlobalClick)
      document.removeEventListener('keydown', handleGlobalKeydown)
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
      toolMenuOpen,
      webSearchEnabled,
      landingGreeting,
      send,
      renderMarkdown,
      onKeydown,
      onCompositionStart,
      onCompositionEnd,
      onInput,
      toggleToolMenu,
      toggleWebSearch,
      agentKey,
      agentName,
      agentIcon,
      isSubagent,
      messagesRef,
      textareaRef,
      toolPickerRef
    }
  }
}
</script>
