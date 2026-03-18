<template>
  <div :class="['chatview', landingMode ? 'landing-mode' : '']">
    <Transition name="landing-fade" mode="out-in">
      <div v-if="!landingMode" class="messages" ref="messagesRef" role="log" aria-live="polite" aria-label="对话消息" aria-relevant="additions">
        <div
          v-for="m in conversation"
          :key="m.id"
          :class="['message', m.role]"
        >
          <!-- typing indicator -->
          <div v-if="m.loading" :class="['bubble', 'typing-card', isSubagent(m) ? 'subagent-bubble' : '']">
            <!-- Search skeleton -->
            <template v-if="m.loadingKind === 'search'">
              <div class="search-skeleton-header">
                <span class="search-spinner" aria-hidden="true">🔍</span>
                <span class="typing-label">{{ m.text || '搜索中...' }}</span>
              </div>
              <div class="search-skeleton-list">
                <div v-for="i in 3" :key="i" class="search-skeleton-card">
                  <div class="sk-card-title"></div>
                  <div class="sk-card-line w-90"></div>
                  <div class="sk-card-line w-72"></div>
                  <div class="sk-card-line w-56"></div>
                </div>
              </div>
            </template>
            <!-- Default thinking indicator -->
            <template v-else>
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
            </template>
          </div>

          <!-- markdown-rendered message -->
          <div v-else :class="['bubble', isSubagent(m) ? 'subagent-bubble' : '', isErrorMessage(m) ? 'error-msg' : '']">
            <!-- Message copy button (assistant only, appears on hover) -->
            <button
              v-if="m.role === 'assistant' && m.text"
              class="msg-copy-btn"
              :aria-label="t('copyMessage')"
              @click="copyMessage(m, $event)"
              title="复制消息"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            </button>
            <template v-if="isSubagent(m)">
              <details class="subagent-details">
                <summary class="subagent-summary" tabindex="0" @keydown.enter="toggleSubagent($event)">
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
                  <div v-if="m.role === 'assistant' && m.thinking" class="thinking-content">
                    {{ truncateThinking(m.thinking) }}
                  </div>
                  <!-- Message copy button for subagent -->
                  <button
                    v-if="m.role === 'assistant' && m.text"
                    class="msg-copy-btn"
                    :aria-label="t('copyMessage')"
                    @click="copyMessage(m, $event)"
                    title="复制消息"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                  </button>
                  <div
                    class="markdown answer"
                    v-html="getCachedRender(m.id, m.text)"
                    @click="handleMarkdownClick"
                    role="presentation"
                  ></div>
                  <!-- Search results with collapse -->
                  <div v-if="m.role === 'assistant' && Array.isArray(m.webSearchResults) && m.webSearchResults.length" class="web-sources">
                    <div class="web-sources-title">
                      {{ m.webSearchResults[0]?.source === 'arxiv' || m.webSearchResults[0]?.source === 'openalex' ? '学术论文' : '搜索结果与出处' }}
                      <button
                        v-if="m.webSearchResults.length > 5"
                        class="collapse-toggle-btn"
                        @click="toggleSearchCollapse(m.id)"
                        :aria-expanded="!collapsedSearchIds.has(m.id)"
                        :aria-label="collapsedSearchIds.has(m.id) ? '展开' : '收起'"
                      >
                        {{ collapsedSearchIds.has(m.id) ? `展开 ${m.webSearchResults.length - 5} 条` : '收起' }}
                      </button>
                    </div>
                    <ul>
                      <li v-for="(r, idx) in getDisplayedSearchResults(m)" :key="`${m.id}-src-${idx}`">
                        <!-- 学术论文显示 -->
                        <template v-if="r.source === 'arxiv' || r.source === 'openalex'">
                          <div class="paper-item">
                            <a :href="r.url" target="_blank" rel="noopener noreferrer" class="paper-title">{{ r.title }}</a>
                            <div class="paper-meta">
                              <span class="paper-authors">{{ r.authors }}</span>
                              <span v-if="r.year" class="paper-year">({{ r.year }})</span>
                              <span v-if="r.venue" class="paper-venue">{{ r.venue }}</span>
                            </div>
                            <div v-if="r.abstract" class="paper-abstract">{{ r.abstract }}</div>
                            <div class="paper-links">
                              <a v-if="r.url" :href="r.url" target="_blank" rel="noopener noreferrer" class="link-btn">论文主页</a>
                              <a v-if="r.pdf_url" :href="r.pdf_url" target="_blank" rel="noopener noreferrer" class="link-btn pdf">PDF</a>
                              <span v-if="r.citations" class="citation-count">{{ r.citations }} 引用</span>
                            </div>
                          </div>
                        </template>
                        <!-- 普通搜索显示 -->
                        <template v-else>
                          <a :href="r.url" target="_blank" rel="noopener noreferrer">{{ r.title || r.url }}</a>
                          <span v-if="r.snippet" class="snippet">{{ r.snippet }}</span>
                        </template>
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
              <div v-if="m.role === 'assistant' && m.thinking" class="thinking-content">
                {{ truncateThinking(m.thinking) }}
              </div>
              <!-- Message copy button -->
              <button
                v-if="m.role === 'assistant' && m.text"
                class="msg-copy-btn"
                :aria-label="t('copyMessage')"
                @click="copyMessage(m, $event)"
                title="复制消息"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
              </button>
              <div
                class="markdown answer"
                v-html="getCachedRender(m.id, m.text)"
                @click="handleMarkdownClick"
                role="presentation"
              ></div>
              <!-- Search results with collapse -->
              <div v-if="m.role === 'assistant' && Array.isArray(m.webSearchResults) && m.webSearchResults.length" class="web-sources">
                <div class="web-sources-title">
                  {{ m.webSearchResults[0]?.source === 'arxiv' || m.webSearchResults[0]?.source === 'openalex' ? '学术论文' : '搜索结果与出处' }}
                  <button
                    v-if="m.webSearchResults.length > 5"
                    class="collapse-toggle-btn"
                    @click="toggleSearchCollapse(m.id)"
                    :aria-expanded="!collapsedSearchIds.has(m.id)"
                    :aria-label="collapsedSearchIds.has(m.id) ? '展开' : '收起'"
                  >
                    {{ collapsedSearchIds.has(m.id) ? `展开 ${m.webSearchResults.length - 5} 条` : '收起' }}
                  </button>
                </div>
                <ul>
                  <li v-for="(r, idx) in getDisplayedSearchResults(m)" :key="`${m.id}-src-${idx}`">
                    <!-- 学术论文显示 -->
                    <template v-if="r.source === 'arxiv' || r.source === 'openalex'">
                      <div class="paper-item">
                        <a :href="r.url" target="_blank" rel="noopener noreferrer" class="paper-title">{{ r.title }}</a>
                        <div class="paper-meta">
                          <span class="paper-authors">{{ r.authors }}</span>
                          <span v-if="r.year" class="paper-year">({{ r.year }})</span>
                          <span v-if="r.venue" class="paper-venue">{{ r.venue }}</span>
                        </div>
                        <div v-if="r.abstract" class="paper-abstract">{{ r.abstract }}</div>
                        <div class="paper-links">
                          <a v-if="r.url" :href="r.url" target="_blank" rel="noopener noreferrer" class="link-btn">论文主页</a>
                          <a v-if="r.pdf_url" :href="r.pdf_url" target="_blank" rel="noopener noreferrer" class="link-btn pdf">PDF</a>
                          <span v-if="r.citations" class="citation-count">{{ r.citations }} 引用</span>
                        </div>
                      </div>
                    </template>
                    <!-- 普通搜索显示 -->
                    <template v-else>
                      <a :href="r.url" target="_blank" rel="noopener noreferrer">{{ r.title || r.url }}</a>
                      <span v-if="r.snippet" class="snippet">{{ r.snippet }}</span>
                    </template>
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

    <!-- Image lightbox -->
    <Teleport to="body">
      <div v-if="lightboxUrl" class="lightbox-overlay" @click="closeLightbox" @keydown.escape="closeLightbox" role="dialog" aria-modal="true" tabindex="-1">
        <img :src="lightboxUrl" class="lightbox-img" @click.stop alt="放大图片" />
        <button class="lightbox-close" @click="closeLightbox" aria-label="关闭">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>
    </Teleport>

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
          <!-- 学术搜索已内置到 agent 工具中，不再显示开关 -->
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
        aria-label="输入消息"
        aria-multiline="true"
        :aria-disabled="!canSend"
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

// Inject copy button into fenced code blocks
const defaultFenceRender = md.renderer.rules.fence || function (tokens, idx, options, env, self) {
  return self.renderToken(tokens, idx, options)
}
md.renderer.rules.fence = function (tokens, idx, options, env, self) {
  const token = tokens[idx]
  const lang = token.info.trim()
  const langLabel = lang ? `<span class="code-lang-label">${lang}</span>` : ''
  const raw = defaultFenceRender.call(this, tokens, idx, options, env, self)
  // Inject copy button before </pre>
  const copyBtn = `<button class="code-copy-btn" data-code="${encodeURIComponent(token.content)}" title="复制代码">复制</button>`
  return raw.replace('</pre>', `${langLabel}${copyBtn}</pre>`)
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
    // Per-message rendered HTML cache: msgId -> { version, html }
    const renderedCache = new Map()
    const text = ref('')
    const collapsedSearchIds = ref(new Set())
    const lightboxUrl = ref('')
    const messagesRef = ref(null)
    const textareaRef = ref(null)
    const toolPickerRef = ref(null)
    const canSend = ref(false)
    const composing = ref(false)
    const landingGreeting = ref('你好，我是 RobotAgent。有什么可以帮你？')
    const toolMenuOpen = ref(false)
    const enabledTools = ref([])

    const translations = {
      zh: {
        copyMessage: '复制消息',
        copied: '已复制!',
        copyFailed: '复制失败',
        expand: '展开',
        collapse: '收起'
      },
      en: {
        copyMessage: 'Copy message',
        copied: 'Copied!',
        copyFailed: 'Copy failed',
        expand: 'Expand',
        collapse: 'Collapse'
      }
    }

    const lang = ref(localStorage.getItem('robotagent_lang') || 'zh')
    function t (key) {
      return translations[lang.value]?.[key] || translations['zh'][key] || key
    }

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
      while (markdownCache.size > 200) {
        const oldestKey = markdownCache.keys().next().value
        markdownCache.delete(oldestKey)
      }
      return rendered
    }

    // Per-message cached rendering with streaming debounce
    // Caches rendered HTML per msgId; during active streaming debounces re-renders to 300ms
    const pendingRenders = {} // msgId -> timer

    function getCachedRender (msgId, text) {
      const raw = String(text || '')
      if (!raw) return ''
      // Cache hit
      if (renderedCache.has(msgId) && renderedCache.get(msgId).raw === raw) {
        return renderedCache.get(msgId).html
      }
      // Debounce re-renders during streaming (when cache exists but text changed)
      if (renderedCache.has(msgId)) {
        if (pendingRenders[msgId]) return renderedCache.get(msgId).html
        pendingRenders[msgId] = setTimeout(() => {
          const cached = renderedCache.get(msgId)
          if (cached) {
            const freshHtml = renderMarkdown(cached.raw)
            cached.html = freshHtml
          }
          delete pendingRenders[msgId]
        }, 300)
        // Return stale HTML while debouncing
        return renderedCache.get(msgId).html
      }
      // No cache — render immediately
      const html = renderMarkdown(raw)
      renderedCache.set(msgId, { raw, html })
      if (renderedCache.size > 100) {
        const firstKey = renderedCache.keys().next().value
        renderedCache.delete(firstKey)
      }
      return html
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

    // 学术搜索已内置到 agent 工具中，不再需要单独的开关

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
        if (lightboxUrl.value) {
          closeLightbox()
        }
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

    function isErrorMessage (msg) {
      const text = String(msg?.text || '')
      return text.startsWith('[后端错误]') || text.startsWith('[网络错误]') || text.startsWith('[错误]')
    }

    function isSubagent (msg) {
      if (!msg || msg.role !== 'assistant') return false
      return agentKey(msg.agent) !== 'main'
    }

    function copyMessage (m, e) {
      const text = String(m.text || '')
      if (!text) return
      navigator.clipboard.writeText(text).then(() => {
        const btn = e.currentTarget
        btn.classList.add('copied')
        const origHTML = btn.innerHTML
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>'
        setTimeout(() => {
          btn.innerHTML = origHTML
          btn.classList.remove('copied')
        }, 2000)
      }).catch(() => {})
    }

    function toggleSearchCollapse (msgId) {
      const s = new Set(collapsedSearchIds.value)
      if (s.has(msgId)) {
        s.delete(msgId)
      } else {
        s.add(msgId)
      }
      collapsedSearchIds.value = s
    }

    function isSearchCollapsed (m) {
      return collapsedSearchIds.value.has(m.id) && m.webSearchResults.length > 5
    }

    function getDisplayedSearchResults (m) {
      if (!m.webSearchResults || !m.webSearchResults.length) return []
      // If collapsed (has id in set) and more than 5 results, show only first 5
      if (collapsedSearchIds.value.has(m.id) && m.webSearchResults.length > 5) {
        return m.webSearchResults.slice(0, 5)
      }
      return m.webSearchResults
    }

    function handleMarkdownClick (e) {
      const img = e.target.closest('img')
      if (img) {
        e.preventDefault()
        lightboxUrl.value = img.src
        nextTick(() => {
          document.body.style.overflow = 'hidden'
        })
      }
    }

    function truncateThinking (text) {
      const s = String(text || '')
      // Show up to 3 lines, with ellipsis if longer
      const lines = s.split('\n').filter(l => l.trim())
      if (lines.length <= 3) return lines.join('\n')
      return lines.slice(0, 3).join('\n') + '\n…'
    }

    function closeLightbox () {
      lightboxUrl.value = ''
      document.body.style.overflow = ''
    }

    function toggleSubagent (e) {
      // Let the native <details>/<summary> handle it
    }

    function handleCodeCopy (e) {
      const btn = e.target.closest('.code-copy-btn')
      if (!btn) return
      const code = decodeURIComponent(btn.getAttribute('data-code') || '')
      if (!code) return
      navigator.clipboard.writeText(code).then(() => {
        btn.textContent = '已复制!'
        btn.classList.add('copied')
        setTimeout(() => {
          btn.textContent = '复制'
          btn.classList.remove('copied')
        }, 2000)
      }).catch(() => {
        btn.textContent = '失败'
        setTimeout(() => {
          btn.textContent = '复制'
        }, 1500)
      })
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
      document.addEventListener('click', handleCodeCopy)
    })

    onBeforeUnmount(() => {
      if (scrollRaf) cancelAnimationFrame(scrollRaf)
      document.removeEventListener('click', handleGlobalClick)
      document.removeEventListener('keydown', handleGlobalKeydown)
      document.removeEventListener('click', handleCodeCopy)
      for (const tid of Object.values(pendingRenders)) {
        clearTimeout(tid)
      }
      for (const k in pendingRenders) {
        delete pendingRenders[k]
      }
    })

    watch(
      () => props.conversation.map(item => `${item.id}:${item.role}:${String(item.text || '').slice(0, 40)}`).join('|'),
      (next) => {
        const firstAssistant = props.conversation.find(item => String(item?.role || '') === 'assistant' && String(item?.text || '').trim())
        if (firstAssistant) {
          landingGreeting.value = String(firstAssistant.text)
        }
      },
      { immediate: true }
    )

    return {
      text,
      canSend,
      toolMenuOpen,
      landingGreeting,
      send,
      renderMarkdown,
      getCachedRender,
      onKeydown,
      onCompositionStart,
      onCompositionEnd,
      onInput,
      toggleToolMenu,
      agentKey,
      agentName,
      agentIcon,
      isSubagent,
      isErrorMessage,
      messagesRef,
      textareaRef,
      toolPickerRef,
      copyMessage,
      toggleSearchCollapse,
      isSearchCollapsed,
      getDisplayedSearchResults,
      handleMarkdownClick,
      closeLightbox,
      truncateThinking,
      lightboxUrl,
      collapsedSearchIds,
      t
    }
  }
}
</script>
