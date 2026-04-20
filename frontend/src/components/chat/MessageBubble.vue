<template>
  <article :class="['message-row', message.role, isSubagent ? 'subagent' : '']">
    <div :class="['bubble-card', isSubagent ? 'subagent-card' : '', isErrorMessage ? 'error-card' : '']" @click="handleBubbleClick">
      <div v-if="message.loading" class="loading-state">
        <div class="loading-title">
          <span class="pulse-dot"></span>
          <span>{{ message.loadingKind === 'search' ? (lang === 'zh' ? '正在检索资料...' : 'Searching references...') : (lang === 'zh' ? '正在执行任务...' : 'Executing task...') }}</span>
        </div>
        <div class="loading-lines">
          <span class="line short"></span>
          <span class="line mid"></span>
          <span class="line long"></span>
        </div>
      </div>

      <template v-else>
        <div v-if="message.role === 'assistant'" class="message-topline">
          <div :class="['agent-chip', `agent-${agentKey}`]">
            <span>{{ agentIcon }}</span>
            <span>{{ agentName }}</span>
          </div>
          <button v-if="message.text" type="button" class="mini-action" @click.stop="copyText(message.text)">
            {{ copied ? (lang === 'zh' ? '已复制' : 'Copied') : (lang === 'zh' ? '复制' : 'Copy') }}
          </button>
        </div>

        <ThinkingTrace v-if="message.role === 'assistant' && message.thinking" :done="Boolean(message.thinkingDone)" />
        <div v-if="message.role === 'assistant' && message.thinking" class="thinking-preview">{{ truncatedThinking }}</div>

        <div
          v-if="message.text"
          class="markdown answer"
          v-html="renderedHtml"
        ></div>

        <section v-if="Array.isArray(message.webSearchResults) && message.webSearchResults.length" class="source-block">
          <header class="source-header">
            <strong>{{ isAcademicSearch ? (lang === 'zh' ? '学术资料' : 'Academic Sources') : (lang === 'zh' ? '搜索出处' : 'Search Sources') }}</strong>
            <button
              v-if="message.webSearchResults.length > 5"
              type="button"
              class="collapse-btn"
              @click.stop="collapsed = !collapsed"
            >
              {{ collapsed ? (lang === 'zh' ? '展开' : 'Expand') : (lang === 'zh' ? '收起' : 'Collapse') }}
            </button>
          </header>
          <ul class="source-list">
            <li v-for="(item, index) in displayedSearchResults" :key="`${message.id}-search-${index}`">
              <template v-if="item.source === 'arxiv' || item.source === 'openalex'">
                <a :href="item.url" target="_blank" rel="noopener noreferrer" class="source-link">{{ item.title }}</a>
                <div class="source-meta">{{ item.authors }} <span v-if="item.year">({{ item.year }})</span></div>
                <p v-if="item.abstract" class="source-snippet">{{ item.abstract }}</p>
              </template>
              <template v-else>
                <a :href="item.url" target="_blank" rel="noopener noreferrer" class="source-link">{{ item.title || item.url }}</a>
                <p v-if="item.snippet" class="source-snippet">{{ item.snippet }}</p>
              </template>
            </li>
          </ul>
        </section>

        <section v-if="Array.isArray(message.ragReferences) && message.ragReferences.length" class="source-block">
          <header class="source-header">
            <strong>{{ lang === 'zh' ? '参考资料' : 'References' }}</strong>
          </header>
          <ul class="source-list compact">
            <li v-for="(item, index) in message.ragReferences" :key="`${message.id}-rag-${index}`">
              <a :href="item.url" target="_blank" rel="noopener noreferrer" class="source-link">{{ item.title || item.url }}</a>
            </li>
          </ul>
        </section>
      </template>
    </div>

    <Teleport to="body">
      <div v-if="lightboxUrl" class="lightbox" @click="closeLightbox">
        <img :src="lightboxUrl" alt="preview" @click.stop />
        <button type="button" class="lightbox-close" @click="closeLightbox">×</button>
      </div>
    </Teleport>
  </article>
</template>

<script>
import { computed, ref } from 'vue'
import MarkdownIt from 'markdown-it'
import markdownItGfm from 'markdown-it-multimd-table'
import markdownItKatex from 'markdown-it-katex'
import hljs from 'highlight.js/lib/core'
import bash from 'highlight.js/lib/languages/bash'
import javascript from 'highlight.js/lib/languages/javascript'
import json from 'highlight.js/lib/languages/json'
import markdown from 'highlight.js/lib/languages/markdown'
import python from 'highlight.js/lib/languages/python'
import xml from 'highlight.js/lib/languages/xml'
import yaml from 'highlight.js/lib/languages/yaml'

import ThinkingTrace from '../ThinkingTrace.vue'
import { resolveAgentKey } from '../../lib/workbench.js'

const SEARCH_RESULTS_COLLAPSE = 5
const renderCache = new Map()

hljs.registerLanguage('bash', bash)
hljs.registerLanguage('shell', bash)
hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('js', javascript)
hljs.registerLanguage('json', json)
hljs.registerLanguage('markdown', markdown)
hljs.registerLanguage('md', markdown)
hljs.registerLanguage('python', python)
hljs.registerLanguage('py', python)
hljs.registerLanguage('xml', xml)
hljs.registerLanguage('html', xml)
hljs.registerLanguage('yaml', yaml)
hljs.registerLanguage('yml', yaml)

const md = new MarkdownIt({
  html: false,
  breaks: true,
  linkify: true,
  typographer: true,
  highlight (source, language) {
    const normalized = String(language || '').trim().toLowerCase()
    const canUseLanguage = normalized && hljs.getLanguage(normalized)
    const highlighted = canUseLanguage
      ? hljs.highlight(source, { language: normalized, ignoreIllegals: true }).value
      : hljs.highlightAuto(source).value

    return `<pre><code class="hljs language-${normalized || 'plain'}">${highlighted}</code></pre>`
  }
})

md.use(markdownItGfm)
md.use(markdownItKatex)

const defaultLinkRender = md.renderer.rules.link_open || function (tokens, idx, options, env, self) {
  return self.renderToken(tokens, idx, options)
}

md.renderer.rules.link_open = function (tokens, idx, options, env, self) {
  const targetIndex = tokens[idx].attrIndex('target')
  if (targetIndex < 0) tokens[idx].attrPush(['target', '_blank'])
  else tokens[idx].attrs[targetIndex][1] = '_blank'

  const relIndex = tokens[idx].attrIndex('rel')
  if (relIndex < 0) tokens[idx].attrPush(['rel', 'noopener noreferrer'])
  else tokens[idx].attrs[relIndex][1] = 'noopener noreferrer'

  return defaultLinkRender(tokens, idx, options, env, self)
}

const defaultFenceRender = md.renderer.rules.fence || function (tokens, idx, options, env, self) {
  return self.renderToken(tokens, idx, options)
}

md.renderer.rules.fence = function (tokens, idx, options, env, self) {
  const token = tokens[idx]
  const lang = token.info.trim()
  const label = lang ? `<span class="code-lang-label">${lang}</span>` : ''
  const raw = defaultFenceRender.call(this, tokens, idx, options, env, self)
  const copyBtn = `<button class="code-copy-btn" data-code="${encodeURIComponent(token.content)}">${lang ? 'Copy' : 'Copy'}</button>`
  return raw.replace('</pre>', `${label}${copyBtn}</pre>`)
}

function preprocessMarkdown(raw) {
  let text = String(raw || '')
  if (text.includes('\\n') && !text.includes('\n')) {
    text = text.replace(/\\n/g, '\n').replace(/\\t/g, '\t')
  }

  return text
    .split(/(```[\s\S]*?```)/g)
    .map((part) => {
      if (part.startsWith('```')) return part
      return part
        .replace(/\\\(([\s\S]*?)\\\)/g, (_m, expr) => `$${String(expr).trim()}$`)
        .replace(/\\\[([\s\S]*?)\\\]/g, (_m, expr) => `$$\n${String(expr).trim()}\n$$`)
    })
    .join('')
}

function renderMarkdown(content) {
  const raw = String(content || '')
  if (!raw) return ''
  if (renderCache.has(raw)) return renderCache.get(raw)
  const html = md.render(preprocessMarkdown(raw))
  renderCache.set(raw, html)
  if (renderCache.size > 200) {
    const firstKey = renderCache.keys().next().value
    renderCache.delete(firstKey)
  }
  return html
}

export default {
  name: 'MessageBubble',
  components: { ThinkingTrace },
  props: {
    message: { type: Object, required: true },
    lang: { type: String, default: 'zh' }
  },
  setup (props) {
    const copied = ref(false)
    const collapsed = ref((props.message?.webSearchResults || []).length > SEARCH_RESULTS_COLLAPSE)
    const lightboxUrl = ref('')

    const agentKey = computed(() => resolveAgentKey(props.message?.agent))
    const isSubagent = computed(() => props.message?.role === 'assistant' && agentKey.value !== 'main')
    const isErrorMessage = computed(() => {
      const text = String(props.message?.text || '')
      return text.startsWith('[后端错误]') || text.startsWith('[网络错误]') || text.startsWith('[错误]')
    })
    const agentName = computed(() => {
      if (agentKey.value === 'simulator') return 'Simulator Agent'
      if (agentKey.value === 'analysis') return 'Analysis Agent'
      return props.lang === 'zh' ? '主代理' : 'Main Agent'
    })
    const agentIcon = computed(() => {
      if (agentKey.value === 'simulator') return '🛠'
      if (agentKey.value === 'analysis') return '📊'
      return '🧠'
    })
    const truncatedThinking = computed(() => {
      const lines = String(props.message?.thinking || '').split('\n').filter(Boolean)
      if (lines.length <= 3) return lines.join('\n')
      return `${lines.slice(0, 3).join('\n')}\n…`
    })
    const renderedHtml = computed(() => renderMarkdown(props.message?.text || ''))
    const displayedSearchResults = computed(() => {
      const results = Array.isArray(props.message?.webSearchResults) ? props.message.webSearchResults : []
      if (collapsed.value && results.length > SEARCH_RESULTS_COLLAPSE) {
        return results.slice(0, SEARCH_RESULTS_COLLAPSE)
      }
      return results
    })
    const isAcademicSearch = computed(() => {
      const first = props.message?.webSearchResults?.[0]
      return first?.source === 'arxiv' || first?.source === 'openalex'
    })

    async function copyText(text) {
      if (!text) return
      try {
        await navigator.clipboard.writeText(String(text))
        copied.value = true
        setTimeout(() => { copied.value = false }, 1500)
      } catch (_) {
      }
    }

    function closeLightbox() {
      lightboxUrl.value = ''
      document.body.style.overflow = ''
    }

    function handleBubbleClick(event) {
      const copyButton = event.target.closest('.code-copy-btn')
      if (copyButton) {
        copyText(decodeURIComponent(copyButton.getAttribute('data-code') || ''))
        return
      }

      const image = event.target.closest('img')
      if (image) {
        lightboxUrl.value = image.src
        document.body.style.overflow = 'hidden'
      }
    }

    return {
      copied,
      collapsed,
      lightboxUrl,
      agentKey,
      isSubagent,
      isErrorMessage,
      agentName,
      agentIcon,
      truncatedThinking,
      renderedHtml,
      displayedSearchResults,
      isAcademicSearch,
      copyText,
      handleBubbleClick,
      closeLightbox
    }
  }
}
</script>

<style scoped>
.message-row {
  display: flex;
}

.message-row.user {
  justify-content: flex-end;
}

.bubble-card {
  width: min(78ch, 100%);
  padding: 16px 18px;
  border-radius: 22px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.035), rgba(255, 255, 255, 0.015)),
    rgba(11, 16, 24, 0.92);
  box-shadow: 0 16px 30px rgba(0, 0, 0, 0.18);
}

.message-row.user .bubble-card {
  background:
    linear-gradient(180deg, rgba(47, 125, 255, 0.17), rgba(47, 125, 255, 0.08)),
    rgba(14, 23, 36, 0.92);
  border-color: rgba(95, 156, 255, 0.35);
}

.subagent-card {
  width: min(64ch, 100%);
}

.error-card {
  border-color: rgba(255, 107, 107, 0.28);
}

.message-topline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.agent-chip,
.mini-action {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
}

.agent-chip {
  padding: 4px 10px;
}

.agent-main {
  color: #cfe0ff;
  background: rgba(67, 143, 255, 0.17);
}

.agent-analysis {
  color: #bdf2d7;
  background: rgba(32, 196, 120, 0.16);
}

.agent-simulator {
  color: #ffd8b7;
  background: rgba(255, 159, 90, 0.16);
}

.mini-action {
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.04);
  color: var(--muted);
  padding: 6px 9px;
  cursor: pointer;
}

.loading-state {
  display: grid;
  gap: 12px;
}

.loading-title {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--muted);
  font-size: 13px;
}

.pulse-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #56a3ff;
  box-shadow: 0 0 0 rgba(86, 163, 255, 0.5);
  animation: pulse 1.25s infinite;
}

.loading-lines {
  display: grid;
  gap: 8px;
}

.line {
  display: block;
  height: 8px;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0.06));
  background-size: 200% 100%;
  animation: shimmer 1.2s linear infinite;
}

.line.short { width: 42%; }
.line.mid { width: 64%; }
.line.long { width: 86%; }

.thinking-preview {
  margin: 0 0 12px;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.035);
  color: var(--muted);
  white-space: pre-wrap;
  font-size: 12px;
  line-height: 1.55;
}

.source-block {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.source-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
  font-size: 12px;
}

.collapse-btn {
  border: none;
  background: transparent;
  color: #8fb7ff;
  cursor: pointer;
  font-size: 12px;
}

.source-list {
  margin: 0;
  padding-left: 18px;
  display: grid;
  gap: 10px;
}

.source-list.compact {
  gap: 6px;
}

.source-link {
  color: #8fb7ff;
  text-decoration: none;
}

.source-link:hover {
  text-decoration: underline;
}

.source-snippet,
.source-meta {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.55;
}

.lightbox {
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: grid;
  place-items: center;
  background: rgba(0, 0, 0, 0.76);
}

.lightbox img {
  max-width: min(88vw, 1080px);
  max-height: 82vh;
  border-radius: 16px;
}

.lightbox-close {
  position: absolute;
  top: 22px;
  right: 22px;
  border: none;
  background: rgba(255, 255, 255, 0.12);
  color: white;
  width: 40px;
  height: 40px;
  border-radius: 999px;
  cursor: pointer;
  font-size: 22px;
}

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(86, 163, 255, 0.48); }
  70% { box-shadow: 0 0 0 10px rgba(86, 163, 255, 0); }
  100% { box-shadow: 0 0 0 0 rgba(86, 163, 255, 0); }
}

@keyframes shimmer {
  from { background-position: 200% 0; }
  to { background-position: -200% 0; }
}
</style>
