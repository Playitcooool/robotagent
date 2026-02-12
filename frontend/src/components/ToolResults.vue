<template>
  <div class="tool-results">
    <div class="header">工具结果</div>
    <div class="body">
        <div v-if="!result && !liveFrame" class="empty">无工具执行信息</div>

        <div v-if="liveFrame && liveFrame.image_url" class="image-wrap">
          <img :src="liveFrame.image_url" alt="pybullet live frame" />
          <div class="live-meta">
            <span>{{ liveFrame.task || 'simulation' }}</span>
            <span v-if="typeof liveFrame.step === 'number'">step {{ liveFrame.step }}/{{ liveFrame.total_steps || '?' }}</span>
            <span>{{ liveFrame.done ? 'done' : 'running' }}</span>
          </div>
        </div>

        <div v-else>
          <div v-if="isImage(result)" class="image-wrap">
            <img :src="getImageUrl(result)" alt="tool image" />
          </div>

          <!-- render markdown for text results so tables/code/math are supported -->
          <div v-if="isText(result)" class="markdown" v-html="renderMarkdown(result)"></div>

          <!-- support list of items or structured JSON -->
          <div v-if="isJson(result)">
            <h4>数据</h4>
            <pre>{{ pretty(result) }}</pre>
          </div>
        </div>
      </div>
  </div>
</template>

<script>
import MarkdownIt from 'markdown-it'
import markdownItGfm from 'markdown-it-multimd-table'
import markdownItKatex from 'markdown-it-katex'
import markdownItHighlightjs from 'markdown-it-highlightjs'
import hljs from 'highlight.js'

const md = new MarkdownIt({ html: false, breaks: true, linkify: true, typographer: true })
md.use(markdownItGfm)
md.use(markdownItKatex)
md.use(markdownItHighlightjs, { auto: true, hljs })

export default {
  name: 'ToolResults',
  props: {
    result: { type: [Object, String, null], default: null },
    liveFrame: { type: [Object, null], default: null }
  },
  methods: {
    isImage (r) {
      if (!r) return false
      if (typeof r === 'string') return r.match(/https?:.*\.(png|jpg|jpeg|gif|svg)/i)
      if (typeof r === 'object') return r.image || r.image_url || r.url
      return false
    },
    getImageUrl (r) {
      if (typeof r === 'string') return r
      return r.image || r.image_url || r.url
    },
    isText (r) { return typeof r === 'string' },
    isJson (r) { return typeof r === 'object' },
    pretty (r) {
      try { return typeof r === 'string' ? r : JSON.stringify(r, null, 2) } catch (e) { return String(r) }
    },
    renderMarkdown (content) {
      return md.render(content || '')
    }
  }
}
</script>

<style scoped>
.live-meta {
  margin-top: 8px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 12px;
  opacity: 0.8;
}
</style>
