<template>
  <div class="tool-results">
    <div class="header">工具结果</div>
    <div class="body">
        <div v-if="hasUsage" class="usage-wrap">
          <div class="usage-head">
            <h4>Token 用量</h4>
            <span class="usage-time">{{ usageTime }}</span>
          </div>
          <div class="usage-grid">
            <div class="usage-cell">
              <span class="k">Prompt</span>
              <strong class="v">{{ tokenUsage.prompt_tokens }}</strong>
            </div>
            <div class="usage-cell">
              <span class="k">Completion</span>
              <strong class="v">{{ tokenUsage.completion_tokens }}</strong>
            </div>
            <div class="usage-cell total">
              <span class="k">Total</span>
              <strong class="v">{{ tokenUsage.total_tokens }}</strong>
            </div>
          </div>
        </div>

        <div v-if="planningSteps.length" class="plan-wrap">
          <div class="plan-head">
            <h4>执行计划</h4>
            <span class="plan-count">{{ planningSteps.length }} 步</span>
          </div>
          <TransitionGroup tag="ol" class="plan-list" name="plan">
            <li
              v-for="(item, idx) in planningSteps"
              :key="item.id || idx"
              :class="['plan-item', item.status === 'in_progress' ? 'active' : item.status === 'completed' ? 'done' : '']"
            >
              <div class="plan-item-top">
                <span :class="['plan-index', item.status === 'completed' ? 'index-done' : '']">
                  <template v-if="item.status === 'completed'">✓</template>
                  <template v-else>{{ idx + 1 }}</template>
                </span>
                <span class="plan-step">{{ item.step }}</span>
              </div>
              <span :class="['plan-status', `status-${item.status}`]">{{ statusLabel(item.status) }}</span>
            </li>
          </TransitionGroup>
        </div>

        <div v-if="timelineItems.length" class="timeline-wrap">
          <div class="timeline-head">
            <h4>执行时间轴</h4>
            <span class="timeline-count">{{ timelineItems.length }} 条</span>
          </div>
          <ul class="timeline-list">
            <li v-for="(item, idx) in timelineItems" :key="`${idx}-${item.timestamp}`" class="timeline-item">
              <div class="timeline-dot"></div>
              <div class="timeline-content">
                <div class="timeline-top">
                  <span class="timeline-title">{{ item.title }}</span>
                  <span class="timeline-time">{{ formatTime(item.timestamp) }}</span>
                </div>
                <div v-if="item.detail" class="timeline-detail">{{ item.detail }}</div>
              </div>
            </li>
          </ul>
        </div>

        <div v-if="!planningSteps.length && !timelineItems.length && !result && !liveFrame" class="empty">无工具执行信息</div>

        <div v-if="liveFrame && liveFrame.image_url" class="image-wrap">
          <img :src="liveFrame.image_url" alt="pybullet live frame" />
          <div class="live-meta">
            <span>{{ liveFrame.task || 'simulation' }}</span>
            <span v-if="typeof liveFrame.step === 'number'">step {{ liveFrame.step }}/{{ liveFrame.total_steps || '?' }}</span>
            <span>{{ liveFrame.done ? '完成' : '运行中' }}</span>
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
    liveFrame: { type: [Object, null], default: null },
    planning: { type: [Object, null], default: null },
    timeline: { type: [Object, null], default: null },
    tokenUsage: { type: [Object, null], default: null }
  },
  computed: {
    planningSteps () {
      const steps = this.planning?.steps
      return Array.isArray(steps) ? steps : []
    },
    timelineItems () {
      const items = this.timeline?.items
      if (!Array.isArray(items)) return []
      return [...items].reverse()
    },
    hasUsage () {
      return Number(this.tokenUsage?.total_tokens || 0) > 0
    },
    usageTime () {
      const ts = Number(this.tokenUsage?.updatedAt || 0)
      const ms = ts * 1000
      if (!Number.isFinite(ms) || ms <= 0) return ''
      return new Date(ms).toLocaleTimeString()
    }
  },
  methods: {
    isImage (r) {
      if (!r) return false
      if (typeof r === 'string') {
        if (r.startsWith('data:image/')) return true
        if (r.startsWith('blob:')) return true
        return r.match(/\.(png|jpg|jpeg|gif|svg)(\?|#|$)/i) || r.startsWith('/api/')
      }
      if (typeof r === 'object') return r.image || r.image_url || r.url
      return false
    },
    getImageUrl (r) {
      if (typeof r === 'string') return r
      return r.image || r.image_url || r.url
    },
    isText (r) { return typeof r === 'string' && !this.isImage(r) },
    isJson (r) { return r !== null && typeof r === 'object' && !this.isImage(r) },
    pretty (r) {
      try { return typeof r === 'string' ? r : JSON.stringify(r, null, 2) } catch (e) { return String(r) }
    },
    statusLabel (status) {
      if (status === 'completed') return '已完成'
      if (status === 'in_progress') return '进行中'
      return '待执行'
    },
    formatTime (timestamp) {
      const ms = Number(timestamp || 0) * 1000
      if (!Number.isFinite(ms) || ms <= 0) return '--:--:--'
      return new Date(ms).toLocaleTimeString()
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
  color: #9aa4b2;
}

.usage-wrap {
  margin-bottom: 12px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  background: #101521;
  padding: 10px;
}

.usage-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 8px;
}

.usage-head h4 {
  margin: 0;
  font-size: 14px;
}

.usage-time {
  font-size: 11px;
  color: #9aa4b2;
}

.usage-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.usage-cell {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 10px;
  background: #0f131b;
  padding: 8px;
}

.usage-cell .k {
  display: block;
  font-size: 11px;
  color: #9aa4b2;
}

.usage-cell .v {
  display: block;
  margin-top: 3px;
  font-size: 16px;
}

.usage-cell.total {
  border-color: rgba(47, 125, 255, 0.4);
  background: #122033;
}

.plan-wrap {
  margin-bottom: 12px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  background: #101521;
  padding: 10px;
}

.plan-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 8px;
}

.plan-head h4 {
  margin: 0;
  font-size: 14px;
}

.plan-count {
  font-size: 12px;
  opacity: 0.7;
}

.plan-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: grid;
  gap: 8px;
}

.plan-item {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 10px;
  padding: 8px;
  background: #0f131b;
  transition: background 0.25s, border-color 0.25s, opacity 0.3s;
}

.plan-item.done {
  opacity: 0.55;
}

.plan-item.active {
  border-color: rgba(47, 125, 255, 0.5);
  background: #122033;
}

.plan-item-top {
  display: flex;
  gap: 8px;
  align-items: flex-start;
}

.plan-index {
  width: 18px;
  height: 18px;
  border-radius: 999px;
  display: inline-grid;
  place-items: center;
  font-size: 11px;
  font-weight: 700;
  background: rgba(47, 125, 255, 0.18);
  color: #9fc1ff;
  flex: 0 0 18px;
  transition: background 0.25s, color 0.25s;
}

.plan-index.index-done {
  background: rgba(32, 196, 120, 0.2);
  color: #a5f3c7;
}

.plan-step {
  font-size: 13px;
  line-height: 1.35;
  word-break: break-word;
}

.plan-status {
  display: inline-flex;
  margin-top: 6px;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 700;
}

.status-pending {
  color: #9aa4b2;
  background: rgba(255, 255, 255, 0.08);
}

.status-in_progress {
  color: #9fc1ff;
  background: rgba(47, 125, 255, 0.2);
}

.status-completed {
  color: #a5f3c7;
  background: rgba(32, 196, 120, 0.2);
}

.timeline-wrap {
  margin-bottom: 12px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  background: #101521;
  padding: 10px;
}

.timeline-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 8px;
}

.timeline-head h4 {
  margin: 0;
  font-size: 14px;
}

.timeline-count {
  font-size: 12px;
  opacity: 0.7;
}

.timeline-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: grid;
  gap: 8px;
}

.timeline-item {
  display: grid;
  grid-template-columns: 12px 1fr;
  gap: 8px;
  align-items: start;
}

.timeline-dot {
  width: 8px;
  height: 8px;
  margin-top: 6px;
  border-radius: 50%;
  background: #2f7dff;
  box-shadow: 0 0 0 3px rgba(47, 125, 255, 0.18);
}

.timeline-content {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 10px;
  padding: 7px 8px;
  background: #0f131b;
}

.timeline-top {
  display: flex;
  justify-content: space-between;
  gap: 8px;
}

.timeline-title {
  font-size: 12px;
  font-weight: 700;
}

.timeline-time {
  font-size: 11px;
  color: #9aa4b2;
  flex: 0 0 auto;
}

.timeline-detail {
  margin-top: 4px;
  font-size: 12px;
  color: #9aa4b2;
  white-space: pre-wrap;
  word-break: break-word;
}

.plan-enter-active {
  transition: opacity 0.22s ease, transform 0.22s ease;
}

.plan-enter-from {
  opacity: 0;
  transform: translateY(-5px);
}
</style>
