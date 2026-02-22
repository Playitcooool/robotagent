<template>
  <div class="tool-results">
    <div class="header">工具结果</div>
    <div class="body">
        <div v-if="planningSteps.length" class="plan-wrap">
          <div class="plan-head">
            <h4>执行计划</h4>
            <span class="plan-count">{{ planningSteps.length }} steps</span>
          </div>
          <ol class="plan-list">
            <li
              v-for="(item, idx) in planningSteps"
              :key="item.id || idx"
              :class="['plan-item', { active: item.status === 'in_progress' }]"
            >
              <div class="plan-item-top">
                <span class="plan-index">{{ idx + 1 }}</span>
                <span class="plan-step">{{ item.step }}</span>
              </div>
              <span :class="['plan-status', `status-${item.status}`]">{{ statusLabel(item.status) }}</span>
            </li>
          </ol>
        </div>

        <div v-if="timelineItems.length" class="timeline-wrap">
          <div class="timeline-head">
            <h4>执行时间轴</h4>
            <span class="timeline-count">{{ timelineItems.length }} events</span>
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
    liveFrame: { type: [Object, null], default: null },
    planning: { type: [Object, null], default: null },
    timeline: { type: [Object, null], default: null }
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
    }
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
    statusLabel (status) {
      if (status === 'completed') return 'completed'
      if (status === 'in_progress') return 'in progress'
      return 'pending'
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
  opacity: 0.8;
}

.plan-wrap {
  margin-bottom: 12px;
  border: 1px solid rgba(13, 122, 102, 0.16);
  border-radius: 10px;
  background: linear-gradient(180deg, #fbfffd, #f3fbf8);
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
  border: 1px solid rgba(31, 42, 51, 0.1);
  border-radius: 8px;
  padding: 8px;
  background: rgba(255, 255, 255, 0.75);
}

.plan-item.active {
  border-color: rgba(224, 156, 9, 0.5);
  background: linear-gradient(180deg, #fff7e9, #fff1d9);
  box-shadow: inset 0 0 0 1px rgba(224, 156, 9, 0.18);
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
  background: rgba(13, 122, 102, 0.12);
  color: #0c6f5d;
  flex: 0 0 18px;
}

.plan-step {
  font-size: 13px;
  line-height: 1.35;
}

.plan-status {
  display: inline-flex;
  margin-top: 6px;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 700;
  text-transform: lowercase;
}

.status-pending {
  color: #6f7882;
  background: rgba(111, 120, 130, 0.15);
}

.status-in_progress {
  color: #8a5a05;
  background: rgba(224, 156, 9, 0.2);
}

.status-completed {
  color: #0c6f5d;
  background: rgba(13, 122, 102, 0.18);
}

.timeline-wrap {
  margin-bottom: 12px;
  border: 1px solid rgba(31, 42, 51, 0.12);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.82);
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
  background: #0d7a66;
  box-shadow: 0 0 0 3px rgba(13, 122, 102, 0.12);
}

.timeline-content {
  border: 1px solid rgba(31, 42, 51, 0.1);
  border-radius: 8px;
  padding: 7px 8px;
  background: #fffefc;
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
  color: #6f7882;
  flex: 0 0 auto;
}

.timeline-detail {
  margin-top: 4px;
  font-size: 12px;
  color: #2f3a42;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
