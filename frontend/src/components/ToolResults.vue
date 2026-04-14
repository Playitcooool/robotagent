<template>
  <div class="tool-results">
    <div class="header">
      <div>
        <div class="title">工具执行结果</div>
        <div class="subtitle">实时展示规划、工具状态与仿真输出</div>
      </div>
      <div class="summary">
        <span v-if="activeTasks.length" class="pill active">{{ activeTasks.length }} 运行中</span>
        <span v-if="planningSteps.length" class="pill">{{ completedPlanningCount }}/{{ planningSteps.length }} 计划</span>
        <span v-if="latestSearchResults.length" class="pill">{{ latestSearchResults.length }} 搜索结果</span>
        <span v-if="latestRagReferences.length" class="pill">{{ latestRagReferences.length }} RAG</span>
      </div>
    </div>

    <div class="body">
      <section v-if="activeTasks.length" class="section">
        <div class="section-title">工具状态</div>
        <div class="status-list">
          <div v-for="task in activeTasks" :key="task.id" class="status-card">
            <div class="status-top">
              <span :class="['agent-badge', task.agentKey]">{{ task.agentLabel }}</span>
              <span class="status-kind">{{ task.loadingKindLabel }}</span>
            </div>
            <div class="status-text">{{ task.statusText }}</div>
          </div>
        </div>
      </section>

      <section v-if="planningSteps.length" class="section">
        <div class="section-title">执行计划</div>
        <ol class="plan-list">
          <li v-for="(item, idx) in planningSteps" :key="item.id || idx" :class="['plan-item', item.status]">
            <span class="plan-index">{{ item.status === 'completed' ? '✓' : idx + 1 }}</span>
            <span class="plan-step">{{ item.step }}</span>
            <span class="plan-status">{{ planStatusLabel(item.status) }}</span>
          </li>
        </ol>
      </section>

      <section v-if="latestSearchResults.length" class="section">
        <div class="section-title">搜索结果</div>
        <ul class="link-list">
          <li v-for="(item, idx) in latestSearchResults" :key="`search-${idx}`">
            <a :href="item.url" target="_blank" rel="noopener noreferrer">{{ item.title || item.url }}</a>
            <div v-if="item.snippet" class="snippet">{{ item.snippet }}</div>
          </li>
        </ul>
      </section>

      <section v-if="latestRagReferences.length" class="section">
        <div class="section-title">参考资料</div>
        <ul class="link-list">
          <li v-for="(item, idx) in latestRagReferences" :key="`rag-${idx}`">
            <a :href="item.url" target="_blank" rel="noopener noreferrer">{{ item.title || item.url }}</a>
          </li>
        </ul>
      </section>

      <section v-if="toolOutputs.length" class="section">
        <div class="section-title">工具输出摘要</div>
        <div class="output-list">
          <div v-for="item in toolOutputs" :key="item.id" class="output-card">
            <div class="status-top">
              <span :class="['agent-badge', item.agentKey]">{{ item.agentLabel }}</span>
              <span class="status-kind">{{ item.outputLabel }}</span>
            </div>
            <pre class="output-text">{{ item.outputText }}</pre>
          </div>
        </div>
      </section>

      <section v-if="liveFrame && liveFrame.image_url" class="section">
        <div class="section-title">仿真画面</div>
        <div class="image-wrap">
          <div v-if="imgLoading" class="img-loading">
            <span>加载中...</span>
          </div>
          <div v-if="imgError" class="img-error">
            <span>图片加载失败</span>
            <button class="retry-btn" @click="retryImage" aria-label="重新加载图片">重试</button>
          </div>
          <img
            ref="imgRef"
            :src="liveFrame.image_url"
            :alt="liveFrame.task ? `仿真画面: ${liveFrame.task}` : '仿真画面'"
            @load="onImgLoad"
            @error="onImgError"
            :class="{ hidden: imgLoading || imgError }"
          />
          <div class="live-meta">
            <span>{{ liveFrame.task || 'simulation' }}</span>
            <span v-if="typeof liveFrame.step === 'number'">
              step {{ liveFrame.step }}/{{ liveFrame.total_steps || '?' }}
            </span>
            <span>{{ liveFrame.done ? '完成' : '运行中' }}</span>
          </div>
        </div>
      </section>

      <div v-if="!hasContent" class="empty">当前没有可展示的工具执行结果。</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ToolResults',
  props: {
    liveFrame: { type: [Object, null], default: null },
    planning: { type: [Object, null], default: null },
    conversation: { type: Array, default: () => [] }
  },
  data () {
    return {
      imgLoading: true,
      imgError: false,
      currentImageUrl: null
    }
  },
  computed: {
    planningSteps () {
      return Array.isArray(this.planning?.steps) ? this.planning.steps : []
    },
    completedPlanningCount () {
      return this.planningSteps.filter(item => item?.status === 'completed').length
    },
    assistantMessages () {
      return (Array.isArray(this.conversation) ? this.conversation : [])
        .filter(item => String(item?.role || '') === 'assistant')
    },
    activeTasks () {
      return this.assistantMessages
        .filter(item => item?.loading)
        .slice(-3)
        .reverse()
        .map(item => ({
          id: item.id,
          agentKey: this.agentKey(item.agent),
          agentLabel: this.agentLabel(item.agent),
          loadingKindLabel: item.loadingKind === 'search' ? '搜索中' : '执行中',
          statusText: String(item.text || '').trim() || '工具正在处理中...'
        }))
    },
    latestSearchResults () {
      const msg = [...this.assistantMessages]
        .reverse()
        .find(item => Array.isArray(item?.webSearchResults) && item.webSearchResults.length)
      return Array.isArray(msg?.webSearchResults) ? msg.webSearchResults.slice(0, 6) : []
    },
    latestRagReferences () {
      const msg = [...this.assistantMessages]
        .reverse()
        .find(item => Array.isArray(item?.ragReferences) && item.ragReferences.length)
      return Array.isArray(msg?.ragReferences) ? msg.ragReferences.slice(0, 6) : []
    },
    toolOutputs () {
      return this.assistantMessages
        .filter((item) => {
          const key = this.agentKey(item.agent)
          return key !== 'main' && !item.loading && String(item.text || '').trim()
        })
        .slice(-4)
        .reverse()
        .map(item => ({
          id: item.id,
          agentKey: this.agentKey(item.agent),
          agentLabel: this.agentLabel(item.agent),
          outputLabel: item.thinking ? '工具回复' : '最新输出',
          outputText: this.truncateText(item.text, 600)
        }))
    },
    hasContent () {
      return Boolean(
        this.activeTasks.length ||
        this.planningSteps.length ||
        this.latestSearchResults.length ||
        this.latestRagReferences.length ||
        this.toolOutputs.length ||
        (this.liveFrame && this.liveFrame.image_url)
      )
    }
  },
  watch: {
    liveFrame: {
      handler (newVal) {
        if (newVal && newVal.image_url) {
          if (newVal.image_url !== this.currentImageUrl) {
            this.currentImageUrl = newVal.image_url
            this.imgLoading = true
            this.imgError = false
          }
        } else {
          this.currentImageUrl = null
          this.imgLoading = true
          this.imgError = false
        }
      },
      immediate: true
    }
  },
  methods: {
    agentKey (agent) {
      const value = String(agent || '').toLowerCase()
      if (value === 'simulator') return 'simulator'
      if (value === 'analysis' || value === 'data-analyzer' || value === 'data_analyzer') return 'analysis'
      return 'main'
    },
    agentLabel (agent) {
      const key = this.agentKey(agent)
      if (key === 'simulator') return 'Simulator'
      if (key === 'analysis') return 'Analysis'
      return 'Main'
    },
    planStatusLabel (status) {
      if (status === 'completed') return '已完成'
      if (status === 'in_progress') return '进行中'
      return '待执行'
    },
    truncateText (text, maxLen = 600) {
      const value = String(text || '').trim()
      if (value.length <= maxLen) return value
      return value.slice(0, maxLen) + '...'
    },
    onImgLoad () {
      this.imgLoading = false
      this.imgError = false
    },
    onImgError () {
      this.imgLoading = false
      this.imgError = true
    },
    retryImage () {
      this.imgLoading = true
      this.imgError = false
      const img = this.$refs.imgRef
      if (img && this.currentImageUrl) {
        const separator = this.currentImageUrl.includes('?') ? '&' : '?'
        img.src = this.currentImageUrl + separator + 'retry=' + Date.now()
      }
    }
  }
}
</script>

<style scoped>
.tool-results {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
  align-items: flex-start;
}

.title {
  font-size: 15px;
  font-weight: 700;
}

.subtitle {
  margin-top: 4px;
  font-size: 12px;
  color: var(--muted);
}

.summary {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-content: flex-start;
  justify-content: flex-end;
}

.pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 11px;
  color: var(--muted);
  background: rgba(255, 255, 255, 0.04);
}

.pill.active {
  color: #dce7ff;
  background: rgba(47, 125, 255, 0.14);
  border-color: rgba(47, 125, 255, 0.35);
}

.body {
  flex: 1;
  min-height: 0;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 12px;
  background:
    radial-gradient(560px 220px at 50% 0%, rgba(47, 125, 255, 0.07), transparent 58%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.01));
  padding: 14px;
}

.section + .section {
  margin-top: 16px;
}

.section-title {
  margin-bottom: 8px;
  font-size: 12px;
  font-weight: 700;
  color: var(--muted);
  letter-spacing: 0.02em;
  text-transform: uppercase;
}

.status-list,
.output-list {
  display: grid;
  gap: 8px;
}

.status-card,
.output-card {
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.015)),
    rgba(255, 255, 255, 0.03);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.status-top {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.agent-badge {
  border-radius: 999px;
  padding: 3px 8px;
  font-size: 11px;
  font-weight: 700;
}

.agent-badge.main {
  background: rgba(255, 255, 255, 0.08);
  color: var(--text);
}

.agent-badge.simulator {
  background: rgba(47, 125, 255, 0.18);
  color: #b9d0ff;
}

.agent-badge.analysis {
  background: rgba(32, 196, 120, 0.18);
  color: #9ff0c3;
}

.status-kind,
.plan-status {
  margin-left: auto;
  font-size: 11px;
  color: var(--muted);
}

.status-text,
.plan-step,
.snippet {
  font-size: 13px;
  line-height: 1.5;
  color: var(--text);
  word-break: break-word;
}

.plan-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: grid;
  gap: 8px;
}

.plan-item {
  display: grid;
  grid-template-columns: 24px 1fr auto;
  gap: 8px;
  align-items: start;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px 12px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.015)),
    rgba(255, 255, 255, 0.03);
}

.plan-index {
  width: 24px;
  height: 24px;
  display: inline-grid;
  place-items: center;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  font-size: 12px;
  font-weight: 700;
}

.plan-item.completed .plan-index {
  background: rgba(32, 196, 120, 0.22);
  color: #a5f3c7;
}

.link-list {
  margin: 0;
  padding-left: 18px;
  display: grid;
  gap: 12px;
}

.link-list a {
  color: var(--accent);
  text-decoration: none;
  word-break: break-word;
}

.link-list a:hover {
  text-decoration: underline;
}

.output-text {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  line-height: 1.5;
  color: var(--text);
}

.image-wrap img {
  width: 100%;
  height: auto;
  border-radius: 12px;
  border: 1px solid var(--line);
  box-shadow: 0 14px 26px rgba(0, 0, 0, 0.18);
}

.image-wrap img.hidden {
  display: none;
}

.img-loading,
.img-error {
  width: 100%;
  aspect-ratio: 4 / 3;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.16);
  border-radius: 10px;
  border: 1px solid var(--line);
  font-size: 13px;
  color: var(--muted);
}

.img-error {
  flex-direction: column;
  gap: 10px;
}

.retry-btn {
  padding: 4px 12px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.1);
  color: var(--text);
  border: 1px solid var(--line);
  cursor: pointer;
  font-size: 12px;
}

.retry-btn:hover {
  background: rgba(255, 255, 255, 0.15);
}

.live-meta {
  margin-top: 8px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 12px;
  color: var(--muted);
}

.empty {
  font-size: 13px;
  color: var(--muted);
}

@media (max-width: 900px) {
  .header {
    flex-direction: column;
  }

  .summary {
    justify-content: flex-start;
  }
}
</style>
