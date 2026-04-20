<template>
  <div class="results-shell">
    <header class="results-header">
      <div>
        <p class="eyebrow">Execution Rail</p>
        <h2>{{ lang === 'zh' ? '执行证据与状态回放' : 'Execution Evidence & State Replay' }}</h2>
      </div>
      <div class="summary-pills">
        <span v-if="activeTasks.length" class="pill active">{{ activeTasks.length }} {{ lang === 'zh' ? '运行中' : 'live' }}</span>
        <span v-if="planningSteps.length" class="pill">{{ completedPlanningCount }}/{{ planningSteps.length }} {{ lang === 'zh' ? '计划' : 'plan' }}</span>
        <span v-if="latestSearchResults.length" class="pill">{{ latestSearchResults.length }} {{ lang === 'zh' ? '出处' : 'sources' }}</span>
      </div>
    </header>

    <div class="results-body">
      <section v-if="activeTasks.length" class="rail-section">
        <div class="section-heading">{{ lang === 'zh' ? '工具状态' : 'Tool Activity' }}</div>
        <div class="card-grid">
          <article v-for="task in activeTasks" :key="task.id" class="rail-card">
            <div class="card-top">
              <span :class="['agent-chip', task.agentKey]">{{ task.agentLabel }}</span>
              <span class="card-status">{{ task.loadingKindLabel }}</span>
            </div>
            <p>{{ task.statusText }}</p>
          </article>
        </div>
      </section>

      <section v-if="planningSteps.length" class="rail-section">
        <div class="section-heading">{{ lang === 'zh' ? '执行计划' : 'Plan Trace' }}</div>
        <ol class="plan-list">
          <li v-for="(item, index) in planningSteps" :key="item.id || index" :class="['plan-item', item.status]">
            <span class="plan-index">{{ item.status === 'completed' ? '✓' : index + 1 }}</span>
            <div class="plan-copy">
              <strong>{{ item.step }}</strong>
              <small>{{ planStatusLabel(item.status) }}</small>
            </div>
          </li>
        </ol>
      </section>

      <section v-if="latestSearchResults.length || latestRagReferences.length" class="rail-section">
        <div class="section-heading">{{ lang === 'zh' ? '证据来源' : 'Evidence Sources' }}</div>
        <div class="stack-list">
          <article v-for="(item, index) in latestSearchResults" :key="`search-${index}`" class="source-card">
            <a :href="item.url" target="_blank" rel="noopener noreferrer">{{ item.title || item.url }}</a>
            <p v-if="item.snippet">{{ item.snippet }}</p>
          </article>
          <article v-for="(item, index) in latestRagReferences" :key="`rag-${index}`" class="source-card compact">
            <a :href="item.url" target="_blank" rel="noopener noreferrer">{{ item.title || item.url }}</a>
          </article>
        </div>
      </section>

      <section v-if="toolOutputs.length" class="rail-section">
        <div class="section-heading">{{ lang === 'zh' ? '子代理输出' : 'Subagent Output' }}</div>
        <div class="stack-list">
          <article v-for="item in toolOutputs" :key="item.id" class="rail-card">
            <div class="card-top">
              <span :class="['agent-chip', item.agentKey]">{{ item.agentLabel }}</span>
              <span class="card-status">{{ item.outputLabel }}</span>
            </div>
            <pre>{{ item.outputText }}</pre>
          </article>
        </div>
      </section>

      <section v-if="liveFrame?.image_url" class="rail-section">
        <div class="section-heading">{{ lang === 'zh' ? '仿真画面' : 'Simulation Feed' }}</div>
        <div class="frame-card">
          <div v-if="imgLoading" class="frame-placeholder">{{ lang === 'zh' ? '画面加载中...' : 'Loading frame...' }}</div>
          <div v-else-if="imgError" class="frame-placeholder error">
            <span>{{ lang === 'zh' ? '画面加载失败' : 'Frame failed to load' }}</span>
            <button type="button" @click="retryImage">{{ lang === 'zh' ? '重试' : 'Retry' }}</button>
          </div>
          <img
            v-show="!imgLoading && !imgError"
            ref="imgRef"
            :src="liveFrame.image_url"
            :alt="liveFrame.task || 'simulation frame'"
            @load="onImgLoad"
            @error="onImgError"
          />
          <div class="frame-meta">
            <span>{{ liveFrame.task || 'simulation' }}</span>
            <span v-if="typeof liveFrame.step === 'number'">step {{ liveFrame.step }}/{{ liveFrame.total_steps || '?' }}</span>
            <span>{{ liveFrame.done ? (lang === 'zh' ? '完成' : 'Done') : (lang === 'zh' ? '运行中' : 'Live') }}</span>
          </div>
        </div>
      </section>

      <div v-if="!hasContent" class="empty-state">
        <strong>{{ lang === 'zh' ? '等待执行证据' : 'Awaiting execution evidence' }}</strong>
        <p>{{ lang === 'zh' ? '规划、搜索结果、子代理输出和仿真帧会在这里按时间汇总。' : 'Plans, sources, subagent output, and simulation frames will collect here as the mission runs.' }}</p>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'

import { deriveResultRailData } from '../lib/workbench.js'
import { useI18n } from '../composables/useI18n.js'

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
  setup () {
    const { lang } = useI18n()
    return { lang }
  },
  computed: {
    planningSteps () {
      return Array.isArray(this.planning?.steps) ? this.planning.steps : []
    },
    completedPlanningCount () {
      return this.planningSteps.filter((item) => item?.status === 'completed').length
    },
    resultRailData () {
      return deriveResultRailData(this.conversation)
    },
    activeTasks () {
      return this.resultRailData.activeTasks
    },
    latestSearchResults () {
      return this.resultRailData.latestSearchResults
    },
    latestRagReferences () {
      return this.resultRailData.latestRagReferences
    },
    toolOutputs () {
      return this.resultRailData.toolOutputs
    },
    hasContent () {
      return Boolean(
        this.activeTasks.length ||
        this.planningSteps.length ||
        this.latestSearchResults.length ||
        this.latestRagReferences.length ||
        this.toolOutputs.length ||
        this.liveFrame?.image_url
      )
    }
  },
  watch: {
    liveFrame: {
      immediate: true,
      handler (nextValue) {
        if (nextValue?.image_url && nextValue.image_url !== this.currentImageUrl) {
          this.currentImageUrl = nextValue.image_url
          this.imgLoading = true
          this.imgError = false
          return
        }

        if (!nextValue?.image_url) {
          this.currentImageUrl = null
          this.imgLoading = true
          this.imgError = false
        }
      }
    }
  },
  methods: {
    planStatusLabel (status) {
      if (status === 'completed') return this.lang === 'zh' ? '已完成' : 'Completed'
      if (status === 'in_progress') return this.lang === 'zh' ? '进行中' : 'In progress'
      return this.lang === 'zh' ? '待执行' : 'Pending'
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
        img.src = `${this.currentImageUrl}${separator}retry=${Date.now()}`
      }
    }
  }
}
</script>

<style scoped>
.results-shell {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.results-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
  padding-bottom: 12px;
}

.eyebrow {
  margin: 0 0 6px;
  color: #8fb7ff;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.results-header h2 {
  margin: 0;
  font-size: 18px;
}

.summary-pills {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.pill {
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 11px;
  border: 1px solid rgba(255, 255, 255, 0.09);
  color: var(--muted);
  background: rgba(255, 255, 255, 0.04);
}

.pill.active {
  color: #dbe9ff;
  background: rgba(47, 125, 255, 0.18);
  border-color: rgba(95, 156, 255, 0.34);
}

.results-body {
  flex: 1;
  min-height: 0;
  overflow: auto;
  display: grid;
  gap: 16px;
  padding-right: 4px;
}

.rail-section {
  display: grid;
  gap: 10px;
}

.section-heading {
  font-size: 12px;
  font-weight: 700;
  color: var(--muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.card-grid,
.stack-list {
  display: grid;
  gap: 10px;
}

.rail-card,
.source-card,
.frame-card {
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.015)),
    rgba(8, 13, 21, 0.9);
  padding: 14px;
}

.card-top {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.agent-chip {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
}

.agent-chip.main {
  color: #cfe0ff;
  background: rgba(67, 143, 255, 0.17);
}

.agent-chip.analysis {
  color: #c2f6dc;
  background: rgba(32, 196, 120, 0.18);
}

.agent-chip.simulator {
  color: #ffd9bb;
  background: rgba(255, 159, 90, 0.18);
}

.card-status,
.plan-copy small,
.frame-meta {
  color: var(--muted);
  font-size: 12px;
}

.plan-list {
  display: grid;
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.plan-item {
  display: grid;
  grid-template-columns: 28px 1fr;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.03);
}

.plan-index {
  display: inline-grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  font-weight: 700;
}

.plan-item.completed .plan-index {
  background: rgba(32, 196, 120, 0.22);
  color: #b7f4d4;
}

.plan-copy {
  display: grid;
  gap: 4px;
}

.plan-copy strong,
.rail-card p,
.source-card p {
  margin: 0;
  line-height: 1.6;
}

.source-card a {
  color: #8fb7ff;
  text-decoration: none;
}

.source-card a:hover {
  text-decoration: underline;
}

.source-card.compact {
  padding-top: 12px;
  padding-bottom: 12px;
}

.frame-card {
  display: grid;
  gap: 12px;
}

.frame-card img {
  width: 100%;
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.frame-placeholder {
  aspect-ratio: 4 / 3;
  display: grid;
  place-items: center;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.03);
  color: var(--muted);
}

.frame-placeholder.error {
  gap: 12px;
  text-align: center;
  padding: 18px;
}

.frame-placeholder button {
  padding: 8px 14px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.05);
  color: var(--text);
  cursor: pointer;
}

.frame-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.empty-state {
  display: grid;
  gap: 6px;
  padding: 18px;
  border-radius: 18px;
  border: 1px dashed rgba(255, 255, 255, 0.12);
  color: var(--muted);
}

.empty-state strong {
  color: var(--text);
}

@media (max-width: 900px) {
  .results-header {
    flex-direction: column;
  }
}
</style>
