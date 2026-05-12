<template>
  <div class="results-shell">
    <header class="results-header">
      <h2>{{ lang === 'zh' ? '仿真画面' : 'Simulation Feed' }}</h2>
    </header>

    <div class="results-body">
      <section v-if="liveFrame?.image_url" class="rail-section frame-section">
        <div class="frame-wrap" @wheel.prevent="onWheel">
          <img class="frame-img" :src="mjpegUrl" :alt="liveFrame.task || 'simulation frame'"
            :style="{ transform: `scale(${frameZoom})` }" />
        </div>
      </section>

      <div v-if="!hasContent" class="empty-state">
        <strong>{{ lang === 'zh' ? '等待仿真画面' : 'Awaiting simulation frame' }}</strong>
        <p>{{ lang === 'zh' ? '仿真工具回传画面后会显示在这里。' : 'Simulation frames will appear here when returned by the simulator.' }}</p>
      </div>
    </div>
  </div>
</template>

<script>
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
      nowSec: Math.floor(Date.now() / 1000),
      frameZoom: 1.0,
      _staleTimer: null
    }
  },
  setup () {
    const { lang } = useI18n()
    return { lang }
  },
  mounted () {
    this._staleTimer = setInterval(() => {
      this.nowSec = Math.floor(Date.now() / 1000)
    }, 2000)
  },
  beforeUnmount () {
    if (this._staleTimer) clearInterval(this._staleTimer)
  },
  computed: {
    hasContent () {
      return Boolean(this.liveFrame?.image_url)
    },
    isStale () {
      const ts = Number(this.liveFrame?.timestamp)
      if (!ts || !Number.isFinite(ts)) return false
      if (this.liveFrame?.done) return false
      return this.nowSec - ts > 10
    },
    mjpegUrl () {
      // Use run_id as cache-buster so stream reconnects when sim is reset
      const rid = this.liveFrame?.run_id || 'default'
      return `/api/sim/mjpeg?rid=${encodeURIComponent(rid)}`
    }
  },
  methods: {
    onWheel (e) {
      const delta = e.deltaY > 0 ? -0.1 : 0.1
      this.frameZoom = Math.max(0.5, Math.min(4.0, this.frameZoom + delta))
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
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding-right: 4px;
}

.rail-section {
  display: grid;
  gap: 10px;
}

.section-heading {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  font-weight: 700;
  color: var(--muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.frame-section {
  gap: 8px;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: center;
  flex: 1;
  width: 100%;
}

.frame-section .frame-wrap {
  width: 100%;
  max-width: 100%;
  aspect-ratio: 4 / 3;
  background: #000;
  border-radius: 14px;
  overflow: hidden;
  position: relative;
}

.frame-section .frame-img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
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

.rail-text {
  margin: 0;
  color: var(--text);
  font-size: 13px;
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.empty-state.compact {
  min-height: auto;
  padding: 14px;
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

.frame-wrap {
  position: relative;
  width: 100%;
  aspect-ratio: 4 / 3;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.frame-img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
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

.stale-badge {
  color: #ffb86b;
  background: rgba(255, 159, 90, 0.15);
  border-radius: 999px;
  padding: 1px 8px;
  font-size: 11px;
  font-weight: 700;
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
