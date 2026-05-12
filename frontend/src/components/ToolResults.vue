<template>
  <div class="results-shell">
    <header class="results-header">
      <div>
        <p class="eyebrow">Execution Rail</p>
        <h2>{{ lang === 'zh' ? '执行结果' : 'Execution Results' }}</h2>
      </div>
    </header>

    <div class="results-body">
      <section v-if="liveFrame?.image_url" class="rail-section frame-section">
        <div class="section-heading">
          <span>{{ lang === 'zh' ? '仿真画面' : 'Simulation Feed' }}</span>
          <span v-if="isStale" class="stale-badge">{{ lang === 'zh' ? '⚠ 画面卡住' : '⚠ Stale' }}</span>
        </div>
        <div
          class="frame-wrap"
          @pointerdown="startCameraDrag"
          @pointermove="moveCameraDrag"
          @pointerup="endCameraDrag"
          @pointercancel="endCameraDrag"
          @contextmenu.prevent
          @wheel.prevent="handleCameraWheel"
        >
          <div class="camera-controls" @mousedown.stop @wheel.stop>
            <button
              type="button"
              class="camera-button"
              :title="lang === 'zh' ? '自动对准' : 'Auto frame'"
              @click="sendCameraAction({ action: 'set_auto' })"
            >A</button>
            <button
              type="button"
              class="camera-button"
              :title="lang === 'zh' ? '重置视角' : 'Reset view'"
              @click="sendCameraAction({ action: 'reset_auto' })"
            >R</button>
          </div>
          <img
            class="frame-img"
            draggable="false"
            :src="mjpegUrl"
            :alt="liveFrame.task || 'simulation frame'"
          />
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
      _staleTimer: null,
      cameraDrag: null,
      lastCameraSend: 0
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
  methods: {
    startCameraDrag (event) {
      if (event.button !== 0 && event.button !== 2) return
      event.currentTarget.setPointerCapture?.(event.pointerId)
      this.cameraDrag = {
        x: event.clientX,
        y: event.clientY,
        pointerId: event.pointerId,
        mode: event.shiftKey || event.button === 2 ? 'pan' : 'orbit'
      }
    },
    moveCameraDrag (event) {
      if (!this.cameraDrag) return
      const dx = event.clientX - this.cameraDrag.x
      const dy = event.clientY - this.cameraDrag.y
      if (Math.abs(dx) + Math.abs(dy) < 2) return
      this.cameraDrag.x = event.clientX
      this.cameraDrag.y = event.clientY

      const now = Date.now()
      if (now - this.lastCameraSend < 40) return
      this.lastCameraSend = now

      if (this.cameraDrag.mode === 'pan') {
        this.sendCameraAction({ action: 'pan', delta_x: -dx, delta_y: dy })
      } else {
        this.sendCameraAction({ action: 'orbit', delta_yaw: dx * 0.35, delta_pitch: -dy * 0.25 })
      }
    },
    endCameraDrag (event) {
      if (event?.currentTarget && this.cameraDrag?.pointerId != null) {
        event.currentTarget.releasePointerCapture?.(this.cameraDrag.pointerId)
      }
      this.cameraDrag = null
    },
    handleCameraWheel (event) {
      const factor = event.deltaY < 0 ? 0.88 : 1.14
      this.sendCameraAction({ action: 'zoom', zoom_factor: factor })
    },
    async sendCameraAction (payload) {
      try {
        await fetch('/api/sim/camera', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
      } catch (error) {
        console.warn('camera action failed', error)
      }
    }
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
  overflow: hidden;
  display: grid;
  gap: 16px;
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
}

.frame-section .frame-wrap {
  width: 100%;
  height: min(72vh, calc(100vh - 190px));
  min-height: 360px;
  background: #000;
  border-radius: 14px;
  overflow: hidden;
  position: relative;
  cursor: grab;
  user-select: none;
  touch-action: none;
}

.frame-section .frame-wrap:active {
  cursor: grabbing;
}

.frame-section .frame-img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  pointer-events: none;
}

.camera-controls {
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 2;
  display: flex;
  gap: 6px;
}

.camera-button {
  display: inline-grid;
  place-items: center;
  width: 30px;
  height: 30px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: rgba(9, 14, 23, 0.78);
  color: #e7eefc;
  font-size: 12px;
  font-weight: 800;
  line-height: 1;
  cursor: pointer;
}

.camera-button:hover {
  background: rgba(34, 87, 160, 0.82);
  border-color: rgba(143, 183, 255, 0.44);
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
