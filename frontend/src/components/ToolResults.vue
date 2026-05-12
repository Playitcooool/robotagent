<template>
  <div class="results-shell">
    <header class="results-header">
      <h2>{{ lang === 'zh' ? '仿真画面' : 'Simulation Feed' }}</h2>
    </header>

    <div class="results-body">
      <section v-if="liveFrame?.image_url" class="rail-section frame-section">
        <div class="section-heading">
          <span>{{ lang === 'zh' ? '仿真画面' : 'Simulation Feed' }}</span>
          <span v-if="isStale" class="stale-badge">{{ lang === 'zh' ? '⚠ 画面卡住' : '⚠ Stale' }}</span>
        </div>
        <div class="frame-stage">
          <div
            class="frame-wrap"
            @pointerdown="startCameraDrag"
            @pointermove="moveCameraDrag"
            @pointerup="endCameraDrag"
            @pointercancel="endCameraDrag"
            @contextmenu.prevent
            @wheel.prevent="handleCameraWheel"
          >
            <div class="camera-toolbar" @mousedown.stop @wheel.stop>
              <button type="button" class="camera-button text" @click="sendCameraAction({ action: 'fit_scene' })">
                {{ lang === 'zh' ? '适配场景' : 'Fit' }}
              </button>
              <button type="button" class="camera-button text" @click="sendCameraAction({ action: 'reset_auto' })">
                {{ lang === 'zh' ? '重置' : 'Reset' }}
              </button>
              <button type="button" class="camera-button text" @click="sendCameraAction({ action: 'preset', preset: 'top' })">
                {{ lang === 'zh' ? '俯视' : 'Top' }}
              </button>
              <button type="button" class="camera-button text" @click="sendCameraAction({ action: 'preset', preset: 'front' })">
                {{ lang === 'zh' ? '正视' : 'Front' }}
              </button>
              <button type="button" class="camera-button text" @click="sendCameraAction({ action: 'preset', preset: 'side' })">
                {{ lang === 'zh' ? '侧视' : 'Side' }}
              </button>
            </div>
            <img
              class="frame-img"
              draggable="false"
              :src="mjpegUrl"
              :alt="liveFrame.task || 'simulation frame'"
            />
          </div>
        </div>
        <div class="camera-panel">
          <button type="button" class="camera-panel-toggle" @click="cameraPanelOpen = !cameraPanelOpen">
            <span>{{ lang === 'zh' ? '摄像头' : 'Camera' }}</span>
            <span class="camera-mode">{{ cameraModeLabel }}</span>
          </button>
          <p v-if="cameraError" class="camera-error">{{ cameraError }}</p>
          <div v-if="cameraPanelOpen" class="camera-grid">
            <label v-for="field in cameraFields" :key="field.key" class="camera-field">
              <span>{{ field.label }}</span>
              <input
                type="number"
                :min="field.min"
                :max="field.max"
                :step="field.step"
                v-model.number="cameraForm[field.key]"
                @input="queueCameraFormSend(field.key)"
              />
            </label>
          </div>
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
      lastCameraSend: 0,
      cameraPanelOpen: false,
      cameraError: '',
      cameraForm: {
        eyeX: 0,
        eyeY: 0,
        eyeZ: 0,
        targetX: 0,
        targetY: 0,
        targetZ: 0,
        distance: 1.2,
        yaw: 45,
        pitch: -30,
        fov: 60
      },
      cameraFormTimer: null,
      pendingCameraField: '',
      syncingCameraForm: false
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
    if (this.cameraFormTimer) clearTimeout(this.cameraFormTimer)
  },
  watch: {
    'liveFrame.camera': {
      handler (camera) {
        this.syncCameraForm(camera)
      },
      immediate: true,
      deep: true
    }
  },
  methods: {
    syncCameraForm (camera) {
      if (!camera) return
      this.syncingCameraForm = true
      const eye = Array.isArray(camera.eye) ? camera.eye : []
      const target = Array.isArray(camera.target) ? camera.target : []
      this.cameraForm = {
        eyeX: this.roundCameraValue(eye[0], this.cameraForm.eyeX),
        eyeY: this.roundCameraValue(eye[1], this.cameraForm.eyeY),
        eyeZ: this.roundCameraValue(eye[2], this.cameraForm.eyeZ),
        targetX: this.roundCameraValue(target[0], this.cameraForm.targetX),
        targetY: this.roundCameraValue(target[1], this.cameraForm.targetY),
        targetZ: this.roundCameraValue(target[2], this.cameraForm.targetZ),
        distance: this.roundCameraValue(camera.distance, this.cameraForm.distance),
        yaw: this.roundCameraValue(camera.yaw, this.cameraForm.yaw),
        pitch: this.roundCameraValue(camera.pitch, this.cameraForm.pitch),
        fov: this.roundCameraValue(camera.fov, this.cameraForm.fov)
      }
      this.$nextTick(() => { this.syncingCameraForm = false })
    },
    roundCameraValue (value, fallback = 0) {
      const num = Number(value)
      if (!Number.isFinite(num)) return fallback
      return Number(num.toFixed(3))
    },
    queueCameraFormSend (fieldKey) {
      if (this.syncingCameraForm) return
      this.pendingCameraField = fieldKey || ''
      if (this.cameraFormTimer) clearTimeout(this.cameraFormTimer)
      this.cameraFormTimer = setTimeout(() => {
        this.cameraFormTimer = null
        this.sendCameraForm()
      }, 180)
    },
    sendCameraForm () {
      const f = this.cameraForm
      const payload = {
        action: 'set_absolute',
        target: [f.targetX, f.targetY, f.targetZ],
        fov: f.fov
      }
      if (this.pendingCameraField?.startsWith('eye')) {
        payload.eye = [f.eyeX, f.eyeY, f.eyeZ]
      } else {
        payload.distance = f.distance
        payload.yaw = f.yaw
        payload.pitch = f.pitch
      }
      this.pendingCameraField = ''
      this.sendCameraAction(payload)
    },
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
        this.cameraError = ''
        const response = await fetch('/api/sim/camera', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
        const data = await response.json().catch(() => ({}))
        if (!response.ok || data.ok === false) {
          throw new Error(data.error || `HTTP ${response.status}`)
        }
      } catch (error) {
        this.cameraError = error?.message || String(error)
        console.warn('camera action failed', error)
      }
    }
  },
  computed: {
    cameraFields () {
      return [
        { key: 'eyeX', label: 'eye x', step: 0.01 },
        { key: 'eyeY', label: 'eye y', step: 0.01 },
        { key: 'eyeZ', label: 'eye z', step: 0.01 },
        { key: 'targetX', label: 'target x', step: 0.01 },
        { key: 'targetY', label: 'target y', step: 0.01 },
        { key: 'targetZ', label: 'target z', step: 0.01 },
        { key: 'distance', label: 'distance', min: 0.15, max: 50, step: 0.05 },
        { key: 'yaw', label: 'yaw', min: 0, max: 360, step: 1 },
        { key: 'pitch', label: 'pitch', min: -89, max: 89, step: 1 },
        { key: 'fov', label: 'fov', min: 25, max: 90, step: 1 }
      ]
    },
    cameraModeLabel () {
      const mode = this.liveFrame?.camera?.mode
      if (mode === 'manual') return this.lang === 'zh' ? '手动视角' : 'Manual'
      return this.lang === 'zh' ? '自动完整取景' : 'Auto fit'
    },
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
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: center;
  flex: 1;
  width: 100%;
}

.frame-stage {
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.frame-section .frame-wrap {
  width: 100%;
  max-width: 960px;
  aspect-ratio: 4 / 3;
  max-height: min(70vh, calc(100vh - 250px));
  min-height: 300px;
  background: #000;
  border-radius: 8px;
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

.camera-toolbar {
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 2;
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
  max-width: calc(100% - 16px);
}

.camera-button {
  display: inline-grid;
  place-items: center;
  min-width: 30px;
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

.camera-button.text {
  width: auto;
  padding: 0 10px;
  white-space: nowrap;
}

.camera-button:hover {
  background: rgba(34, 87, 160, 0.82);
  border-color: rgba(143, 183, 255, 0.44);
}

.camera-panel {
  display: grid;
  gap: 8px;
  max-width: 960px;
  width: 100%;
  margin: 0 auto;
}

.camera-panel-toggle {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
  height: 34px;
  padding: 0 12px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.04);
  color: var(--text);
  cursor: pointer;
}

.camera-mode {
  color: var(--muted);
  font-size: 12px;
}

.camera-error {
  margin: 0;
  padding: 7px 10px;
  border-radius: 8px;
  background: rgba(255, 91, 91, 0.12);
  color: #ffb3b3;
  font-size: 12px;
}

.camera-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(96px, 1fr));
  gap: 8px;
}

.camera-field {
  display: grid;
  gap: 4px;
  color: var(--muted);
  font-size: 11px;
}

.camera-field input {
  width: 100%;
  min-width: 0;
  height: 30px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(6, 10, 16, 0.86);
  color: var(--text);
  padding: 0 8px;
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
