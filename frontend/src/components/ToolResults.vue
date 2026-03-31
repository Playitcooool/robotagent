<template>
  <div class="tool-results">
    <div class="header">仿真画面</div>
    <div class="body">
      <div v-if="liveFrame && liveFrame.image_url" class="image-wrap">
        <div v-if="imgLoading" class="img-loading">
          <span>加载中...</span>
        </div>
        <div v-if="imgError" class="img-error">
          <span>图片加载失败</span>
          <button class="retry-btn" @click="retryImage">重试</button>
        </div>
        <img
          ref="imgRef"
          :src="liveFrame.image_url"
          alt="simulation frame"
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
      <div v-else class="empty">等待 MCP 仿真画面...</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ToolResults',
  props: {
    liveFrame: { type: [Object, null], default: null }
  },
  data () {
    return {
      imgLoading: true,
      imgError: false,
      currentImageUrl: null
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
        }
      },
      immediate: true
    }
  },
  methods: {
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
}

.header {
  margin-bottom: 10px;
  font-size: 15px;
  font-weight: 600;
}

.body {
  flex: 1;
  overflow: auto;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 10px;
  background: #101521;
  padding: 12px;
}

.image-wrap img {
  width: 100%;
  height: auto;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.1);
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
  background: #0a0e1a;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  font-size: 13px;
  color: #9aa4b2;
}

.img-error {
  flex-direction: column;
  gap: 10px;
}

.retry-btn {
  padding: 4px 12px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.1);
  color: #9aa4b2;
  border: 1px solid rgba(255, 255, 255, 0.15);
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
  color: #9aa4b2;
}

.empty {
  font-size: 12px;
  color: #9aa4b2;
}
</style>
