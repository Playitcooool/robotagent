<template>
  <div class="tool-results">
    <div class="header">仿真画面</div>
    <div class="body">
      <div v-if="liveFrame && liveFrame.image_url" class="image-wrap">
        <img :src="liveFrame.image_url" alt="simulation frame" />
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
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.1);
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
