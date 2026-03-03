<template>
  <div class="thinking-chip">
    <span class="title">{{ done ? '思考过程' : '思考中' }}</span>
    <span class="chevron">▾</span>
  </div>
</template>

<script>
export default {
  name: 'ThinkingTrace',
  props: {
    done: { type: Boolean, default: false }
  }
}
</script>

<style scoped>
.thinking-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px dashed rgba(255, 255, 255, 0.18);
  border-radius: 10px;
  margin-bottom: 10px;
  background: rgba(255, 255, 255, 0.03);
  color: #9aa4b2;
  font-size: 12px;
  line-height: 1;
  padding: 7px 10px;
}

.thinking-chip .title {
  font-weight: 600;
}

.thinking-chip .chevron {
  color: #c2c9d3;
  font-size: 12px;
}
</style>
