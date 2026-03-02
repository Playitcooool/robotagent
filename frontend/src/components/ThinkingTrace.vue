<template>
  <div class="thinking-wrap" v-if="contentText">
    <button class="thinking-head" type="button" @click="expanded = !expanded">
      <span class="title">思考过程</span>
      <span class="status" v-if="!done">生成中...</span>
      <span :class="['chevron', expanded ? 'open' : '']">▾</span>
    </button>
    <div v-if="expanded" class="thinking-body">{{ contentText }}</div>
    <div v-if="expanded && truncated" class="thinking-note">已省略部分思考内容</div>
  </div>
</template>

<script>
import { ref, watch, computed } from 'vue'

export default {
  name: 'ThinkingTrace',
  props: {
    content: { type: String, default: '' },
    done: { type: Boolean, default: false },
    truncated: { type: Boolean, default: false }
  },
  setup (props) {
    const expanded = ref(true)
    const contentText = computed(() => String(props.content || '').trim())

    watch(
      () => props.done,
      (v) => {
        if (v) expanded.value = false
      },
      { immediate: true }
    )

    return {
      expanded,
      contentText
    }
  }
}
</script>

<style scoped>
.thinking-wrap {
  border: 1px dashed rgba(255, 255, 255, 0.18);
  border-radius: 10px;
  margin-bottom: 10px;
  background: rgba(255, 255, 255, 0.03);
}

.thinking-head {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  border: 0;
  background: transparent;
  color: #9aa4b2;
  padding: 8px 10px;
  cursor: pointer;
  font-size: 12px;
}

.thinking-head .title {
  font-weight: 600;
}

.thinking-head .status {
  opacity: 0.9;
}

.thinking-head .chevron {
  color: #c2c9d3;
  transition: transform 0.16s ease;
}

.thinking-head .chevron.open {
  transform: rotate(180deg);
}

.thinking-body {
  border-top: 1px dashed rgba(255, 255, 255, 0.12);
  padding: 8px 10px 10px;
  color: #a9b2bf;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}

.thinking-note {
  border-top: 1px dashed rgba(255, 255, 255, 0.12);
  padding: 6px 10px 10px;
  color: #8f98a6;
  font-size: 11px;
}
</style>
