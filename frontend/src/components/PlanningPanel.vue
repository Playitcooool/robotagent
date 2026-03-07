<template>
  <div v-if="steps.length" class="planning-panel">
    <button class="planning-head" type="button" @click="toggleExpanded">
      <span class="title">计划</span>
      <span class="count">{{ doneCount }}/{{ steps.length }}</span>
      <span :class="['caret', expanded ? 'open' : '']">▾</span>
    </button>
    <div v-if="expanded" class="planning-body">
      <ol class="planning-list">
        <li v-for="(item, idx) in steps" :key="item.id || idx" :class="['planning-item', item.status]">
          <span class="idx">
            <template v-if="item.status === 'completed'">✓</template>
            <template v-else>{{ idx + 1 }}</template>
          </span>
          <span class="step">{{ item.step }}</span>
          <span class="status">{{ statusLabel(item.status) }}</span>
        </li>
      </ol>
    </div>
  </div>
</template>

<script>
import { computed, ref, watch, onBeforeUnmount } from 'vue'

export default {
  name: 'PlanningPanel',
  props: {
    planning: { type: [Object, null], default: null }
  },
  setup (props) {
    const expanded = ref(false)
    let collapseTimer = null

    const steps = computed(() => {
      const incoming = props.planning?.steps
      return Array.isArray(incoming) ? incoming : []
    })

    const doneCount = computed(
      () => steps.value.filter(item => item?.status === 'completed').length
    )

    watch(
      () => props.planning?.updatedAt,
      () => {
        if (!steps.value.length) {
          expanded.value = false
          return
        }
        const hasActive = steps.value.some(item => item?.status === 'in_progress')
        if (hasActive) {
          if (collapseTimer) clearTimeout(collapseTimer)
          expanded.value = true
          return
        }
        if (collapseTimer) clearTimeout(collapseTimer)
        collapseTimer = setTimeout(() => {
          expanded.value = false
        }, 1000)
      }
    )

    function toggleExpanded () {
      expanded.value = !expanded.value
    }

    onBeforeUnmount(() => {
      if (collapseTimer) clearTimeout(collapseTimer)
    })

    function statusLabel (status) {
      if (status === 'completed') return '已完成'
      if (status === 'in_progress') return '进行中'
      return '待执行'
    }

    return { expanded, steps, doneCount, toggleExpanded, statusLabel }
  }
}
</script>

<style scoped>
.planning-panel {
  margin: 8px 14px 0;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  background: #101521;
}

.planning-head {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 8px;
  border: none;
  padding: 7px 10px;
  background: transparent;
  color: #dbe2ed;
  cursor: pointer;
}

.title {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.01em;
}

.count {
  font-size: 11px;
  color: #9aa4b2;
}

.caret {
  margin-left: auto;
  color: #9aa4b2;
  transition: transform 0.2s ease;
}

.caret.open {
  transform: rotate(180deg);
}

.planning-body {
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  padding: 8px 10px 10px;
}

.planning-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: grid;
  gap: 6px;
}

.planning-item {
  display: grid;
  grid-template-columns: 18px 1fr auto;
  gap: 8px;
  align-items: start;
  font-size: 12px;
  line-height: 1.35;
}

.planning-item .idx {
  width: 18px;
  height: 18px;
  border-radius: 999px;
  display: inline-grid;
  place-items: center;
  background: rgba(255, 255, 255, 0.1);
  color: #c5cedb;
  font-size: 11px;
  font-weight: 700;
}

.planning-item .step {
  color: #dbe2ed;
  word-break: break-word;
}

.planning-item .status {
  border-radius: 999px;
  padding: 1px 7px;
  font-size: 10px;
  font-weight: 700;
  white-space: nowrap;
}

.planning-item.in_progress .status {
  color: #9fc1ff;
  background: rgba(47, 125, 255, 0.2);
}

.planning-item.completed .idx {
  background: rgba(32, 196, 120, 0.22);
  color: #a5f3c7;
}

.planning-item.completed .status {
  color: #a5f3c7;
  background: rgba(32, 196, 120, 0.2);
}

.planning-item.pending .status {
  color: #aeb8c8;
  background: rgba(255, 255, 255, 0.08);
}
</style>
