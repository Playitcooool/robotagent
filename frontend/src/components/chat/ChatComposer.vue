<template>
  <form :class="['composer-shell', landingMode ? 'landing' : '']" @submit.prevent="submit">
    <div class="composer-surface">
      <textarea
        ref="textareaRef"
        :value="modelValue"
        :placeholder="lang === 'zh' ? '输入机器人任务目标。Enter 发送，Shift+Enter 换行' : 'Describe the robot mission. Enter to send, Shift+Enter for newline'"
        rows="1"
        @input="onInput"
        @keydown="onKeydown"
        @compositionstart="composing = true"
        @compositionend="composing = false"
      ></textarea>

      <div class="composer-actions">
        <button
          type="button"
          :class="['search-toggle', searchEnabled ? 'active' : '']"
          :title="searchEnabled ? (lang === 'zh' ? '联网搜索已开启' : 'Web search on') : (lang === 'zh' ? '联网搜索已关闭' : 'Web search off')"
          :aria-pressed="searchEnabled ? 'true' : 'false'"
          @click="$emit('toggle-search')"
        >
          <span class="search-dot"></span>
          <span>{{ lang === 'zh' ? '联网' : 'Search' }}</span>
        </button>
        <button
          :type="isSending ? 'button' : 'submit'"
          :class="['send-btn', isSending ? 'stop' : '']"
          :disabled="!isSending && !canSend"
          :title="isSending ? (lang === 'zh' ? '中断' : 'Stop') : (lang === 'zh' ? '发送' : 'Send')"
          @click="isSending ? $emit('stop') : null"
        >
          <span v-if="isSending" class="stop-icon"></span>
          <span v-else>{{ lang === 'zh' ? '发送' : 'Send' }}</span>
        </button>
      </div>
    </div>
  </form>
</template>

<script>
import { nextTick, onMounted, ref, watch } from 'vue'

const TEXTAREA_MAX_HEIGHT = 160

export default {
  name: 'ChatComposer',
  props: {
    modelValue: { type: String, default: '' },
    landingMode: { type: Boolean, default: false },
    canSend: { type: Boolean, default: false },
    isSending: { type: Boolean, default: false },
    lang: { type: String, default: 'zh' },
    searchEnabled: { type: Boolean, default: false }
  },
  emits: ['update:modelValue', 'send', 'stop', 'toggle-search'],
  setup (props, { emit }) {
    const textareaRef = ref(null)
    const composing = ref(false)

    function resize() {
      const element = textareaRef.value
      if (!element) return
      element.style.height = 'auto'
      element.style.height = `${Math.min(element.scrollHeight, TEXTAREA_MAX_HEIGHT)}px`
    }

    function onInput(event) {
      emit('update:modelValue', event.target.value)
      resize()
    }

    function onKeydown(event) {
      if (composing.value || event.isComposing || event.keyCode === 229) return
      if (props.isSending && event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        emit('stop')
        return
      }
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        emit('send')
      }
    }

    function submit() {
      if (props.isSending) {
        emit('stop')
        return
      }
      emit('send')
    }

    watch(() => props.modelValue, () => nextTick(resize))
    onMounted(() => resize())

    return {
      textareaRef,
      composing,
      onInput,
      onKeydown,
      submit
    }
  }
}
</script>

<style scoped>
.composer-shell {
  padding: 12px 14px 16px;
}

.composer-shell.landing {
  width: min(860px, 100%);
  margin: 0 auto;
  padding: 0;
}

.composer-surface {
  display: grid;
  gap: 10px;
  padding: 13px 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 18px;
  background:
    radial-gradient(620px 220px at 50% 0%, rgba(47, 125, 255, 0.12), transparent 58%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.015)),
    rgba(10, 15, 24, 0.95);
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
}

textarea {
  min-height: 42px;
  max-height: 160px;
  width: 100%;
  border: none;
  outline: none;
  resize: none;
  background: transparent;
  color: var(--text);
  font: inherit;
  line-height: 1.55;
}

.composer-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  justify-content: flex-end;
}

.search-toggle {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 38px;
  padding: 0 12px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.04);
  color: var(--muted);
  font-size: 13px;
  font-weight: 650;
  cursor: pointer;
}

.search-toggle.active {
  border-color: rgba(86, 163, 255, 0.58);
  background: rgba(47, 125, 255, 0.14);
  color: var(--text);
}

.search-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: currentColor;
  opacity: 0.5;
}

.search-toggle.active .search-dot {
  opacity: 1;
  background: #56a3ff;
}

.send-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 72px;
  height: 38px;
  padding: 0 16px;
  border: 1px solid rgba(86, 163, 255, 0.48);
  border-radius: 10px;
  background: rgba(47, 125, 255, 0.16);
  color: #d9e9ff;
  font-size: 13px;
  line-height: 1;
  font-weight: 650;
  cursor: pointer;
  box-shadow: none;
}

.send-btn:hover:not(:disabled) {
  background: rgba(47, 125, 255, 0.24);
  border-color: rgba(86, 163, 255, 0.72);
}

.send-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
  box-shadow: none;
}

.send-btn.stop {
  min-width: 44px;
  border-color: rgba(224, 82, 82, 0.5);
  background: rgba(224, 82, 82, 0.14);
  color: #ffd9d9;
}

.send-btn.stop:hover {
  background: rgba(224, 82, 82, 0.22);
  border-color: rgba(224, 82, 82, 0.72);
}

.stop-icon {
  width: 12px;
  height: 12px;
  border-radius: 1px;
  background: currentColor;
}

@media (max-width: 720px) {
  .composer-actions {
    justify-content: space-between;
  }

  .search-toggle {
    flex: 1;
    justify-content: center;
  }
}
</style>
