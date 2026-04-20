<template>
  <form :class="['composer-shell', landingMode ? 'landing' : '']" @submit.prevent="$emit('send')">
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
        <div class="hint">
          <span>{{ lang === 'zh' ? '任务下达' : 'Mission Dispatch' }}</span>
          <small>{{ lang === 'zh' ? '规划 / 分析 / 仿真会在右栏联动展示' : 'Planning / analysis / simulation sync to the right rail.' }}</small>
        </div>

        <button type="submit" class="send-btn" :disabled="!canSend">
          {{ lang === 'zh' ? '执行任务' : 'Run Mission' }}
        </button>
      </div>
    </div>
  </form>
</template>

<script>
import { nextTick, onMounted, ref, watch } from 'vue'

const TEXTAREA_MAX_HEIGHT = 220

export default {
  name: 'ChatComposer',
  props: {
    modelValue: { type: String, default: '' },
    landingMode: { type: Boolean, default: false },
    canSend: { type: Boolean, default: false },
    lang: { type: String, default: 'zh' }
  },
  emits: ['update:modelValue', 'send'],
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
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        emit('send')
      }
    }

    watch(() => props.modelValue, () => nextTick(resize))
    onMounted(() => resize())

    return {
      textareaRef,
      composing,
      onInput,
      onKeydown
    }
  }
}
</script>

<style scoped>
.composer-shell {
  padding: 18px 18px 22px;
}

.composer-shell.landing {
  width: min(940px, 100%);
  margin: 0 auto;
  padding: 0;
}

.composer-surface {
  display: grid;
  gap: 14px;
  padding: 18px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 22px;
  background:
    radial-gradient(620px 220px at 50% 0%, rgba(47, 125, 255, 0.12), transparent 58%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.015)),
    rgba(10, 15, 24, 0.95);
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
}

textarea {
  min-height: 56px;
  max-height: 220px;
  width: 100%;
  border: none;
  outline: none;
  resize: none;
  background: transparent;
  color: var(--text);
  font: inherit;
  line-height: 1.7;
}

.composer-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.hint {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.hint span {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #8fb7ff;
  font-weight: 700;
}

.hint small {
  color: var(--muted);
  font-size: 12px;
}

.send-btn {
  min-width: 142px;
  padding: 12px 20px;
  border: none;
  border-radius: 999px;
  background: linear-gradient(180deg, #56a3ff, #2f7dff);
  color: white;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 14px 28px rgba(47, 125, 255, 0.28);
}

.send-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
  box-shadow: none;
}

@media (max-width: 720px) {
  .composer-actions {
    flex-direction: column;
    align-items: stretch;
  }

  .send-btn {
    width: 100%;
  }
}
</style>
