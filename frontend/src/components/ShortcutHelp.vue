<template>
  <Teleport to="body">
    <div v-if="visible" class="shortcut-overlay" @click.self="$emit('close')">
      <div class="shortcut-modal">
        <div class="modal-header">
          <h2>{{ lang === 'zh' ? '键盘快捷键' : 'Keyboard Shortcuts' }}</h2>
          <button class="close-btn" @click="$emit('close')">×</button>
        </div>
        <div class="modal-body">
          <div class="shortcut-group">
            <h3>{{ lang === 'zh' ? '全局' : 'General' }}</h3>
            <dl class="shortcut-list">
              <div class="shortcut-item">
                <dt><kbd>?</kbd></dt>
                <dd>{{ lang === 'zh' ? '显示此帮助面板' : 'Show this help' }}</dd>
              </div>
              <div class="shortcut-item">
                <dt><kbd>Esc</kbd></dt>
                <dd>{{ lang === 'zh' ? '关闭面板/取消' : 'Close panel / Cancel' }}</dd>
              </div>
              <div class="shortcut-item">
                <dt><kbd>Ctrl + K</kbd></dt>
                <dd>{{ lang === 'zh' ? '聚焦搜索框' : 'Focus search' }}</dd>
              </div>
            </dl>
          </div>
          <div class="shortcut-group">
            <h3>{{ lang === 'zh' ? '对话' : 'Chat' }}</h3>
            <dl class="shortcut-list">
              <div class="shortcut-item">
                <dt><kbd>Ctrl + Enter</kbd></dt>
                <dd>{{ lang === 'zh' ? '发送消息' : 'Send message' }}</dd>
              </div>
            </dl>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script>
export default {
  name: 'ShortcutHelp',
  props: {
    visible: { type: Boolean, default: false },
    lang: { type: String, default: 'zh' }
  },
  emits: ['close']
}
</script>

<style scoped>
.shortcut-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.shortcut-modal {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  min-width: 320px;
  max-width: 480px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--line);
}

.modal-header h2 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
}

.close-btn {
  border: none;
  background: transparent;
  color: var(--muted);
  font-size: 20px;
  cursor: pointer;
  padding: 0;
  line-height: 1;
}

.close-btn:hover {
  color: var(--text);
}

.modal-body {
  padding: 16px 20px;
}

.shortcut-group {
  margin-bottom: 16px;
}

.shortcut-group:last-child {
  margin-bottom: 0;
}

.shortcut-group h3 {
  margin: 0 0 10px;
  font-size: 12px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.shortcut-list {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.shortcut-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.shortcut-item dt {
  min-width: 100px;
}

.shortcut-item dd {
  margin: 0;
  color: var(--muted);
  font-size: 13px;
}

kbd {
  display: inline-block;
  padding: 4px 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--input-bg);
  color: var(--text);
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
}
</style>
