<template>
  <div ref="messagesRef" class="message-list" role="log" aria-live="polite" aria-label="对话消息" aria-relevant="additions">
    <MessageBubble
      v-for="message in conversation"
      :key="message.id"
      :message="message"
      :lang="lang"
    />
  </div>
</template>

<script>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

import MessageBubble from './MessageBubble.vue'

export default {
  name: 'MessageList',
  components: { MessageBubble },
  props: {
    conversation: { type: Array, default: () => [] },
    lang: { type: String, default: 'zh' }
  },
  setup (props) {
    const messagesRef = ref(null)
    let scrollRaf = 0

    function scrollToBottom() {
      const element = messagesRef.value
      if (!element) return
      element.scrollTop = element.scrollHeight
    }

    function scheduleScroll() {
      if (scrollRaf) cancelAnimationFrame(scrollRaf)
      scrollRaf = requestAnimationFrame(() => nextTick(scrollToBottom))
    }

    watch(
      () => props.conversation.slice(-3).map((item) => {
        const textLen = String(item?.text || '').length
        const thinkLen = String(item?.thinking || '').length
        return `${item?.id}:${textLen}:${thinkLen}:${item?.loading ? 1 : 0}:${item?.thinkingDone ? 1 : 0}`
      }).join('|'),
      scheduleScroll
    )

    onMounted(scrollToBottom)
    onBeforeUnmount(() => {
      if (scrollRaf) cancelAnimationFrame(scrollRaf)
    })

    return { messagesRef }
  }
}
</script>

<style scoped>
.message-list {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 22px 22px 8px;
  display: grid;
  gap: 16px;
}
</style>
