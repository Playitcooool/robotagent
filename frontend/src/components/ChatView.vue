<template>
  <section :class="['chat-panel', landingMode ? 'landing' : 'dialogue']">
    <div class="chat-content">
      <LandingHero v-if="landingMode" :lang="lang" @prompt="applyPrompt" />
      <MessageList v-else :conversation="conversation" :lang="lang" />
      <PlanningPanel v-if="!landingMode" :planning="planning" />
    </div>

    <ChatComposer
      v-model="text"
      :landingMode="landingMode"
      :canSend="canSend"
      :lang="lang"
      @send="send"
    />
  </section>
</template>

<script>
import { computed, ref } from 'vue'

import { useI18n } from '../composables/useI18n.js'
import PlanningPanel from './PlanningPanel.vue'
import ChatComposer from './chat/ChatComposer.vue'
import LandingHero from './chat/LandingHero.vue'
import MessageList from './chat/MessageList.vue'

export default {
  name: 'ChatView',
  components: { ChatComposer, LandingHero, MessageList, PlanningPanel },
  props: {
    conversation: { type: Array, default: () => [] },
    planning: { type: [Object, null], default: null },
    landingMode: { type: Boolean, default: false }
  },
  emits: ['sendMessage'],
  setup (_, { emit }) {
    const { lang } = useI18n()
    const text = ref('')
    const canSend = computed(() => text.value.trim().length > 0)

    function send() {
      if (!text.value.trim()) return
      emit('sendMessage', { text: text.value.trim(), enabledTools: [] })
      text.value = ''
    }

    function applyPrompt(prompt) {
      text.value = prompt
    }

    return {
      lang,
      text,
      canSend,
      send,
      applyPrompt
    }
  }
}
</script>

<style scoped>
.chat-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.chat-panel.landing {
  justify-content: center;
  padding: 28px 24px 24px;
}

.chat-panel.dialogue {
  min-height: 0;
}

.chat-content {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
</style>
