<template>
  <div class="app-grid">
    <aside class="left">
      <Sidebar @selectMessage="onSelectMessage" ref="sidebar" />
    </aside>

    <main class="center">
      <ChatView :conversation="conversation" @sendMessage="onSendMessage" />
    </main>

    <aside class="right">
      <ToolResults :result="toolResult" />
    </aside>
  </div>
</template>

<script>
import { ref } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatView from './components/ChatView.vue'
import ToolResults from './components/ToolResults.vue'

export default {
  name: 'App',
  components: { Sidebar, ChatView, ToolResults },
  setup () {
    // conversation state shared between components
    const conversation = ref([
      { id: 1, role: 'assistant', text: '欢迎使用 RobotAgent！请选择左侧会话或发起新对话。' }
    ])

    const toolResult = ref(null)

    function onSelectMessage(message) {
      // show message content in middle panel (replace conversation with a focused view)
      conversation.value = [message]
      toolResult.value = null
    }

    async function onSendMessage(text) {
      // append user message immediately
      const userMsg = { id: Date.now(), role: 'user', text }
      conversation.value.push(userMsg)

      // call backend API (assumes /api/chat/send exists). If not, mock response.
      try {
        const res = await fetch('/api/chat/send', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        })
        if (res.ok) {
          const data = await res.json()
          // expected: { reply: '...', tool_result: {...} }
          if (data.reply) conversation.value.push({ id: Date.now()+1, role: 'assistant', text: data.reply })
          if (data.tool_result) toolResult.value = data.tool_result
        } else {
          // fallback: mocked assistant reply
          conversation.value.push({ id: Date.now()+2, role: 'assistant', text: '无法连接后端，已在本地模拟回复：' + text })
        }
      } catch (e) {
        conversation.value.push({ id: Date.now()+2, role: 'assistant', text: '网络错误（本地模拟回复）：' + text })
      }
    }

    return { conversation, toolResult, onSelectMessage, onSendMessage }
  }
}
</script>
