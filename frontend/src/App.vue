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

        if (!res.ok) {
          conversation.value.push({ id: Date.now()+2, role: 'assistant', text: '无法连接后端，已在本地模拟回复：' + text })
          return
        }

        // Prepare an assistant placeholder to update incrementally
        const assistantId = Date.now() + 1
        conversation.value.push({ id: assistantId, role: 'assistant', text: '' })

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buf = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })

          const lines = buf.split('\n')
          buf = lines.pop() // keep the last partial line

          for (const line of lines) {
            if (!line.trim()) continue
            let obj = null
            try {
              obj = JSON.parse(line)
            } catch (err) {
              continue
            }

            if (obj.type === 'delta') {
              // Update the assistant placeholder with the latest text
              const idx = conversation.value.findIndex(m => m.id === assistantId)
              if (idx !== -1) {
                // Replace or append text (agent may stream full content each time)
                conversation.value[idx].text = obj.text
              } else {
                conversation.value.push({ id: assistantId, role: 'assistant', text: obj.text })
              }
            } else if (obj.type === 'error') {
              const idx = conversation.value.findIndex(m => m.id === assistantId)
              const errText = `[后端错误] ${obj.error}`
              if (idx !== -1) conversation.value[idx].text = (conversation.value[idx].text || '') + '\n' + errText
              else conversation.value.push({ id: assistantId, role: 'assistant', text: errText })
            } else if (obj.type === 'done') {
              // finalization can be handled here if needed
            }
          }
        }

        // process any remaining buffered line
        if (buf.trim()) {
          try {
            const obj = JSON.parse(buf)
            if (obj && obj.type === 'delta') {
              const idx = conversation.value.findIndex(m => m.role === 'assistant' && m.id === assistantId)
              if (idx !== -1) conversation.value[idx].text = obj.text
              else conversation.value.push({ id: assistantId, role: 'assistant', text: obj.text })
            }
          } catch (err) {
            // ignore
          }
        }

      } catch (e) {
        conversation.value.push({ id: Date.now()+2, role: 'assistant', text: '网络错误（本地模拟回复）：' + text })
      }
    }

    return { conversation, toolResult, onSelectMessage, onSendMessage }
  }
}
</script>
