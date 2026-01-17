<template>
  <div class="app-grid">
    <aside class="left">
      <Sidebar @selectMessage="onSelectMessage" ref="sidebar" />
    </aside>

    <main class="center">
      <ChatView :conversation="conversation" :assistantDraft="assistantDraft" @sendMessage="onSendMessage" />
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

    // (streaming assistant placeholder will be inserted directly into `conversation`)

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

      // create assistant placeholder immediately so UI shows typing animation
      const assistantId = Date.now() + 1
      conversation.value.push({ id: assistantId, role: 'assistant', text: '', loading: true })
      console.debug('App: pushed assistant placeholder', assistantId, conversation.value.map(m=>({id:m.id,role:m.role,text:m.text,loading:m.loading})))
      console.debug('dqdaadadadad')
      // call backend API (assumes /api/chat/send exists). If not, mock response.
      try {
        const res = await fetch('/api/chat/send', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        })

        if (!res.ok) {
          // update the existing assistant placeholder with an error message
          const idxErr = conversation.value.findIndex(m => m.id === assistantId)
          const errText = '无法连接后端（http ' + res.status + '）'
          if (idxErr !== -1) {
            conversation.value[idxErr].loading = false
            conversation.value[idxErr].text = errText
          } else {
            conversation.value.push({ id: Date.now()+2, role: 'assistant', text: errText })
          }
          return
        }
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
              // Update the assistant placeholder in conversation with the latest text
              const idx = conversation.value.findIndex(m => m.id === assistantId)
              if (idx !== -1) {
                if (conversation.value[idx].loading) {
                  conversation.value[idx].loading = false
                  console.debug('App: first token arrived, clearing loading for', assistantId)
                }
                conversation.value[idx].text = obj.text
              } else {
                conversation.value.push({ id: assistantId, role: 'assistant', text: obj.text })
              }
            } else if (obj.type === 'error') {
              const errText = `[后端错误] ${obj.error}`
              const idxErr2 = conversation.value.findIndex(m => m.id === assistantId)
              if (idxErr2 !== -1) {
                conversation.value[idxErr2].loading = false
                conversation.value[idxErr2].text = errText
              } else {
                conversation.value.push({ id: Date.now()+3, role: 'assistant', text: errText })
              }
            } else if (obj.type === 'done') {
              // stream finished — ensure loading cleared
              const idxDone = conversation.value.findIndex(m => m.id === assistantId)
              if (idxDone !== -1) conversation.value[idxDone].loading = false
            }
          }
        }

        // process any remaining buffered line
        if (buf.trim()) {
          try {
            const obj = JSON.parse(buf)
            if (obj && obj.type === 'delta') {
              const idxRem = conversation.value.findIndex(m => m.id === assistantId)
              if (idxRem !== -1) {
                conversation.value[idxRem].loading = false
                conversation.value[idxRem].text = obj.text
              } else {
                conversation.value.push({ id: assistantId, role: 'assistant', text: obj.text })
              }
            }
          } catch (err) {
            // ignore
          }
        }

      } catch (e) {
        const idxNet = conversation.value.findIndex(m => m.role === 'assistant' && m.loading)
        const errMsg = '[网络错误] ' + String(e)
        if (idxNet !== -1) {
          conversation.value[idxNet].loading = false
          conversation.value[idxNet].text = errMsg
        } else {
          conversation.value.push({ id: Date.now()+2, role: 'assistant', text: errMsg })
        }
      }
    }

    return { conversation, toolResult, onSelectMessage, onSendMessage }
  }
}
</script>
