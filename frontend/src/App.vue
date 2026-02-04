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

    // helper: per-assistant streaming buffers (for progressive reveal)
    const assistantStreams = {}

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
      // init stream buffer state
      assistantStreams[assistantId] = {
        buffer: '',
        displayed: '',
        interval: null
      }
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
        // guard: some environments may not expose a streaming body
        if (!res.body || !res.body.getReader) {
          const textBody = await res.text()
          const idxNoStream = conversation.value.findIndex(m => m.id === assistantId)
          const txt = textBody || '[后端返回空响应]'
          if (idxNoStream !== -1) {
            conversation.value[idxNoStream].loading = false
            conversation.value[idxNoStream].text = txt
          } else {
            conversation.value.push({ id: assistantId, role: 'assistant', text: txt })
          }
          return
        }

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buf = ''

        // Keep the original user text for prefix-stripping if backend echoes it
        const originalUserText = text || ''

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
              let incoming = obj.text || ''
              if (idx !== -1) {
                const currentlyShown = conversation.value[idx].text || ''
                if ((!currentlyShown || currentlyShown === '') && originalUserText && incoming.startsWith(originalUserText)) {
                  incoming = incoming.slice(originalUserText.length)
                  incoming = incoming.replace(/^\s*[:：\-–—]?\s*/, '')
                }

                if (conversation.value[idx].loading) {
                  conversation.value[idx].loading = false
                  console.debug('App: first token arrived, clearing loading for', assistantId)
                }

                // put incoming into buffer and decide immediate vs progressive reveal
                let st = assistantStreams[assistantId]
                if (!st) {
                  st = assistantStreams[assistantId] = { buffer: '', displayed: '', interval: null }
                }

                const prev = st.buffer || ''
                // if incoming is only a small growth over previous buffer, treat as live streaming and update immediately
                const growth = incoming.length - prev.length
                if (incoming.startsWith(prev) && growth > 0 && growth <= 50) {
                  // immediate append for smooth streaming
                  st.buffer = incoming
                  st.displayed = incoming
                  const idx2 = conversation.value.findIndex(m => m.id === assistantId)
                  if (idx2 !== -1) conversation.value[idx2].text = st.displayed
                } else if (incoming.length > prev.length && incoming.startsWith(prev) && growth > 50) {
                  // large chunk appended — show progressively
                  st.buffer = incoming
                  if (!st.interval) {
                    st.interval = setInterval(() => {
                      try {
                        const nextLen = Math.min(st.displayed.length + 6, st.buffer.length)
                        if (nextLen > st.displayed.length) {
                          st.displayed = st.buffer.slice(0, nextLen)
                          const idx2 = conversation.value.findIndex(m => m.id === assistantId)
                          if (idx2 !== -1) conversation.value[idx2].text = st.displayed
                        }
                      } catch (e) {}
                    }, 25)
                  }
                } else {
                  // not a simple append (e.g., backend rewrote or first chunk) — replace displayed
                  st.buffer = incoming
                  st.displayed = incoming
                  const idx2 = conversation.value.findIndex(m => m.id === assistantId)
                  if (idx2 !== -1) conversation.value[idx2].text = st.displayed
                }
              } else {
                // push sanitized assistant message and init stream state
                if (originalUserText && incoming.startsWith(originalUserText)) {
                  incoming = incoming.slice(originalUserText.length).replace(/^\s*[:：\-–—]?\s*/, '')
                }
                conversation.value.push({ id: assistantId, role: 'assistant', text: incoming })
                assistantStreams[assistantId] = { buffer: incoming, displayed: incoming, interval: null }
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
              // stream finished — ensure loading cleared and flush buffers
              const idxDone = conversation.value.findIndex(m => m.id === assistantId)
              if (idxDone !== -1) {
                conversation.value[idxDone].loading = false
                const stDone = assistantStreams[assistantId]
                if (stDone) {
                  // finalize display
                  conversation.value[idxDone].text = stDone.buffer || stDone.displayed || conversation.value[idxDone].text
                  if (stDone.interval) {
                    clearInterval(stDone.interval)
                    stDone.interval = null
                  }
                  // cleanup
                  delete assistantStreams[assistantId]
                }
              }
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
