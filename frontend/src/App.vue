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

const WELCOME_TEXT = '欢迎使用 RobotAgent！请选择左侧会话或发起新对话。'

export default {
  name: 'App',
  components: { Sidebar, ChatView, ToolResults },
  setup () {
    const conversation = ref([
      { id: 1, role: 'assistant', text: WELCOME_TEXT }
    ])

    const assistantStreams = {}
    const toolResult = ref(null)

    function resetConversation () {
      conversation.value = [{ id: Date.now(), role: 'assistant', text: WELCOME_TEXT }]
      toolResult.value = null
    }

    function onSelectMessage (message) {
      if (!message || message.__newConversation) {
        resetConversation()
        return
      }
      conversation.value = [message]
      toolResult.value = null
    }

    async function onSendMessage(text) {
      const userMsg = { id: Date.now(), role: 'user', text }
      conversation.value.push(userMsg)

      const assistantId = Date.now() + 1
      conversation.value.push({ id: assistantId, role: 'assistant', text: '', loading: true })
      assistantStreams[assistantId] = {
        buffer: '',
        displayed: '',
        interval: null
      }

      try {
        const res = await fetch('/api/chat/send', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        })

        if (!res.ok) {
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
        const originalUserText = text || ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })

          const lines = buf.split('\n')
          buf = lines.pop()

          for (const line of lines) {
            if (!line.trim()) continue
            let obj = null
            try {
              obj = JSON.parse(line)
            } catch (err) {
              continue
            }

            if (obj.type === 'delta') {
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
                }

                let st = assistantStreams[assistantId]
                if (!st) {
                  st = assistantStreams[assistantId] = { buffer: '', displayed: '', interval: null }
                }

                const prev = st.buffer || ''
                const growth = incoming.length - prev.length
                if (incoming.startsWith(prev) && growth > 0 && growth <= 50) {
                  st.buffer = incoming
                  st.displayed = incoming
                  const idx2 = conversation.value.findIndex(m => m.id === assistantId)
                  if (idx2 !== -1) conversation.value[idx2].text = st.displayed
                } else if (incoming.length > prev.length && incoming.startsWith(prev) && growth > 50) {
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
                      } catch (e) {
                        // ignore interval update errors
                      }
                    }, 25)
                  }
                } else {
                  st.buffer = incoming
                  st.displayed = incoming
                  const idx2 = conversation.value.findIndex(m => m.id === assistantId)
                  if (idx2 !== -1) conversation.value[idx2].text = st.displayed
                }
              } else {
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
            } else if (obj.type === 'tool') {
              toolResult.value = obj.result ?? obj.data ?? obj.payload ?? null
            } else if (obj.type === 'done') {
              const idxDone = conversation.value.findIndex(m => m.id === assistantId)
              if (idxDone !== -1) {
                conversation.value[idxDone].loading = false
                const stDone = assistantStreams[assistantId]
                if (stDone) {
                  conversation.value[idxDone].text = stDone.buffer || stDone.displayed || conversation.value[idxDone].text
                  if (stDone.interval) {
                    clearInterval(stDone.interval)
                    stDone.interval = null
                  }
                  delete assistantStreams[assistantId]
                }
              }
            }
          }
        }

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
            // ignore invalid trailing line
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
