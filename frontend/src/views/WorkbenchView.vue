<template>
  <section
    :class="['workbench', toolPanelOpen ? 'has-results' : 'no-results']"
    :style="{ gridTemplateColumns }"
  >
    <aside class="session-rail">
      <Sidebar
        :reloadToken="sidebarReloadToken"
        :currentSessionId="currentSessionId"
        :authUser="auth.authUser.value"
        :authToken="auth.authToken.value"
        :lang="preferences.lang.value"
        @selectSession="handleSelectSession"
        @sessionDeleted="handleSessionDeleted"
        @logout="auth.onLogout"
      />
    </aside>

    <div class="resize-handle left-handle" @mousedown="startResize('left', $event)"></div>

    <main class="conversation-stage">
      <ChatView
        :conversation="conversation"
        :planning="planningState"
        :landingMode="landingMode"
        @sendMessage="sendMessage"
      />
      <button
        class="results-toggle"
        type="button"
        :aria-pressed="toolPanelOpen"
        :title="toolPanelOpen ? (preferences.lang.value === 'zh' ? '收起结果' : 'Hide results') : (preferences.lang.value === 'zh' ? '展开结果' : 'Show results')"
        @click="toolPanelOpen = !toolPanelOpen"
      >
        {{ toolPanelOpen ? '▸' : '◂' }}
      </button>
    </main>

    <div v-if="toolPanelOpen" class="resize-handle right-handle" @mousedown="startResize('right', $event)"></div>

    <aside v-if="toolPanelOpen" class="results-rail">
      <ToolResults
        :liveFrame="liveFrame"
        :planning="planningState"
        :conversation="conversation"
      />
    </aside>
  </section>
</template>

<script>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import ChatView from '../components/ChatView.vue'
import Sidebar from '../components/Sidebar.vue'
import ToolResults from '../components/ToolResults.vue'
import { useWorkbenchStore } from '../composables/useWorkbenchStore.js'

export default {
  name: 'WorkbenchView',
  components: { ChatView, Sidebar, ToolResults },
  setup () {
    const route = useRoute()
    const router = useRouter()
    const workbench = useWorkbenchStore()
    const toolPanelOpen = ref(false)
    const leftWidth = ref(290)
    const rightWidth = ref(360)
    let resizeState = null

    const gridTemplateColumns = computed(() => {
      if (toolPanelOpen.value) {
        return `${leftWidth.value}px 8px minmax(360px, 1fr) 8px ${rightWidth.value}px`
      }
      return `${leftWidth.value}px 8px minmax(360px, 1fr)`
    })

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value))
    }

    function startResize(target, event) {
      event.preventDefault()
      resizeState = {
        target,
        startX: event.clientX,
        startLeft: leftWidth.value,
        startRight: rightWidth.value
      }
      document.body.classList.add('is-resizing')
      window.addEventListener('mousemove', onResize)
      window.addEventListener('mouseup', stopResize)
    }

    function onResize(event) {
      if (!resizeState) return
      const delta = event.clientX - resizeState.startX
      if (resizeState.target === 'left') {
        leftWidth.value = clamp(resizeState.startLeft + delta, 220, 420)
        return
      }
      rightWidth.value = clamp(resizeState.startRight - delta, 280, 560)
    }

    function stopResize() {
      resizeState = null
      document.body.classList.remove('is-resizing')
      window.removeEventListener('mousemove', onResize)
      window.removeEventListener('mouseup', stopResize)
    }

    async function handleSelectSession(session) {
      await workbench.selectSession(session)
      if (!session || session.__newConversation) {
        router.replace('/chat')
        return
      }
      router.replace(`/chat/${encodeURIComponent(session.session_id)}`)
    }

    function handleSessionDeleted(sessionId) {
      workbench.onSessionDeleted(sessionId)
      if (String(sessionId || '') === String(route.params.sessionId || '')) {
        router.replace('/chat')
      }
    }

    onMounted(async () => {
      if (route.params.sessionId) {
        await workbench.selectSession({ session_id: route.params.sessionId })
      }
    })

    watch(() => route.params.sessionId, async (sessionId, previous) => {
      if (sessionId && sessionId !== previous && sessionId !== workbench.currentSessionId.value) {
        await workbench.selectSession({ session_id: sessionId })
      }

      if (!sessionId && previous) {
        workbench.resetConversation(`session_${Date.now()}`)
      }
    })

    watch(workbench.hasToolPanelContent, (hasContent, hadContent) => {
      if (hasContent && !hadContent) toolPanelOpen.value = true
    })

    onBeforeUnmount(stopResize)

    return {
      ...workbench,
      toolPanelOpen,
      gridTemplateColumns,
      startResize,
      handleSelectSession,
      handleSessionDeleted
    }
  }
}
</script>

<style scoped>
.workbench {
  position: relative;
  display: grid;
  gap: 8px;
  height: calc(100vh - 72px);
  padding: 18px;
}

.results-toggle {
  position: absolute;
  top: 50%;
  right: -1px;
  z-index: 8;
  transform: translateY(-50%);
  display: inline-grid;
  place-items: center;
  width: 24px;
  height: 46px;
  border: 1px solid rgba(95, 156, 255, 0.32);
  border-right: none;
  border-radius: 12px 0 0 12px;
  background: rgba(12, 18, 29, 0.92);
  color: #dce8ff;
  padding: 0;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  box-shadow: var(--shadow-sm);
}

.session-rail,
.conversation-stage,
.results-rail {
  min-height: 0;
  border-radius: 24px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.035), rgba(255, 255, 255, 0.015)),
    rgba(12, 16, 24, 0.9);
  box-shadow: var(--shadow-sm);
}

.session-rail,
.results-rail {
  padding: 14px;
}

.conversation-stage {
  position: relative;
  overflow: hidden;
  background:
    radial-gradient(880px 360px at 50% 0%, rgba(47, 125, 255, 0.12), transparent 56%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.015)),
    rgba(12, 16, 24, 0.96);
}

.resize-handle {
  min-height: 0;
  border-radius: 999px;
  cursor: col-resize;
  background: transparent;
  transition: background 0.16s ease;
}

.resize-handle:hover {
  background: rgba(95, 156, 255, 0.2);
}

:global(body.is-resizing) {
  cursor: col-resize;
  user-select: none;
}

@media (max-width: 1280px) {
  .workbench {
    grid-template-columns: 268px 8px minmax(0, 1fr) !important;
  }

  .results-rail {
    grid-column: 1 / -1;
    min-height: 320px;
  }
}

@media (max-width: 900px) {
  .workbench {
    grid-template-columns: 1fr !important;
    height: auto;
    min-height: calc(100vh - 72px);
  }

  .resize-handle {
    display: none;
  }

  .session-rail,
  .results-rail,
  .conversation-stage {
    min-height: 320px;
  }
}
</style>
