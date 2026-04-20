<template>
  <section :class="['workbench', showToolPanel ? 'has-results' : 'no-results']">
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

    <main class="conversation-stage">
      <ChatView
        :conversation="conversation"
        :planning="planningState"
        :landingMode="landingMode"
        @sendMessage="sendMessage"
      />
    </main>

    <aside v-if="showToolPanel" class="results-rail">
      <ToolResults
        :liveFrame="liveFrame"
        :planning="planningState"
        :conversation="conversation"
      />
    </aside>
  </section>
</template>

<script>
import { onMounted, watch } from 'vue'
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

    return {
      ...workbench,
      handleSelectSession,
      handleSessionDeleted
    }
  }
}
</script>

<style scoped>
.workbench {
  display: grid;
  grid-template-columns: 290px minmax(0, 1fr) 360px;
  gap: 16px;
  height: calc(100vh - 72px);
  padding: 18px;
}

.workbench.no-results {
  grid-template-columns: 290px minmax(0, 1fr);
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
  overflow: hidden;
  background:
    radial-gradient(880px 360px at 50% 0%, rgba(47, 125, 255, 0.12), transparent 56%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.015)),
    rgba(12, 16, 24, 0.96);
}

@media (max-width: 1280px) {
  .workbench,
  .workbench.no-results {
    grid-template-columns: 268px minmax(0, 1fr);
  }

  .results-rail {
    grid-column: 1 / -1;
    min-height: 320px;
  }
}

@media (max-width: 900px) {
  .workbench,
  .workbench.no-results {
    grid-template-columns: 1fr;
    height: auto;
    min-height: calc(100vh - 72px);
  }

  .session-rail,
  .results-rail,
  .conversation-stage {
    min-height: 320px;
  }
}
</style>
