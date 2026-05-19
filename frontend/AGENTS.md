# Frontend Guide

This folder is the Vue 3 + Vite workbench UI for RobotAgent.

- `src/App.vue`, `src/main.js`, and `src/router/` set up the app shell and routes.
- `src/views/WorkbenchView.vue` owns the main workbench layout.
- `src/components/ChatView.vue` and `src/components/chat/` implement the chat surface, landing state, messages, and composer.
- `src/components/PlanningPanel.vue`, `ThinkingTrace.vue`, and `ToolResults.vue` render agent planning, reasoning traces, and tool outputs.
- `src/composables/useSSE.js` consumes backend NDJSON/SSE-style chat streaming.
- `src/composables/useAuth.js`, `usePreferences.js`, `useWorkbenchStore.js`, and `useI18n.js` hold client-side state and API helpers.
- `src/assets/styles.css` is the main styling surface.
- `tests/` contains Vite/Vitest-style frontend unit tests.

Run frontend commands from this directory with `rtk npm ...`; common checks are `rtk npm run build` and targeted tests if the project has the needed test runner installed.

Preserve the backend stream contract from `backend/stream_utils.py` and `backend/app.py` when editing chat rendering. Keep UI changes consistent with the existing workbench style rather than introducing a separate landing-page pattern.
