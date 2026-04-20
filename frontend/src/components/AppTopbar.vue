<template>
  <header class="topbar">
    <div class="topbar-brand">
      <RouterLink class="brand-lockup" to="/chat">
        <span class="brand-mark">RA</span>
        <span class="brand-copy">
          <strong>RobotAgent</strong>
          <small>Mission Workbench</small>
        </span>
      </RouterLink>

      <nav class="nav">
        <button
          :class="['nav-btn', route.path.startsWith('/chat') ? 'active' : '']"
          type="button"
          @click="router.push('/chat')"
        >
          {{ t('chat') }}
        </button>
        <button
          :class="['nav-btn', route.path === '/about' ? 'active' : '']"
          type="button"
          @click="router.push('/about')"
        >
          {{ t('about') }}
        </button>
      </nav>
    </div>

    <div class="topbar-actions">
      <div class="status-chip">
        <span class="status-dot"></span>
        <span>{{ simStreamActive ? (lang === 'zh' ? '仿真在线' : 'Simulation Live') : (lang === 'zh' ? '控制台就绪' : 'Console Ready') }}</span>
      </div>

      <button class="icon-btn" type="button" :title="t('switchLang')" @click="$emit('toggle-lang')">
        {{ lang === 'zh' ? 'EN' : '中' }}
      </button>

      <button class="icon-btn" type="button" :title="t('switchTheme')" @click="$emit('toggle-theme')">
        {{ isDark ? '☀' : '◐' }}
      </button>

      <label class="font-control">
        <span>{{ lang === 'zh' ? '字号' : 'Type' }}</span>
        <select :value="fontSize" @change="$emit('set-font-size', Number($event.target.value))">
          <option value="13">S</option>
          <option value="15">M</option>
          <option value="18">L</option>
        </select>
      </label>

      <div class="user-chip">
        <span class="user-name">{{ authUser?.username }}</span>
        <button class="logout-btn" type="button" @click="$emit('logout')">{{ t('logout') }}</button>
      </div>
    </div>
  </header>
</template>

<script>
import { useRoute, useRouter, RouterLink } from 'vue-router'
import { useI18n } from '../composables/useI18n.js'

export default {
  name: 'AppTopbar',
  components: { RouterLink },
  props: {
    authUser: { type: Object, default: null },
    isDark: { type: Boolean, default: true },
    lang: { type: String, default: 'zh' },
    fontSize: { type: Number, default: 15 },
    simStreamActive: { type: Boolean, default: false }
  },
  emits: ['toggle-lang', 'toggle-theme', 'set-font-size', 'logout'],
  setup () {
    const route = useRoute()
    const router = useRouter()
    const { t } = useI18n()
    return { route, router, t }
  }
}
</script>

<style scoped>
.topbar {
  position: sticky;
  top: 0;
  z-index: 30;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 14px 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  background:
    linear-gradient(180deg, rgba(8, 11, 18, 0.94), rgba(8, 11, 18, 0.88)),
    var(--panel);
  backdrop-filter: blur(18px);
}

.topbar-brand,
.topbar-actions,
.nav {
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand-lockup {
  display: flex;
  align-items: center;
  gap: 12px;
  text-decoration: none;
  color: var(--text);
}

.brand-mark {
  display: inline-grid;
  place-items: center;
  width: 36px;
  height: 36px;
  border-radius: 12px;
  background:
    linear-gradient(145deg, rgba(82, 174, 255, 0.95), rgba(41, 111, 255, 0.85));
  color: white;
  font-weight: 800;
  letter-spacing: 0.04em;
  box-shadow: 0 10px 30px rgba(47, 125, 255, 0.24);
}

.brand-copy {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.brand-copy strong {
  font-size: 15px;
  letter-spacing: 0.02em;
}

.brand-copy small {
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.14em;
}

.nav-btn,
.icon-btn,
.logout-btn,
.font-control select {
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.04);
  color: var(--text);
  border-radius: 999px;
  cursor: pointer;
}

.nav-btn {
  padding: 8px 14px;
  font-size: 13px;
}

.nav-btn.active {
  background: rgba(79, 145, 255, 0.18);
  border-color: rgba(95, 156, 255, 0.5);
  color: #dce8ff;
}

.status-chip,
.user-chip,
.font-control {
  display: flex;
  align-items: center;
  gap: 8px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.03);
  border-radius: 999px;
  padding: 7px 11px;
  font-size: 12px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: linear-gradient(180deg, #56e39f, #20c478);
  box-shadow: 0 0 14px rgba(32, 196, 120, 0.55);
}

.icon-btn {
  width: 36px;
  height: 36px;
  display: inline-grid;
  place-items: center;
  font-size: 13px;
}

.font-control span {
  color: var(--muted);
}

.font-control select {
  padding: 4px 10px;
}

.user-name {
  max-width: 140px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.logout-btn {
  padding: 6px 10px;
}

@media (max-width: 1080px) {
  .topbar {
    flex-direction: column;
    align-items: stretch;
  }

  .topbar-brand,
  .topbar-actions {
    justify-content: space-between;
    flex-wrap: wrap;
  }
}

@media (max-width: 720px) {
  .status-chip {
    display: none;
  }

  .brand-copy small {
    display: none;
  }
}
</style>
