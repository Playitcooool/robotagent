<template>
  <div class="auth-card">
    <div class="hero-mark">RA</div>
    <h1>RobotAgent {{ mode === 'login' ? (lang === 'zh' ? '登录' : 'Access') : (lang === 'zh' ? '创建账户' : 'Create Account') }}</h1>
    <p class="hint">{{ lang === 'zh' ? '进入机器人任务工作台，查看会话、规划、证据和仿真回放。' : 'Enter the robot mission workbench to review sessions, plans, evidence, and simulation replay.' }}</p>

    <div class="tabs">
      <button :class="{ active: mode === 'login' }" @click="mode = 'login'">{{ lang === 'zh' ? '登录' : 'Login' }}</button>
      <button :class="{ active: mode === 'register' }" @click="mode = 'register'">{{ lang === 'zh' ? '注册' : 'Register' }}</button>
    </div>

    <form @submit.prevent="submit">
      <label for="auth-username">
        {{ lang === 'zh' ? '用户名' : 'Username' }}
        <input id="auth-username" v-model.trim="username" type="text" autocomplete="username" :placeholder="lang === 'zh' ? '3-32位，字母/数字/._-' : '3-32 chars, letters / numbers / . _ -'" />
      </label>

      <label for="auth-password">
        {{ lang === 'zh' ? '密码' : 'Password' }}
        <input id="auth-password" v-model="password" type="password" autocomplete="current-password" :placeholder="lang === 'zh' ? '至少6位' : 'At least 6 characters'" />
      </label>

      <label v-if="mode === 'register'" for="auth-confirm-password">
        {{ lang === 'zh' ? '确认密码' : 'Confirm Password' }}
        <input id="auth-confirm-password" v-model="confirmPassword" type="password" autocomplete="new-password" :placeholder="lang === 'zh' ? '再次输入密码' : 'Repeat your password'" />
      </label>

      <p v-if="errorText" class="error">{{ errorText }}</p>
      <button class="submit" type="submit" :disabled="loading">{{ loading ? (lang === 'zh' ? '提交中...' : 'Submitting...') : mode === 'login' ? (lang === 'zh' ? '进入控制台' : 'Enter Console') : (lang === 'zh' ? '创建并进入' : 'Create & Enter') }}</button>
    </form>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useI18n } from '../composables/useI18n.js'

export default {
  name: 'AuthView',
  emits: ['authed'],
  setup (_, { emit }) {
    const { lang } = useI18n()
    const mode = ref('login')
    const username = ref('')
    const password = ref('')
    const confirmPassword = ref('')
    const errorText = ref('')
    const loading = ref(false)

    function validate () {
      const u = username.value
      const p = password.value
      if (!u || !p) return '请输入用户名和密码'
      if (!/^[A-Za-z0-9._-]{3,32}$/.test(u)) return '用户名格式不合法'
      if (p.length < 6) return '密码至少 6 位'
      if (mode.value === 'register' && p !== confirmPassword.value) return '两次密码不一致'
      return ''
    }

    async function submit () {
      errorText.value = ''
      const err = validate()
      if (err) {
        errorText.value = err
        return
      }

      const endpoint = mode.value === 'login' ? '/api/auth/login' : '/api/auth/register'
      loading.value = true
      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: username.value, password: password.value })
        })

        const data = await res.json().catch(() => ({}))
        if (!res.ok) {
          errorText.value = data.detail || `请求失败 (${res.status})`
          return
        }

        if (!data?.token || !data?.user) {
          errorText.value = '后端未返回有效登录信息'
          return
        }

        emit('authed', data)
      } catch (e) {
        errorText.value = `网络错误: ${String(e)}`
      } finally {
        loading.value = false
      }
    }

    return {
      lang,
      mode,
      username,
      password,
      confirmPassword,
      errorText,
      loading,
      submit
    }
  }
}
</script>

<style scoped>
.auth-card {
  position: relative;
}

.hero-mark {
  display: inline-grid;
  place-items: center;
  width: 52px;
  height: 52px;
  border-radius: 18px;
  background: linear-gradient(145deg, rgba(86, 163, 255, 0.95), rgba(47, 125, 255, 0.82));
  color: white;
  font-weight: 800;
  letter-spacing: 0.04em;
  margin-bottom: 18px;
  box-shadow: 0 16px 36px rgba(47, 125, 255, 0.28);
}

.auth-card h1 {
  margin: 0 0 10px;
  font-size: 28px;
  line-height: 1.05;
  letter-spacing: -0.04em;
}

.hint {
  margin: 0 0 18px;
  color: var(--muted);
  line-height: 1.7;
}

.tabs {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 18px;
}

.tabs button,
.submit {
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 999px;
  cursor: pointer;
}

.tabs button {
  padding: 10px 12px;
  background: rgba(255, 255, 255, 0.04);
  color: var(--muted);
}

.tabs button.active {
  color: var(--text);
  background: rgba(47, 125, 255, 0.16);
  border-color: rgba(95, 156, 255, 0.34);
}

form {
  display: grid;
  gap: 14px;
}

label {
  display: grid;
  gap: 7px;
  font-size: 13px;
  color: var(--muted);
}

input {
  width: 100%;
  padding: 12px 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  background: var(--input-bg);
  color: var(--text);
}

input:focus {
  outline: none;
  border-color: rgba(95, 156, 255, 0.34);
  box-shadow: 0 0 0 4px rgba(47, 125, 255, 0.12);
}

.error {
  margin: 0;
  color: var(--danger);
  font-size: 12px;
}

.submit {
  padding: 13px 16px;
  background: linear-gradient(180deg, #56a3ff, #2f7dff);
  color: white;
  font-weight: 700;
  box-shadow: 0 18px 34px rgba(47, 125, 255, 0.24);
}

.submit:disabled {
  opacity: 0.55;
  cursor: not-allowed;
  box-shadow: none;
}
</style>
