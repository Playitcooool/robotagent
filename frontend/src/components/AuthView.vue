<template>
  <div class="auth-card">
    <h1>RobotAgent 登录</h1>
    <p class="hint">登录后可访问对话历史与会话能力</p>

    <div class="tabs">
      <button :class="{ active: mode === 'login' }" @click="mode = 'login'">登录</button>
      <button :class="{ active: mode === 'register' }" @click="mode = 'register'">注册</button>
    </div>

    <form @submit.prevent="submit">
      <label>
        用户名
        <input v-model.trim="username" type="text" autocomplete="username" placeholder="3-32位，字母/数字/._-" />
      </label>

      <label>
        密码
        <input v-model="password" type="password" autocomplete="current-password" placeholder="至少6位" />
      </label>

      <label v-if="mode === 'register'">
        确认密码
        <input v-model="confirmPassword" type="password" autocomplete="new-password" placeholder="再次输入密码" />
      </label>

      <p v-if="errorText" class="error">{{ errorText }}</p>
      <button class="submit" type="submit" :disabled="loading">{{ loading ? '提交中...' : mode === 'login' ? '登录' : '注册' }}</button>
    </form>
  </div>
</template>

<script>
import { ref } from 'vue'

export default {
  name: 'AuthView',
  emits: ['authed'],
  setup (_, { emit }) {
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
