import { ref } from 'vue'

// Constants (matching App.vue constants)
const FRAME_CLOSE_DELAY = 800

// Retry constants (借鉴 Claude Code 指数退避 + jitter)
const BASE_RECONNECT_DELAY = 500      // 基础延迟 (ms)
const MAX_RECONNECT_DELAY = 30000      // 最大延迟 (ms)
const MAX_RECONNECT_ATTEMPTS = 8      // 最大重试次数
const JITTER_FACTOR = 0.25            // 25% 随机化

/**
 * 计算指数退避延迟 (带 jitter)
 * 借鉴 Claude Code: services/api/withRetry.ts
 */
function calculateReconnectDelay (attempt) {
  const baseDelay = Math.min(BASE_RECONNECT_DELAY * Math.pow(2, attempt), MAX_RECONNECT_DELAY)
  const jitter = Math.random() * JITTER_FACTOR * baseDelay
  return baseDelay + jitter
}

export function useSSE (authToken) {
  const liveFrame = ref(null)
  const simStreamActive = ref(false)

  let liveFrameEventSource = null
  let liveFrameStartTimestamp = 0
  let reconnectAttempt = 0  // Non-reactive: internal retry counter
  let reconnectTimeoutId = null

  function stopLiveFrameStream () {
    if (reconnectTimeoutId) {
      clearTimeout(reconnectTimeoutId)
      reconnectTimeoutId = null
    }
    if (liveFrameEventSource) {
      liveFrameEventSource.close()
      liveFrameEventSource = null
    }
    simStreamActive.value = false
    reconnectAttempt = 0  // 重置重试计数
  }

  function startLiveFrameStream () {
    if (!authToken.value) return
    stopLiveFrameStream()

    liveFrameStartTimestamp = Date.now() / 1000
    const eventSource = new EventSource(`/api/sim/stream?since=${liveFrameStartTimestamp}`)

    eventSource.addEventListener('frame', (event) => {
      try {
        const payload = JSON.parse(event.data)
        if (
          payload.has_frame &&
          typeof payload.timestamp === 'number' &&
          payload.timestamp >= liveFrameStartTimestamp
        ) {
          simStreamActive.value = true
          liveFrame.value = payload
          // 连接成功后重置重试计数
          reconnectAttempt = 0
          if (payload.done) {
            setTimeout(() => {
              if (liveFrameEventSource === eventSource) {
                stopLiveFrameStream()
              }
            }, FRAME_CLOSE_DELAY)
          }
        }
      } catch (e) {
        console.error('Failed to parse frame:', e)
      }
    })

    eventSource.onerror = () => {
      eventSource.close()
      liveFrameEventSource = null
      const tokenAtError = authToken.value

      // 检查是否超过最大重试次数
      if (tokenAtError && simStreamActive.value && reconnectAttempt < MAX_RECONNECT_ATTEMPTS) {
        const delay = calculateReconnectDelay(reconnectAttempt)
        reconnectAttempt++
        console.log(`[SSE] Connection lost, retrying in ${delay.toFixed(0)}ms (attempt ${reconnectAttempt}/${MAX_RECONNECT_ATTEMPTS})`)

        reconnectTimeoutId = setTimeout(() => {
          if (liveFrameEventSource === null && authToken.value && simStreamActive.value) {
            startLiveFrameStream()
          }
        }, delay)
      } else if (reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
        console.log('[SSE] Max reconnection attempts reached, giving up')
        simStreamActive.value = false
      }
    }

    liveFrameEventSource = eventSource
  }

  return {
    liveFrame,
    simStreamActive,
    startLiveFrameStream,
    stopLiveFrameStream
  }
}
