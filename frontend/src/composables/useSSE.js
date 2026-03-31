import { ref } from 'vue'

// Constants (matching App.vue constants)
const FRAME_CLOSE_DELAY = 800
const STREAM_RECONNECT_DELAY = 1000

export function useSSE (authToken) {
  const liveFrame = ref(null)
  const simStreamActive = ref(false)

  let liveFrameEventSource = null
  let liveFrameStartTimestamp = 0

  function stopLiveFrameStream () {
    if (liveFrameEventSource) {
      liveFrameEventSource.close()
      liveFrameEventSource = null
    }
    simStreamActive.value = false
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
      // 清空引用后再重连
      liveFrameEventSource = null
      const tokenAtError = authToken.value
      if (tokenAtError && simStreamActive.value) {
        setTimeout(() => {
          if (liveFrameEventSource === null && authToken.value && simStreamActive.value) {
            startLiveFrameStream()
          }
        }, STREAM_RECONNECT_DELAY)
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
