import { ref } from 'vue'

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
            }, 800)
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
      if (authToken.value && simStreamActive.value) {
        setTimeout(() => {
          if (liveFrameEventSource === null && authToken.value && simStreamActive.value) {
            startLiveFrameStream()
          }
        }, 1000)
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
