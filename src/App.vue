<script setup lang="ts">
import { ref } from 'vue'
import { ChatInterface } from '@galvanized-pukeko/vue-ui'
import type { Tool } from '@ag-ui/client'
import WebcamPanel from './components/WebcamPanel.vue'

const webcamPanelRef = ref<InstanceType<typeof WebcamPanel> | null>(null)

const clientTools: Tool[] = [
  {
    name: 'capture_image',
    description: 'Capture a photo from the robot webcam. Returns the current image of the robot and its surroundings as seen from above.',
    parameters: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
]

const clientToolHandlers = {
  capture_image: async () => {
    if (!webcamPanelRef.value) {
      return JSON.stringify({ error: 'Webcam not initialized' })
    }
    const frame = webcamPanelRef.value.captureFrame()
    if (!frame) {
      return JSON.stringify({ error: 'Failed to capture frame. Is the camera active?' })
    }

    // Extracted format from captureFrame(): "data:image/jpeg;base64,/9j/4..."
    const match = frame.match(/^data:(image\/[a-zA-Z0-9+.-]+);base64,([^"]*)$/)
    if (match) {
      return JSON.stringify({
        mimeType: match[1],
        data: match[2]
      })
    }
    return JSON.stringify({ error: 'Invalid frame format captured' })
  }
}

const ROBOT_HOST = import.meta.env.VITE_ROBOT_HOST ?? '192.168.4.1'

async function emergencyStop() {
  // Best-effort, fire-and-forget. The robot's tiny HTTP server doesn't send
  // CORS headers, so we use no-cors and ignore the (opaque) response.
  try {
    await fetch(`http://${ROBOT_HOST}/control?var=robot&val=8`, { mode: 'no-cors' })
  } catch {
    /* ignore */
  }
}
</script>

<template>
  <div class="robot-controller">
    <header class="robot-header">
      <h1>Pukeko Robot Controller</h1>
    </header>
    <main class="robot-panels">
      <section class="panel webcam-section">
        <h2>Camera Feed</h2>
        <WebcamPanel ref="webcamPanelRef" />
      </section>
      <section class="panel chat-section">
        <ChatInterface
          :clientTools="clientTools"
          :clientToolHandlers="clientToolHandlers"
        />
      </section>
    </main>
    <button
      type="button"
      class="emergency-stop"
      aria-label="Emergency stop"
      title="Emergency stop"
      @click="emergencyStop"
    >
      STOP
    </button>
  </div>
</template>

<style scoped>
.robot-controller {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg-primary, #1a1a2e);
  color: var(--text-primary, #e0e0e0);
}

.robot-header {
  padding: 0.75rem 1.5rem;
  background: var(--bg-secondary, #16213e);
  border-bottom: 1px solid var(--border-color, #0f3460);
}

.robot-header h1 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
}

.robot-panels {
  flex: 1;
  display: flex;
  gap: 1rem;
  padding: 1rem;
  overflow: hidden;
}

.panel {
  flex: 1;
  min-width: 0;
  background: var(--bg-secondary, #16213e);
  border-radius: 8px;
  border: 1px solid var(--border-color, #0f3460);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.panel h2 {
  margin: 0;
  padding: 0.75rem 1rem;
  font-size: 0.9rem;
  font-weight: 600;
  border-bottom: 1px solid var(--border-color, #0f3460);
}

.emergency-stop {
  position: fixed;
  bottom: 1.5rem;
  left: 1.5rem;
  width: 6rem;
  height: 6rem;
  border-radius: 50%;
  background: #d32f2f;
  color: #fff;
  font-size: 1.4rem;
  font-weight: 900;
  letter-spacing: 0.1em;
  border: 4px solid #fff;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5), inset 0 -4px 0 rgba(0, 0, 0, 0.3);
  cursor: pointer;
  z-index: 9999;
  font-family: inherit;
}

.emergency-stop:hover {
  background: #e53935;
}

.emergency-stop:active {
  background: #b71c1c;
  transform: scale(0.96);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5), inset 0 -2px 0 rgba(0, 0, 0, 0.3);
}
</style>
