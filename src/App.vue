<script setup lang="ts">
import { onMounted, ref } from 'vue'
import {
  ChatInterface,
  PkButton,
  PkLogo,
  PkNavHeader,
  PkWebcamPanel,
} from '@galvanized-pukeko/vue-ui'
import type { Tool } from '@ag-ui/client'

const webcamPanelRef = ref<InstanceType<typeof PkWebcamPanel> | null>(null)
const chatInterfaceRef = ref<InstanceType<typeof ChatInterface> | null>(null)

const ROBOT_HOST = import.meta.env.VITE_ROBOT_HOST ?? '192.168.4.1'

function robotUrl(path: string): string {
  return `http://${ROBOT_HOST}${path}`
}

const stepsParameter = {
  type: 'object' as const,
  properties: {
    steps: {
      type: 'integer' as const,
      minimum: 1,
      maximum: 10,
      description:
        'Number of cycles to run (1-10, defaults to 1). 1 forward/backward cycle ≈ 1.5 cm; 1 turn cycle ≈ 15°; 6 turn cycles ≈ 90°.',
    },
  },
  required: [],
}

const motionDescriptions = {
  move_forward:
    'Walk the robot forward. Optional `steps` (1-10). ~1.5 cm per cycle. Automatically captures Before/After camera frames and ultrasonic readings on every call.',
  move_backward:
    'Walk the robot backward. Optional `steps` (1-10). ~1.5 cm per cycle. Automatically captures Before/After camera frames and ultrasonic readings on every call.',
  turn_left:
    'Rotate the robot left in place. Optional `steps` (1-10). ~15° per cycle; 6 ≈ 90°. Automatically captures Before/After camera frames and ultrasonic readings on every call.',
  turn_right:
    'Rotate the robot right in place. Optional `steps` (1-10). ~15° per cycle; 6 ≈ 90°. Automatically captures Before/After camera frames and ultrasonic readings on every call.',
} as const

const clientTools: Tool[] = [
  {
    name: 'capture_image',
    description:
      'Capture a photo from the robot webcam. Returns the current image of the robot and its surroundings as seen from above.',
    parameters: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'move_forward',
    description: motionDescriptions.move_forward,
    parameters: stepsParameter,
  },
  {
    name: 'move_backward',
    description: motionDescriptions.move_backward,
    parameters: stepsParameter,
  },
  {
    name: 'turn_left',
    description: motionDescriptions.turn_left,
    parameters: stepsParameter,
  },
  {
    name: 'turn_right',
    description: motionDescriptions.turn_right,
    parameters: stepsParameter,
  },
]

function frameToEnvelope(frame: string | null): { mimeType: string; data: string } | null {
  if (!frame) return null
  const match = frame.match(/^data:(image\/[a-zA-Z0-9+.-]+);base64,([^"]*)$/)
  if (!match) return null
  return { mimeType: match[1], data: match[2] }
}

function coerceSteps(args: unknown): number {
  if (args && typeof args === 'object' && 'steps' in args) {
    const raw = (args as { steps?: unknown }).steps
    if (typeof raw === 'number' && Number.isFinite(raw) && raw >= 1) {
      return Math.min(10, Math.floor(raw))
    }
  }
  return 1
}

async function runMotion(
  toolName: keyof typeof motionDescriptions,
  endpoint: string,
  args: unknown
): Promise<string> {
  if (!webcamPanelRef.value) {
    return JSON.stringify({ error: 'Webcam not initialized' })
  }
  const steps = coerceSteps(args)
  const motionLabel = steps === 1 ? toolName : `${toolName} (steps=${steps})`

  const beforeFrame = webcamPanelRef.value.captureFrame()
  if (!beforeFrame) {
    return JSON.stringify({ error: 'Failed to capture Before frame. Is the camera active?', motion: motionLabel })
  }

  const query = steps > 1 ? `?steps=${steps}` : ''
  try {
    const res = await fetch(robotUrl(`${endpoint}${query}`))
    if (!res.ok) {
      return JSON.stringify({
        error: `Robot returned HTTP ${res.status} for ${endpoint}`,
        motion: motionLabel,
      })
    }
    await res.text()
  } catch (err) {
    const message = err instanceof Error ? err.message : 'unknown error'
    return JSON.stringify({
      error: `Failed to reach robot at ${ROBOT_HOST}: ${message}`,
      motion: motionLabel,
    })
  }

  const afterFrame = webcamPanelRef.value.captureFrame()
  if (!afterFrame) {
    return JSON.stringify({ error: 'Failed to capture After frame.', motion: motionLabel })
  }

  let compositeUrl: string | null
  try {
    compositeUrl = await webcamPanelRef.value.composeBeforeAfter(beforeFrame, afterFrame)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'compose error'
    return JSON.stringify({ error: `Failed to compose Before/After image: ${message}`, motion: motionLabel })
  }

  const envelope = frameToEnvelope(compositeUrl)
  if (!envelope) {
    return JSON.stringify({ error: 'Invalid composite frame format', motion: motionLabel })
  }

  return JSON.stringify({
    ...envelope,
    motion: motionLabel,
  })
}

const clientToolHandlers = {
  capture_image: async () => {
    if (!webcamPanelRef.value) {
      return JSON.stringify({ error: 'Webcam not initialized' })
    }
    const frame = webcamPanelRef.value.captureFrame()
    const envelope = frameToEnvelope(frame)
    if (envelope) return JSON.stringify(envelope)
    return JSON.stringify({ error: 'Failed to capture frame. Is the camera active?' })
  },
  move_forward: (args: unknown) => runMotion('move_forward', '/forward', args),
  move_backward: (args: unknown) => runMotion('move_backward', '/backward', args),
  turn_left: (args: unknown) => runMotion('turn_left', '/turn_left', args),
  turn_right: (args: unknown) => runMotion('turn_right', '/turn_right', args),
}

function emergencyStop() {
  // Halt the agent itself — the real runaway risk. This aborts any in-flight
  // model stream and blocks the tool loop from resuming, so no further motion
  // commands are issued. (The robot firmware can't interrupt a motion mid-cycle
  // anyway, so poking it is pointless; stopping the agent is what matters.)
  chatInterfaceRef.value?.stop()
}

function startNewConversation() {
  chatInterfaceRef.value?.clearHistory()
}

// Provider/model label shown in the nav header, fetched live from the AG-UI
// server's /info endpoint so it always reflects the running profile (including
// env overrides), rather than a duplicated build-time constant.
const agentLabel = ref('')

async function loadAgentInfo() {
  try {
    const buildUrl = typeof __AGUI_URL__ !== 'undefined' ? __AGUI_URL__ : ''
    let agUiUrl = buildUrl
    if (!agUiUrl) {
      const cfgRes = await fetch('/config.json')
      if (cfgRes.ok) agUiUrl = (await cfgRes.json()).agUiUrl ?? ''
    }
    if (!agUiUrl) return
    const base = agUiUrl.replace(/\/agents\/.*$/, '')
    const res = await fetch(`${base}/info`)
    if (!res.ok) return
    const info = (await res.json()) as { provider?: string | null; model?: string | null }
    agentLabel.value = [info.provider, info.model].filter(Boolean).join(' ')
  } catch (err) {
    console.warn('[App] Failed to load agent info:', err)
  }
}

onMounted(loadAgentInfo)
</script>

<template>
  <div class="robot-controller">
    <PkNavHeader>
      <template #logo>
        <PkLogo />
        <span class="app-title">Pukeko Robot Controller</span>
      </template>
      <template #nav-controls>
        <span v-if="agentLabel" class="agent-label">Model: {{ agentLabel }}</span>
        <PkButton class="header-action" @click="startNewConversation">New conversation</PkButton>
        <PkButton class="header-action stop" @click="emergencyStop">Emergency stop</PkButton>
      </template>
    </PkNavHeader>
    <main class="robot-panels">
      <section class="panel webcam-section">
        <h2>Camera Feed</h2>
        <PkWebcamPanel ref="webcamPanelRef" />
      </section>
      <section class="panel chat-section">
        <ChatInterface
          ref="chatInterfaceRef"
          :clientTools="clientTools"
          :clientToolHandlers="clientToolHandlers"
        />
      </section>
    </main>
  </div>
</template>

<style scoped>
.robot-controller {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.app-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--main-text-color);
  margin-left: var(--padding-third);
}

.agent-label {
  /* Noticeable gap between the model label and the action buttons. */
  display: flex;
  align-items: center;
  margin-right: 2rem;
  font-size: 1.1rem;
  font-weight: 500;
  color: var(--main-text-color);
  white-space: nowrap;
  opacity: 0.85;
}

.header-action.stop :deep(button) {
  background: linear-gradient(#d32f2f, #b71c1c);
  color: #fff;
  border-color: #b71c1c;
}

.header-action.stop:hover :deep(button) {
  background: linear-gradient(#e53935, #c62828);
}

.robot-panels {
  flex: 1;
  display: flex;
  gap: var(--padding-twothird);
  padding: var(--padding-twothird);
  overflow: hidden;
}

.panel {
  flex: 1;
  min-width: 0;
  background: var(--bg-input-idle);
  border-radius: var(--border-radius-small-box);
  border: var(--line-separator-subtle);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.panel h2 {
  margin: 0;
  padding: var(--padding-third) var(--padding-twothird);
  font-size: 0.9rem;
  font-weight: 600;
  border-bottom: var(--line-separator-subtle);
}

.stop {
  background: salmon;
}
</style>
