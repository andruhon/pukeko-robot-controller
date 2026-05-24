<script setup lang="ts">
import { ref } from 'vue'
import { ChatInterface } from '@galvanized-pukeko/vue-ui'
import type { Tool } from '@ag-ui/client'
import WebcamPanel from './components/WebcamPanel.vue'

const webcamPanelRef = ref<InstanceType<typeof WebcamPanel> | null>(null)

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

async function readDistance(): Promise<string | null> {
  try {
    const res = await fetch(robotUrl('/distance'))
    if (!res.ok) return null
    const text = (await res.text()).trim()
    return text || null
  } catch {
    return null
  }
}

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

  const distanceBefore = await readDistance()
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
    // Consume body so the connection closes cleanly; we don't need the content.
    await res.text()
  } catch (err) {
    const message = err instanceof Error ? err.message : 'unknown error'
    return JSON.stringify({
      error: `Failed to reach robot at ${ROBOT_HOST}: ${message}`,
      motion: motionLabel,
    })
  }

  const distanceAfter = await readDistance()
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
    distanceBefore: distanceBefore ?? undefined,
    distanceAfter: distanceAfter ?? undefined,
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

async function emergencyStop() {
  // Best-effort, fire-and-forget. The robot is on the same network and now
  // sends CORS headers, but for an emergency stop we don't care about the
  // response either way.
  try {
    await fetch(robotUrl('/stop'))
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
