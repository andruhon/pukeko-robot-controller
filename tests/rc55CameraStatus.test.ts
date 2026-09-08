import { describe, it, expect, vi } from 'vitest'
import {
  createWorldCapabilities,
  createWorldSession,
  namedCameraFailure,
  runRecipe,
  type RobotCapabilities,
  type WorldHosts,
  type WorldId,
} from '../src/robotSession/index.js'
import { ACEBOTT_QD021_PRESET } from '../src/agent/robotPresets/index.js'
import type { RobotToolDef } from '../src/agent/robotPresets/index.js'
import { z } from 'zod'

// RC-55. A capture failure could not name its cause, and a camera that will
// never come up still burned the whole readiness deadline on EVERY tool call —
// five seconds, per call, to say nothing the model could act on.
//
// Two properties are under test here, and they are deliberately separate:
//
//   1. WORDING — the failure names its cause, in vue-ui's vocabulary, and the
//      capture_image and motion paths say the SAME thing about the same camera.
//   2. TIMING — a cause that waiting cannot fix is answered immediately, while
//      a camera that is genuinely starting still gets the full deadline.
//
// A spec that only checked wording would pass with the five-second stall fully
// intact, which is the substantive half of the defect. So every fail-fast spec
// below asserts the message AND that it arrived without the deadline being
// advanced, and the specs in "what still waits" hold the other side: they fail
// if the deadline is skipped for a camera that deserved it.
//
// Every expected string is written out BY HAND rather than read back off
// `captureFailureMessage`. That is the point: reading the expected value off the
// module under test lets an implementation that stops consulting vue-ui satisfy
// these by agreeing with itself. These strings are vue-ui's published
// vocabulary, so they are a contract with the installed package, and mutating
// that package's dist is what proves they are load-bearing.

const HOSTS: WorldHosts = { robotHost: '10.0.0.7', emulatorHost: '127.0.0.1:9099' }

const LIVE_FRAME = 'data:image/png;base64,LIVEWEBCAMFRAME'

// vue-ui's CAPTURE_FAILURE_BY_STATUS, transcribed by hand.
const DENIED_MESSAGE = 'Failed to capture frame. Camera permission was denied.'
const NO_DEVICE_MESSAGE = 'Failed to capture frame. No camera device was found.'
const BUSY_MESSAGE = 'Failed to capture frame. The camera is in use by another application.'
const STARTING_MESSAGE = 'Failed to capture frame. The camera has not finished starting.'
const IDLE_MESSAGE = 'Failed to capture frame. The camera is not running.'
// vue-ui's CAPTURE_IMAGE_FAILED_ERROR — the frozen fallback, kept for a status
// that names nothing a caller can act on.
const FROZEN_MESSAGE = 'Failed to capture frame. Is the camera active?'
// The robot's own pre-capture guard. NOT part of RC-55 and asserted here only to
// pin that it survives: it is the honest answer when the reason is unknown.
const NOT_INITIALIZED = 'Webcam not initialized'

// 'HI' as bytes; base64 'SEk=' computed independently of the encoder under test.
const FRAME_BYTES = Uint8Array.from([0x48, 0x49])
const SNAPSHOT_DATA_URL = 'data:image/jpeg;base64,SEk='

function makeFetch(opts: { captureStatus?: number } = {}) {
  const calls: string[] = []
  const fn = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    calls.push(url)
    if (url.endsWith('/capture')) {
      const status = opts.captureStatus ?? 200
      return {
        ok: status >= 200 && status < 300,
        status,
        headers: {
          get: (name: string) =>
            name.toLowerCase() === 'content-type' ? 'image/jpeg' : null,
        },
        arrayBuffer: async () => FRAME_BYTES.buffer.slice(0),
      } as unknown as Response
    }
    return { ok: true, status: 200, text: async () => 'ok' } as unknown as Response
  })
  return { fn: fn as unknown as typeof fetch, calls }
}

/**
 * A panel reporting `status`, streaming only when `isActive` is set.
 *
 * `neverDecodes` reproduces a live stream whose <video> has not decoded: every
 * frame is the `data:,` a 0x0 canvas serialises to — truthy, and not an image.
 */
function makePanel(opts: {
  status?: 'idle' | 'starting' | 'live' | 'denied' | 'no-device' | 'busy' | 'error'
  isActive?: boolean
  neverDecodes?: boolean
}) {
  const panel = {
    isActive: opts.isActive ?? false,
    cameraStatus: opts.status,
    captureFrame: vi.fn(() => {
      if (!panel.isActive) return 'data:,'
      return opts.neverDecodes ? 'data:,' : LIVE_FRAME
    }),
    composeBeforeAfter: vi.fn(async (_b: string, _a: string) => 'data:image/jpeg;base64,COMPOSITE'),
  }
  return panel
}

function sessionOver(
  panel: ReturnType<typeof makePanel>,
  opts: { worldId?: WorldId; cameraReadyTimeoutMs?: number; captureStatus?: number } = {}
) {
  const worldId = opts.worldId ?? 'real'
  const { fn, calls } = makeFetch({ captureStatus: opts.captureStatus })
  const capabilities = createWorldCapabilities({
    getWorldId: () => worldId,
    hosts: HOSTS,
    getWebcamPanel: () => panel,
    fetch: fn,
    cameraReadyPollMs: 1,
    cameraReadyTimeoutMs: opts.cameraReadyTimeoutMs ?? 100,
  })
  const session = createWorldSession({
    worldId,
    hosts: HOSTS,
    presetId: ACEBOTT_QD021_PRESET.id,
    capabilities,
  })
  return { session, capabilities, calls }
}

describe('RC-55: a capture failure names its cause', () => {
  // The three statuses that cannot improve, with the message each must produce.
  const UNIMPROVABLE = [
    ['denied', DENIED_MESSAGE],
    ['no-device', NO_DEVICE_MESSAGE],
    ['busy', BUSY_MESSAGE],
  ] as const

  describe('capture_image fails IMMEDIATELY on a cause that waiting cannot fix', () => {
    for (const [status, message] of UNIMPROVABLE) {
      it(`answers '${status}' without spending the readiness deadline`, async () => {
        vi.useFakeTimers()
        try {
          // Never goes live, and says why. Before RC-55 this cost the full
          // deadline and then answered 'Webcam not initialized'.
          const panel = makePanel({ status })
          const { session } = sessionOver(panel, { cameraReadyTimeoutMs: 100 })

          const settled: string[] = []
          void session.captureImage().then((r) => settled.push(r))

          // NOT advancing the clock. Only microtasks are flushed, so a settled
          // promise here means no timer was ever waited on. This is the
          // assertion that fails if the fail-fast path regresses to polling —
          // the wording assertion below would still pass in that world.
          await vi.advanceTimersByTimeAsync(0)
          expect(settled).toHaveLength(1)
          expect(JSON.parse(settled[0])).toEqual({ error: message })
        } finally {
          vi.useRealTimers()
        }
      })
    }

    it('does not fetch, move, or capture a frame on the way out', async () => {
      const panel = makePanel({ status: 'denied' })
      const { session, calls } = sessionOver(panel)

      expect(JSON.parse(await session.captureImage())).toEqual({ error: DENIED_MESSAGE })
      expect(calls).toEqual([])
      expect(panel.captureFrame).not.toHaveBeenCalled()
    })

    it('stops waiting when a LIVE stream dies mid-call', async () => {
      // The second wait, not the readiness one. A call can begin against a
      // perfectly good stream and lose the device part-way through — another
      // application takes it — and the frame wait would then spend the whole
      // remaining budget on a camera that has already reported why it stopped.
      vi.useFakeTimers()
      try {
        const panel = makePanel({ status: 'live', isActive: true, neverDecodes: true })
        const { session } = sessionOver(panel, { cameraReadyTimeoutMs: 1000 })

        const settled: string[] = []
        void session.captureImage().then((r) => settled.push(r))

        await vi.advanceTimersByTimeAsync(50)
        // Correctly still polling: a live stream that has not decoded yet is
        // exactly what the deadline is for.
        expect(settled).toEqual([])

        panel.cameraStatus = 'busy'
        await vi.advanceTimersByTimeAsync(5)

        // Answered at ~55 ms of a 1000 ms budget.
        expect(settled).toHaveLength(1)
        expect(JSON.parse(settled[0])).toEqual({ error: BUSY_MESSAGE })
      } finally {
        vi.useRealTimers()
      }
    })
  })

  describe('what still waits — the RC-53 startup window is NOT sacrificed', () => {
    it('a starting camera still gets the whole deadline, then names that', async () => {
      vi.useFakeTimers()
      try {
        const panel = makePanel({ status: 'starting' })
        const { session } = sessionOver(panel, { cameraReadyTimeoutMs: 100 })

        const settled: string[] = []
        void session.captureImage().then((r) => settled.push(r))

        // The control for every fail-fast spec above: same shape, same fake,
        // and it is STILL WAITING at the instant those had already answered. If
        // fail-fast ever widened to "any status that is not live", this is the
        // spec that goes red.
        await vi.advanceTimersByTimeAsync(0)
        expect(settled).toEqual([])
        await vi.advanceTimersByTimeAsync(90)
        expect(settled).toEqual([])

        await vi.advanceTimersByTimeAsync(20)
        expect(settled).toHaveLength(1)
        expect(JSON.parse(settled[0])).toEqual({ error: STARTING_MESSAGE })
      } finally {
        vi.useRealTimers()
      }
    })

    it('a LIVE stream that never decodes waits, then fails for a reason that is not "starting"', async () => {
      vi.useFakeTimers()
      try {
        // Acceptance: this case must stay distinguishable from the startup one,
        // because the two call for different actions from the model.
        const panel = makePanel({ status: 'live', isActive: true, neverDecodes: true })
        const { session } = sessionOver(panel, { cameraReadyTimeoutMs: 100 })

        const settled: string[] = []
        void session.captureImage().then((r) => settled.push(r))

        await vi.advanceTimersByTimeAsync(90)
        expect(settled).toEqual([])

        await vi.advanceTimersByTimeAsync(20)
        expect(settled).toHaveLength(1)
        const error = JSON.parse(settled[0]).error as string
        expect(error).toBe(FROZEN_MESSAGE)
        // The discrimination the acceptance criterion actually asks for.
        expect(error).not.toBe(STARTING_MESSAGE)
      } finally {
        vi.useRealTimers()
      }
    })
  })

  describe('the capture_image and motion paths agree about the same camera', () => {
    for (const [status, message] of UNIMPROVABLE) {
      it(`both name '${status}' identically`, async () => {
        const panel = makePanel({ status })
        const { session, calls } = sessionOver(panel)

        const capture = JSON.parse(await session.captureImage())
        const motion = JSON.parse(await session.clientToolHandlers.move_forward({}))

        expect(capture).toEqual({ error: message })
        // The motion guard fires before the label is computed, so no `motion`
        // field — unchanged from before RC-55.
        expect(motion).toEqual({ error: message })
        // A student told the camera was denied must not also find the robot has
        // walked: the guard is still ahead of every endpoint.
        expect(calls).toEqual([])
      })
    }
  })

  describe("'Webcam not initialized' survives for a camera that cannot say why", () => {
    // This path is NOT RC-55's to change. These specs exist so that a later
    // widening of the naming rule cannot silently swallow it.
    it('capture_image keeps it when the panel reports no status at all', async () => {
      const panel = makePanel({})
      // A panel too old to expose the status, or any hand-rolled stand-in.
      panel.cameraStatus = undefined
      const { session } = sessionOver(panel)

      expect(JSON.parse(await session.captureImage())).toEqual({ error: NOT_INITIALIZED })
    })

    it("keeps it for 'error', which names no cause a caller can act on", async () => {
      const panel = makePanel({ status: 'error' })
      const { session } = sessionOver(panel)

      expect(JSON.parse(await session.captureImage())).toEqual({ error: NOT_INITIALIZED })
    })

    it('namedCameraFailure returns null exactly when vue-ui names no cause', () => {
      // The rule that protects the guard above. `live` and `error` fall back to
      // the frozen message, so they must read as "no better sentence exists"
      // rather than as a cause.
      const of = (status?: string) =>
        namedCameraFailure({ cameraStatus: () => status as never })

      expect(of(undefined)).toBeNull()
      expect(of('live')).toBeNull()
      expect(of('error')).toBeNull()

      expect(of('denied')).toBe(DENIED_MESSAGE)
      expect(of('no-device')).toBe(NO_DEVICE_MESSAGE)
      expect(of('busy')).toBe(BUSY_MESSAGE)
      expect(of('starting')).toBe(STARTING_MESSAGE)
      expect(of('idle')).toBe(IDLE_MESSAGE)
    })
  })

  describe('the simulated world ignores the camera entirely', () => {
    // The panel stays MOUNTED in the simulated world (composeBeforeAfter lives on
    // it) and still holds a camera status. That status describes a device this
    // world does not use: its frames come over HTTP.
    it('captures over HTTP even though the panel reports denied', async () => {
      const panel = makePanel({ status: 'denied' })
      const { session, calls } = sessionOver(panel, { worldId: 'simulated' })

      const out = JSON.parse(await session.captureImage())
      expect(out).toEqual({ mimeType: 'image/jpeg', data: 'SEk=' })
      expect(calls).toEqual(['http://127.0.0.1:9099/capture'])
      expect(SNAPSHOT_DATA_URL).toBe('data:image/jpeg;base64,SEk=')
    })

    it('does not blame the camera when a simulated capture FAILS', async () => {
      // This is the spec that actually holds the exclusion, and the happy-path
      // one above is not it: a successful capture never reads `cameraStatus` at
      // all, so it stays green with the exclusion deleted. The exclusion only
      // bites when the HTTP frame fails and vue-ui reaches for a cause — at
      // which point a leaked panel status would tell the model the camera
      // permission was denied, about a world with no camera in it, while the
      // real fault is an unreachable emulator.
      const panel = makePanel({ status: 'denied' })
      const { session } = sessionOver(panel, { worldId: 'simulated', captureStatus: 500 })

      const error = JSON.parse(await session.captureImage()).error as string
      expect(error).toBe(FROZEN_MESSAGE)
      expect(error).not.toBe(DENIED_MESSAGE)
    })
  })

  describe('the motion interpreter guard, in isolation', () => {
    const DEF: RobotToolDef = {
      name: 'move_forward',
      description: 'test',
      zodSchema: z.object({ steps: z.number().int().min(1).max(10).optional() }),
      fulfillment: 'client',
      clientEndpoint: '/forward',
      recipe: [
        { step: 'captureFrame', as: 'before', failMessage: 'no before frame' },
        { step: 'returnImage', from: 'before' },
      ],
    }

    function capsWith(status?: string): RobotCapabilities {
      return {
        isReady: () => false,
        cameraStatus: () => status as never,
        captureFrame: () => null,
        composeBeforeAfter: async () => null,
        fetch: (async () => {
          throw new Error('the guard must fire before any fetch')
        }) as unknown as typeof fetch,
        robotUrl: (p: string) => `http://10.0.0.7${p}`,
        robotHost: '10.0.0.7',
      }
    }

    it('names the cause when the camera has one', async () => {
      expect(JSON.parse(await runRecipe(DEF, {}, capsWith('denied')))).toEqual({
        error: DENIED_MESSAGE,
      })
    })

    it('falls back to the unchanged guard when it does not', async () => {
      expect(JSON.parse(await runRecipe(DEF, {}, capsWith(undefined)))).toEqual({
        error: NOT_INITIALIZED,
      })
    })
  })
})
