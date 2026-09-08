import { describe, it, expect, vi } from 'vitest'
import { defineComponent, h, nextTick, ref } from 'vue'
import { mount } from '@vue/test-utils'
import { PkWebcamPanel } from '@galvanized-pukeko/vue-ui'
import {
  captureUrlForWorld,
  createWorldCapabilities,
  createWorldSession,
  hostForWorld,
  resolveHost,
  DEFAULT_EMULATOR_HOST,
  DEFAULT_ROBOT_HOST,
  WORLDS,
  type WorldHosts,
  type WorldId,
} from '../src/robotSession/index.js'
import { ACEBOTT_QD021_PRESET } from '../src/agent/robotPresets/index.js'
import { makeSession } from '../src/App.vue'
import WorldPicker from '../src/components/WorldPicker.vue'

// RC-44 acceptance. The choice of world is a robot-TARGET choice: the emulated
// world only advances when the motion endpoints actually reach the emulator, so
// selecting it has to repoint the motion URLs as well as the frames.
//
// Every expected value below is written out by hand — the literal hosts, the
// literal URLs, the literal base64 of the bytes the fake server serves. None of
// them is read back off the module under test, so a helper that starts
// returning the wrong host cannot satisfy them by agreeing with itself.

/**
 * Hosts deliberately UNLIKE the shipped defaults, so a test that passed by
 * accidentally hitting a default would fail here.
 */
const HOSTS: WorldHosts = { robotHost: '10.0.0.7', emulatorHost: '127.0.0.1:9099' }

// 'HI' as bytes; base64 'SEk=' computed independently of the encoder under test.
const FRAME_BYTES = Uint8Array.from([0x48, 0x49])
const FRAME_DATA_URL = 'data:image/jpeg;base64,SEk='

/** A fake `fetch` that serves a JPEG for /capture and 200 OK for anything else. */
function makeFetch(opts?: { captureStatus?: number }) {
  const calls: string[] = []
  const fn = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    calls.push(url)
    if (url.endsWith('/capture')) {
      const status = opts?.captureStatus ?? 200
      return {
        ok: status >= 200 && status < 300,
        status,
        headers: { get: (name: string) => (name.toLowerCase() === 'content-type' ? 'image/jpeg' : null) },
        arrayBuffer: async () => FRAME_BYTES.buffer.slice(0),
      } as unknown as Response
    }
    return { ok: true, status: 200, text: async () => 'ok' } as unknown as Response
  })
  return { fn: fn as unknown as typeof fetch, calls }
}

/**
 * A stand-in for the mounted <PkWebcamPanel>, streaming.
 *
 * RC-53 added `isActive` to model the panel's real exposed surface: the panel
 * has always had it, and the capabilities have to read it now that readiness
 * means the stream is flowing rather than that the component exists. `true`
 * here is the settled state these RC-44 specs were always describing — the
 * starting-up state they never exercised is covered on its own below.
 */
function makePanel() {
  const composeBeforeAfter = vi.fn(
    async (_b: string, _a: string) => 'data:image/jpeg;base64,COMPOSITEBYTES'
  )
  return {
    isActive: true,
    captureFrame: vi.fn(() => 'data:image/png;base64,LIVEWEBCAMFRAME'),
    composeBeforeAfter,
  }
}

const MOVE_FORWARD = ACEBOTT_QD021_PRESET.tools.find((t) => t.name === 'move_forward')!

// --- host + URL resolution -------------------------------------------------

describe('RC-44 world hosts', () => {
  it('resolves each world to its own host', () => {
    expect(hostForWorld('real', HOSTS)).toBe('10.0.0.7')
    expect(hostForWorld('simulated', HOSTS)).toBe('127.0.0.1:9099')
  })

  it('gives the simulated world a snapshot URL and the real robot none', () => {
    expect(captureUrlForWorld('simulated', HOSTS)).toBe('http://127.0.0.1:9099/capture')
    // The real robot's frames come off the mounted panel's canvas, not HTTP.
    expect(captureUrlForWorld('real', HOSTS)).toBeNull()
  })

  it('ships defaults that match the robot AP and the emulator port', () => {
    // Written out rather than imported-and-compared: these two literals are the
    // contract with VITE_ROBOT_HOST and with ROBOT_EMULATOR_PORT's own default.
    expect(DEFAULT_ROBOT_HOST).toBe('192.168.4.1')
    expect(DEFAULT_EMULATOR_HOST).toBe('localhost:8081')
  })

  it('falls back when the env var is unset or blank, and trims when it is set', () => {
    expect(resolveHost(undefined, 'localhost:8081')).toBe('localhost:8081')
    expect(resolveHost('', 'localhost:8081')).toBe('localhost:8081')
    expect(resolveHost('   ', 'localhost:8081')).toBe('localhost:8081')
    expect(resolveHost(' sim.example:1234 ', 'localhost:8081')).toBe('sim.example:1234')
  })
})

// --- motion URLs follow the world -----------------------------------------

describe('RC-44 selecting a world repoints the MOTION urls', () => {
  async function runForwardIn(worldId: WorldId) {
    const panel = makePanel()
    const { fn, calls } = makeFetch()
    const capabilities = createWorldCapabilities({
      getWorldId: () => worldId,
      hosts: HOSTS,
      getWebcamPanel: () => panel,
      fetch: fn,
    })
    const session = createWorldSession({
      worldId,
      hosts: HOSTS,
      presetId: ACEBOTT_QD021_PRESET.id,
      capabilities,
    })
    const result = JSON.parse(await session.runMotion(MOVE_FORWARD, { steps: 2 }))
    return { calls, result, session }
  }

  it('sends the simulated world to the emulator host', async () => {
    const { calls, result } = await runForwardIn('simulated')

    // The exact wire calls, in order: Before frame, drive, halt, After frame.
    expect(calls).toEqual([
      'http://127.0.0.1:9099/capture',
      'http://127.0.0.1:9099/forward?steps=2',
      'http://127.0.0.1:9099/stop',
      'http://127.0.0.1:9099/capture',
    ])
    expect(result.motion).toBe('move_forward (steps=2)')
    // Nothing leaked to the real robot's address.
    expect(calls.some((url) => url.includes('10.0.0.7'))).toBe(false)
  })

  it('sends the real robot to the robot host and never fetches a frame', async () => {
    const { calls, result } = await runForwardIn('real')

    expect(calls).toEqual(['http://10.0.0.7/forward?steps=2', 'http://10.0.0.7/stop'])
    expect(result.motion).toBe('move_forward (steps=2)')
    // The webcam path takes no HTTP frame fetch at all.
    expect(calls.some((url) => url.endsWith('/capture'))).toBe(false)
  })

  it('exposes the selected world in robotUrl and robotHost', () => {
    const panel = makePanel()
    const { fn } = makeFetch()
    const capabilities = createWorldCapabilities({
      getWorldId: () => 'simulated',
      hosts: HOSTS,
      getWebcamPanel: () => panel,
      fetch: fn,
    })
    const sim = createWorldSession({ worldId: 'simulated', hosts: HOSTS, capabilities })
    const real = createWorldSession({ worldId: 'real', hosts: HOSTS, capabilities })

    expect(sim.robotHost).toBe('127.0.0.1:9099')
    expect(sim.robotUrl('/turn_left')).toBe('http://127.0.0.1:9099/turn_left')
    expect(real.robotHost).toBe('10.0.0.7')
    expect(real.robotUrl('/turn_left')).toBe('http://10.0.0.7/turn_left')
  })
})

// --- frames follow the world ----------------------------------------------

describe('RC-44 selecting a world repoints the CAPTURE source', () => {
  it('fetches the emulator snapshot and returns it as a data URL', async () => {
    const panel = makePanel()
    const { fn, calls } = makeFetch()
    const frames: (string | null)[] = []
    const capabilities = createWorldCapabilities({
      getWorldId: () => 'simulated',
      hosts: HOSTS,
      getWebcamPanel: () => panel,
      fetch: fn,
      onSimulatedFrame: (frame) => frames.push(frame),
    })

    const frame = await capabilities.captureFrame()

    expect(calls).toEqual(['http://127.0.0.1:9099/capture'])
    expect(frame).toBe(FRAME_DATA_URL)
    // The viewport is refreshed with the same frame the agent just received —
    // event-driven, off a capture the app was making anyway.
    expect(frames).toEqual([FRAME_DATA_URL])
    expect(panel.captureFrame).not.toHaveBeenCalled()
  })

  it('reads the real robot off the mounted webcam panel, with no fetch', async () => {
    const panel = makePanel()
    const { fn, calls } = makeFetch()
    const frames: (string | null)[] = []
    const capabilities = createWorldCapabilities({
      getWorldId: () => 'real',
      hosts: HOSTS,
      getWebcamPanel: () => panel,
      fetch: fn,
      onSimulatedFrame: (frame) => frames.push(frame),
    })

    const frame = await capabilities.captureFrame()

    expect(frame).toBe('data:image/png;base64,LIVEWEBCAMFRAME')
    expect(calls).toEqual([])
    expect(frames).toEqual([])
  })

  it('reports an unreachable emulator as a null frame, not a thrown error', async () => {
    const panel = makePanel()
    const { fn } = makeFetch({ captureStatus: 503 })
    const frames: (string | null)[] = []
    const capabilities = createWorldCapabilities({
      getWorldId: () => 'simulated',
      hosts: HOSTS,
      getWebcamPanel: () => panel,
      fetch: fn,
      onSimulatedFrame: (frame) => frames.push(frame),
    })

    expect(await capabilities.captureFrame()).toBeNull()
    // null is what the UI keys on to say "the simulator is not running" rather
    // than blaming the camera. The frozen envelope text is untouched.
    expect(frames).toEqual([null])
  })

  it('the capture URL follows the CURRENT world, not the one at construction', async () => {
    // One capabilities object for the app's lifetime, as App.vue builds it: the
    // snapshot source is constructed once and must still address whichever
    // world is selected now.
    const panel = makePanel()
    const { fn, calls } = makeFetch()
    let worldId: WorldId = 'real'
    const capabilities = createWorldCapabilities({
      getWorldId: () => worldId,
      hosts: HOSTS,
      getWebcamPanel: () => panel,
      fetch: fn,
    })

    // Constructed while 'real' is selected — and 'real' has no snapshot target.
    expect(capabilities.isReady()).toBe(true)
    expect(await capabilities.captureFrame()).toBe('data:image/png;base64,LIVEWEBCAMFRAME')
    expect(calls).toEqual([])

    worldId = 'simulated'
    expect(await capabilities.captureFrame()).toBe(FRAME_DATA_URL)
    expect(calls).toEqual(['http://127.0.0.1:9099/capture'])

    // ...and back again, on the same object.
    worldId = 'real'
    expect(await capabilities.captureFrame()).toBe('data:image/png;base64,LIVEWEBCAMFRAME')
    expect(calls).toEqual(['http://127.0.0.1:9099/capture'])
  })

  it('is not ready in either world while the webcam panel is unmounted', () => {
    // composeBeforeAfter lives on the panel and is needed for the motion
    // composite in BOTH worlds, so an unmounted panel is not ready even when
    // the frames would come from the emulator.
    const { fn } = makeFetch()
    for (const worldId of ['real', 'simulated'] as const) {
      const capabilities = createWorldCapabilities({
        getWorldId: () => worldId,
        hosts: HOSTS,
        getWebcamPanel: () => null,
        fetch: fn,
      })
      expect(capabilities.isReady()).toBe(false)
    }
  })

  it('composes simulated frames through the still-mounted webcam panel', async () => {
    // Trap 1: the panel is kept mounted precisely so this keeps working when
    // the frames are simulated and the camera is stopped.
    const panel = makePanel()
    const { fn } = makeFetch()
    const capabilities = createWorldCapabilities({
      getWorldId: () => 'simulated',
      hosts: HOSTS,
      getWebcamPanel: () => panel,
      fetch: fn,
    })

    const composite = await capabilities.composeBeforeAfter(
      'data:image/jpeg;base64,SIMBEFORE',
      'data:image/jpeg;base64,SIMAFTER'
    )

    expect(panel.composeBeforeAfter).toHaveBeenCalledWith(
      'data:image/jpeg;base64,SIMBEFORE',
      'data:image/jpeg;base64,SIMAFTER'
    )
    expect(composite).toBe('data:image/jpeg;base64,COMPOSITEBYTES')
  })
})

// --- RC-53: readiness means the stream is flowing --------------------------

describe('RC-53 the real world is ready when its camera is streaming', () => {
  // The defect this pins: selecting the real world MOUNTS the panel at once,
  // but getUserMedia resolves later and the <video> decodes its first frame
  // later still. Readiness used to be satisfied by the mount, so a capture
  // issued in between was answered "Is the camera active?" about a camera that
  // was in the act of becoming active. Measured at a 23 ms margin in the
  // browser e2e — which is why the pin belongs here, at a level where the
  // timing is controlled, rather than in a browser run that has to lose a race
  // to notice.
  //
  // Every frame literal below is written out by hand rather than read back off
  // the panel fake, so a capture path that stops producing frames cannot
  // satisfy these by agreeing with itself.
  const LIVE_FRAME = 'data:image/png;base64,LIVEWEBCAMFRAME'

  /**
   * A panel that starts NOT streaming, exactly as a freshly mounted one does.
   *
   * `goLive()` is called by the test on a timer, so the transition happens
   * while a capture is already in flight — the actual shape of the race. Before
   * it, `captureFrame` returns what a real panel returns with a zero-size
   * <video>: the string a 0x0 canvas serialises to. It is truthy, it is not an
   * image, and accepting it is the failure mode one frame further on.
   */
  function makeStartingPanel(opts: { blankFramesWhenLive?: number } = {}) {
    let blanksLeft = opts.blankFramesWhenLive ?? 0
    const panel = {
      isActive: false,
      captureFrame: vi.fn(() => {
        if (!panel.isActive) return 'data:,'
        if (blanksLeft > 0) {
          blanksLeft--
          return 'data:,'
        }
        return LIVE_FRAME
      }),
      composeBeforeAfter: vi.fn(
        async (_b: string, _a: string) => 'data:image/jpeg;base64,COMPOSITEBYTES'
      ),
    }
    return panel
  }

  function realWorldCapabilities(
    panel: ReturnType<typeof makeStartingPanel>,
    overrides?: { cameraReadyTimeoutMs?: number }
  ) {
    const { fn } = makeFetch()
    return createWorldCapabilities({
      getWorldId: () => 'real',
      hosts: HOSTS,
      getWebcamPanel: () => panel,
      fetch: fn,
      cameraReadyPollMs: 1,
      ...overrides,
    })
  }

  it('is NOT ready while the panel is mounted but its stream has not started', () => {
    const panel = makeStartingPanel()
    const capabilities = realWorldCapabilities(panel)

    // The whole defect in one assertion: mounted, and not ready. Before RC-53
    // this returned true, which is what let a capture through into the gap.
    expect(capabilities.isReady()).toBe(false)

    panel.isActive = true
    expect(capabilities.isReady()).toBe(true)
  })

  it('answers a capture issued before the camera is live with a FRAME, not an error', async () => {
    const panel = makeStartingPanel()
    const capabilities = realWorldCapabilities(panel)
    const session = createWorldSession({
      worldId: 'real',
      hosts: HOSTS,
      presetId: ACEBOTT_QD021_PRESET.id,
      capabilities,
    })

    // The stream comes up while the capture is already waiting — the ordering
    // the e2e hit by 23 ms.
    const live = setTimeout(() => {
      panel.isActive = true
    }, 20)

    const result = JSON.parse(await session.captureImage())
    clearTimeout(live)

    // Not 'Webcam not initialized', and not 'Failed to capture frame. Is the
    // camera active?' — an actual image envelope.
    expect(result).toEqual({ mimeType: 'image/png', data: 'LIVEWEBCAMFRAME' })
  })

  it('keeps waiting when the stream is live but the video has not decoded a frame yet', async () => {
    // isActive flips as soon as getUserMedia resolves, which is BEFORE the
    // <video> reports a size. Drawing then yields 'data:,' from a 0x0 canvas.
    // A readiness signal that stopped at isActive would hand that on and
    // reproduce the same user-visible failure, one frame later.
    const panel = makeStartingPanel({ blankFramesWhenLive: 3 })
    const capabilities = realWorldCapabilities(panel)
    panel.isActive = true

    expect(await capabilities.captureFrame()).toBe(LIVE_FRAME)
    // Three blanks refused, then the real frame: the wait outlasted them.
    expect(panel.captureFrame).toHaveBeenCalledTimes(4)
  })

  it('gives up at the deadline and blames the camera, rather than waiting forever', async () => {
    // A camera that is denied, absent, or held by another application never
    // goes live. The wait is bounded, so this is still an answer — and by the
    // time it is given, "the camera had not started yet" is no longer the
    // reason: it was given 50 ms and never started at all.
    const panel = makeStartingPanel()
    const capabilities = realWorldCapabilities(panel, { cameraReadyTimeoutMs: 50 })
    const session = createWorldSession({
      worldId: 'real',
      hosts: HOSTS,
      presetId: ACEBOTT_QD021_PRESET.id,
      capabilities,
    })

    const result = JSON.parse(await session.captureImage())

    expect(result).toEqual({ error: 'Webcam not initialized' })
    expect(panel.isActive).toBe(false)
  })

  it('reads isActive off the REAL panel as a boolean, through the ref App.vue holds', async () => {
    // The one way this fix can degrade silently. `isActive` is a `ref()` inside
    // <PkWebcamPanel>, and the package's own .d.ts types the exposed member as
    // `Ref<boolean, boolean>`. If it arrived here as the ref OBJECT rather than
    // its value it would be truthy always, `isReady()` would go back to meaning
    // "mounted", and every test above would still pass because they hand in a
    // plain boolean. Nothing else in the suite — and no browser run — would
    // notice.
    //
    // So this mounts the real published component behind a real template ref,
    // which is precisely what App.vue's `webcamPanelRef` is, and asserts the
    // unwrapping actually happens. jsdom has no `navigator.mediaDevices`, so
    // the panel's startCamera fails and leaves the stream down: a false that
    // has to be a real boolean false, not an object.
    const panelRef = ref<InstanceType<typeof PkWebcamPanel> | null>(null)
    const Parent = defineComponent({
      setup: () => () => h(PkWebcamPanel, { ref: panelRef }),
    })
    const wrapper = mount(Parent)
    await nextTick()

    expect(panelRef.value).not.toBeNull()
    expect(typeof panelRef.value!.isActive).toBe('boolean')
    expect(panelRef.value!.isActive).toBe(false)

    // ...and the capabilities read that same value, so an unmounted camera is
    // reported not-ready rather than ready-because-truthy.
    const { fn } = makeFetch()
    const capabilities = createWorldCapabilities({
      getWorldId: () => 'real',
      hosts: HOSTS,
      getWebcamPanel: () => panelRef.value,
      fetch: fn,
    })
    expect(capabilities.isReady()).toBe(false)

    wrapper.unmount()
  })

  it('does not make the simulated world wait on a camera it never uses', async () => {
    // The simulated world reads frames over HTTP and stops the camera stream
    // outright, so a panel with isActive false is entirely normal there and
    // must not gate anything.
    const panel = makeStartingPanel()
    const { fn, calls } = makeFetch()
    const capabilities = createWorldCapabilities({
      getWorldId: () => 'simulated',
      hosts: HOSTS,
      getWebcamPanel: () => panel,
      fetch: fn,
      cameraReadyPollMs: 1,
      cameraReadyTimeoutMs: 50,
    })

    expect(capabilities.isReady()).toBe(true)
    expect(await capabilities.captureFrame()).toBe(FRAME_DATA_URL)
    expect(calls).toEqual(['http://127.0.0.1:9099/capture'])
  })

  // --- the wait's OTHER caller, and the deadline it spends ------------------

  /**
   * A panel that goes live and whose <video> then never decodes: every frame is
   * the `data:,` a 0x0 canvas serialises to. This is the only shape that
   * reaches the SECOND wait of a tool call — the first is satisfied the moment
   * the stream is live — and the second wait is where the deadlines used to
   * compose instead of bound.
   */
  function makeNeverDecodingPanel() {
    return makeStartingPanel({ blankFramesWhenLive: Number.POSITIVE_INFINITY })
  }

  /**
   * A real-world session over `panel`, the capabilities object behind it, and
   * the URLs its fetch was asked for.
   *
   * The capabilities come back as well as the session because the seam and the
   * direct capture are two different callers of the same object, and telling
   * them apart is the whole point of the deadline-scope specs below.
   */
  function realWorldSession(
    panel: ReturnType<typeof makeStartingPanel>,
    overrides?: { cameraReadyTimeoutMs?: number }
  ) {
    const { fn, calls } = makeFetch()
    const capabilities = createWorldCapabilities({
      getWorldId: () => 'real',
      hosts: HOSTS,
      getWebcamPanel: () => panel,
      fetch: fn,
      cameraReadyPollMs: 1,
      ...overrides,
    })
    const session = createWorldSession({
      worldId: 'real',
      hosts: HOSTS,
      presetId: ACEBOTT_QD021_PRESET.id,
      capabilities,
    })
    return { session, capabilities, calls }
  }

  it('waits for the camera on the MOTION path too, not only for capture_image', async () => {
    // The half of the fix that nothing held. Deleting the wait from runMotion
    // leaves the whole suite green while restoring THIS node's original defect
    // on the motion path: runRecipe gates on isReady() and returns 'Webcam not
    // initialized' before it runs a single step, so a motion tool issued during
    // camera startup is refused for precisely the reason RC-53 exists to
    // remove.
    const panel = makeStartingPanel()
    const { session, calls } = realWorldSession(panel)

    // The stream comes up while the tool call is already waiting.
    const live = setTimeout(() => {
      panel.isActive = true
    }, 20)

    // Through the handler map, because that is the entry point CopilotKit
    // actually calls — a wait on a method nothing routes to would prove nothing.
    const result = JSON.parse(await session.clientToolHandlers.move_forward({}))
    clearTimeout(live)

    expect(result).toEqual({
      mimeType: 'image/jpeg',
      data: 'COMPOSITEBYTES',
      motion: 'move_forward',
    })
    // ...and the robot was actually driven, rather than the call being answered
    // by an error that happens to parse.
    expect(calls).toEqual(['http://10.0.0.7/forward', 'http://10.0.0.7/stop'])
  })

  it('bounds a capture_image call by ONE deadline, not one per wait', async () => {
    // The deadlines used to COMPOSE rather than bound: the readiness wait could
    // spend the whole budget and the capture that followed started a fresh one.
    // Measured at 181 ms against a configured 100 ms — just under 2x, and just
    // under 3x on the motion path below.
    //
    // Fake timers rather than a wall-clock margin: the clock advances by exact
    // amounts, so this asserts "settled within one deadline of simulated time"
    // rather than "fast enough on this machine today".
    vi.useFakeTimers()
    try {
      const panel = makeNeverDecodingPanel()
      const { session } = realWorldSession(panel, { cameraReadyTimeoutMs: 100 })
      // Live at 80 ms — late enough that a second, fresh 100 ms deadline would
      // run on to 180 ms, and early enough to leave the first one 20 ms.
      setTimeout(() => {
        panel.isActive = true
      }, 80)

      const settled: string[] = []
      void session.captureImage().then((r) => {
        settled.push(r)
      })

      await vi.advanceTimersByTimeAsync(90)
      // Still waiting: the shared deadline shortens the call, it does not make
      // the second wait give up the moment the stream goes live.
      expect(settled).toEqual([])

      await vi.advanceTimersByTimeAsync(30)
      expect(settled).toHaveLength(1)
      // vue-ui's frozen envelope string, which is what a capture failure says
      // when its cause is not knowable. This panel reports no camera status, so
      // there is nothing to name — RC-55 changed which sentence is chosen, not
      // this one. A panel that DOES report a cause is covered in
      // rc55CameraStatus.test.ts.
      expect(JSON.parse(settled[0])).toEqual({
        error: 'Failed to capture frame. Is the camera active?',
      })
    } finally {
      vi.useRealTimers()
    }
  })

  it('bounds a MOTION call by one deadline too, and names what expired', async () => {
    // The 3x path: the readiness wait, the Before frame, the After frame, each
    // formerly free to start a deadline of its own.
    vi.useFakeTimers()
    try {
      const panel = makeNeverDecodingPanel()
      const { session, calls } = realWorldSession(panel, { cameraReadyTimeoutMs: 100 })
      setTimeout(() => {
        panel.isActive = true
      }, 80)

      const settled: string[] = []
      void session.clientToolHandlers.move_forward({}).then((r) => {
        settled.push(r)
      })

      await vi.advanceTimersByTimeAsync(90)
      expect(settled).toEqual([])

      await vi.advanceTimersByTimeAsync(30)
      expect(settled).toHaveLength(1)
      // The message is asserted verbatim because it is the whole deliverable
      // here: after a deadline has been waited out, "Is the camera active?"
      // points the model at the one explanation the wait already ruled out.
      expect(JSON.parse(settled[0])).toEqual({
        error:
          'The camera stream produced no usable frame within the deadline, so no Before frame was captured and the robot has not moved.',
        motion: 'move_forward',
      })
      // The message claims the robot has not moved. That is a claim about the
      // wire, so it is checked against the wire.
      expect(calls).toEqual([])
    } finally {
      vi.useRealTimers()
    }
  })

  it('still captures the After frame when the motion outlasted the deadline', async () => {
    // The cost of sharing one deadline, and the one place it could bite a real
    // user. A long walk finishes AFTER the call's window has closed, so the
    // After frame runs against an expired deadline — and must still get its
    // frame, because by then the camera has been streaming the whole time. A
    // wait that checked the deadline BEFORE attempting would fail this motion
    // outright and report a camera fault for a camera that was working.
    vi.useFakeTimers()
    try {
      const panel = makeStartingPanel()
      panel.isActive = true // already streaming: the ordinary case
      const calls: string[] = []
      const slowFetch = vi.fn(async (input: RequestInfo | URL) => {
        calls.push(String(input))
        // Each leg of the walk outlasts the whole camera deadline on its own.
        await new Promise((resolve) => setTimeout(resolve, 150))
        return { ok: true, status: 200, text: async () => 'ok' } as unknown as Response
      })
      const capabilities = createWorldCapabilities({
        getWorldId: () => 'real',
        hosts: HOSTS,
        getWebcamPanel: () => panel,
        fetch: slowFetch as unknown as typeof fetch,
        cameraReadyPollMs: 1,
        cameraReadyTimeoutMs: 100,
      })
      const session = createWorldSession({
        worldId: 'real',
        hosts: HOSTS,
        presetId: ACEBOTT_QD021_PRESET.id,
        capabilities,
      })

      const settled: string[] = []
      void session.clientToolHandlers.move_forward({}).then((r) => {
        settled.push(r)
      })

      await vi.advanceTimersByTimeAsync(400)

      expect(settled).toHaveLength(1)
      expect(JSON.parse(settled[0])).toEqual({
        mimeType: 'image/jpeg',
        data: 'COMPOSITEBYTES',
        motion: 'move_forward',
      })
      expect(calls).toEqual(['http://10.0.0.7/forward', 'http://10.0.0.7/stop'])
    } finally {
      vi.useRealTimers()
    }
  })

  it('honours the INJECTED deadline, so a shorter one really is shorter', async () => {
    // The override exists to make these specs fast, and nothing held it:
    // ignoring it and always using the production 5 s left every spec green,
    // because 5 s fits inside vitest's 10 s ceiling. The deadline VALUE was
    // therefore untested — the specs pinned the message at expiry, not when
    // expiry came.
    vi.useFakeTimers()
    try {
      // Never goes live at all: a denied or absent camera.
      const panel = makeStartingPanel()
      const { session } = realWorldSession(panel, { cameraReadyTimeoutMs: 100 })

      const settled: string[] = []
      void session.captureImage().then((r) => {
        settled.push(r)
      })

      await vi.advanceTimersByTimeAsync(90)
      expect(settled).toEqual([])

      await vi.advanceTimersByTimeAsync(20)
      expect(settled).toHaveLength(1)
      expect(JSON.parse(settled[0])).toEqual({ error: 'Webcam not initialized' })
    } finally {
      vi.useRealTimers()
    }
  })

  // --- the deadline belongs to ONE call --------------------------------------
  //
  // Sharing one deadline between the two waits of a call is the fix above. The
  // two specs here hold the other half of it: the deadline is scoped to the
  // call that opened it. Both use ONE capabilities object, because that is the
  // object App.vue builds once for the app's lifetime and every session reuses
  // — a spec that builds a fresh one per call cannot see either failure.

  it('does not leave a spent deadline behind for the next capture', async () => {
    // The trap a per-app deadline lays. A capabilities object outlives every
    // call made through it, so a deadline stored on it is a permanently-past
    // instant the moment the first call ends: the next capture measures itself
    // against a window that closed, takes a single attempt and reports no
    // frame — for a camera that is streaming perfectly well.
    //
    // Real timers, and a first call that genuinely EXHAUSTS its budget rather
    // than one that succeeds: a first call that finished early would leave a
    // stored deadline still in the future, and this spec would pass over the
    // defect it exists to catch.
    const panel = makeStartingPanel({ blankFramesWhenLive: 3 })
    const { session, capabilities } = realWorldSession(panel, { cameraReadyTimeoutMs: 50 })

    // Call one: the camera never comes up, so the wait spends the whole 50 ms
    // and the call fails at its gate without ever reaching a capture.
    expect(JSON.parse(await session.captureImage())).toEqual({ error: 'Webcam not initialized' })
    expect(panel.captureFrame).not.toHaveBeenCalled()

    // The camera comes up afterwards — the student pressed Retry, or the
    // sensor finally finished spinning up.
    panel.isActive = true

    // A capture issued now is a NEW piece of work and gets a budget of its own:
    // three undecodable frames polled through, then the real one. This is the
    // same assertion the fresh-object spec above makes, on an object that has
    // already served a call.
    expect(await capabilities.captureFrame()).toBe(LIVE_FRAME)
    expect(panel.captureFrame).toHaveBeenCalledTimes(4)
  })

  it('gives two overlapping calls a budget each, not one they share', async () => {
    // The second consequence of storing the deadline: a later call overwrites
    // the budget of one still in flight. The first call then runs on past its
    // own deadline to the second call's — measured at 161 ms against 100 ms —
    // so "one deadline per tool call" would hold only while calls are
    // serialised, and nothing serialises them.
    vi.useFakeTimers()
    try {
      const panel = makeNeverDecodingPanel()
      const { session } = realWorldSession(panel, { cameraReadyTimeoutMs: 100 })
      // Live at 80 ms, decoding never: the only shape that reaches the second
      // wait, so both calls are still inside their captures when they overlap.
      setTimeout(() => {
        panel.isActive = true
      }, 80)

      const first: string[] = []
      const second: string[] = []
      void session.captureImage().then((r) => {
        first.push(r)
      })
      // A second call opens 60 ms in, while the first is still waiting. Its own
      // deadline therefore falls at 160 ms — 60 ms past the first call's.
      setTimeout(() => {
        void session.captureImage().then((r) => {
          second.push(r)
        })
      }, 60)

      await vi.advanceTimersByTimeAsync(130)
      // The first call is 30 ms past its own deadline and must have answered.
      expect(first).toHaveLength(1)
      expect(JSON.parse(first[0])).toEqual({
        error: 'Failed to capture frame. Is the camera active?',
      })
      // ...and the second is still inside its own, which started 60 ms later.
      // Without this the spec would only say the first call was fast, not that
      // the two budgets are separate.
      expect(second).toEqual([])

      await vi.advanceTimersByTimeAsync(50)
      expect(second).toHaveLength(1)
      expect(JSON.parse(second[0])).toEqual({
        error: 'Failed to capture frame. Is the camera active?',
      })
    } finally {
      vi.useRealTimers()
    }
  })

  it('names the AFTER frame when that is the capture that failed', async () => {
    // The other message on the motion path, and the one a control mutation
    // showed was unasserted: the suite stayed green while this string said
    // something else entirely. It is pinned here verbatim so it cannot drift
    // unnoticed. The wording itself is a candidate for the vue-ui-boundary
    // node, not for this spec to pre-empt.
    //
    // Reaching it needs a call where the Before frame lands, the walk happens,
    // and the camera has stopped producing usable frames by the time the After
    // frame is taken — which is why the panel yields exactly one good frame.
    vi.useFakeTimers()
    try {
      let goodFramesLeft = 1
      const panel = {
        isActive: true,
        // 'data:,' is what a 0x0 canvas serialises to: truthy, and not an image.
        captureFrame: vi.fn(() => (goodFramesLeft-- > 0 ? LIVE_FRAME : 'data:,')),
        composeBeforeAfter: vi.fn(
          async (_b: string, _a: string) => 'data:image/jpeg;base64,COMPOSITEBYTES'
        ),
      }
      const calls: string[] = []
      const slowFetch = vi.fn(async (input: RequestInfo | URL) => {
        calls.push(String(input))
        // Each leg outlasts the whole camera deadline, so the After frame is
        // taken well after the call's window has closed and gets its one shot.
        await new Promise((resolve) => setTimeout(resolve, 150))
        return { ok: true, status: 200, text: async () => 'ok' } as unknown as Response
      })
      const capabilities = createWorldCapabilities({
        getWorldId: () => 'real',
        hosts: HOSTS,
        getWebcamPanel: () => panel,
        fetch: slowFetch as unknown as typeof fetch,
        cameraReadyPollMs: 1,
        cameraReadyTimeoutMs: 100,
      })
      const session = createWorldSession({
        worldId: 'real',
        hosts: HOSTS,
        presetId: ACEBOTT_QD021_PRESET.id,
        capabilities,
      })

      const settled: string[] = []
      void session.clientToolHandlers.move_forward({}).then((r) => {
        settled.push(r)
      })

      await vi.advanceTimersByTimeAsync(400)

      expect(settled).toHaveLength(1)
      expect(JSON.parse(settled[0])).toEqual({
        error: 'Failed to capture After frame.',
        motion: 'move_forward',
      })
      // It really was the AFTER frame: the Before frame was taken and the robot
      // was driven before the failure, so this is not the Before-frame message
      // arriving under another name.
      expect(calls).toEqual(['http://10.0.0.7/forward', 'http://10.0.0.7/stop'])
      expect(panel.captureFrame).toHaveBeenCalledTimes(2)
      expect(panel.composeBeforeAfter).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })
})

// --- the App.vue wiring seam ----------------------------------------------

describe('RC-44 App.vue makeSession passes the selection through', () => {
  // The world layer below this is covered thoroughly; these six lines of
  // pass-through were not. A single wrong identifier here — a hardcoded world
  // id, the wrong host set — would build every session for the real robot
  // however the picker is set, and leave the whole suite green. So each field
  // is asserted where it lands, against hand-written literals.

  function fixture(worldId: WorldId) {
    const panel = makePanel()
    const { fn, calls } = makeFetch()
    const capabilities = createWorldCapabilities({
      getWorldId: () => worldId,
      hosts: HOSTS,
      getWebcamPanel: () => panel,
      fetch: fn,
    })
    return { calls, capabilities, panel }
  }

  it('builds a simulated session against the emulator host', () => {
    const { capabilities } = fixture('simulated')

    const session = makeSession({
      worldId: 'simulated',
      presetId: ACEBOTT_QD021_PRESET.id,
      hosts: HOSTS,
      capabilities,
    })

    expect(session.robotHost).toBe('127.0.0.1:9099')
    expect(session.robotUrl('/forward')).toBe('http://127.0.0.1:9099/forward')
  })

  it('builds a real-robot session against the robot host', () => {
    const { capabilities } = fixture('real')

    const session = makeSession({
      worldId: 'real',
      presetId: ACEBOTT_QD021_PRESET.id,
      hosts: HOSTS,
      capabilities,
    })

    expect(session.robotHost).toBe('10.0.0.7')
    expect(session.robotUrl('/forward')).toBe('http://10.0.0.7/forward')
  })

  it('drives the emulator end to end for a simulated selection', async () => {
    // The pass-through proved on the wire rather than on a property: the
    // capabilities and the world both have to arrive for these URLs to appear.
    const { capabilities, calls } = fixture('simulated')

    const session = makeSession({
      worldId: 'simulated',
      presetId: ACEBOTT_QD021_PRESET.id,
      hosts: HOSTS,
      capabilities,
    })
    await session.runMotion(MOVE_FORWARD, { steps: 2 })

    expect(calls).toEqual([
      'http://127.0.0.1:9099/capture',
      'http://127.0.0.1:9099/forward?steps=2',
      'http://127.0.0.1:9099/stop',
      'http://127.0.0.1:9099/capture',
    ])
  })

  it('passes the preset id through to the session tool list', () => {
    const { capabilities } = fixture('real')

    const session = makeSession({
      worldId: 'real',
      presetId: ACEBOTT_QD021_PRESET.id,
      hosts: HOSTS,
      capabilities,
    })

    expect(session.presetId).toBe('ACEBOTT-QD021')
    expect(session.clientTools.map((t) => t.name)).toEqual([
      'capture_image',
      'move_forward',
      'move_backward',
      'turn_left',
      'turn_right',
    ])
  })

  it('passes the AG-UI url through, so the agent label is fetched from it', async () => {
    const { capabilities, calls } = fixture('real')

    const session = makeSession({
      worldId: 'real',
      presetId: ACEBOTT_QD021_PRESET.id,
      hosts: HOSTS,
      capabilities,
      agUiUrl: 'http://agui.example:4321/agents/default/run',
    })
    await session.loadAgentInfo()

    // /info is derived from the url that was handed in; a dropped agUiUrl would
    // instead fall back to fetching '/config.json'.
    expect(calls).toEqual(['http://agui.example:4321/info'])
  })
})

// --- the picker ------------------------------------------------------------

describe('WorldPicker', () => {
  it('offers both worlds, with the real robot first and selected by default', () => {
    const wrapper = mount(WorldPicker, { props: { worlds: WORLDS, modelValue: 'real' } })
    const options = wrapper.findAll('option')
    expect(options.map((o) => o.attributes('value'))).toEqual(['real', 'simulated'])
    expect(options.map((o) => o.text())).toEqual(['Real robot', 'Simulated world'])
    expect((wrapper.find('select').element as HTMLSelectElement).value).toBe('real')
  })

  it('emits the chosen world id on change', async () => {
    const wrapper = mount(WorldPicker, { props: { worlds: WORLDS, modelValue: 'real' } })
    await wrapper.find('select').setValue('simulated')
    expect(wrapper.emitted('update:modelValue')).toEqual([['simulated']])
  })
})
