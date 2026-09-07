// RC-44: which world the student is driving — the real hardware, or the
// simulated grid world served by `robot-emulator/`.
//
// This is a robot-TARGET selector, not a camera selector. The emulated world
// only advances when `/forward`, `/turn_left` and the rest actually REACH the
// emulator, so a switch that repointed only the image source would leave the
// student watching a static picture while their commands flew off to hardware
// that is not there. Both the motion URLs and the frames follow the choice, and
// that is why the whole thing hangs off one host per world.
//
// Everything here is plain, Vue-free and injectable so the wiring is unit-
// testable without mounting App.vue — the same reason RobotSession itself was
// extracted (RC-7). App.vue supplies the reactive getters and the real fetch.
import {
  createHttpSnapshotCaptureSource,
  frameToEnvelope,
  type ImageCaptureSource,
} from '@galvanized-pukeko/vue-ui';
import { RobotSession, type RobotSessionOptions } from './RobotSession.js';
import type { BrowserCapabilities } from './interpreter.js';

/** The real Acebott biped on its access point. */
export const REAL_ROBOT_WORLD_ID = 'real';
/** The `robot-emulator/` grid world. */
export const SIMULATED_WORLD_ID = 'simulated';

export type WorldId = typeof REAL_ROBOT_WORLD_ID | typeof SIMULATED_WORLD_ID;

export interface WorldOption {
  id: WorldId;
  /** The picker's option label. */
  name: string;
  /** The short badge the Cockpit shows so the active world is obvious at a glance. */
  badge: string;
}

/**
 * The selectable worlds, real first — the default, and today's behaviour.
 * Ordered as the picker renders them.
 */
export const WORLDS: readonly WorldOption[] = [
  { id: REAL_ROBOT_WORLD_ID, name: 'Real robot', badge: 'Real robot' },
  { id: SIMULATED_WORLD_ID, name: 'Simulated world', badge: 'Simulated world' },
];

/** The Acebott's own access-point address, unchanged from before RC-44. */
export const DEFAULT_ROBOT_HOST = '192.168.4.1';

/**
 * Where `pnpm run emulator` listens by default — ROBOT_EMULATOR_PORT's own
 * default, 8081 (see robot-emulator/index.ts). Overridden at build time by
 * VITE_ROBOT_EMULATOR_HOST, exactly as VITE_ROBOT_HOST overrides the robot's.
 */
export const DEFAULT_EMULATOR_HOST = 'localhost:8081';

/** The emulator's rendered-frame endpoint. */
export const CAPTURE_PATH = '/capture';

/**
 * RC-53. How long a capture waits for the real world's camera to start
 * producing frames before it reports a failure.
 *
 * Selecting the real world mounts the panel synchronously, but neither the
 * panel's `getUserMedia` call nor the video element's first decoded frame is
 * synchronous. Measured against Chromium's fake device the stream went live
 * 133 ms after the run started while a capture was issued at 110 ms, so the
 * capture lost by 23 ms; real hardware is slower still, because a sensor has to
 * spin up. That window used to fail the capture outright. It now waits.
 *
 * 5 s matches vue-ui's own DEFAULT_HTTP_SNAPSHOT_TIMEOUT_MS — the deadline the
 * other capture source in this stack already uses — and is far longer than any
 * camera start observed here. Exhausting it is therefore a REAL failure (the
 * camera is denied, absent, or held by another application), not a warm-up,
 * which is what makes the "Is the camera active?" the model then sees an honest
 * question rather than a race reported as a fault.
 */
export const CAMERA_READY_TIMEOUT_MS = 5_000;

/**
 * How often the bounded wait re-checks. Short enough that the wait costs at
 * most about one frame beyond the camera actually being ready.
 */
export const CAMERA_READY_POLL_MS = 25;

/** The hosts the two worlds live on, resolved from env by the caller. */
export interface WorldHosts {
  robotHost: string;
  emulatorHost: string;
}

/**
 * Read a host out of a build-time env var, falling back to its default. An
 * empty-but-present var counts as unset, matching App.vue's `?? host`
 * convention and `resolveSeedPreset`.
 */
export function resolveHost(raw: string | undefined, fallback: string): string {
  return raw && raw.trim().length > 0 ? raw.trim() : fallback;
}

/** The host every HTTP call for `worldId` goes to — motion endpoints included. */
export function hostForWorld(worldId: WorldId, hosts: WorldHosts): string {
  return worldId === SIMULATED_WORLD_ID ? hosts.emulatorHost : hosts.robotHost;
}

/**
 * The snapshot endpoint to fetch frames from for `worldId`, or **null** when
 * the world has none. The real robot's frames come off the mounted
 * <PkWebcamPanel>'s canvas rather than over HTTP, so `null` there is the honest
 * answer, and it is also the value createHttpSnapshotCaptureSource reads as "no
 * target configured" — its `isReady()` reports false and it never fetches.
 */
export function captureUrlForWorld(worldId: WorldId, hosts: WorldHosts): string | null {
  if (worldId !== SIMULATED_WORLD_ID) return null;
  return `http://${hosts.emulatorHost}${CAPTURE_PATH}`;
}

/** The bits of a mounted <PkWebcamPanel> the capabilities need. */
export interface WebcamPanelLike {
  /**
   * Whether the panel's media stream is live: `getUserMedia` has resolved and
   * the <video> element is wired to the stream. FALSE throughout the panel's
   * first moments — the window RC-53 closes — and false again after
   * `stopCamera()`, which is what the simulated world does to the panel it
   * keeps mounted.
   *
   * Required rather than optional on purpose. There is no honest default for
   * an absent value: read as live it silently reinstates the race for every
   * caller that forgets it, and read as not-live it strands them. Requiring it
   * makes the type-checker enumerate the construction sites instead.
   */
  isActive: boolean;
  captureFrame(): string | null;
  composeBeforeAfter(before: string, after: string): Promise<string | null>;
}

export interface WorldCapabilitiesDeps {
  /** The world selected RIGHT NOW — read per capture, never captured at construction. */
  getWorldId: () => WorldId;
  hosts: WorldHosts;
  /** The mounted panel, read lazily: it need not exist yet at wiring time. */
  getWebcamPanel: () => WebcamPanelLike | null | undefined;
  fetch: typeof fetch;
  /**
   * Called with every frame the simulated world produced — the resolved data
   * URL, or null when the fetch failed. This is how the Cockpit's simulated
   * viewport stays current WITHOUT a timer: the app already fetches a frame for
   * each capture_image and twice per motion recipe, so the viewport simply
   * follows the frames the agent was going to pull anyway. Do not "fix" this
   * into a polling loop — an idle simulator has nothing new to show, and a poll
   * would burn a render per tick to prove it.
   */
  onSimulatedFrame?: (frame: string | null) => void;
  /** Injected in tests so the snapshot source can be observed; defaults to the real one. */
  createSnapshotSource?: typeof createHttpSnapshotCaptureSource;
  /**
   * Overrides {@link CAMERA_READY_TIMEOUT_MS}. Injected so a test can exercise
   * the deadline in milliseconds rather than seconds — the timeout is part of
   * the behaviour, so a test that never reaches it has not covered it.
   */
  cameraReadyTimeoutMs?: number;
  /** Overrides {@link CAMERA_READY_POLL_MS}, for the same reason. */
  cameraReadyPollMs?: number;
}

const sleep = (ms: number): Promise<void> =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * The panel's current frame if it is one the rest of the stack can actually
 * use, else null.
 *
 * The validator is `frameToEnvelope` — vue-ui's own parser, the one
 * `capture_image`'s envelope is built with — so this predicate cannot disagree
 * with its consumer about what counts as a frame. That is load-bearing, not
 * tidiness: a live stream whose <video> has not decoded its metadata yet still
 * reports `videoWidth` 0, and a 0x0 canvas serialises to the string `data:,`,
 * which is truthy and is not an image. A hand-rolled check that accepted it
 * would reproduce exactly the failure this node exists to remove, one frame
 * later.
 */
function decodableFrame(panel: WebcamPanelLike): string | null {
  let frame: string | null;
  try {
    frame = panel.captureFrame();
  } catch {
    // A capture can throw while the element is being torn down. Treat it as
    // "not yet" and let the deadline decide — an escaping rejection would be
    // reported as a failed run rather than as a failed capture.
    return null;
  }
  return frameToEnvelope(frame) ? frame : null;
}

/**
 * The browser capabilities App.vue hands every RobotSession, dispatching each
 * capture on the CURRENTLY selected world.
 *
 * Built once for the app's lifetime and reused across session re-instantiation:
 * the HTTP snapshot source reads its URL through a getter on every capture, so
 * one source serves both worlds and keeps addressing whichever is selected now.
 *
 * `composeBeforeAfter` always goes to the webcam panel, in both worlds. It
 * draws on that component's hidden canvas rather than on the camera stream, so
 * it composes simulated frames perfectly well — provided the component is still
 * mounted. A `v-if` that removed it would null the ref and break every motion
 * composite with the whole suite green, which is why App.vue keeps the panel
 * mounted and merely stops it streaming.
 */
export function createWorldCapabilities(deps: WorldCapabilitiesDeps): BrowserCapabilities {
  const makeSource = deps.createSnapshotSource ?? createHttpSnapshotCaptureSource;
  const snapshotSource: ImageCaptureSource = makeSource({
    getUrl: () => captureUrlForWorld(deps.getWorldId(), deps.hosts),
    fetch: deps.fetch,
  });

  async function captureSimulatedFrame(): Promise<string | null> {
    const frame = await snapshotSource.captureFrame();
    deps.onSimulatedFrame?.(frame);
    return frame;
  }

  const readyTimeoutMs = deps.cameraReadyTimeoutMs ?? CAMERA_READY_TIMEOUT_MS;
  const readyPollMs = deps.cameraReadyPollMs ?? CAMERA_READY_POLL_MS;

  // RC-53 — WHAT "READY" MEANS HERE, so the next reader does not have to
  // re-derive which of two meanings is in force.
  //
  // Readiness is THE MEDIA STREAM IS FLOWING. It is NOT "the webcam panel is
  // mounted", which is what this predicate used to say. Those are two different
  // instants: mounting is synchronous and immediate, whereas the panel's
  // getUserMedia and its first decoded frame are neither, so "mounted" reported
  // ready for a window in which every capture came back empty and the model was
  // told the camera was inactive when it was merely starting.
  //
  // The capture path agrees with this definition rather than merely being
  // guarded by it: `whenReady` below lets a caller wait for the stream instead
  // of being refused, and `captureRealFrame` waits again for a frame that
  // actually decodes. Both are bounded by CAMERA_READY_TIMEOUT_MS, so a camera
  // that is genuinely absent still fails, and fails for that reason.
  function isReady(): boolean {
    const panel = deps.getWebcamPanel();
    // The panel is required in BOTH worlds: composeBeforeAfter lives on it.
    if (panel == null) return false;
    // The simulated world reads frames over HTTP and never touches the camera,
    // so what it needs ready is a configured snapshot target.
    if (deps.getWorldId() === SIMULATED_WORLD_ID) return snapshotSource.isReady();
    return panel.isActive;
  }

  /**
   * Resolve once readiness holds, or once the deadline passes — never reject.
   * Callers gate on `isReady()` afterwards, so a timeout surfaces as the
   * ordinary not-ready failure rather than as a thrown run error.
   */
  async function whenReady(): Promise<void> {
    const deadline = Date.now() + readyTimeoutMs;
    while (!isReady() && Date.now() < deadline) {
      await sleep(readyPollMs);
    }
  }

  /**
   * The real world's frame: the first one the panel produces that actually
   * decodes, or null once the deadline passes. See `decodableFrame` for why a
   * live stream is not on its own enough.
   */
  async function captureRealFrame(): Promise<string | null> {
    const deadline = Date.now() + readyTimeoutMs;
    for (;;) {
      const panel = deps.getWebcamPanel();
      if (panel != null && panel.isActive) {
        const frame = decodableFrame(panel);
        if (frame != null) return frame;
      }
      if (Date.now() >= deadline) return null;
      await sleep(readyPollMs);
    }
  }

  return {
    isReady,
    whenReady,
    captureFrame: () => {
      if (deps.getWorldId() === SIMULATED_WORLD_ID) return captureSimulatedFrame();
      return captureRealFrame();
    },
    composeBeforeAfter: (before, after) =>
      deps.getWebcamPanel()?.composeBeforeAfter(before, after) ?? Promise.resolve(null),
    fetch: deps.fetch,
  };
}

export interface WorldSessionOptions extends Omit<RobotSessionOptions, 'robotHost'> {
  worldId: WorldId;
  hosts: WorldHosts;
}

/**
 * A RobotSession pointed at `worldId`'s host, so every motion URL the recipe
 * interpreter builds addresses that world.
 *
 * `robotHost` is readonly by design, so switching worlds means re-instantiating
 * the session — the same move a preset switch already makes (see App.vue's
 * `makeSession` + the `:key` remount of <CopilotKitProvider>), and for the same
 * reason: a conversation describing a world that no longer exists is worse than
 * a clean start.
 */
export function createWorldSession(options: WorldSessionOptions): RobotSession {
  const { worldId, hosts, ...rest } = options;
  return new RobotSession({ ...rest, robotHost: hostForWorld(worldId, hosts) });
}
