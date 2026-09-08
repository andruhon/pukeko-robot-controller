// The recipe interpreter (RC-7). A single generic runner that fulfils a
// client-side robot tool by executing its declarative `recipe` (see
// robotPresets/types.ts) against a browser capability context. This is the
// piece that used to be App.vue's hardcoded `runMotion`: the multi-call
// procedure (Before frame → drive endpoint → /stop → After frame → compose →
// return one image) is now DATA on the preset, and this interpreter is the one
// place that turns that data into behaviour. It's a plain module function with
// no Vue/DOM dependency of its own — every side-effecting capability is
// injected via `RobotCapabilities` — so it is unit-testable without mounting
// the SFC (the whole point of the node).
import {
  frameToEnvelope,
  captureFailureMessage,
  CAPTURE_IMAGE_FAILED_ERROR,
  type WebcamStatus,
} from '@galvanized-pukeko/vue-ui';
import type { RobotToolDef, RecipeStep, HttpPath } from './../agent/robotPresets/types.js';

// The browser-side capabilities the interpreter needs. App.vue supplies these
// backed by the mounted <PkWebcamPanel> ref and the real `fetch`; tests supply
// fakes. `robotUrl` + `robotHost` are added by RobotSession from its config.
export interface RobotCapabilities {
  // Whether frames can be captured RIGHT NOW. Guards the pre-motion "Webcam not
  // initialized" case exactly as the old runMotion did. RC-53: for the real
  // world this means the media stream is flowing, not merely that the panel is
  // mounted — see worlds.ts, which is where the definition is stated.
  isReady(): boolean;
  // RC-53: open ONE tool call's camera window. Resolves once `isReady()` holds
  // or once a bounded deadline passes, and hands back the capture bound to that
  // same deadline, so the two waits inside one call measure themselves against
  // one instant instead of each starting a fresh budget.
  //
  // Optional because the interpreter's contract is "ask, then act" and a
  // capability set with no startup window (every test fake, the HTTP snapshot
  // source) has nothing to wait for. RobotSession opens one before either gate
  // so a tool call arriving during the camera's startup waits for the stream
  // instead of being refused for not having one yet.
  //
  // It returns the capture rather than storing it because a deadline is
  // per-CALL state and this object is per-APP: stored on the capabilities it
  // would outlive the call that set it, leaving the next capture measuring
  // itself against a window that closed, and a second call would overwrite the
  // budget of one still in flight. A capture handed back is reachable only
  // through the call that was given it.
  beginCall?(): Promise<CallScopedCapabilities>;
  // May be synchronous (the mounted <PkWebcamPanel>, which draws off a canvas
  // it already has) or asynchronous (an HTTP-backed source, which must fetch
  // the frame). This mirrors vue-ui's own ImageCaptureSource, which allows
  // exactly this union — and `runRecipe` MUST await it. A synchronous-only
  // signature here fails asymmetrically and silently for an async source:
  // capture_image goes through vue-ui's captureImageResult, which awaits, so
  // it works; the motion recipes would store the un-awaited Promise, which is
  // truthy, so the null guard passes and a Promise reaches composeBeforeAfter
  // dressed as a `data:` URL. Nothing throws and the composite is simply wrong.
  captureFrame(): string | null | Promise<string | null>;
  // RC-55: why the camera does or does not have frames, so a failed capture can
  // name its cause instead of asking the model a question the code has already
  // answered. Structurally the same optional member as vue-ui's own
  // ImageCaptureSource.cameraStatus, so the object RobotSession hands to
  // `captureImageResult` satisfies that shape without an adapter.
  //
  // A method, not a property, so it is read at capture time: the status changes
  // underneath a capability set that lives as long as the app does.
  //
  // Optional because a capability set may have no camera to report on — every
  // test fake, and the simulated world, whose frames come over HTTP. Absent, the
  // envelope keeps the frozen message byte for byte.
  cameraStatus?(): WebcamStatus | null | undefined;
  composeBeforeAfter(before: string, after: string): Promise<string | null>;
  fetch: typeof fetch;
  robotUrl(path: string): string;
  robotHost: string;
}

/**
 * The capture-failure message for `caps`' current camera status when that
 * status names a cause, else **null**.
 *
 * Null is what keeps `'Webcam not initialized'` in place for every caller that
 * cannot say why: it means "vue-ui has no better sentence than the frozen one",
 * not "the camera is fine".
 *
 * Membership is DERIVED by asking vue-ui rather than by listing the statuses it
 * maps. That is the whole point — `captureFailureMessage` is the vocabulary
 * authority (RC-55), and a local list of which statuses it happens to name today
 * would be a second copy of that map, free to drift the moment vue-ui adds a
 * status or rewords one. Comparing its answer against the frozen fallback asks
 * the authority the question instead of re-deriving it.
 */
export function namedCameraFailure(
  caps: Pick<RobotCapabilities, 'cameraStatus'>
): string | null {
  const status = caps.cameraStatus?.();
  if (!status) return null;
  const message = captureFailureMessage(status);
  return message === CAPTURE_IMAGE_FAILED_ERROR ? null : message;
}

// The subset App.vue actually provides; RobotSession fills in robotUrl/robotHost.
export type BrowserCapabilities = Omit<RobotCapabilities, 'robotUrl' | 'robotHost'>;

// RC-53: what `beginCall` hands back — the part of the capability set that is
// scoped to one tool call rather than to the app. Only the capture, because the
// camera deadline is the only per-call thing there is. Deliberately the same
// signature as RobotCapabilities' own, so vue-ui's zero-argument
// ImageCaptureSource shape is still satisfied by the object RobotSession builds
// for the call: the deadline travels in the closure, not in a parameter.
export type CallScopedCapabilities = Pick<RobotCapabilities, 'captureFrame'>;

// Parse a `{ mimeType, data }` image envelope out of a `data:` URL, or null if
// the string isn't a well-formed base64 image data URL. Promoted verbatim into
// @galvanized-pukeko/vue-ui with the shared capture_image tool (PLAT-18);
// re-exported here so the interpreter's recipe steps and existing robot-side
// importers keep their path.
export { frameToEnvelope };

// Clamp a tool's `steps` argument to the firmware-supported 1..10 integer
// range, defaulting to 1. Pure; moved here from App.vue verbatim.
export function coerceSteps(args: unknown): number {
  if (args && typeof args === 'object' && 'steps' in args) {
    const raw = (args as { steps?: unknown }).steps;
    if (typeof raw === 'number' && Number.isFinite(raw) && raw >= 1) {
      return Math.min(10, Math.floor(raw));
    }
  }
  return 1;
}

// The label echoed back to the model as the `motion` field: bare tool name for
// a single cycle, `name (steps=N)` for a multi-cycle call. Matches old runMotion.
function motionLabelFor(toolName: string, steps: number): string {
  return steps === 1 ? toolName : `${toolName} (steps=${steps})`;
}

function resolveHttpPath(path: HttpPath, def: RobotToolDef): string {
  if (typeof path === 'string') return path;
  // path.fromDef === 'clientEndpoint'
  const endpoint = def.clientEndpoint;
  if (!endpoint) {
    throw new Error(
      `Recipe for '${def.name}' references clientEndpoint but the tool has none.`
    );
  }
  return endpoint;
}

// Run a client tool's recipe, returning the JSON string the AG-UI client hands
// back to the model — either the success image envelope
// (`{ mimeType, data, motion }`) or a `{ error, motion? }` object. Byte-for-byte
// equivalent to the pre-RC-7 App.vue runMotion for the QD021 MOTION_RECIPE.
export async function runRecipe(
  def: RobotToolDef,
  args: unknown,
  caps: RobotCapabilities
): Promise<string> {
  if (!def.recipe) {
    return JSON.stringify({ error: `Tool '${def.name}' has no recipe to run.` });
  }
  if (!caps.isReady()) {
    // No motion label here — matches the original guard, which fired before
    // the label was computed.
    //
    // RC-55: when the camera can say WHY it has no frames, say that instead.
    // This is the same sentence `RobotSession.captureImage` returns for the same
    // camera, which is what "the two paths agree about the same failure" means:
    // a student who is told the permission was denied by one tool is not told
    // the webcam was never initialised by the other.
    //
    // The fallback is unchanged and load-bearing. A capability set that reports
    // no status — every test fake, the simulated world, a panel too old to
    // expose one — still gets `'Webcam not initialized'`, because that is the
    // honest answer when the reason is genuinely unknown.
    return JSON.stringify({ error: namedCameraFailure(caps) ?? 'Webcam not initialized' });
  }

  const steps = coerceSteps(args);
  const motion = motionLabelFor(def.name, steps);
  const slots: Record<string, string | null> = {};

  for (const raw of def.recipe) {
    const step: RecipeStep = raw;
    switch (step.step) {
      case 'captureFrame': {
        // `await` is load-bearing, not cosmetic: see RobotCapabilities above.
        const frame = await caps.captureFrame();
        if (!frame) {
          return JSON.stringify({ error: step.failMessage, motion });
        }
        slots[step.as] = frame;
        break;
      }
      case 'http': {
        const path = resolveHttpPath(step.path, def);
        const query = step.withSteps && steps > 1 ? `?steps=${steps}` : '';
        const url = caps.robotUrl(`${path}${query}`);
        if (step.optional) {
          // Best-effort side effect (the /stop halt). Its HTTP status is
          // ignored and a throw is logged and stepped over — exactly the old
          // fire-and-forget `await fetch(robotUrl('/stop'))` behaviour.
          try {
            await caps.fetch(url);
          } catch (err) {
            console.warn(`[RobotSession] optional step ${path} failed after ${motion}:`, err);
          }
          break;
        }
        try {
          const res = await caps.fetch(url);
          if (!res.ok) {
            return JSON.stringify({
              error: `Robot returned HTTP ${res.status} for ${path}`,
              motion,
            });
          }
          await res.text();
        } catch (err) {
          const message = err instanceof Error ? err.message : 'unknown error';
          return JSON.stringify({
            error: `Failed to reach robot at ${caps.robotHost}: ${message}`,
            motion,
          });
        }
        break;
      }
      case 'compose': {
        try {
          slots[step.as] = await caps.composeBeforeAfter(
            slots[step.before] ?? '',
            slots[step.after] ?? ''
          );
        } catch (err) {
          const message = err instanceof Error ? err.message : 'compose error';
          return JSON.stringify({
            error: `Failed to compose Before/After image: ${message}`,
            motion,
          });
        }
        break;
      }
      case 'returnImage': {
        const envelope = frameToEnvelope(slots[step.from] ?? null);
        if (!envelope) {
          return JSON.stringify({ error: 'Invalid composite frame format', motion });
        }
        return JSON.stringify({ ...envelope, motion });
      }
    }
  }

  // A well-formed recipe ends with a returnImage step; reaching here means it
  // didn't produce a return value.
  return JSON.stringify({ error: `Recipe for '${def.name}' produced no result.`, motion });
}
