// RobotSession (RC-7): the unit-testable service extracted out of App.vue.
// It owns everything that used to live inline in the SFC — robotUrl, the
// recipe interpreter, runMotion/capture_image fulfilment, the client tool
// declarations + handlers the chat engine is fed (PLAT-13: as CopilotKit
// `frontendTools` for the headless surface), and the /info agent-label fetch —
// leaving App.vue to only instantiate this, supply the browser capabilities
// (webcam ref + fetch), and render. Mirrors how galvanized-pukeko factors its
// chatService.ts / configService.ts. Because every side effect is injected via
// BrowserCapabilities, runMotion is now reachable (and asserted) in tests
// without mounting the component.
import { ref, type Ref } from 'vue';
import {
  captureImageResult,
  createCaptureImageToolDeclaration,
  type Tool,
} from '@galvanized-pukeko/vue-ui';
import type { VueFrontendTool } from '@copilotkit/vue/v2';
import { DEFAULT_ROBOT_PRESET_ID, getClientToolDefs } from '../agent/robotPresets/index.js';
import type { RobotToolDef } from '../agent/robotPresets/index.js';
import {
  runRecipe,
  namedCameraFailure,
  type BrowserCapabilities,
  type RobotCapabilities,
} from './interpreter.js';

// The robot-flavoured model-facing description for the shared `capture_image`
// tool (PLAT-18: declaration + handler now live in @galvanized-pukeko/vue-ui;
// this override is the only robot-specific part left). Byte-identical to the
// pre-PLAT-18 inline declaration so the model's prompt surface is unchanged.
const CAPTURE_IMAGE_ROBOT_DESCRIPTION =
  'Capture a photo from the robot webcam. Returns the current image of the robot and its surroundings as seen from above.';

// A client-tool handler as RobotSession exposes it (and as the bespoke path
// consumed it): parsed args in, JSON envelope string out.
export type ClientToolHandler = (args: unknown) => Promise<string>;

// PLAT-13: adapt one of this session's AG-UI tool declarations (a hand-written
// JSON Schema `parameters` value) into the `parameters` shape CopilotKit's
// `FrontendTool` takes — a Standard Schema. CopilotKit serializes that schema
// into the AG-UI run-input tool declaration via `createToolSchema`, which
// PREFERS the Standard *JSON Schema* V1 protocol
// (`schema['~standard'].jsonSchema.input({target})`) over converting a zod
// schema. Handing it the preset's own JSON Schema through that protocol keeps
// the model-facing declaration BYTE-IDENTICAL to what the bespoke path sent —
// RC-1's client-authoritative schema/description text (see acebottQd021.ts's
// header on why the client text is load-bearing) — instead of drifting to a
// zod-derived serialization of the *server* schema. `input()` returns a clone
// so CopilotKit's post-processing can never mutate the preset data.
//
// `validate` is deliberately accept-all: the bespoke engine never validated
// client-tool args browser-side either (CopilotKit parses the raw JSON args
// itself, and the recipe interpreter's `coerceSteps` clamps out-of-range
// input), so validating here would CHANGE behaviour, not preserve it.
//
// WARNING (review M1): `createToolSchema` post-processes this emission before
// it hits the wire — it strips a top-level `$schema`, recursively DELETES
// every `additionalProperties` key, and force-defaults `type`/`properties`.
// That is a no-op for today's preset schemas (none use those keys), but a
// future preset that sets e.g. `additionalProperties: false` would silently
// lose it on the wire: byte-identity would break with all tests green except
// the wire-shape test in tests/robotSession.test.ts, which replicates that
// post-processing to pin the actual declaration. Revisit both if a preset
// ever needs those keys.
function jsonSchemaAsParameters(
  jsonSchema: Record<string, unknown>
): NonNullable<VueFrontendTool['parameters']> {
  return {
    '~standard': {
      version: 1,
      vendor: 'pukeko-robot-controller',
      validate: (value: unknown) => ({ value: value as Record<string, unknown> }),
      jsonSchema: { input: () => structuredClone(jsonSchema) },
    },
  } as unknown as NonNullable<VueFrontendTool['parameters']>;
}

export interface RobotSessionOptions {
  robotHost: string;
  // Browser side effects (webcam capture/compose + fetch). Supplied by App.vue
  // from the mounted <PkWebcamPanel> ref; faked in tests.
  capabilities: BrowserCapabilities;
  presetId?: string;
  // The build-time AG-UI URL (App.vue's `__AGUI_URL__` define). When empty,
  // loadAgentInfo falls back to /config.json. Injected so loadAgentInfo is
  // testable without the Vite global.
  agUiUrl?: string;
}

export class RobotSession {
  readonly robotHost: string;
  readonly presetId: string;
  // Provider/model label for the nav header, populated by loadAgentInfo().
  readonly agentLabel: Ref<string> = ref('');

  private readonly caps: RobotCapabilities;
  private readonly agUiUrl: string;
  private readonly motionToolDefs: RobotToolDef[];

  constructor(options: RobotSessionOptions) {
    this.robotHost = options.robotHost;
    this.presetId = options.presetId ?? DEFAULT_ROBOT_PRESET_ID;
    this.agUiUrl = options.agUiUrl ?? '';
    this.motionToolDefs = getClientToolDefs(this.presetId);
    // Compose the full capability context the interpreter needs by adding the
    // host-derived robotUrl on top of the injected browser capabilities.
    this.caps = {
      ...options.capabilities,
      robotHost: this.robotHost,
      robotUrl: (path: string) => this.robotUrl(path),
    };
  }

  robotUrl(path: string): string {
    return `http://${this.robotHost}${path}`;
  }

  // RC-53: the one seam every client-fulfilled tool call enters the browser
  // through, and therefore the one place the startup wait belongs.
  //
  // Both gates below it — runRecipe's own, and captureImageResult's inside
  // vue-ui — ask `isReady()` and refuse outright when the answer is no. That is
  // the right shape for a camera that is absent, and the wrong one for a camera
  // that is starting, which is the state the panel is in for the first frames
  // after the real world is selected. Waiting here, rather than loosening
  // either gate, is what lets readiness keep meaning "the stream is flowing"
  // while a capture issued inside the startup window still returns a frame.
  //
  // It is bounded and never rejects, so a camera that really is unavailable
  // still reaches its gate and still fails there — only later, and for a reason
  // that is no longer "you asked too early".
  //
  // What comes back is the capability set THIS call must use: `beginCall` hands
  // over a capture bound to the deadline it just opened, so both waits inside
  // one call measure themselves against one instant. Handing it over rather
  // than storing it on the long-lived capabilities is what keeps the deadline
  // per-call — the merged object below is a local of one call, dropped when
  // that call returns OR throws, so a spent deadline cannot reach the next
  // caller and two overlapping calls hold one budget each.
  private async capsForCall(): Promise<RobotCapabilities> {
    const scoped = await this.caps.beginCall?.();
    if (!scoped) return this.caps;
    // Drop `beginCall` from the per-call view. Leaving it on would let anything downstream open a
    // SECOND deadline window mid-call and carry on against it, which is the failure this per-call
    // scoping exists to remove — and it would be quieter than the bug it replaced, because there is
    // no longer a stored slot for a probe to catch it in.
    const { beginCall, ...rest } = this.caps;
    return { ...rest, ...scoped };
  }

  // Fulfil one client-side motion tool by running its recipe. Returns the JSON
  // string handed back to the model.
  async runMotion(def: RobotToolDef, args: unknown): Promise<string> {
    return runRecipe(def, args, await this.capsForCall());
  }

  // The generic single-frame capability (capture_image). PLAT-18: delegates to
  // the shared capture layer in @galvanized-pukeko/vue-ui — `this.caps`
  // (isReady + captureFrame, and since RC-55 the optional cameraStatus) already
  // satisfies its ImageCaptureSource shape. The envelope SHAPE
  // ({mimeType,data}/{error}) is frozen there (RC-14 renderers key on it), and
  // every error string still comes from that package: RC-55 changed which of
  // its sentences a failure gets, never the wording, and never this repo's.
  async captureImage(): Promise<string> {
    const caps = await this.capsForCall();
    // RC-55 — why this gate sits HERE, ahead of vue-ui, rather than inside it.
    //
    // `captureImageResult` asks `isReady()` FIRST and only consults
    // `cameraStatus()` after a capture has come back empty. That ordering is
    // right for vue-ui's own panel source, whose `isReady()` means "a panel is
    // mounted", so a denied camera reaches the cause-naming branch. It is the
    // wrong ordering for THIS repo, because RC-53 deliberately narrowed
    // `isReady()` to "the media stream is flowing" (see worlds.ts, which argues
    // that at length). A denied camera is therefore not-ready here, and would
    // short-circuit to `'Webcam not initialized'` — a sentence that names
    // nothing — without `cameraStatus()` ever being read.
    //
    // So the robot answers for the case it can name and delegates every other
    // case unchanged. Widening `isReady()` back to "panel mounted" would get the
    // message right by making readiness lie, and would let a motion recipe drive
    // the robot on a camera that cannot see it.
    if (!caps.isReady()) {
      const named = namedCameraFailure(caps);
      if (named) return JSON.stringify({ error: named });
    }
    return captureImageResult(caps);
  }

  // The AG-UI run-input tool declarations — the shared capture_image
  // (robot-flavoured description) plus the active preset's client-fulfilled
  // motion tools, in preset order. Same shape App.vue used to build inline
  // (RC-1's client<->server parity is preserved: name/description/parameters
  // come straight from the preset). PLAT-13: no longer handed to a chat
  // component directly — `frontendTools` below folds these declarations and
  // `clientToolHandlers` into the CopilotKit registration shape.
  get clientTools(): Tool[] {
    return [
      createCaptureImageToolDeclaration({ description: CAPTURE_IMAGE_ROBOT_DESCRIPTION }),
      ...this.motionToolDefs.map((def) => ({
        name: def.name,
        // clientDescription (when set) is authoritative for what the model
        // sees for a client-fulfilled tool — see RobotToolDef.
        description: def.clientDescription ?? def.description,
        parameters: def.jsonSchema,
      })),
    ];
  }

  // Handlers for the tools above, keyed by name.
  get clientToolHandlers(): Record<string, ClientToolHandler> {
    return {
      capture_image: () => this.captureImage(),
      ...Object.fromEntries(
        this.motionToolDefs.map((def) => {
          if (!def.recipe) {
            throw new Error(`Client-fulfilled tool '${def.name}' is missing a recipe.`);
          }
          return [def.name, (args: unknown) => this.runMotion(def, args)];
        })
      ),
    };
  }

  // PLAT-13: the CopilotKit registration of this session's client tools — the
  // exact `clientTools` declarations above folded together with their
  // `clientToolHandlers`, in the `frontendTools` shape `CopilotKitProvider`
  // takes. CopilotKit declares these in every AG-UI run-input; the gaunt-sloth
  // server binds each as a `metadata.client` interrupt stub, suspends the graph
  // at the call, CopilotKit runs the handler and re-runs the agent with the
  // result as a trailing `tool` message, and the server resumes the suspended
  // run (the C-a flow PLAT-18 proved live). `wrapHandler` lets the host wrap
  // each real handler — App.vue passes the EXT-6 tool-firing tracker so the
  // Tool Belt pulse spans the handler's actual work window, exactly as it
  // wrapped `clientToolHandlers` on the bespoke path.
  //
  // Build this ONCE per session and hand the same array to the provider:
  // CopilotKitProvider requires `frontendTools` to be a stable array (App.vue
  // remounts the provider per preset switch, so per-session is stable enough).
  frontendTools(options?: {
    wrapHandler?: (name: string, handler: ClientToolHandler) => ClientToolHandler;
  }): VueFrontendTool[] {
    const wrap = options?.wrapHandler ?? ((_name, handler) => handler);
    const handlers = this.clientToolHandlers;
    return this.clientTools.map((tool) => {
      const handler = wrap(tool.name, handlers[tool.name]);
      return {
        name: tool.name,
        description: tool.description,
        parameters: jsonSchemaAsParameters(tool.parameters as Record<string, unknown>),
        handler: (args: Record<string, unknown>) => handler(args),
      };
    });
  }

  // Provider/model label fetched live from the AG-UI server's /info endpoint so
  // the nav header always reflects the running profile. Uses the injected
  // agUiUrl, falling back to /config.json. Sets `agentLabel`.
  async loadAgentInfo(): Promise<void> {
    try {
      let agUiUrl = this.agUiUrl;
      if (!agUiUrl) {
        const cfgRes = await this.caps.fetch('/config.json');
        if (cfgRes.ok) agUiUrl = ((await cfgRes.json()).agUiUrl as string) ?? '';
      }
      if (!agUiUrl) return;
      const base = agUiUrl.replace(/\/agents\/.*$/, '');
      const res = await this.caps.fetch(`${base}/info`);
      if (!res.ok) return;
      const info = (await res.json()) as { provider?: string | null; model?: string | null };
      this.agentLabel.value = [info.provider, info.model].filter(Boolean).join(' ');
    } catch (err) {
      console.warn('[RobotSession] Failed to load agent info:', err);
    }
  }
}
