// The ACEBOTT-QD021 preset — RC-1's first (and, for this increment, only)
// robot preset. Reproduces byte-for-byte the tool set that used to be
// hardcoded across robotTools.ts and App.vue: move_forward, move_backward,
// turn_left, turn_right, stop, read_distance, read_status, finish_task.
//
// Leaf module: only 'zod' + ../robotToolNames.js (also a leaf). Safe to
// import from the browser bundle (App.vue) as well as the server.
import { z } from 'zod';
import { FINISH_TOOL_NAME, MOTION_TOOL_NAMES } from '../robotToolNames.js';
import type { RecipeStep, RobotPreset } from './types.js';

// IMPORTANT: gaunt-sloth's AG-UI server treats a client-declared run-input
// tool as *authoritative* for client-fulfilled tools — it drops the
// server's own config.tools entry by name and binds the client's
// name/description/parameters straight through to the model
// (buildClientToolStub in @gaunt-sloth/agent's apiAgUiModule.ts). So for
// move_forward/move_backward/turn_left/turn_right, the `jsonSchema` below
// (not `zodSchema`) is what the model actually sees. Pre-RC-1, the
// server's zod description and the client's hand-written JSON Schema
// description for `steps` had already drifted apart (different wording).
// Preserving that exact (pre-existing) split verbatim here — rather than
// consolidating onto one string — is what makes this a true byte-for-byte
// reproduction of prior model-facing behaviour, not just prior server-side
// code.
//
// RC-49: the two figures below used to be conflated as "1 forward/backward
// cycle ≈ 1.5 cm", which was false — the gait is asymmetric and the robot
// profile has always said backward is 1.3 cm (see physical.ts). The prose was
// the copy that was wrong, so the prose is what changed; the profile's numbers
// are the hardware's and are never bent to make a text line up.
const STEPS_DESCRIPTION_SERVER =
  'Number of cycles to run. Defaults to 1; capped at 10 by the firmware. Calibration: 1 forward cycle ≈ 1.5 cm; 1 backward cycle ≈ 1.3 cm; 6 turn cycles ≈ 90° (~15° per turn cycle).';
const STEPS_DESCRIPTION_CLIENT =
  'Number of cycles to run (1-10, defaults to 1). 1 forward cycle ≈ 1.5 cm; 1 backward cycle ≈ 1.3 cm; 1 turn cycle ≈ 15°; 6 turn cycles ≈ 90°.';

const stepsZodSchema = z.object({
  steps: z
    .number()
    .int()
    .min(1)
    .max(10)
    .optional()
    .describe(STEPS_DESCRIPTION_SERVER),
});

const stepsJsonSchema = {
  type: 'object' as const,
  properties: {
    steps: {
      type: 'integer' as const,
      minimum: 1,
      maximum: 10,
      description: STEPS_DESCRIPTION_CLIENT,
    },
  },
  required: [],
};

const emptyZodSchema = z.object({});

const finishZodSchema = z.object({
  status: z
    .enum(['success', 'failed', 'need_input'])
    .describe(
      'success = objective met; failed = cannot proceed; need_input = must ask the operator.'
    ),
  summary: z
    .string()
    .min(1)
    .describe('One line: what was achieved, or what is blocking and what you tried.'),
});

const MOTION_DESCRIPTION_TAIL =
  'Automatically captures Before/After camera frames on every call (no need to call capture_image around it). It does NOT read the distance sensor — call read_distance yourself when you need range.';

// RC-7: the client-side fulfilment recipe shared by all four QD021 motion
// tools. What used to be App.vue's hardcoded `runMotion` sequence now lives
// here as DATA, run by the generic recipe interpreter (src/robotSession/):
// capture a Before frame, drive this tool's own motion endpoint, halt with
// /stop, capture an After frame, compose them, and return the single image.
// The motion http step resolves its path from the def's `clientEndpoint`, so
// one recipe serves /forward, /backward, /turn_left and /turn_right.
//
// The `/stop` step: whether a *bounded* `steps` motion still needs an explicit
// halt is a QD021-firmware GAIT fact — its walk/turn gaits free-run until told
// to stop, so a bounded move must be terminated or the robot keeps going while
// the After frame is captured. That is exactly why the halt belongs here as
// recipe DATA and not as a hardcoded call: a robot whose gait self-terminates
// after `steps` cycles would simply drop this one line and behave correctly.
//
// RC-53: the Before-frame message says what happened rather than asking
// whether the camera is active. By the time a capture fails, the call has
// already waited out the camera-ready deadline (worlds.ts), so "is the camera
// active?" is a question the code has just spent five seconds answering — and
// it points a model at the one explanation that has been ruled out. What the
// model needs instead is the two facts it must decide on: no frame arrived in
// time, and the robot has not moved, because this step runs before the motion
// endpoint is called.
const MOTION_RECIPE: RecipeStep[] = [
  {
    step: 'captureFrame',
    as: 'before',
    failMessage:
      'The camera stream produced no usable frame within the deadline, so no Before frame was captured and the robot has not moved.',
  },
  { step: 'http', path: { fromDef: 'clientEndpoint' }, withSteps: true },
  { step: 'http', path: '/stop', optional: true },
  { step: 'captureFrame', as: 'after', failMessage: 'Failed to capture After frame.' },
  { step: 'compose', before: 'before', after: 'after', as: 'composite' },
  { step: 'returnImage', from: 'composite' },
];

export const ACEBOTT_QD021_PRESET: RobotPreset = {
  id: 'ACEBOTT-QD021',
  name: 'Acebott QD021 (biped)',
  hardwareId: 'ACEBOTT-QD021',
  // No `physical` block, and that is the point rather than an omission: the
  // measurements in DEFAULT_ROBOT_PHYSICAL_PROFILE (physical.ts) ARE this
  // robot's, taken from the same hardware the `steps` calibration text above
  // was measured on. Restating them here would be a second copy of numbers
  // that must agree, i.e. a drift waiting to happen. A preset for a DIFFERENT
  // robot states only the fields where it differs.
  tools: [
    {
      name: MOTION_TOOL_NAMES[0], // 'move_forward'
      // Pre-RC-1, robotTools.ts's (server) and App.vue's (client) text for
      // this one tool had already drifted — "for multiple cycles" only
      // appeared in the server copy. Preserved verbatim on each side via
      // clientDescription (see RobotToolDef) rather than silently unified.
      description: `Walk the robot forward. Optional \`steps\` (1-10) for multiple cycles. ~1.5 cm per cycle. ${MOTION_DESCRIPTION_TAIL}`,
      clientDescription: `Walk the robot forward. Optional \`steps\` (1-10). ~1.5 cm per cycle. ${MOTION_DESCRIPTION_TAIL}`,
      zodSchema: stepsZodSchema,
      jsonSchema: stepsJsonSchema,
      fulfillment: 'client',
      clientEndpoint: '/forward',
      recipe: MOTION_RECIPE,
    },
    {
      name: MOTION_TOOL_NAMES[1], // 'move_backward'
      // RC-49: 1.3, not 1.5 — the gait is asymmetric (physical.ts's
      // `backwardPerCycleCm`), and this text said 1.5 for as long as it existed.
      description: `Walk the robot backward. Optional \`steps\` (1-10). ~1.3 cm per cycle. ${MOTION_DESCRIPTION_TAIL}`,
      zodSchema: stepsZodSchema,
      jsonSchema: stepsJsonSchema,
      fulfillment: 'client',
      clientEndpoint: '/backward',
      recipe: MOTION_RECIPE,
    },
    {
      name: MOTION_TOOL_NAMES[2], // 'turn_left'
      description: `Rotate the robot left in place. Optional \`steps\` (1-10). ~15° per cycle; 6 ≈ 90°. ${MOTION_DESCRIPTION_TAIL}`,
      zodSchema: stepsZodSchema,
      jsonSchema: stepsJsonSchema,
      fulfillment: 'client',
      clientEndpoint: '/turn_left',
      recipe: MOTION_RECIPE,
    },
    {
      name: MOTION_TOOL_NAMES[3], // 'turn_right'
      description: `Rotate the robot right in place. Optional \`steps\` (1-10). ~15° per cycle; 6 ≈ 90°. ${MOTION_DESCRIPTION_TAIL}`,
      zodSchema: stepsZodSchema,
      jsonSchema: stepsJsonSchema,
      fulfillment: 'client',
      clientEndpoint: '/turn_right',
      recipe: MOTION_RECIPE,
    },
    {
      name: 'stop',
      description: 'Immediately halt all robot motion.',
      zodSchema: emptyZodSchema,
      fulfillment: 'server',
      serverPath: '/stop',
    },
    {
      name: 'read_distance',
      description:
        'Read the ultrasonic distance sensor. Returns distance to the nearest obstacle in centimetres ("-1.0" on read failure). Trust this reading: when it disagrees with what the camera shows you (e.g. visually facing a box but reading ~70 cm), the robot is almost certainly mis-aimed — the cone is shooting past the target. Use as an alignment check after every small heading adjustment, not just once at the start of an approach.',
      zodSchema: emptyZodSchema,
      fulfillment: 'server',
      serverPath: '/distance',
    },
    {
      name: 'read_status',
      description:
        'Cheap "is the robot alive" probe. Returns JSON {uptimeMs, lastCommand, lastSteps, lastCommandAtMs, lastDistanceCm}. Useful before issuing a longer sequence of moves; null fields mean the matching endpoint hasn\'t been called yet, and uptimeMs resets to 0 on every robot reboot.',
      zodSchema: emptyZodSchema,
      fulfillment: 'server',
      serverPath: '/status',
    },
    {
      name: FINISH_TOOL_NAME, // 'finish_task'
      description:
        'Call this to END the task — it is the ONLY way to finish. ' +
        'status="success" when the objective is met, "failed" when you cannot proceed, ' +
        '"need_input" when you must ask the operator (e.g. robot out of frame or unreachable). ' +
        'Always include a one-line summary. Never end the task by just going silent. ' +
        'Do at least one real action (capture_image / read_distance / a move) before ending ' +
        'with failed or need_input.',
      zodSchema: finishZodSchema,
      fulfillment: 'server',
      returnDirect: true,
    },
  ],
};
