import { tool } from '@langchain/core/tools';
import type { StructuredToolInterface } from '@langchain/core/tools';
import { z } from 'zod';

const TIMEOUT_MS = 30_000;

async function callRobot(host: string, path: string, query?: string): Promise<string> {
  const url = `http://${host}${path}${query ? `?${query}` : ''}`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(TIMEOUT_MS) });
    if (!res.ok) {
      return `Robot returned HTTP ${res.status}`;
    }
    return (await res.text()).trim();
  } catch (err) {
    if (err instanceof Error) {
      if (err.name === 'TimeoutError' || err.name === 'AbortError') {
        return `Robot did not respond within ${TIMEOUT_MS}ms — is it powered on and on the same network as ${host}?`;
      }
      return `Failed to reach robot at ${host}: ${err.message}`;
    }
    return `Failed to reach robot at ${host}`;
  }
}

const stepsSchema = z.object({
  steps: z
    .number()
    .int()
    .min(1)
    .max(10)
    .optional()
    .describe(
      'Number of cycles to run. Defaults to 1; capped at 10 by the firmware. Calibration: 1 forward/backward cycle ≈ 1.5 cm; 6 turn cycles ≈ 90° (~15° per turn cycle).'
    ),
});

export const MOTION_TOOL_NAMES = [
  'move_forward',
  'move_backward',
  'turn_left',
  'turn_right',
] as const;

export type MotionToolName = (typeof MOTION_TOOL_NAMES)[number];

export function createRobotTools(host: string): StructuredToolInterface[] {
  // Motion tools are CLIENT-fulfilled: the browser handler reads /distance,
  // captures a Before frame, sends the motion to the robot, captures an After
  // frame, composes them into one image, and returns the envelope. The server
  // body is just a stub — AG-UI routes the call to the browser before it ever
  // reaches the function below.
  const clientMotion = (name: MotionToolName, description: string) => {
    const t = tool(async () => 'Client tool stub executed on server', {
      name,
      description,
      schema: stepsSchema,
    }) as StructuredToolInterface;
    (t as unknown as { metadata: Record<string, unknown> }).metadata = { client: true };
    return t;
  };

  return [
    clientMotion(
      'move_forward',
      'Walk the robot forward. Optional `steps` (1-10) for multiple cycles. ~1.5 cm per cycle. Automatically captures Before/After camera frames and ultrasonic readings on every call — you do not need to call capture_image or read_distance around it.'
    ),
    clientMotion(
      'move_backward',
      'Walk the robot backward. Optional `steps` (1-10). ~1.5 cm per cycle. Automatically captures Before/After camera frames and ultrasonic readings on every call.'
    ),
    clientMotion(
      'turn_left',
      'Rotate the robot left in place. Optional `steps` (1-10). ~15° per cycle; 6 ≈ 90°. Automatically captures Before/After camera frames and ultrasonic readings on every call.'
    ),
    clientMotion(
      'turn_right',
      'Rotate the robot right in place. Optional `steps` (1-10). ~15° per cycle; 6 ≈ 90°. Automatically captures Before/After camera frames and ultrasonic readings on every call.'
    ),
    tool(async () => callRobot(host, '/stop'), {
      name: 'stop',
      description: 'Immediately halt all robot motion.',
      schema: z.object({}),
    }) as StructuredToolInterface,
    tool(async () => callRobot(host, '/distance'), {
      name: 'read_distance',
      description:
        'Read the ultrasonic distance sensor. Returns distance to the nearest obstacle in centimetres ("-1.0" on read failure). Trust this reading: when it disagrees with what the camera shows you (e.g. visually facing a box but reading ~70 cm), the robot is almost certainly mis-aimed — the cone is shooting past the target. Use as an alignment check after every small heading adjustment, not just once at the start of an approach.',
      schema: z.object({}),
    }) as StructuredToolInterface,
    tool(async () => callRobot(host, '/status'), {
      name: 'read_status',
      description:
        'Cheap "is the robot alive" probe. Returns JSON {uptimeMs, lastCommand, lastSteps, lastCommandAtMs, lastDistanceCm}. Useful before issuing a longer sequence of moves; null fields mean the matching endpoint hasn\'t been called yet, and uptimeMs resets to 0 on every robot reboot.',
      schema: z.object({}),
    }) as StructuredToolInterface,
  ];
}
