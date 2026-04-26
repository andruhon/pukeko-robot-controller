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
      'Number of cycles to run. Defaults to 1; capped at 10 by the firmware. Calibration: 1 forward/backward cycle ≈ 1.5 cm; 8 turn cycles ≈ 90°.'
    ),
});

export function createRobotTools(host: string): StructuredToolInterface[] {
  const movement = (name: string, path: string, description: string) =>
    tool(
      async (args: { steps?: number }) => {
        const query = args.steps && args.steps > 1 ? `steps=${args.steps}` : undefined;
        return callRobot(host, path, query);
      },
      { name, description, schema: stepsSchema }
    ) as StructuredToolInterface;

  return [
    movement(
      'move_forward',
      '/forward',
      'Walk the robot forward. Optional `steps` (1-10) for multiple cycles. ~1.5 cm per cycle.'
    ),
    movement(
      'move_backward',
      '/backward',
      'Walk the robot backward. Optional `steps` (1-10). ~1.5 cm per cycle.'
    ),
    movement(
      'turn_left',
      '/turn_left',
      'Rotate the robot left in place. Optional `steps` (1-10). ~11° per cycle; 8 ≈ 90°.'
    ),
    movement(
      'turn_right',
      '/turn_right',
      'Rotate the robot right in place. Optional `steps` (1-10). ~11° per cycle; 8 ≈ 90°.'
    ),
    tool(async () => callRobot(host, '/stop'), {
      name: 'stop',
      description: 'Immediately halt all robot motion.',
      schema: z.object({}),
    }) as StructuredToolInterface,
    tool(async () => callRobot(host, '/distance'), {
      name: 'read_distance',
      description:
        'Read the ultrasonic distance sensor. Returns distance to the nearest obstacle in centimetres ("-1.0" on read failure).',
      schema: z.object({}),
    }) as StructuredToolInterface,
  ];
}
