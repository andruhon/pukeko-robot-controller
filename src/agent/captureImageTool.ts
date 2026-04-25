import { tool } from '@langchain/core/tools';
import type { StructuredToolInterface } from '@langchain/core/tools';
import { z } from 'zod';

export const captureImageTool = tool(
  async () => {
    return 'Client tool stub executed on server';
  },
  {
    name: 'capture_image',
    description:
      'Capture a photo from the robot webcam. Returns the current image of the robot and its surroundings as seen from above.',
    schema: z.object({}),
  }
) as StructuredToolInterface;

(captureImageTool as unknown as { metadata: Record<string, unknown> }).metadata = {
  client: true,
};
