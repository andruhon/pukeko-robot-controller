import { startAgUiServer } from '@gaunt-sloth/api';
import { DEFAULT_CONFIG, type GthConfig } from '@gaunt-sloth/core/config.js';
import { captureImageTool } from '../src/agent/captureImageTool.js';
import { createFrontendImageInjectionMiddleware } from '../src/agent/frontendImageInjectionMiddleware.js';
import { createMotionSummarizationMiddleware } from '../src/agent/motionSummarizationMiddleware.js';
import { createRobotTools } from '../src/agent/robotTools.js';
import { createLlm } from './createLlm.js';

const ROBOT_HOST = process.env.ROBOT_HOST ?? '192.168.4.1';
const PORT = 3000;

const { provider, llm } = createLlm();
console.log(`[server] Using LLM provider: ${provider}`);

const config = {
  ...DEFAULT_CONFIG,
  llm,
  noDefaultPrompts: true,
  tools: [captureImageTool, ...createRobotTools(ROBOT_HOST)],
  middleware: [
    createFrontendImageInjectionMiddleware({ provider }),
    createMotionSummarizationMiddleware({ llm }),
  ],
  commands: {
    ...DEFAULT_CONFIG.commands,
    api: {
      ...DEFAULT_CONFIG.commands.api,
      filesystem: 'none',
      port: PORT,
      cors: {
        allowOrigin: 'http://localhost:5173',
        allowMethods: 'POST, GET, OPTIONS',
        allowHeaders: 'Content-Type, Accept',
      },
    },
  },
} as unknown as GthConfig;

await startAgUiServer(config, PORT);
