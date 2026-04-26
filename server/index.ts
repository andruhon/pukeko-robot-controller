import { ChatAnthropic } from '@langchain/anthropic';
import { startAgUiServer } from '@gaunt-sloth/api';
import { DEFAULT_CONFIG, type GthConfig } from '@gaunt-sloth/core/config.js';
import { captureImageTool } from '../src/agent/captureImageTool.js';
import { frontendImageInjectionMiddleware } from '../src/agent/frontendImageInjectionMiddleware.js';
import { createRobotTools } from '../src/agent/robotTools.js';

const ROBOT_HOST = process.env.ROBOT_HOST ?? '192.168.4.1';
const PORT = 3000;

const llm = new ChatAnthropic({ model: 'claude-sonnet-4-6' });

const config = {
  ...DEFAULT_CONFIG,
  llm,
  noDefaultPrompts: true,
  tools: [captureImageTool, ...createRobotTools(ROBOT_HOST)],
  middleware: [frontendImageInjectionMiddleware],
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
