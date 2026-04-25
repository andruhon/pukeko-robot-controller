import { ChatAnthropic } from '@langchain/anthropic';
import { startAgUiServer } from '@gaunt-sloth/api';
import { DEFAULT_CONFIG, type GthConfig } from '@gaunt-sloth/core/config.js';
import { captureImageTool } from '../src/agent/captureImageTool.js';
import { frontendImageInjectionMiddleware } from '../src/agent/frontendImageInjectionMiddleware.js';

const ROBOT_HOST = process.env.ROBOT_HOST ?? '192.168.4.1';
const PORT = 3000;

const llm = new ChatAnthropic({ model: 'claude-sonnet-4-6' });

const config = {
  ...DEFAULT_CONFIG,
  llm,
  noDefaultPrompts: true,
  tools: [captureImageTool],
  middleware: [frontendImageInjectionMiddleware],
  customTools: {
    move_forward: {
      command: `curl -sf "http://${ROBOT_HOST}/control?var=robot&val=1" && echo 'Robot moved forward one step'`,
      description:
        'Move the robot forward one step. The robot takes a single step forward then auto-stops.',
    },
    move_backward: {
      command: `curl -sf "http://${ROBOT_HOST}/control?var=robot&val=2" && echo 'Robot moved backward one step'`,
      description:
        'Move the robot backward one step. The robot takes a single step backward then auto-stops.',
    },
    turn_left: {
      command: `curl -sf "http://${ROBOT_HOST}/control?var=robot&val=3" && echo 'Robot turned left'`,
      description: 'Turn the robot left. The robot performs a single left turn then auto-stops.',
    },
    turn_right: {
      command: `curl -sf "http://${ROBOT_HOST}/control?var=robot&val=4" && echo 'Robot turned right'`,
      description: 'Turn the robot right. The robot performs a single right turn then auto-stops.',
    },
    stop: {
      command: `curl -sf "http://${ROBOT_HOST}/control?var=robot&val=8" && echo 'Robot stopped'`,
      description: 'Immediately halt all robot motion.',
    },
    read_distance: {
      command: `curl -sf "http://${ROBOT_HOST}/control?var=sensor&val=distance"`,
      description:
        'Read the ultrasonic distance sensor. Returns distance to the nearest obstacle in centimeters.',
    },
  },
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
