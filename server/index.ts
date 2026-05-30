import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { startAgUiServer } from '@gaunt-sloth/api';
import { DEFAULT_CONFIG, type GthConfig } from '@gaunt-sloth/core/config.js';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { captureImageTool } from '../src/agent/captureImageTool.js';
import { createFrontendImageInjectionMiddleware } from '../src/agent/frontendImageInjectionMiddleware.js';
import { createMotionSummarizationMiddleware } from '../src/agent/motionSummarizationMiddleware.js';
import { createContextPrunerMiddleware } from '../src/agent/contextPrunerMiddleware.js';
import { createLazyToolRecoveryMiddleware } from '../src/agent/lazyToolRecoveryMiddleware.js';
import { createObservabilityMiddleware } from './observabilityMiddleware.js';
import { createRobotTools } from '../src/agent/robotTools.js';
import { loadConfig } from './loadConfig.js';
import type { MiddlewareEntry, PukekoProfile } from '../src/lib/config.js';
import { createLlm } from './createLlm.js';

const PORT = 3000;
const DEFAULT_ROBOT_HOST = '192.168.4.1';
const DEFAULT_SYSTEM_PROMPT_FILE = 'system-prompt.md';
const DEFAULT_SUMMARY_PROMPT_FILE = 'summarization-prompt.md';

// Read a prompt file relative to the project root. Returns undefined when the
// file is absent so callers can fall back to a baked-in default.
function readPromptFileOrUndefined(relPath: string): string | undefined {
  const abs = resolve(process.cwd(), relPath);
  if (!existsSync(abs)) return undefined;
  return readFileSync(abs, 'utf8');
}

const { configPath, profileName, profile } = await loadConfig();
console.log(
  `[server] Loaded profile '${profileName}'${configPath ? ` from ${configPath}` : ' (fallback defaults — no config file found)'}.`
);

const robotHost = profile.robot?.host ?? DEFAULT_ROBOT_HOST;
const { provider, llm } = createLlm(profile.llm);
console.log(
  `[server] LLM: ${provider} / ${profile.llm.model}${profile.llm.baseUrl ? ` @ ${profile.llm.baseUrl}` : ''}; robot host: ${robotHost}`
);

function buildMiddleware(entries: MiddlewareEntry[] | undefined, llm: BaseChatModel, profile: PukekoProfile): unknown[] {
  const list = entries ?? ['frontend-images', 'motion-summary'];
  const built: unknown[] = [];
  const enabled: string[] = [];
  for (const entry of list) {
    if (typeof entry === 'string') {
      switch (entry) {
        case 'frontend-images':
          built.push(createFrontendImageInjectionMiddleware({ provider }));
          enabled.push(entry);
          break;
        case 'motion-summary': {
          const summaryPrompt = readPromptFileOrUndefined(
            profile.summaryPromptPath ?? DEFAULT_SUMMARY_PROMPT_FILE
          );
          built.push(createMotionSummarizationMiddleware({ llm, summaryPrompt }));
          enabled.push(entry);
          break;
        }
        case 'context-pruner': {
          const summaryPrompt = readPromptFileOrUndefined(
            profile.summaryPromptPath ?? DEFAULT_SUMMARY_PROMPT_FILE
          );
          built.push(
            createContextPrunerMiddleware({
              llm,
              summaryPrompt,
              ...(profile.contextPruner ?? {}),
            })
          );
          enabled.push(entry);
          break;
        }
        case 'lazy-tool-recovery':
          built.push(createLazyToolRecoveryMiddleware(profile.lazyToolRecovery ?? {}));
          enabled.push(entry);
          break;
        case 'observability': {
          const obs = profile.observability;
          if (!obs?.verbose) {
            console.warn(
              `[server] middleware 'observability' listed but profile.observability.verbose is falsy — skipping.`
            );
            break;
          }
          built.push(
            createObservabilityMiddleware({
              dumpDir: obs.dumpDir ?? './logs',
              dumpImages: obs.dumpImages ?? true,
            })
          );
          enabled.push(entry);
          break;
        }
        default:
          console.warn(`[server] Unknown built-in middleware id: '${entry}' — skipped.`);
      }
    } else if (entry && typeof entry === 'object') {
      built.push(entry);
      enabled.push('<custom>');
    }
  }
  console.log(`[server] Middleware: ${enabled.join(', ') || '(none)'}`);
  return built;
}

const middleware = buildMiddleware(profile.middleware, llm, profile);

const config = {
  ...DEFAULT_CONFIG,
  llm,
  // Surfaced by the AG-UI server's /info endpoint so the frontend can show the
  // active model (provider comes from the model's _llmType()).
  modelDisplayName: profile.llm.model,
  noDefaultPrompts: true,
  // The behavioural system prompt rides in gaunt-sloth's `projectGuidelines`
  // slot (a configurable filename), letting us use a clean `system-prompt.md`
  // instead of the hardcoded `.gsloth.system.md`.
  projectGuidelines: profile.systemPromptPath ?? DEFAULT_SYSTEM_PROMPT_FILE,
  tools: [captureImageTool, ...createRobotTools(robotHost)],
  middleware,
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
