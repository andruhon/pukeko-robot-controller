import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { startAgUiServer } from '@gaunt-sloth/agent';
import { DEFAULT_CONFIG, type GthConfig } from '@gaunt-sloth/core/config.js';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { createFrontendImageInjectionMiddleware } from '../src/agent/frontendImageInjectionMiddleware.js';
import { createMotionSummarizationMiddleware } from '../src/agent/motionSummarizationMiddleware.js';
import { createContextPrunerMiddleware } from '../src/agent/contextPrunerMiddleware.js';
import { createLazyToolRecoveryMiddleware } from '../src/agent/lazyToolRecoveryMiddleware.js';
import { createObservabilityMiddleware } from './observabilityMiddleware.js';
import { createRobotTools } from '../src/agent/robotTools.js';
import { DEFAULT_ROBOT_PRESET_ID } from '../src/agent/robotPresets/index.js';
import { DEFAULT_MIDDLEWARE, loadConfig } from './loadConfig.js';
import type { MiddlewareEntry, PukekoProfile } from '../src/lib/config.js';
import { createLlm } from './createLlm.js';

// OPS-8: port + web CORS origin are env-overridable (loaded from `.env` via the
// `--env-file-if-exists=.env` flag in the `server` npm script; inline env wins).
// Unset falls back to today's 3000 / http://localhost:5173.
const PORT = Number(process.env.AGUI_PORT) || 3000;
const WEB_ORIGIN = process.env.WEB_ORIGIN || 'http://localhost:5173';
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
const robotPreset = profile.robot?.preset ?? DEFAULT_ROBOT_PRESET_ID;
const { provider, llm } = createLlm(profile.llm);
console.log(
  `[server] LLM: ${provider} / ${profile.llm.model}${profile.llm.baseUrl ? ` @ ${profile.llm.baseUrl}` : ''}; robot host: ${robotHost}; robot preset: ${robotPreset}`
);

function buildMiddleware(entries: MiddlewareEntry[] | undefined, llm: BaseChatModel, profile: PukekoProfile): unknown[] {
  // The default lives in loadConfig.ts, next to the fallback profile that uses
  // it and the RC-62 check that has to predict it — an inline copy here is a
  // list this function could quietly build while the check reasoned about a
  // different one.
  const list = entries ?? DEFAULT_MIDDLEWARE;
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
  // Cap the agent's super-steps per run. gaunt-sloth defaults to 1000 (sensible
  // for long coding chains); for this embodied loop that just lets a stuck run
  // grind ~1000 local inferences. 500 makes a stuck loop fail faster.
  recursionLimit: 500,
  // Surfaced by the AG-UI server's /info endpoint so the frontend can show the
  // active model (provider comes from the model's _llmType()).
  modelDisplayName: profile.llm.model,
  noDefaultPrompts: true,
  // The behavioural system prompt rides in gaunt-sloth's `prompts.guidelines`
  // slot (GS2-43: replaces the removed flat `projectGuidelines` key; a bare
  // string is `{ path }` shorthand), letting us use a clean `system-prompt.md`
  // instead of the hardcoded `.gsloth.system.md`.
  prompts: { guidelines: profile.systemPromptPath ?? DEFAULT_SYSTEM_PROMPT_FILE },
  // PLAT-18: no static capture_image stub any more. The UI declares the shared
  // client tool (from @galvanized-pukeko/vue-ui) in every AG-UI run-input, and
  // the gaunt-sloth server binds run-input tools as metadata.client interrupt
  // stubs itself (apiAgUiModule buildClientToolStub) — the static stub here was
  // already being filtered out in favour of that dynamic binding on every UI
  // run. Server-side tools below are the real (server-fulfilled) robot tools.
  tools: [...createRobotTools(robotHost, robotPreset)],
  middleware,
  // Route the AG-UI backend to the lean agent (plain createAgent, no deepagents).
  // The deep backend's /large_tool_results offload writes oversized tool results
  // to a virtual FS; under `filesystem: 'none'` that write is denied on every
  // camera capture, spamming the model into a loop. Lean has no such offload, so
  // the embodied capture->move loop runs clean. Our own robot tools ride in
  // `tools` above and are unaffected by the backend choice.
  agent: { backend: 'lean' },
  commands: {
    ...DEFAULT_CONFIG.commands,
    api: {
      ...DEFAULT_CONFIG.commands.api,
      filesystem: 'none',
      port: PORT,
      cors: {
        allowOrigin: WEB_ORIGIN,
        allowMethods: 'POST, GET, OPTIONS',
        allowHeaders: 'Content-Type, Accept',
      },
    },
  },
} as unknown as GthConfig;

await startAgUiServer(config, PORT);
