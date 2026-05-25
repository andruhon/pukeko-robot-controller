// Declarative configuration for the robot controller.
//
// Users author either `pukeko.config.ts`, `pukeko.config.js`, or
// `pukeko.config.json` at the project root. The loader in `loadConfig.ts`
// resolves a profile and applies env-var overrides on top.

export type LlmProvider = 'ollama' | 'anthropic';

export interface LlmSpec {
  provider: LlmProvider;
  model: string;
  baseUrl?: string;
}

export type BuiltinMiddlewareId =
  | 'frontend-images'
  | 'motion-summary'
  | 'observability';

// Arbitrary middleware object the user can pass in from a .ts/.js config.
// Kept loose; the loader doesn't introspect it.
export type MiddlewareObject = Record<string, unknown>;

export type MiddlewareEntry = BuiltinMiddlewareId | MiddlewareObject;

export interface ObservabilityOptions {
  verbose: boolean;
  dumpDir?: string;
  dumpImages?: boolean;
}

export interface RobotOptions {
  host?: string;
}

export interface PukekoProfile {
  llm: LlmSpec;
  // Path to the agent's behavioural system prompt, resolved from the project
  // root. Wired into gaunt-sloth's `projectGuidelines` slot. Defaults to
  // `system-prompt.md`.
  systemPromptPath?: string;
  // Path to the motion-summarization prompt, resolved from the project root.
  // Defaults to `summarization-prompt.md`; the middleware keeps an identical
  // baked-in copy as a fallback if the file is missing.
  summaryPromptPath?: string;
  middleware?: MiddlewareEntry[];
  observability?: ObservabilityOptions;
  robot?: RobotOptions;
}

export interface PukekoConfig {
  defaultProfile?: string;
  profiles: Record<string, PukekoProfile>;
}

// Identity helper that gives type inference + IDE completion when authoring
// `pukeko.config.ts`. Pure pass-through at runtime.
export function defineConfig(cfg: PukekoConfig): PukekoConfig {
  return cfg;
}
