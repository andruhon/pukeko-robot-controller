// Declarative configuration for the robot controller.
//
// Users author either `pukeko.config.ts`, `pukeko.config.js`, or
// `pukeko.config.json` at the project root. The loader in `loadConfig.ts`
// resolves a profile and applies env-var overrides on top.

export type LlmProvider = 'ollama' | 'anthropic' | 'openai' | 'openrouter' | 'google';

/**
 * RC-50 — generation options for the local (Ollama) provider, set per profile and
 * forwarded to `ChatOllama` by `server/createLlm.ts`.
 *
 * Ollama-only. Each provider gets its own sibling key on `LlmSpec` when there is a
 * measured need for one, because their parameter vocabularies do not line up; a
 * single shared bag would have to pick one provider's spelling and translate for
 * the rest. Ollama is the only one built today.
 *
 * Names are ChatOllama's camelCase, which it maps to ollama's snake_case on the
 * way out (`repeatLastN` becomes `repeat_last_n` in the `/api/chat` body).
 *
 * **The rule for adding a key here: `ChatOllama.invocationParams()` must read it
 * off the instance.** That is what decides whether a profile setting it changes
 * the request at all. `stop` is the counter-example and is deliberately absent —
 * ChatOllama does have a constructor-level `stop` field, but `invocationParams`
 * sends the per-call `options.stop` and never the instance's, so exposing it
 * would add a knob wired to nothing, which is the exact defect this node exists
 * to remove.
 *
 * Every key is optional and unset means "send nothing", leaving ollama's own
 * default in place — a profile that sets none produces byte-for-byte the request
 * this repo sent before any of this existed.
 */
export interface OllamaGenerationOptions {
  /**
   * Whether the model thinks before replying.
   *
   * A switch, not a budget: on gemma4:12b the same prompt returned 0 thinking
   * characters with this off, against 297 with it on.
   *
   * Deliberately typed `boolean` and not widened to a string level. The ollama
   * server does accept `'low'` / `'high'`, but @langchain/ollama types the field
   * as `boolean`, so reaching a level would need a cast at a typed boundary — and
   * a cast there would advertise support we cannot back: the one sample we have
   * came out backwards (`'low'` produced 671 thinking characters, `'high'` 334),
   * so there is no evidence a graded budget exists to expose. Widen this only on
   * a measurement that actually shows an ordering.
   */
  think?: boolean;
  /** Context window in tokens (`num_ctx`). */
  numCtx?: number;
  /** Cap on the tokens generated in one reply (`num_predict`). */
  numPredict?: number;
  /** Sampling temperature. gemma4's own default is 1. */
  temperature?: number;
  /** Top-k sampling (`top_k`). gemma4's own default is 64. */
  topK?: number;
  /** Nucleus sampling (`top_p`). gemma4's own default is 0.95. */
  topP?: number;
  /** Penalty applied to tokens already seen (`repeat_penalty`). */
  repeatPenalty?: number;
  /**
   * How many recent tokens `repeatPenalty` looks back over (`repeat_last_n`).
   * Ollama's default is a 64-token window, so a repeated passage longer than the
   * window is invisible to the penalty.
   */
  repeatLastN?: number;
  /** One-off penalty for tokens already present (`presence_penalty`). */
  presencePenalty?: number;
  /** Penalty scaled by how often a token has appeared (`frequency_penalty`). */
  frequencyPenalty?: number;
  /** Sampling seed — fix it to make a run reproducible. */
  seed?: number;
}

export interface LlmSpec {
  provider: LlmProvider;
  model: string;
  baseUrl?: string;
  /**
   * Enable Anthropic prompt caching (`cache_control: { type: 'ephemeral' }`).
   * Anthropic-only — ignored by the other providers. Recommended for every
   * anthropic profile: the large system prompt + tool schemas (and as much of the
   * message-history prefix as survives the pruning middleware) are re-read at ~0.1x
   * instead of billed as full input tokens each turn.
   */
  cache?: boolean;
  /**
   * Generation options for the local model. Read only on an `ollama` profile and
   * ignored by the other providers.
   */
  ollama?: OllamaGenerationOptions;
}

export type BuiltinMiddlewareId =
  | 'frontend-images'
  | 'motion-summary'
  | 'context-pruner'
  | 'lazy-tool-recovery'
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
  // Named robot preset (RC-1) selecting the tool set for this hardware
  // variant — see src/agent/robotPresets/. Defaults to 'ACEBOTT-QD021'
  // (DEFAULT_ROBOT_PRESET_ID) when unset.
  preset?: string;
}

export interface ContextPrunerProfileOpts {
  maxContextTokens?: number;
  summarizeAtFraction?: number;
  keepLatestImages?: number;
  imageTokenBudget?: number;
}

export interface LazyToolRecoveryProfileOpts {
  maxRecoveries?: number;
  skipClassifier?: boolean;
  // Re-prompt on ANY no-tool reply (not just ones that name a tool). The
  // model must call some tool every turn — a real action or finish_task to end.
  // The Ollama-path equivalent of forcing tool_choice. Default false.
  force?: boolean;
}

export interface PukekoProfile {
  llm: LlmSpec;
  // Path to the agent's behavioural system prompt, resolved from the project
  // root. Wired into gaunt-sloth's `prompts.guidelines` slot. Defaults to
  // `system-prompt.md`.
  systemPromptPath?: string;
  // Path to the motion-summarization prompt, resolved from the project root.
  // Defaults to `summarization-prompt.md`; the middleware keeps an identical
  // baked-in copy as a fallback if the file is missing.
  summaryPromptPath?: string;
  middleware?: MiddlewareEntry[];
  observability?: ObservabilityOptions;
  robot?: RobotOptions;
  contextPruner?: ContextPrunerProfileOpts;
  lazyToolRecovery?: LazyToolRecoveryProfileOpts;
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
