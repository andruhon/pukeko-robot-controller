import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import type {
  LlmProvider,
  MiddlewareEntry,
  PukekoConfig,
  PukekoProfile,
} from '../src/lib/config.js';
import { DEFAULT_MAX_CONTEXT_TOKENS, textTokens } from '../src/agent/contextPrunerMiddleware.js';

const CONFIG_FILENAMES = [
  'pukeko.config.ts',
  'pukeko.config.js',
  'pukeko.config.mjs',
  'pukeko.config.json',
] as const;

// The middleware a profile gets when it names none. One definition, read by all
// three places that need it: the fallback profile below, `buildMiddleware` in
// server/index.ts which actually builds them, and the RC-62 check below, which
// can only ask whether the pruner's budget is in force if it knows what an
// unset `middleware` resolves to.
//
// RC-16: 'context-pruner' (not 'motion-summary') — RC-9/RC-12 moved every named
// profile off motion-summary, so a fresh checkout with no config file never
// routes through the deprecated motion-summarization middleware.
export const DEFAULT_MIDDLEWARE: readonly MiddlewareEntry[] = [
  'frontend-images',
  'context-pruner',
];

const FALLBACK_PROFILE: PukekoProfile = {
  llm: { provider: 'ollama', model: 'gemma4:31b' },
  middleware: [...DEFAULT_MIDDLEWARE],
};

/**
 * The context window a stock ollama server serves when the request carries no
 * `num_ctx`.
 *
 * Measured on ollama 0.33.2 (RC-62) with `OLLAMA_CONTEXT_LENGTH` unset and no
 * `num_ctx` in the modelfile: a freshly loaded gemma4:12b reports `CONTEXT 4096`
 * in `ollama ps`. `ollama show` is the wrong instrument for this — it reports the
 * model's ARCHITECTURE context length (262144 for gemma4), which is what the
 * model could support and not what the server allocates.
 */
export const OLLAMA_DEFAULT_NUM_CTX = 4096;

/**
 * The behavioural system prompt a profile gets when it names no
 * `systemPromptPath`, wired into gaunt-sloth's `prompts.guidelines` slot by
 * `server/index.ts`.
 *
 * It lives here rather than beside that call site for the same reason
 * `DEFAULT_MIDDLEWARE` does: the check below has to size the prompt the server
 * will actually load, and two copies of the filename are a way for the loader
 * and the check to drift apart while each looks right on its own.
 */
export const DEFAULT_SYSTEM_PROMPT_FILE = 'system-prompt.md';

const FALLBACK_CONFIG: PukekoConfig = {
  defaultProfile: 'default',
  profiles: { default: FALLBACK_PROFILE },
};

export interface ResolvedConfig {
  configPath: string | null;
  profileName: string;
  profile: PukekoProfile;
}

async function importConfigModule(absPath: string): Promise<PukekoConfig> {
  // `node --import=tsx` registers tsx for both .ts and .js imports, so a
  // dynamic import of either works. JSON is handled separately.
  const mod = await import(pathToFileURL(absPath).href);
  const cfg = (mod.default ?? mod.config ?? mod) as PukekoConfig;
  if (!cfg || typeof cfg !== 'object' || !cfg.profiles) {
    throw new Error(
      `Invalid pukeko config at ${absPath}: missing 'profiles' object.`
    );
  }
  return cfg;
}

function readJsonConfig(absPath: string): PukekoConfig {
  const raw = readFileSync(absPath, 'utf8');
  const cfg = JSON.parse(raw) as PukekoConfig;
  if (!cfg.profiles) {
    throw new Error(`Invalid pukeko config at ${absPath}: missing 'profiles' object.`);
  }
  return cfg;
}

function applyEnvOverrides(profile: PukekoProfile): PukekoProfile {
  const next: PukekoProfile = {
    ...profile,
    llm: { ...profile.llm },
    robot: { ...(profile.robot ?? {}) },
    observability: profile.observability ? { ...profile.observability } : undefined,
  };

  if (process.env.LLM_PROVIDER) {
    next.llm.provider = process.env.LLM_PROVIDER as LlmProvider;
  }
  if (next.llm.provider === 'ollama') {
    if (process.env.OLLAMA_MODEL) next.llm.model = process.env.OLLAMA_MODEL;
    if (process.env.OLLAMA_BASE_URL) next.llm.baseUrl = process.env.OLLAMA_BASE_URL;
  } else if (next.llm.provider === 'anthropic') {
    if (process.env.ANTHROPIC_MODEL) next.llm.model = process.env.ANTHROPIC_MODEL;
  } else if (next.llm.provider === 'openai') {
    if (process.env.OPENAI_MODEL) next.llm.model = process.env.OPENAI_MODEL;
    if (process.env.OPENAI_BASE_URL) next.llm.baseUrl = process.env.OPENAI_BASE_URL;
  } else if (next.llm.provider === 'openrouter') {
    if (process.env.OPENROUTER_MODEL) next.llm.model = process.env.OPENROUTER_MODEL;
    if (process.env.OPENROUTER_BASE_URL) next.llm.baseUrl = process.env.OPENROUTER_BASE_URL;
  } else if (next.llm.provider === 'google') {
    if (process.env.GOOGLE_MODEL) next.llm.model = process.env.GOOGLE_MODEL;
  }
  if (process.env.ROBOT_HOST) {
    next.robot = { ...(next.robot ?? {}), host: process.env.ROBOT_HOST };
  }
  if (process.env.ROBOT_PRESET) {
    next.robot = { ...(next.robot ?? {}), preset: process.env.ROBOT_PRESET };
  }
  if (process.env.PUKEKO_DUMP_DIR) {
    next.observability = {
      verbose: next.observability?.verbose ?? true,
      dumpImages: next.observability?.dumpImages ?? true,
      ...next.observability,
      dumpDir: process.env.PUKEKO_DUMP_DIR,
    };
  }
  if (process.env.PUKEKO_VERBOSE === '1') {
    next.observability = {
      dumpImages: true,
      ...next.observability,
      verbose: true,
    };
  }

  return next;
}

/**
 * Tokens the system prompt will cost, on the pruner's own scale — or 0 when no
 * prompt file can be read.
 *
 * The prompt is the piece the budget arithmetic below was missing.
 * `estimateTokens` sums over `state.messages` only, and the composed prompt is
 * handed to `createAgent` as `systemPrompt`, applied at model-call time outside
 * that array, so it costs real window tokens that no component was counting.
 *
 * **Derived per profile, never a constant**, because `systemPromptPath` lets a
 * profile point at a file of any size. A hardcoded figure would be true for the
 * shipped prompt and false for everyone who overrides it — which is the same
 * defect this check exists to report, one layer down.
 *
 * `root` follows the documented convention for the path (`src/lib/config.ts`:
 * resolved from the project root) and equals `process.cwd()` in production,
 * where `server/index.ts` calls `loadConfig()` with no argument. Note gaunt-
 * sloth resolves the `prompts.guidelines` path itself, so this reads the file it
 * is pointed at rather than mirroring that resolution.
 *
 * Unreadable — absent, a directory, no permission — yields 0 rather than
 * throwing. A missing prompt file must not turn a startup warning into a startup
 * failure, and the caller says so in the text when it happens instead of
 * printing a number it could not compute.
 */
function systemPromptTokens(profile: PukekoProfile, root: string): number {
  try {
    const relPath = profile.systemPromptPath ?? DEFAULT_SYSTEM_PROMPT_FILE;
    return textTokens(readFileSync(resolve(root, relPath), 'utf8'));
  } catch {
    return 0;
  }
}

/**
 * RC-62 — does this profile's context window agree with the pruner's budget?
 *
 * Returns the warning text, or `null` when the two agree. Two components here
 * both believe they are managing context: `context-pruner` sizes the history it
 * hands over against `maxContextTokens`, and the ollama server decides how much
 * of that history the model is actually shown. Nothing made them compare notes,
 * and the failure is silent in both directions — the pruner logs a history it
 * considers within budget, and ollama drops the overflow with no error and no
 * log line of its own. The shipped local profile disagreed by a factor of five
 * (30000 against 4096) for as long as both existed.
 *
 * **The comparison is against `maxContextTokens`, not against the summarize
 * threshold** (`summarizeAtFraction × maxContextTokens`). The threshold is where
 * compression is *attempted*; `maxContextTokens` is the size the pruner permits,
 * and a history is regularly above the threshold — that is what triggers the
 * summary — so a window sized to the threshold would still be overrun.
 *
 * **And the requirement is `maxContextTokens` PLUS the system prompt, which is
 * not in it.** The pruner budgets `state.messages`; the prompt is applied
 * outside that array, so a window sized to the budget exactly is already short
 * by the length of the prompt. Comparing against the budget alone made this
 * check bless the very configuration its own remedy produced — a user who
 * raised `numCtx` to `maxContextTokens`, as it told them to, still truncated,
 * and it then said nothing. `systemPromptTokens` above is why the number is
 * derived from the profile rather than written down here.
 *
 * Ollama-only: it is the one provider whose window this repo sets, and the
 * hosted providers have no `num_ctx` to compare against. Pruner-only: with
 * `context-pruner` absent from the stack, `contextPruner` on the profile is
 * inert and there is no second opinion to disagree with.
 *
 * It warns rather than throwing. The disagreeing shape is what a fresh checkout
 * with no config file produces (no `llm.ollama` block at all), so refusing to
 * start would turn a degraded run into no run.
 */
export function contextWindowWarning(
  profileName: string,
  profile: PukekoProfile,
  root: string = process.cwd()
): string | null {
  if (profile.llm.provider !== 'ollama') return null;
  const middleware = profile.middleware ?? DEFAULT_MIDDLEWARE;
  if (!middleware.includes('context-pruner')) return null;

  const budget = profile.contextPruner?.maxContextTokens ?? DEFAULT_MAX_CONTEXT_TOKENS;
  const configured = profile.llm.ollama?.numCtx;
  const window = configured ?? OLLAMA_DEFAULT_NUM_CTX;
  const promptFile = profile.systemPromptPath ?? DEFAULT_SYSTEM_PROMPT_FILE;
  const promptTokens = systemPromptTokens(profile, root);
  const required = budget + promptTokens;
  if (window >= required) return null;

  // Name the condition that actually holds, rather than a paraphrase that
  // covers both: an unset `numCtx` and one set too low are different facts
  // about the profile and have different fixes, and only one of them lets this
  // warning speak about the server's own default.
  const condition =
    configured === undefined
      ? `sends no llm.ollama.numCtx, so ollama applies its own default — ${OLLAMA_DEFAULT_NUM_CTX} on a stock server, unless OLLAMA_CONTEXT_LENGTH or the model's own modelfile raises it`
      : `sets llm.ollama.numCtx=${configured}`;

  // Say where the requirement comes from, and say what the number does and does
  // not cover. The prompt is the piece the budget never counted, so it is the
  // piece the sentence has to name; and when the file cannot be read there is no
  // figure to quote, which is a different sentence rather than a zero.
  const requirement =
    promptTokens > 0
      ? `context-pruner sizes this profile's history against maxContextTokens=${budget}, and the ` +
        `system prompt adds about ${promptTokens} on top of it, because it is sent outside that ` +
        `history and has never been part of that budget. Those ${promptTokens} cover ` +
        `${promptFile} alone, at the pruner's own four-characters-per-token estimate — the ` +
        `composed prompt also carries tool descriptions this repo does not size, so ${required} ` +
        `is a floor rather than an exact requirement.`
      : `context-pruner sizes this profile's history against maxContextTokens=${budget}, and no ` +
        `prompt file could be read at ${promptFile}, so nothing is counted here for the system ` +
        `prompt — which is sent outside that history and has never been part of that budget. ` +
        `${required} is therefore the pruner's budget alone, and whatever prompt and tool ` +
        `descriptions the server does compose sit above it.`;

  // Do NOT quote a difference here as an upper bound on what is lost.
  // `maxContextTokens` is not enforced anywhere: it sets the summarize threshold
  // and a log line, and the pruner's own notes record rebuilds landing above it.
  // A sentence promising "up to N tokens" would be the same false claim this
  // node is fixing, with a corrected number in it.
  const consequence =
    `Past the window ollama discards from the HEAD — where the opening instruction, the framing ` +
    `and the pruner's own summary sit — with no error, and how much it discards is not bounded ` +
    `by the gap between these numbers: maxContextTokens sets the summarize threshold rather than ` +
    `a ceiling the pruner enforces, so a rebuilt history can land above it.`;

  // Only offer a lower budget when the arithmetic leaves one to offer: with a
  // prompt longer than the whole window, a bigger window is the only fix.
  const loweredBudget = window - promptTokens;
  const remedy =
    `Raise llm.ollama.numCtx to at least ${required}` +
    (loweredBudget > 0
      ? `, or lower contextPruner.maxContextTokens to ${loweredBudget} or less.`
      : `.`) +
    // On the no-config path neither of those keys exists to be edited, and
    // OLLAMA_CONTEXT_LENGTH — the one thing that path can set — appeared only in
    // the condition clause above. Offer it exactly where it applies: it moves
    // the default this profile is falling back to.
    (configured === undefined
      ? ` With no config file, neither key exists to edit and OLLAMA_CONTEXT_LENGTH on the ollama ` +
        `server is the lever that path has — it sets the default this profile is falling back to.`
      : ``);

  return `[config] profile '${profileName}' ${condition}, while it needs a window of at least ${required} tokens: ${requirement} ${consequence} ${remedy}`;
}

function locateConfigFile(cwd: string): string | null {
  for (const name of CONFIG_FILENAMES) {
    const candidate = resolve(cwd, name);
    if (existsSync(candidate)) return candidate;
  }
  return null;
}

export async function loadConfig(cwd: string = process.cwd()): Promise<ResolvedConfig> {
  const configPath = locateConfigFile(cwd);
  let cfg: PukekoConfig;
  if (configPath) {
    cfg = configPath.endsWith('.json')
      ? readJsonConfig(configPath)
      : await importConfigModule(configPath);
  } else {
    cfg = FALLBACK_CONFIG;
  }

  const requested = process.env.PUKEKO_PROFILE ?? cfg.defaultProfile;
  const available = Object.keys(cfg.profiles);
  const profileName =
    requested && available.includes(requested) ? requested : available[0];
  if (!profileName) {
    throw new Error(`pukeko config at ${configPath} has no profiles defined.`);
  }
  if (requested && requested !== profileName) {
    console.warn(
      `[config] PUKEKO_PROFILE='${requested}' not found; falling back to '${profileName}'. Available: ${available.join(', ')}`
    );
  }

  const profile = applyEnvOverrides(cfg.profiles[profileName]);
  // RC-62. Checked AFTER the env overrides, since one of them can change the
  // model. Once per process without a dedupe set, unlike the pruner's
  // once-per-thread warning: this runs at config load, which happens once per
  // server start, so the "once" is the call site rather than bookkeeping.
  const windowWarning = contextWindowWarning(profileName, profile, cwd);
  if (windowWarning) console.warn(windowWarning);
  return { configPath, profileName, profile };
}

// Re-exported for tests.
export { FALLBACK_CONFIG, FALLBACK_PROFILE, CONFIG_FILENAMES };
export type { MiddlewareEntry };
