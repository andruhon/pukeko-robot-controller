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

/**
 * The profile a checkout with no config file runs.
 *
 * **A fallback is a floor, not a recommendation** — it is chosen for the weakest
 * machine that should still work, because the one user who reaches it is by
 * definition the user who configured nothing. `gemma4:12b` at a `numCtx` of
 * 49152 is the same pair the example profiles ship, so the repo gives one answer
 * about what a local run is.
 *
 * **Why the smaller model, when this decides what runs on ANY machine.**
 * `gemma4:31b` is ~21.5 GB loaded — that is a property of the model, identical
 * everywhere, and only the card decides whether it fits. Under about 22 GiB it
 * spills to the CPU, and widening the window makes the spill worse because KV
 * cache displaces weights, so even a 24 GiB card holds it only barely at the
 * window below. Measured on the dev box (Radeon RX 9060 XT, 15.9 GiB): a third
 * of 31b on the CPU and 17–19 s to first token, against 12b loading whole at
 * 9.2 GB and ~4 s. Nothing has measured 31b fitting anything.
 *
 * The two mistakes are not symmetric. Too small costs answer quality on a big
 * machine, and that machine opts up with `OLLAMA_MODEL=gemma4:31b` — one env var,
 * no config file needed. Too large costs a small machine a run that looks hung,
 * with nothing on screen saying why.
 *
 * **The cost of this choice, for whoever is choosing.** On a card too small for
 * 12b — under roughly 10 GiB once the window below is allocated — the model
 * spills to the CPU and first token takes many seconds. The window is the first
 * number to bring down there, and it has to come down TOGETHER with
 * `contextPruner.maxContextTokens`: they are one decision, and lowering the
 * window alone is the exact disagreement `contextWindowWarning` below exists to
 * report. So the remedy on a small machine is a config file (copy
 * `pukeko.config.example.ts`), not an env var. Note that setting `numCtx` here
 * also means `OLLAMA_CONTEXT_LENGTH` no longer reaches a no-config run: an
 * explicit `num_ctx` in the request overrides the server default. That lever
 * only ever let this path change one of the two numbers, which is why losing it
 * is a cost worth paying rather than a regression.
 *
 * **49152 rather than the bare floor**, for the reason written out beside the
 * example profiles: the floor is `maxContextTokens` + the system prompt + the
 * bound tool descriptions, and the check below can only see the first two. A
 * window sized to what the check asks for can still truncate.
 */
const FALLBACK_PROFILE: PukekoProfile = {
  llm: {
    provider: 'ollama',
    model: 'gemma4:12b',
    ollama: { numCtx: 49152 },
  },
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
 * Tokens the system prompt will cost, on the pruner's own scale — or `null`
 * when no prompt file can be read.
 *
 * **`null` and 0 are different facts, so they are different return values.**
 * `null` means the file could not be read and the real cost is unknown, above
 * whatever is counted; 0 means the file was read and is empty, so the prompt
 * genuinely costs nothing. Both once returned 0, which made the caller describe
 * an empty file as unreadable — a false statement about a configuration that is
 * fine.
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
 * Unreadable — absent, a directory, no permission — yields `null` rather than
 * throwing. A missing prompt file must not turn a startup warning into a startup
 * failure, and the caller says so in the text when it happens instead of
 * printing a number it could not compute.
 */
function systemPromptTokens(profile: PukekoProfile, root: string): number | null {
  try {
    const relPath = profile.systemPromptPath ?? DEFAULT_SYSTEM_PROMPT_FILE;
    return textTokens(readFileSync(resolve(root, relPath), 'utf8'));
  } catch {
    return null;
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
 * It warns rather than throwing, and RC-63 did not change that. The reason has
 * moved, though: the disagreeing shape used to be what a fresh checkout with no
 * config file produced, and `FALLBACK_PROFILE` now sets a window that agrees, so
 * this no longer fires on the repo's own default. What it still fires on is a
 * config file someone wrote — the commonest shape being an ollama profile with
 * no `llm.ollama` block — and turning that into a startup failure would make a
 * degraded run no run at all, on a configuration whose owner may well have
 * raised the window on the server instead.
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
  // An unreadable prompt costs an unknown amount, not zero — but the comparison
  // has to use a number, so it uses the only defensible one and the text below
  // says which of the two cases produced it.
  const promptCost = promptTokens ?? 0;
  const required = budget + promptCost;
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
  // piece the sentence has to name — and "counted", "empty" and "unreadable" are
  // three different facts about that file, each with its own fix, so they get
  // three sentences rather than one that is true of only one of them.
  const budgetClause = `context-pruner sizes this profile's history against maxContextTokens=${budget}`;
  // The floor caveat belongs on every branch, because the term it names is
  // missing from every branch. Bound tool descriptions cost real window and this
  // config cannot size them: the tool set belongs to whoever builds the agent.
  // On the shipped robot preset they measure about 1285 tokens.
  const floorClause =
    `${required} does not count the tool descriptions bound alongside the prompt — the tool set ` +
    `belongs to whoever builds the agent rather than to this config — so treat it as a floor and ` +
    `not as an exact requirement.`;
  const requirement =
    promptTokens === null
      ? `${budgetClause}, and no prompt file could be read at ${promptFile}, so nothing is ` +
        `counted here for the system prompt — which is sent outside that history and has never ` +
        `been part of that budget. ${required} is therefore the pruner's budget alone. ` +
        `${floorClause}`
      : promptTokens === 0
        ? `${budgetClause}, and ${promptFile} was read and is empty, so the system prompt costs ` +
          `nothing here — a prompt is sent outside that history and has never been part of that ` +
          `budget, so any content in that file would cost window on top of it. ${required} is ` +
          `therefore the pruner's budget alone. ${floorClause}`
        : `${budgetClause}, and the system prompt adds about ${promptTokens} on top of it, ` +
          `because it is sent outside that history and has never been part of that budget. Those ` +
          `${promptTokens} cover ${promptFile} alone, at the pruner's own four-characters-per-` +
          `token estimate. ${floorClause}`;

  // Do NOT quote a difference here as an upper bound on what is lost.
  // `maxContextTokens` is not enforced anywhere: with `summarizeAtFraction` it
  // sets the point where a summary is attempted, plus a log line, and the
  // pruner's own notes record rebuilds landing above it. A sentence promising
  // "up to N tokens" would be the same false claim this node is fixing, with a
  // corrected number in it — and so would naming `maxContextTokens` AS the
  // threshold, which is `summarizeAtFraction` times it and never the number in
  // the same sentence.
  const consequence =
    `Past the window ollama discards from the HEAD — where the opening instruction, the framing ` +
    `and the pruner's own summary sit — with no error, and how much it discards is not bounded ` +
    `by the gap between these numbers: maxContextTokens is not a ceiling the pruner enforces. ` +
    `Together with summarizeAtFraction it sets the point at which a summary is ATTEMPTED, which ` +
    `is a fraction of ${budget} and not ${budget} itself, so a rebuilt history can land above ` +
    `maxContextTokens as well.`;

  // Only offer a lower budget when the arithmetic leaves one to offer: with a
  // prompt longer than the whole window, a bigger window is the only fix.
  const loweredBudget = window - promptCost;
  const remedy =
    `Raise llm.ollama.numCtx to at least ${required}` +
    (loweredBudget > 0
      ? `, or lower contextPruner.maxContextTokens to ${loweredBudget} or less.`
      : `.`) +
    // OLLAMA_CONTEXT_LENGTH appeared only in the condition clause above, so it
    // is offered here as a remedy too.
    //
    // RC-63: it is no longer THIS repo's no-config lever. `FALLBACK_PROFILE`
    // now sends an explicit `num_ctx`, which overrides the server default, so a
    // run with no config file neither needs that env var nor is affected by it —
    // and cannot reach this branch at all. The sentence stays because it is a
    // conditional, not a claim about the current run: this function is handed a
    // profile and a root, never a `configPath`, so it cannot know which case it
    // is in, and any caller that builds an ollama profile with no window and no
    // file to edit still has exactly this one lever.
    //
    // But this branch is gated on an unset `numCtx`, NOT on the absence of a
    // config file — the two are different conditions and this function is never
    // told which one holds. The commonest way to reach here is a config file
    // that simply omits `llm.ollama`, where both keys are perfectly editable, so
    // asserting that neither exists to edit was false on exactly the path most
    // people are on. Stated as a conditional, the sentence is true on every
    // firing without the function needing a fact it does not have.
    (configured === undefined
      ? ` Both of those are keys in a config file; where a run has none to edit, ` +
        `OLLAMA_CONTEXT_LENGTH on the ollama server raises the default this profile is falling ` +
        `back to.`
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
  // RC-62. Checked AFTER the env overrides, and the reason is the PROVIDER, not
  // the model: this check never reads `llm.model`. `LLM_PROVIDER` is the one
  // override that reaches it, and it reaches the very first line — a hosted
  // profile flipped to ollama has no `num_ctx` to compare and must be reported,
  // while the check run against the pre-override profile would return null on
  // the hosted provider and say nothing. The ordering is pinned by a spec in
  // `tests/contextWindowAgreement.test.ts`; moving this line above
  // `applyEnvOverrides` reds it.
  //
  // Once per process without a dedupe set, unlike the pruner's once-per-thread
  // warning: this runs at config load, which happens once per server start, so
  // the "once" is the call site rather than bookkeeping.
  const windowWarning = contextWindowWarning(profileName, profile, cwd);
  if (windowWarning) console.warn(windowWarning);
  return { configPath, profileName, profile };
}

// Re-exported for tests.
export { FALLBACK_CONFIG, FALLBACK_PROFILE, CONFIG_FILENAMES };
export type { MiddlewareEntry };
