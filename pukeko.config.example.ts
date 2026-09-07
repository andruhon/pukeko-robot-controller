import { defineConfig } from './src/lib/config.js'

// Sample profiles — select one with `PUKEKO_PROFILE=<name> npm run server`.
// Env vars (OLLAMA_MODEL, ANTHROPIC_MODEL, ROBOT_HOST, PUKEKO_VERBOSE=1, ...) override
// individual fields on top of the chosen profile.
//
// Unified middleware stack — the same on every profile:
//   'frontend-images' — surfaces the robot camera frame to the model and the web client.
//   'context-pruner'  — bounds cost: mechanically drops old image BYTES (keepLatestImages)
//                       and summarizes lazily only past summarizeAtFraction × maxContextTokens.
//                       Prefer this over 'motion-summary': motion-summary's per-turn summary
//                       LLM call fails on Anthropic (unpaired tool_use/tool_result → 400) and so
//                       never compresses, leaving the image history to grow unbounded → huge bills.
//   'observability'   — debug logging: per-turn messages/response (+ images) under dumpDir.
// Local (Ollama) models additionally get 'lazy-tool-recovery' (force): small models narrate a
//   tool instead of calling it; this re-prompts so the call streams for real. Do NOT add it to
//   hosted models — it would force a tool onto legitimate plain-text replies.
const PRUNER_LOCAL = { maxContextTokens: 30_000, summarizeAtFraction: 0.7, keepLatestImages: 1, imageTokenBudget: 800 }
const PRUNER_HOSTED = { maxContextTokens: 130_000, summarizeAtFraction: 0.7, keepLatestImages: 1, imageTokenBudget: 800 }
const OBSERVABILITY = { verbose: true, dumpDir: './logs', dumpImages: true }

export default defineConfig({
  defaultProfile: 'gemma-default',
  profiles: {
    'gemma-default': {
      llm: { provider: 'ollama', model: 'gemma4:31b' },
      middleware: ['frontend-images', 'context-pruner', 'observability', 'lazy-tool-recovery'],
      contextPruner: PRUNER_LOCAL,
      observability: OBSERVABILITY,
      lazyToolRecovery: { force: true },
      // Both prompt files default to the repo root and may be overridden per profile:
      // systemPromptPath: 'system-prompt.md',        // behavioural prompt (gaunt-sloth prompts.guidelines)
      // summaryPromptPath: 'summarization-prompt.md', // context-pruner's lazy-summary prompt
      // robot: { host: '192.168.4.1', preset: 'ACEBOTT-QD021' }, // overridable with ROBOT_HOST / ROBOT_PRESET
    },

    // Same local model and same middleware as 'gemma-default', differing ONLY in
    // `llm.ollama` — the generation options, which reach the /api/chat request
    // (RC-50). Held equal that way, running one profile against the other is a
    // clean comparison of the sampling settings and nothing else.
    //
    // The VALUES here are illustrative, not recommended: which settings actually
    // help is an open question that needs a human driving the robot and recording
    // both halves — time-to-complete AND whether the robot still reaches the
    // target. Omit a key to leave ollama's own default in place; a profile that
    // sets none behaves exactly like 'gemma-default'.
    'gemma-tuned': {
      llm: {
        provider: 'ollama',
        model: 'gemma4:31b',
        ollama: {
          // A switch, not a budget — this suppresses thinking rather than shortening it.
          think: false,
          temperature: 0.6, // the model's own default is 1
          topK: 64,
          topP: 0.95,
          repeatPenalty: 1.15,
          // Ollama's default repeat_last_n is a 64-token window, so a repeated
          // passage longer than that is invisible to repeatPenalty. Widen it past
          // the length of whatever is repeating.
          repeatLastN: 512,
          // `numCtx` is available too and is deliberately NOT set here: it changes
          // what the model can see rather than how it samples, so setting it would
          // confound the comparison this profile exists for. It also has to agree
          // with contextPruner — the pruner summarizes at summarizeAtFraction ×
          // maxContextTokens (21k with PRUNER_LOCAL), and ollama silently truncates
          // a prompt longer than num_ctx, so a smaller window drops context with no
          // error. Move the two together or not at all.
        },
      },
      middleware: ['frontend-images', 'context-pruner', 'observability', 'lazy-tool-recovery'],
      contextPruner: PRUNER_LOCAL,
      observability: OBSERVABILITY,
      lazyToolRecovery: { force: true },
    },

    'gpt-5.5': {
      llm: { provider: 'openai', model: 'gpt-5.5' },
      middleware: ['frontend-images', 'context-pruner', 'observability'],
      contextPruner: PRUNER_HOSTED,
      observability: OBSERVABILITY,
    },

    openrouter: {
      llm: { provider: 'openrouter', model: 'google/gemini-2.5-pro' },
      middleware: ['frontend-images', 'context-pruner', 'observability'],
      contextPruner: PRUNER_HOSTED,
      observability: OBSERVABILITY,
    },

    anthropic: {
      // `cache: true` enables Anthropic prompt caching (system prompt + tool schemas re-read at ~0.1x).
      llm: { provider: 'anthropic', model: 'claude-sonnet-4-6', cache: true },
      middleware: ['frontend-images', 'context-pruner', 'observability'],
      contextPruner: PRUNER_HOSTED,
      observability: OBSERVABILITY,
    },
  },
})
