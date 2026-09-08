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
      // gemma4:12b rather than 31b, measured on the dev box (Radeon RX 9060 XT,
      // 15.9 GiB): 12b loads whole into VRAM — 8.4 GB at the 32768-token window
      // below, 100% on the GPU, ~4 s to first token. 31b is 21.5 GB loaded, so a
      // third of it runs on the CPU at ANY window, and widening the window makes
      // the spill worse (66% → 60% on GPU) because KV cache displaces weights;
      // 17–19 s to first token. On a card with room for it, 31b is a defensible
      // choice — but then set it deliberately and expect the CPU spill.
      llm: {
        provider: 'ollama',
        model: 'gemma4:12b',
        ollama: {
          // The window the model is actually given (`num_ctx`), and the one key
          // in this bag that must cover `contextPruner.maxContextTokens` PLUS
          // the system prompt. The pruner lets history grow to that cap and
          // budgets only `state.messages`; the system prompt is sent outside
          // that array and costs window on top of it — roughly 1900 tokens for
          // the shipped `system-prompt.md`, more if `systemPromptPath` points
          // somewhere longer. Ollama then serves only its own window and
          // silently discards the rest FROM THE HEAD — the opening instruction,
          // the framing and the pruner's own summary go first, with no error.
          // Left unset, a stock server serves 4096 against the 30000-token
          // budget below; `loadConfig` warns on any such disagreement, sizing
          // the prompt from the profile's own file rather than assuming this
          // one.
          //
          // So the minimum here is not 30000 — it is 30000 plus the prompt, and
          // the ~2800 tokens between that budget and 32768 are what pays for it.
          // THAT MARGIN IS LOAD-BEARING: do not tidy this number down to the
          // budget. 32768 costs nothing to prefer — ollama's granted window is a
          // power of two anyway, and on 12b, 4096 → 32768 cost 0.3 GB and stayed
          // 100% on the GPU.
          numCtx: 32768,
        },
      },
      middleware: ['frontend-images', 'context-pruner', 'observability', 'lazy-tool-recovery'],
      contextPruner: PRUNER_LOCAL,
      observability: OBSERVABILITY,
      lazyToolRecovery: { force: true },
      // Both prompt files default to the repo root and may be overridden per profile:
      // systemPromptPath: 'system-prompt.md',        // behavioural prompt (gaunt-sloth prompts.guidelines)
      // summaryPromptPath: 'summarization-prompt.md', // context-pruner's lazy-summary prompt
      // robot: { host: '192.168.4.1', preset: 'ACEBOTT-QD021' }, // overridable with ROBOT_HOST / ROBOT_PRESET
    },

    // Same local model, same window and same middleware as 'gemma-default',
    // differing ONLY in the SAMPLING options inside `llm.ollama`, which reach the
    // /api/chat request (RC-50). Held equal that way, running one profile against
    // the other is a clean comparison of the sampling settings and nothing else.
    // `numCtx` is in that bag but is not one of the variables: it changes what the
    // model can SEE rather than how it draws tokens, so it is set identically here
    // — a window that differed between the two would be the loudest difference in
    // the experiment while looking like a sampling knob.
    //
    // The VALUES here are illustrative, not recommended: which settings actually
    // help is an open question that needs a human driving the robot and recording
    // both halves — time-to-complete AND whether the robot still reaches the
    // target. Omit a sampling key to leave ollama's own default in place; strip
    // them all and what is left — numCtx alone — is 'gemma-default'.
    'gemma-tuned': {
      llm: {
        provider: 'ollama',
        model: 'gemma4:12b',
        ollama: {
          // Held identical to 'gemma-default' — see the note above the profile,
          // and the reasoning beside that profile's own numCtx.
          numCtx: 32768,
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
