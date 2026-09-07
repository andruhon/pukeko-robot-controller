import { ChatAnthropic } from '@langchain/anthropic';
import { ChatGoogle } from '@langchain/google/node';
import { ChatOllama } from '@langchain/ollama';
import { ChatOpenAI } from '@langchain/openai';
import { ChatOpenRouter } from '@langchain/openrouter';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { LlmProvider, LlmSpec } from '../src/lib/config.js';
import { ScriptedRobotChatModel } from './test-support/scriptedRobotModel.js';

// Client-fulfilled tools (capture_image, motions) trigger a langgraph
// interrupt(). When the model batches several tool calls into one assistant
// message, the interrupt fires mid-batch and the history ends up with a
// tool_use that has no immediately-following tool_result — which the
// tool-calling providers reject on the next turn. Disabling parallel tool
// use keeps it to one tool call per assistant turn. (Anthropic exposes this
// via invocationKwargs.tool_choice; the OpenAI-shaped providers take it as a
// request-body param via modelKwargs.)
//
// We deliberately do NOT force a tool every turn. Forcing tool_choice
// ("required"/"any") was tried to make the "narrate-but-don't-call" stall
// impossible, but on capable hosted models it backfires badly: the model can
// then never emit a plain-text reply (every turn is a tool-only message, so the
// UI never shows the agent "talking"), and with no "just answer / stop" escape
// hatch it loops on a tool — usually capture_image — when uncertain, instead of
// reading a sensor or calling finish_task. So we use tool_choice "auto": the
// model talks when it should, acts when it should, and ends naturally (via
// finish_task or a final text reply). The narrate-but-don't-call case is covered
// by the `lazy-tool-recovery` middleware instead. NOTE: ChatOllama does not
// support tool_choice at all (`tool_choice?: never`), so the Ollama/Gemma path
// is unset anyway and leans on lazy-tool-recovery.
const NO_PARALLEL_TOOLS = { parallel_tool_calls: false, tool_choice: 'auto' } as const;

export type { LlmProvider, LlmSpec };

export interface LlmSelection {
  provider: LlmProvider;
  llm: BaseChatModel;
}

export function createLlm(spec: LlmSpec): LlmSelection {
  // E2E seam: a deterministic scripted tool-calling model (move_forward →
  // finish_task), no network. Guarded by env so it can never engage in normal
  // runs. See server/test-support/scriptedRobotModel.ts.
  if (process.env.PUKEKO_FAKE_LLM === '1') {
    return { provider: spec.provider, llm: new ScriptedRobotChatModel() };
  }

  if (spec.provider === 'ollama') {
    return {
      provider: 'ollama',
      // RC-50: generation options (temperature, repeatLastN, think, …) come from
      // the profile's `llm.ollama`. They are spread FIRST so that baseUrl and
      // model — which say *what is being called*, not how it should generate —
      // are written last and cannot be shadowed: a `pukeko.config.json` is parsed
      // at runtime and never meets the type checker, so a key the interface does
      // not declare can still arrive in that bag.
      //
      // With no options set the spread contributes nothing and this is exactly
      // the two-field constructor argument it has always been. That is not a
      // side effect to rely on quietly — every existing profile is on that path
      // and every smoke observation was taken there, so `tests/createLlm.test.ts`
      // asserts the resulting request body whole.
      llm: new ChatOllama({
        ...spec.ollama,
        baseUrl: spec.baseUrl ?? 'http://localhost:11434',
        model: spec.model,
      }),
    };
  }

  if (spec.provider === 'anthropic') {
    return {
      provider: 'anthropic',
      // See NO_PARALLEL_TOOLS — Anthropic spells both controls on tool_choice:
      // type "auto" lets the model talk or act per turn (not forced);
      // disable_parallel_tool_use keeps it to one call so the interrupt ordering
      // holds.
      llm: new ChatAnthropic({
        model: spec.model,
        invocationKwargs: {
          tool_choice: { type: 'auto', disable_parallel_tool_use: true },
        },
        // Prompt caching (opt-in via `cache: true` on the profile's llm). A single
        // top-level cache_control makes @langchain/anthropic place — and advance across
        // turns — the cache breakpoint automatically, so the stable system prompt + tool
        // schemas are re-read at ~0.1x instead of billed as full input tokens every turn.
        ...(spec.cache ? { cache_control: { type: 'ephemeral' as const } } : {}),
      }),
    };
  }

  if (spec.provider === 'openai') {
    return {
      provider: 'openai',
      // apiKey falls back to OPENAI_API_KEY; baseUrl (if set) lets users point
      // at an OpenAI-compatible endpoint.
      llm: new ChatOpenAI({
        model: spec.model,
        modelKwargs: NO_PARALLEL_TOOLS,
        ...(spec.baseUrl ? { configuration: { baseURL: spec.baseUrl } } : {}),
      }),
    };
  }

  if (spec.provider === 'openrouter') {
    return {
      provider: 'openrouter',
      // apiKey falls back to OPENROUTER_API_KEY; baseUrl overrides the default
      // https://openrouter.ai/api/v1.
      llm: new ChatOpenRouter({
        model: spec.model,
        modelKwargs: NO_PARALLEL_TOOLS,
        ...(spec.baseUrl ? { baseURL: spec.baseUrl } : {}),
      }),
    };
  }

  if (spec.provider === 'google') {
    return {
      provider: 'google',
      // Native Google AI Studio (Gemini). apiKey falls back to GOOGLE_API_KEY;
      // platformType 'gai' selects the AI Studio endpoint (not Vertex).
      // NOTE on forced tool choice: unlike the OpenAI/Anthropic paths above,
      // ChatGoogle only honours `tool_choice` as a per-call option (mapped to
      // Gemini's functionCallingConfig), not as a constructor field — and
      // setting it via `.withConfig`/`.bind` would hand the engine a
      // RunnableBinding instead of a bindable BaseChatModel. So this path is
      // unforced and leans on Gemini's native tool-calling reliability (the
      // small-model `lazy-tool-recovery` net is available per-profile if a
      // given Gemini model ever narrates instead of calling).
      llm: new ChatGoogle({
        model: spec.model,
        apiKey: process.env.GOOGLE_API_KEY,
        platformType: 'gai',
      }),
    };
  }

  throw new Error(
    `Unknown LLM provider: ${spec.provider}. Expected 'ollama', 'anthropic', 'openai', 'openrouter', or 'google'.`
  );
}
