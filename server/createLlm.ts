import { ChatAnthropic } from '@langchain/anthropic';
import { ChatOllama } from '@langchain/ollama';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';

export type LlmProvider = 'ollama' | 'anthropic';

export interface LlmSelection {
  provider: LlmProvider;
  llm: BaseChatModel;
}

export function createLlm(): LlmSelection {
  const provider = (process.env.LLM_PROVIDER ?? 'ollama') as LlmProvider;

  if (provider === 'ollama') {
    return {
      provider,
      llm: new ChatOllama({
        baseUrl: process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434',
        model: process.env.OLLAMA_MODEL ?? 'gemma4:31b',
      }),
    };
  }

  if (provider === 'anthropic') {
    return {
      provider,
      // disable_parallel_tool_use: client-fulfilled tools (capture_image)
      // trigger langgraph interrupt(). When the model batches two tool calls
      // in one assistant message — e.g. read_status + capture_image — the
      // interrupt fires mid-batch and the message history ends up with one
      // tool_use that has no immediately-following tool_result, which
      // Anthropic rejects on the next turn.
      llm: new ChatAnthropic({
        model: process.env.ANTHROPIC_MODEL ?? 'claude-sonnet-4-6',
        invocationKwargs: {
          tool_choice: { type: 'auto', disable_parallel_tool_use: true },
        },
      }),
    };
  }

  throw new Error(`Unknown LLM_PROVIDER=${provider}. Expected 'ollama' or 'anthropic'.`);
}
