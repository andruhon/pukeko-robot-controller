import { createMiddleware } from 'langchain';
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  RemoveMessage,
  isAIMessage,
  isHumanMessage,
  isToolMessage,
  isSystemMessage,
  type BaseMessage,
} from '@langchain/core/messages';
import { REMOVE_ALL_MESSAGES } from '@langchain/langgraph';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { MOTION_TOOL_NAMES } from './robotTools.js';

const MOTION_NAMES: ReadonlySet<string> = new Set(MOTION_TOOL_NAMES);

// thread_id → in-flight summary Promise. afterModel kicks off the summary call
// the moment the assistant decides to move; beforeModel on the next turn awaits
// the same Promise and rewrites the message history. The two hooks straddle
// the slow browser round-trip (capture → move → capture → compose), giving the
// summarization wall-clock parallelism with the motion itself.
const pendingSummaries = new Map<string, Promise<string>>();

const SUMMARY_SYSTEM_PROMPT = `You are compressing a robot-control conversation log so a small local model can stay on task.

Produce a concise summary covering:
- The user's original objective (state it verbatim if short, otherwise quote the goal).
- Robot state evolution: which motion commands ran, sensor readings before/after, what the assistant decided each step.
- Constraints, obstacles, or hypotheses raised so far.

Rules:
- DO NOT describe image content. Do not write "the photo shows...".
- DO NOT include base64 data, image URLs, or content blocks.
- Maximum 8 short sentences.
- Past tense. Plain text only.`;

interface MaybeToolCall {
  name?: unknown;
}

function isMotionToolCall(msg: unknown): boolean {
  if (!msg || typeof msg !== 'object') return false;
  // Be permissive: state.messages entries may be serialized plain objects
  // depending on how LangGraph round-trips them between hooks.
  const tcs = (msg as { tool_calls?: unknown }).tool_calls;
  if (!Array.isArray(tcs)) return false;
  return tcs.some(
    (tc: MaybeToolCall) => typeof tc?.name === 'string' && MOTION_NAMES.has(tc.name)
  );
}

interface MaybeBlock {
  type?: string;
  text?: string;
}

function stripImageBlocks(msg: BaseMessage): BaseMessage {
  if (typeof msg.content === 'string') return msg;
  if (!Array.isArray(msg.content)) return msg;
  const original = msg.content as MaybeBlock[];
  const textOnly = original.filter(
    (b) => b && b.type !== 'image' && b.type !== 'image_url'
  );
  if (textOnly.length === original.length) return msg;
  const newContent = (textOnly.length === 0 ? '[image omitted]' : textOnly) as unknown as BaseMessage['content'];
  if (isAIMessage(msg)) {
    return new AIMessage({ content: newContent, tool_calls: (msg as AIMessage).tool_calls, name: msg.name });
  }
  if (isHumanMessage(msg)) {
    return new HumanMessage({ content: newContent, name: msg.name });
  }
  if (isToolMessage(msg)) {
    return new ToolMessage({
      content: typeof newContent === 'string' ? newContent : JSON.stringify(newContent),
      tool_call_id: (msg as ToolMessage).tool_call_id,
      name: msg.name,
    });
  }
  if (isSystemMessage(msg)) {
    return new SystemMessage({ content: newContent, name: msg.name });
  }
  return msg;
}

function extractText(content: BaseMessage['content']): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return (content as MaybeBlock[])
    .filter((b) => b && b.type === 'text' && typeof b.text === 'string')
    .map((b) => b.text as string)
    .join(' ')
    .trim();
}

export interface MotionSummarizationOptions {
  llm: BaseChatModel;
}

export function createMotionSummarizationMiddleware(opts: MotionSummarizationOptions) {
  return createMiddleware({
    name: 'motion-summarization',

    afterModel: async (state, runtime) => {
      const messages = (state.messages || []) as BaseMessage[];
      if (messages.length === 0) return undefined;
      const last = messages[messages.length - 1];
      if (!isMotionToolCall(last)) return undefined;

      const threadId = runtime?.configurable?.thread_id ?? '__default__';
      if (pendingSummaries.has(threadId)) return undefined;

      const sanitized = messages.map(stripImageBlocks);

      const promise = (async () => {
        try {
          // Detach from the main agent's run config — otherwise tokens
          // streamed by this parallel call hit the now-closed StreamMessages
          // controller for the original turn and spam ERR_INVALID_STATE.
          const result = await opts.llm.invoke(
            [
              new SystemMessage(SUMMARY_SYSTEM_PROMPT),
              ...sanitized,
              new HumanMessage('Write the summary now.'),
            ],
            { callbacks: [], tags: ['motion-summarization'] }
          );
          return extractText(result.content);
        } catch (err) {
          console.error('[motion-summarization] LLM call failed:', err);
          return '';
        }
      })();
      pendingSummaries.set(threadId, promise);
      console.log(`[motion-summarization] kicked off (thread=${threadId}, history=${messages.length})`);
      return undefined;
    },

    beforeModel: async (state, runtime) => {
      const threadId = runtime?.configurable?.thread_id ?? '__default__';
      const pending = pendingSummaries.get(threadId);
      if (!pending) return undefined;
      const summary = await pending;
      pendingSummaries.delete(threadId);
      if (!summary) return undefined;

      const messages = (state.messages || []) as BaseMessage[];
      const firstHumanIdx = messages.findIndex((m) => isHumanMessage(m));
      let lastMotionIdx = -1;
      for (let i = messages.length - 1; i >= 0; i--) {
        if (isMotionToolCall(messages[i])) {
          lastMotionIdx = i;
          break;
        }
      }
      if (firstHumanIdx < 0 || lastMotionIdx < 0 || lastMotionIdx <= firstHumanIdx + 1) {
        return undefined;
      }

      const head = messages.slice(0, firstHumanIdx + 1);
      const tail = messages.slice(lastMotionIdx);
      const replaced: BaseMessage[] = [
        ...head,
        new SystemMessage(`Summary of prior steps:\n${summary}`),
        ...tail,
      ];

      console.log(
        `[motion-summarization] applied (thread=${threadId}, ${messages.length} → ${replaced.length} messages, ${summary.length} chars)`
      );
      return {
        messages: [new RemoveMessage({ id: REMOVE_ALL_MESSAGES }), ...replaced],
      };
    },
  });
}

// Exposed for tests; do not call from app code.
export const __pendingSummariesForTest = pendingSummaries;
