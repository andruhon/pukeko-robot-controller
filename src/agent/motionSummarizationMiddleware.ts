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

// thread_id → ordered list of motions issued, newest last. Appended to the
// summary verbatim so the recent-move history is deterministic (immune to LLM
// summary drift). The newest entry is the in-flight motion — marked pending
// until the next turn observes its result.
interface MotionEntry {
  label: string;
  pending: boolean;
}
const motionLogByThread = new Map<string, MotionEntry[]>();
const MAX_MOTION_LOG = 5;

// Baked-in fallback. Kept identical to `summarization-prompt.md` at the repo
// root, which the server loads and passes in via `summaryPrompt`. This copy is
// used only when that file is missing.
const DEFAULT_SUMMARY_PROMPT = `You are compressing a robot-control conversation log so a small local model can stay on task. The summary REPLACES the detailed history, so capture the operator's understanding so far — conclusions, not a play-by-play.

Cover, in a few terse sentences:
- The user's objective (verbatim if short).
- What has been learned about the controls in this camera view: which on-screen direction each turn produces (and whether turn_left/turn_right are inverted here), which end is the robot's face, and the rough movement scale.
- Where the robot currently is and which way it is facing relative to the target, and the intended next move.
- Open questions, obstacles, or sensor caveats (e.g. a flat or thin target the ultrasonic can't see).

Rules:
- Write conclusions and current state, NOT a list of the commands issued — recent moves are tracked separately and appended for you.
- Do NOT describe raw image content ("the photo shows..."), and do NOT include base64 data or image URLs.
- Plain text, terse, present tense.`;

interface MaybeToolCall {
  name?: unknown;
  args?: unknown;
}

// Human-readable label for the motion a message just issued, e.g.
// "turn_right (steps=3)". Returns null when the message has no motion call.
function motionLabel(msg: unknown): string | null {
  if (!msg || typeof msg !== 'object') return null;
  const tcs = (msg as { tool_calls?: unknown }).tool_calls;
  if (!Array.isArray(tcs)) return null;
  for (const tc of tcs as MaybeToolCall[]) {
    if (typeof tc?.name === 'string' && MOTION_NAMES.has(tc.name)) {
      const rawSteps = (tc.args as { steps?: unknown } | undefined)?.steps;
      const n =
        typeof rawSteps === 'number' && Number.isFinite(rawSteps) && rawSteps >= 1
          ? Math.floor(rawSteps)
          : 1;
      return n > 1 ? `${tc.name} (steps=${n})` : tc.name;
    }
  }
  return null;
}

// Record a freshly-issued motion: the previously pending one is now resolved
// (its result was observed last turn), and this one becomes the new pending.
function recordMotion(threadId: string, label: string): void {
  const log = motionLogByThread.get(threadId) ?? [];
  for (const e of log) e.pending = false;
  log.push({ label, pending: true });
  while (log.length > MAX_MOTION_LOG) log.shift();
  motionLogByThread.set(threadId, log);
}

function formatMotionLog(threadId: string): string {
  const log = motionLogByThread.get(threadId);
  if (!log || log.length === 0) return '';
  const lines = log.map(
    (e) =>
      `- ${e.label}${e.pending ? ' (pending — its result is the latest Before/After frame below)' : ''}`
  );
  return `Recent motions (newest last):\n${lines.join('\n')}`;
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
  // Override for the summarization system prompt. Falls back to
  // DEFAULT_SUMMARY_PROMPT when omitted.
  summaryPrompt?: string;
}

export function createMotionSummarizationMiddleware(opts: MotionSummarizationOptions) {
  const summaryPrompt = opts.summaryPrompt ?? DEFAULT_SUMMARY_PROMPT;
  return createMiddleware({
    name: 'motion-summarization',

    afterModel: async (state, runtime) => {
      const messages = (state.messages || []) as BaseMessage[];
      if (messages.length === 0) return undefined;
      const last = messages[messages.length - 1];
      if (!isMotionToolCall(last)) return undefined;

      const threadId = runtime?.configurable?.thread_id ?? '__default__';
      // Log the motion for the deterministic recent-moves list — always, even
      // when a summary is already in flight (the guard below only skips the
      // expensive LLM call, not the bookkeeping).
      recordMotion(threadId, motionLabel(last) ?? 'motion');
      if (pendingSummaries.has(threadId)) return undefined;

      const sanitized = messages.map(stripImageBlocks);

      const promise = (async () => {
        try {
          // Detach from the main agent's run config — otherwise tokens
          // streamed by this parallel call hit the now-closed StreamMessages
          // controller for the original turn and spam ERR_INVALID_STATE.
          const result = await opts.llm.invoke(
            [
              new SystemMessage(summaryPrompt),
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
      const motionList = formatMotionLog(threadId);
      const summaryBody = motionList
        ? `Summary of prior steps:\n${summary}\n\n${motionList}`
        : `Summary of prior steps:\n${summary}`;
      const replaced: BaseMessage[] = [
        ...head,
        new SystemMessage(summaryBody),
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
export const __motionLogForTest = motionLogByThread;
