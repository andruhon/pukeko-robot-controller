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
import {
  formatPinnedState,
  isMotionToolCall,
  observeAssistantMessage,
  __motionLogForTest,
} from './motionLog.js';
import { toolFreeModel } from './toolFreeModel.js';

// thread_id → in-flight summary Promise. afterModel kicks off the summary call
// the moment the assistant decides to move; beforeModel on the next turn awaits
// the same Promise and rewrites the message history. The two hooks straddle
// the slow browser round-trip (capture → move → capture → compose), giving the
// summarization wall-clock parallelism with the motion itself.
const pendingSummaries = new Map<string, Promise<string>>();

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

interface MaybeBlock {
  type?: string;
  text?: string;
}

// Drop `image` / `image_url` blocks from a message's content array. Used only on
// the transient input handed to the summarizer: the summary is text, and a frame
// would spend the small local model's whole budget describing what it must not
// describe.
//
// The message is COPIED, never re-described — a prototype-preserving clone of its
// own property descriptors with only `content` replaced, the same shape as the
// three strips in contextPrunerMiddleware. The literal rebuild this replaces
// named at most three fields per type, so `id`, `status`, `artifact`,
// `response_metadata` and `additional_kwargs` were dropped from every message it
// touched, and a streamed AIMessageChunk was flattened into a plain AIMessage.
//
// Its severity is lower than the strips in contextPrunerMiddleware, and the
// reason is recorded so nobody inflates it: this output feeds the summarizer's
// own LLM call and nothing else. It is never checkpointed and never becomes
// conversation state, so a dropped field does not persist past that call. What
// makes it worth fixing anyway is that the loss is silent and open-ended — every
// field added upstream later goes the same way with nothing to say so.
//
// `lc_kwargs` is replaced alongside `content`, as in every strip here: a
// descriptor clone shares that bag with the source BY REFERENCE and it still
// holds the original content, so the frames this function exists to drop would
// stay reachable through the copy, and any serializer resolving values from
// `lc_kwargs` rather than the live field would put them straight back into
// whatever it wrote — a trace, a log, the summarizer payload itself.
//
// The four-type gate is deliberate rather than an artifact of the rebuild: a
// message of any other type passes through untouched today, and widening that
// here would be a behaviour change this function was not asked to make. The
// checks are duck-typed and must stay so — this repo can resolve more than one
// `@langchain/core`, and a message built by the other copy answers `_getType()`
// correctly while failing every class check (RC-21, RC-58).
//
// Returns the same instance when nothing changed. The source is never mutated.
export function stripImageBlocks(msg: BaseMessage): BaseMessage {
  if (typeof msg.content === 'string') return msg;
  if (!Array.isArray(msg.content)) return msg;
  const original = msg.content as MaybeBlock[];
  const textOnly = original.filter(
    (b) => b && b.type !== 'image' && b.type !== 'image_url'
  );
  if (textOnly.length === original.length) return msg;
  const handled =
    isAIMessage(msg) || isHumanMessage(msg) || isToolMessage(msg) || isSystemMessage(msg);
  if (!handled) return msg;

  const stripped = textOnly.length === 0 ? '[image omitted]' : textOnly;
  // A ToolMessage carries the survivors as a JSON string, exactly as the rebuild
  // did. The transformation this function performs is unchanged; only the way
  // the result is carried onto the message is.
  const newContent = (
    isToolMessage(msg) && typeof stripped !== 'string' ? JSON.stringify(stripped) : stripped
  ) as unknown as BaseMessage['content'];

  const copy = Object.create(
    Object.getPrototypeOf(msg) as object,
    Object.getOwnPropertyDescriptors(msg)
  ) as BaseMessage;
  copy.content = newContent;
  copy.lc_kwargs = { ...msg.lc_kwargs, content: newContent };
  return copy;
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

interface MaybeToolUseBlock extends MaybeBlock {
  id?: string;
}

// Drop `tool_use` content blocks (Anthropic's native AIMessage content shape)
// whose id is not in `keepIds`, mirroring the `.tool_calls` filter so a stripped
// AIMessage never carries an unpaired tool_use in *either* representation.
// Also reports the ids of the surviving `tool_use` blocks, so the caller can
// keep their `tool_result`s even when the id lives only in `content` (not
// mirrored into `.tool_calls`) — the actual Anthropic-native wire shape.
function filterToolUseBlocks(
  content: BaseMessage['content'],
  keepIds: Set<string>
): { content: BaseMessage['content']; changed: boolean; keptBlockIds: string[] } {
  if (!Array.isArray(content)) return { content, changed: false, keptBlockIds: [] };
  const blocks = content as MaybeToolUseBlock[];
  const kept = blocks.filter(
    (b) => !(b && b.type === 'tool_use' && (!b.id || !keepIds.has(b.id)))
  );
  const keptBlockIds = kept
    .filter((b) => b && b.type === 'tool_use' && typeof b.id === 'string')
    .map((b) => b.id as string);
  if (kept.length === blocks.length) return { content, changed: false, keptBlockIds };
  return { content: kept as unknown as BaseMessage['content'], changed: true, keptBlockIds };
}

// Remove tool-call pairs that are not complete, so the rebuilt history is always
// Anthropic-valid (it rejects a `tool_use` that is not immediately followed by
// its matching `tool_result` with `INVALID_TOOL_RESULTS`). The trailing, just-
// emitted motion tool call is the common offender: its `tool_result` does not
// exist yet at summarization time, so it can only be stripped, not paired.
//
// Pure and side-effect free (mirrors the pruning helpers in
// contextPrunerMiddleware). Fully-paired histories pass through with their
// original message instances preserved; a message that loses a call comes back
// as a copy of itself, never a rebuild — see the AIMessage arm.
export function stripUnpairedToolCalls(messages: BaseMessage[]): BaseMessage[] {
  // A tool_call is "resolved" iff some ToolMessage carries its id.
  const resolvedIds = new Set<string>();
  for (const m of messages) {
    if (isToolMessage(m)) {
      const id = (m as ToolMessage).tool_call_id;
      if (id) resolvedIds.add(id);
    }
  }

  const out: BaseMessage[] = [];
  const keptCallIds = new Set<string>();

  for (const m of messages) {
    if (isAIMessage(m)) {
      const ai = m as AIMessage;
      const calls = ai.tool_calls ?? [];
      const keptCalls = calls.filter((tc) => tc.id != null && resolvedIds.has(tc.id));
      const {
        content: keptContent,
        changed: contentChanged,
        keptBlockIds,
      } = filterToolUseBlocks(ai.content, resolvedIds);

      // A surviving tool_use may live in `.tool_calls`, in the `content` blocks,
      // or both. Track EVERY kept id from both representations so the two can't
      // diverge — otherwise a content-block tool_use kept here whose id is not
      // mirrored in `.tool_calls` would later have its `tool_result` dropped,
      // re-creating the exact unpaired-tool_use INVALID_TOOL_RESULTS shape.
      const recordKept = () => {
        for (const tc of keptCalls) if (tc.id) keptCallIds.add(tc.id);
        for (const id of keptBlockIds) keptCallIds.add(id);
      };

      if (keptCalls.length === calls.length && !contentChanged) {
        recordKept();
        out.push(m);
        continue;
      }

      // Something was stripped. If nothing meaningful survives (no tool_calls
      // and no text/blocks), drop the message entirely — an empty AIMessage is
      // itself invalid for Anthropic.
      const hasContent =
        (typeof keptContent === 'string' && keptContent.length > 0) ||
        (Array.isArray(keptContent) && keptContent.length > 0);
      if (keptCalls.length === 0 && !hasContent) continue;

      recordKept();
      // COPIED, never re-described — a prototype-preserving clone of the
      // message's own property descriptors with only the two fields this
      // function rewrites replaced: the same shape as stripImageBlocks above
      // and the three strips in contextPrunerMiddleware. The literal rebuild
      // this replaces named five fields by hand, so `response_metadata`,
      // `usage_metadata` and `invalid_tool_calls` were dropped from every
      // message it touched, a streamed AIMessageChunk lost `tool_call_chunks`
      // and was flattened into a plain AIMessage, and every field added
      // upstream later would have gone the same way. It was also the one
      // rebuild left downstream of stripImageBlocks, so "every field survives
      // the strip" held while "every field survives the pipeline" did not.
      //
      // `lc_kwargs` is replaced on the same terms as every strip here: a
      // descriptor clone shares that bag with the source BY REFERENCE, and it
      // still holds the unpaired `tool_calls` and `tool_use` block this
      // function exists to drop. Here that is sharper than a retained frame —
      // a serializer resolving values from `lc_kwargs` rather than the live
      // field would put the unpaired call straight back into the summarizer
      // payload, which is the INVALID_TOOL_RESULTS shape this whole function
      // exists to prevent.
      //
      // Only `.tool_calls` and the `tool_use` content blocks are filtered. A
      // chunk's `tool_call_chunks` is carried across whole, like every other
      // field: narrowing that too would be a behaviour change this function
      // was not asked to make.
      const copy = Object.create(
        Object.getPrototypeOf(ai) as object,
        Object.getOwnPropertyDescriptors(ai)
      ) as AIMessage;
      copy.content = keptContent;
      copy.tool_calls = keptCalls;
      copy.lc_kwargs = { ...ai.lc_kwargs, content: keptContent, tool_calls: keptCalls };
      out.push(copy);
      continue;
    }

    if (isToolMessage(m)) {
      // Keep a tool_result only if its emitting tool_use survived above.
      const id = (m as ToolMessage).tool_call_id;
      if (id && keptCallIds.has(id)) out.push(m);
      continue;
    }

    out.push(m);
  }

  return out;
}

// Assemble the exact message array sent to the summarization LLM: strip image
// blocks, remove any unpaired tool-call pairs, then wrap with the summary system
// prompt and the "write it now" nudge. Pure and testable without an LLM.
export function buildSummarizationMessages(
  messages: BaseMessage[],
  summaryPrompt: string
): BaseMessage[] {
  const sanitized = messages.map(stripImageBlocks);
  const paired = stripUnpairedToolCalls(sanitized);
  return [
    new SystemMessage(summaryPrompt),
    ...paired,
    new HumanMessage('Write the summary now.'),
  ];
}

export interface MotionSummarizationOptions {
  llm: BaseChatModel;
  // Override for the summarization system prompt. Falls back to
  // DEFAULT_SUMMARY_PROMPT when omitted.
  summaryPrompt?: string;
}

export function createMotionSummarizationMiddleware(opts: MotionSummarizationOptions) {
  const summaryPrompt = opts.summaryPrompt ?? DEFAULT_SUMMARY_PROMPT;
  // This call sends a transcript and no `tools`; the agent model's baked-in
  // tool_choice / parallel_tool_calls are rejected on a tool-less request.
  // Built once, not per call: rebuilding a provider model allocates a client.
  const summaryLlm = toolFreeModel(opts.llm);
  return createMiddleware({
    name: 'motion-summarization',

    afterModel: async (state, runtime) => {
      const messages = (state.messages || []) as BaseMessage[];
      if (messages.length === 0) return undefined;
      const last = messages[messages.length - 1];

      const threadId = runtime?.configurable?.thread_id ?? '__default__';
      // Durable per-thread bookkeeping (recent-motion log, give-up gate, pinned
      // calibration) — every assistant turn, even when a summary is already in
      // flight (the guard below only skips the expensive LLM call).
      observeAssistantMessage(threadId, last);

      // Summarization only fires after a motion: its slow browser round-trip is
      // the wall-clock cover for the summary call.
      if (!isMotionToolCall(last)) return undefined;
      if (pendingSummaries.has(threadId)) return undefined;

      // Build the LLM input off the full history: strip images AND remove any
      // unpaired tool-call pairs (the just-emitted motion tool_use has no result
      // yet), so Anthropic doesn't 400 with INVALID_TOOL_RESULTS.
      const summarizationInput = buildSummarizationMessages(messages, summaryPrompt);

      const promise = (async () => {
        try {
          // Detach from the main agent's run config — otherwise tokens
          // streamed by this parallel call hit the now-closed StreamMessages
          // controller for the original turn and spam ERR_INVALID_STATE.
          const result = await summaryLlm.invoke(summarizationInput, {
            callbacks: [],
            tags: ['motion-summarization'],
          });
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
      const pinned = formatPinnedState(threadId);
      const summaryBody = pinned
        ? `Summary of prior steps:\n${summary}\n\n${pinned}`
        : `Summary of prior steps:\n${summary}`;
      // RC-16: the summary rides as a clearly-marked HumanMessage, NEVER a
      // SystemMessage. This message lands at index ≥ 1 of the rebuilt history,
      // and @langchain/anthropic rejects any non-first system message
      // ("System messages are only permitted as the first passed message." —
      // the PLAT-13 crash). Hoisting into a first-position system message is
      // no fix either: the lean backend's composed prompt is passed to
      // createAgent as `systemPrompt` (applied at model-call time, outside
      // state.messages), so a state-level SystemMessage would still land
      // behind it, i.e. non-first. A user-role recap is valid at any index on
      // every backend (GS2-64 content-preserving precedent), and consecutive
      // user turns already occur live (ToolMessage → injected composite
      // HumanMessage), so this introduces no new wire shape.
      const replaced: BaseMessage[] = [
        ...head,
        new HumanMessage(`[Motion summary]\n${summaryBody}`),
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

// Exposed for tests; do not call from app code. The motion log itself now lives
// in ./motionLog — re-exported here so existing tests keep their import path.
export const __pendingSummariesForTest = pendingSummaries;
export { __motionLogForTest };
