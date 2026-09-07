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
  type BaseMessage,
} from '@langchain/core/messages';
import { REMOVE_ALL_MESSAGES } from '@langchain/langgraph';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { MOTION_TOOL_NAMES } from './robotToolNames.js';
import {
  formatPinnedState,
  isMotionToolCall,
  observeAssistantMessage,
} from './motionLog.js';
import { toolFreeModel } from './toolFreeModel.js';

const IMAGE_TOOL_NAMES: ReadonlySet<string> = new Set([...MOTION_TOOL_NAMES, 'capture_image']);

// Per-thread guard against re-entrant summarization. beforeModel is async; if a
// second request lands on the same thread mid-flight the second call awaits the
// first instead of issuing a duplicate LLM round-trip.
const inflightSummaries = new Map<string, Promise<string>>();

const DEFAULT_SUMMARY_PROMPT = `You are compressing the early portion of a robot-control conversation so a small local model can stay on task within its context budget. The summary REPLACES the detailed history that came before it, so capture the operator's understanding so far — conclusions, not a play-by-play.

Cover, in a few terse sentences:
- The user's objective (verbatim if short).
- What has been learned about the controls in this camera view: which on-screen direction each turn produces (and whether turn_left/turn_right are inverted here), which end is the robot's face, and the rough movement scale.
- Where the robot currently is and which way it is facing relative to the target.
- Open questions, obstacles, or sensor caveats (e.g. a flat or thin target the ultrasonic can't see).

Rules:
- Write conclusions and current state, NOT a list of the commands issued.
- Do NOT describe raw image content ("the photo shows..."), and do NOT include base64 data or image URLs.
- Plain text, terse, present tense.`;

export interface ContextPrunerOptions {
  llm: BaseChatModel;
  // Override for the summarization system prompt. Falls back to a baked-in
  // default that mirrors the existing motion-summarization wording.
  summaryPrompt?: string;
  // Hard cap on tokens we want the LLM to receive. Default tuned for Gemma 31b.
  maxContextTokens?: number;
  // Fraction of maxContextTokens at which we synchronously summarize the head
  // before letting the next LLM call go through.
  summarizeAtFraction?: number;
  // How many of the most-recent image-bearing HumanMessages keep their image
  // blocks. Older ones become text-only.
  keepLatestImages?: number;
  // Flat per-image-block charge used by the token estimator. Approximate;
  // Ollama's actual image tokenization differs by model.
  imageTokenBudget?: number;
}

interface MaybeBlock {
  type?: string;
  text?: string;
}

// ────────────────────────────────────────────────────────────────────────────
// Pruning helpers
// ────────────────────────────────────────────────────────────────────────────

// Drop the base64 `data` field from a motion / capture ToolMessage's JSON
// content. The same image is re-emitted one message later as an `image_url` /
// `image` block by frontendImageInjectionMiddleware; the bytes inside the
// ToolMessage are pure dead weight to the model.
//
// The message is COPIED, never re-described (same shape as
// stripReasoningContent below). A rebuild from an object literal keeps only the
// fields the literal happens to name, and the casualty that matters most here
// is `status`: this strip runs on EVERY motion/capture tool result, not only
// old ones, so a result carrying `status: 'error'` came back `undefined` and a
// failed motion became indistinguishable from a completed one. `artifact`,
// `response_metadata` and `additional_kwargs` went the same way.
//
// `lc_kwargs` is replaced too, unlike in stripReasoningContent, because here the
// field being rewritten IS the payload this function exists to free. A plain
// descriptor copy shares the source's `lc_kwargs` by reference, and that object
// still holds the original content string — so the base64 frame this call just
// dropped would stay reachable for the life of the thread (one per motion), and
// any serializer that resolved values from `lc_kwargs` rather than the live
// instance field would put the bytes straight back into the checkpoint. The
// replacement is a fresh object; the source is never mutated.
//
// Returns the same instance when nothing changed: the caller counts strips by
// reference identity.
function stripToolMessageImageData(msg: ToolMessage): ToolMessage {
  if (typeof msg.content !== 'string') return msg;
  if (!msg.name || !IMAGE_TOOL_NAMES.has(msg.name)) return msg;
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(msg.content) as Record<string, unknown>;
  } catch {
    return msg;
  }
  if (typeof parsed !== 'object' || parsed === null) return msg;
  if (typeof parsed.data !== 'string' || parsed.data.length === 0) return msg;
  const { data: _dropped, ...rest } = parsed;
  void _dropped;
  const nextContent = JSON.stringify({ ...rest, dataDropped: true });
  const copy = Object.create(
    Object.getPrototypeOf(msg) as object,
    Object.getOwnPropertyDescriptors(msg)
  ) as ToolMessage;
  copy.content = nextContent;
  copy.lc_kwargs = { ...msg.lc_kwargs, content: nextContent };
  return copy;
}

function hasImageBlock(content: unknown): boolean {
  if (!Array.isArray(content)) return false;
  return (content as MaybeBlock[]).some(
    (b) => b && (b.type === 'image' || b.type === 'image_url')
  );
}

// Strip image blocks out of a HumanMessage's content array, keeping the
// leading text caption. If nothing useful survives, return a single
// "[image dropped]" text block so the model still sees the slot.
//
// Copied rather than re-described, for the same reason as the two strips around
// it: the literal rebuild this replaces named only `id`, `content` and `name`,
// so `additional_kwargs` and `response_metadata` were dropped from every
// image-bearing turn that aged out. `lc_kwargs` is replaced alongside `content`
// so the image blocks being dropped do not stay reachable through the source's
// shared bag — see stripToolMessageImageData for the full reasoning.
//
// Returns the same instance when nothing changed: the caller counts strips by
// reference identity.
function pruneImageBlocksInHumanMessage(msg: HumanMessage): HumanMessage {
  if (!Array.isArray(msg.content)) return msg;
  const textOnly = (msg.content as MaybeBlock[]).filter(
    (b) => b && b.type !== 'image' && b.type !== 'image_url'
  );
  if (textOnly.length === msg.content.length) return msg;
  const newContent =
    textOnly.length === 0
      ? ([{ type: 'text', text: '[image dropped]' }] as unknown as HumanMessage['content'])
      : (textOnly as unknown as HumanMessage['content']);
  const copy = Object.create(
    Object.getPrototypeOf(msg) as object,
    Object.getOwnPropertyDescriptors(msg)
  ) as HumanMessage;
  copy.content = newContent;
  copy.lc_kwargs = { ...msg.lc_kwargs, content: newContent };
  return copy;
}

// Clear `additional_kwargs.reasoning_content` (Anthropic extended-thinking,
// Ollama Qwen3 / deepseek-r1) while preserving every other field of the
// message — not just the other additional_kwargs keys.
//
// The message is COPIED, never re-described: a prototype-preserving clone of
// its own property descriptors, with only `additional_kwargs` replaced. A
// rebuild from an object literal silently drops whatever the literal forgot,
// and two of the casualties are load-bearing. `response_metadata` is where the
// OpenAI Responses API carries the reasoning-item ids that a following tool
// call must be paired with, and the robot's default profile is an OpenAI one.
// `tool_call_chunks` lives only on AIMessageChunk, which `isAIMessage` admits
// and a literal rebuild would flatten into a plain AIMessage. Copying also
// keeps the message on the prototype it arrived with, rather than re-minting it
// under this module's copy of @langchain/core — this repo resolves two.
//
// Returns the same instance when nothing changed: the caller counts strips by
// reference identity. The source is never mutated — only the clone's
// `additional_kwargs` is replaced, and with a fresh object.
function stripReasoningContent(msg: AIMessage): AIMessage {
  const ak = msg.additional_kwargs as Record<string, unknown> | undefined;
  if (!ak || ak.reasoning_content == null) return msg;
  const { reasoning_content: _dropped, ...rest } = ak;
  void _dropped;
  const copy = Object.create(
    Object.getPrototypeOf(msg) as object,
    Object.getOwnPropertyDescriptors(msg)
  ) as AIMessage;
  copy.additional_kwargs = rest;
  return copy;
}

// Newest-first list of indices of HumanMessages that carry an image block.
function findImageHumanMessageIndices(messages: BaseMessage[]): number[] {
  const out: number[] = [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (isHumanMessage(m) && hasImageBlock(m.content)) out.push(i);
  }
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// Token estimator (cheap heuristic — no tokenizer dependency)
// ────────────────────────────────────────────────────────────────────────────

function textTokens(text: string): number {
  // ~4 chars per token is the canonical rough estimate for English; reasonable
  // for our prompt style across cl100k/o200k/Gemma's SentencePiece.
  return Math.ceil(text.length / 4);
}

function contentTokens(content: BaseMessage['content'], imageBudget: number): number {
  if (typeof content === 'string') return textTokens(content);
  if (!Array.isArray(content)) return 0;
  let total = 0;
  for (const block of content as MaybeBlock[]) {
    if (!block) continue;
    if (block.type === 'image' || block.type === 'image_url') {
      total += imageBudget;
    } else if (typeof block.text === 'string') {
      total += textTokens(block.text);
    }
  }
  return total;
}

export function estimateTokens(messages: BaseMessage[], imageBudget: number): number {
  let total = 0;
  for (const m of messages) {
    total += contentTokens(m.content, imageBudget);
    // Tool calls on AIMessages cost real tokens too — the name + serialized
    // args are sent verbatim.
    if (isAIMessage(m)) {
      const tcs = (m as AIMessage).tool_calls ?? [];
      for (const tc of tcs) {
        total += textTokens((tc.name ?? '') + JSON.stringify(tc.args ?? {}));
      }
      const reasoning = (m.additional_kwargs as Record<string, unknown> | undefined)
        ?.reasoning_content;
      if (typeof reasoning === 'string') total += textTokens(reasoning);
    }
    // Per-message envelope overhead (role token, separators) — rough.
    total += 4;
  }
  return total;
}

// ────────────────────────────────────────────────────────────────────────────
// Mechanical prune
// ────────────────────────────────────────────────────────────────────────────

interface PruneStats {
  toolImageDataStripped: number;
  humanImagesStripped: number;
  reasoningStripped: number;
}

// `keptImageIdx` is the prune's own answer to "which image-bearing turns did I
// elect to keep", newest-first, as indices into the RETURNED array. It is
// returned rather than recomputed by the caller because the summarize boundary
// depends on it. Note what the argument is NOT: recomputing it at the summarize
// site gives the same answer today, because the strip has already cleared the
// image blocks of every non-kept turn, so the two cannot currently disagree.
// The point is that the agreement rests on a coupling inside this function that
// is invisible from the call site and that nobody has written down as an
// invariant. Returning the indices removes the dependency on it, rather than
// fixing a drift that exists — two mechanisms disagreeing about which frames
// matter is the defect the boundary rule below exists to close, and this keeps
// the question from having a second answer at all.
//
// The indices stay valid because every step here is a `map` — messages are
// copied in place, never inserted or removed, so position is preserved.
function mechanicalPrune(
  messages: BaseMessage[],
  keepLatestImages: number
): { messages: BaseMessage[]; stats: PruneStats; keptImageIdx: number[] } {
  const stats: PruneStats = {
    toolImageDataStripped: 0,
    humanImagesStripped: 0,
    reasoningStripped: 0,
  };

  // Step 1: strip ToolMessage `data` everywhere.
  let next: BaseMessage[] = messages.map((m) => {
    if (isToolMessage(m)) {
      const stripped = stripToolMessageImageData(m);
      if (stripped !== m) stats.toolImageDataStripped++;
      return stripped;
    }
    return m;
  });

  // Step 2: keep the latest N image HumanMessages, prune image blocks from the rest.
  // Kept and pruned come from ONE split of the same newest-first list, so the
  // two can never disagree about which side a turn fell on.
  const imageIdxNewestFirst = findImageHumanMessageIndices(next);
  const splitAt = Math.max(0, keepLatestImages);
  const keptImageIdx = imageIdxNewestFirst.slice(0, splitAt);
  const toPrune = new Set(imageIdxNewestFirst.slice(splitAt));
  if (toPrune.size > 0) {
    next = next.map((m, i) => {
      if (!toPrune.has(i)) return m;
      if (!isHumanMessage(m)) return m;
      const pruned = pruneImageBlocksInHumanMessage(m);
      if (pruned !== m) stats.humanImagesStripped++;
      return pruned;
    });
  }

  // Step 3: strip reasoning_content from every AIMessage except the last one
  // in the list. The last AI message belongs to the in-flight turn; keeping
  // its reasoning intact satisfies Anthropic extended-thinking's mid-round
  // requirement (boundaries between turns are always HumanMessages anyway).
  let lastAiIdx = -1;
  for (let i = next.length - 1; i >= 0; i--) {
    if (isAIMessage(next[i])) {
      lastAiIdx = i;
      break;
    }
  }
  next = next.map((m, i) => {
    if (i === lastAiIdx) return m;
    if (!isAIMessage(m)) return m;
    const stripped = stripReasoningContent(m);
    if (stripped !== m) stats.reasoningStripped++;
    return stripped;
  });

  return { messages: next, stats, keptImageIdx };
}

// ────────────────────────────────────────────────────────────────────────────
// Summarization
// ────────────────────────────────────────────────────────────────────────────

function extractText(content: BaseMessage['content']): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return (content as MaybeBlock[])
    .filter((b) => b && b.type === 'text' && typeof b.text === 'string')
    .map((b) => b.text as string)
    .join(' ')
    .trim();
}

// Final image strip used only on the LLM input we send to the summarizer —
// even the "latest" frame is irrelevant when summarizing text history.
function dropAllImageBlocks(msg: BaseMessage): BaseMessage {
  if (isHumanMessage(msg) && Array.isArray(msg.content)) {
    return pruneImageBlocksInHumanMessage(msg);
  }
  if (isToolMessage(msg)) {
    return stripToolMessageImageData(msg);
  }
  return msg;
}

// `llm` MUST be the tool-free model (see toolFreeModel): this call sends a
// transcript and no `tools`, and the agent model's baked-in tool_choice /
// parallel_tool_calls are rejected outright by OpenAI on a tool-less request.
async function runSummary(
  llm: BaseChatModel,
  summaryPrompt: string,
  head: BaseMessage[]
): Promise<string> {
  const sanitized = head.map(dropAllImageBlocks);
  const result = await llm.invoke(
    [
      new SystemMessage(summaryPrompt),
      ...sanitized,
      new HumanMessage('Write the summary now.'),
    ],
    { callbacks: [], tags: ['context-pruner-summary'] }
  );
  return extractText(result.content);
}

// ────────────────────────────────────────────────────────────────────────────
// Middleware
// ────────────────────────────────────────────────────────────────────────────

export function createContextPrunerMiddleware(opts: ContextPrunerOptions) {
  const summaryPrompt = opts.summaryPrompt ?? DEFAULT_SUMMARY_PROMPT;
  const maxContextTokens = opts.maxContextTokens ?? 30_000;
  const summarizeAtFraction = opts.summarizeAtFraction ?? 0.7;
  const keepLatestImages = Math.max(0, opts.keepLatestImages ?? 1);
  const imageTokenBudget = opts.imageTokenBudget ?? 800;
  const summarizeThreshold = Math.floor(summarizeAtFraction * maxContextTokens);
  // Built once, not per call: rebuilding a provider model allocates a client.
  const summaryLlm = toolFreeModel(opts.llm);

  return createMiddleware({
    name: 'context-pruner',

    // Record durable per-thread state (recent-motion log, give-up gate, pinned
    // calibration) for every assistant turn. Unlike motion-summarization this
    // middleware doesn't summarize on each motion, so the bookkeeping has its
    // own hook rather than riding the summary kick-off.
    afterModel: async (state, runtime) => {
      const messages = (state.messages || []) as BaseMessage[];
      if (messages.length === 0) return undefined;
      const threadId = runtime?.configurable?.thread_id ?? '__default__';
      observeAssistantMessage(threadId, messages[messages.length - 1]);
      return undefined;
    },

    beforeModel: async (state, runtime) => {
      const messages = (state.messages || []) as BaseMessage[];
      if (messages.length === 0) return undefined;

      const threadId = runtime?.configurable?.thread_id ?? '__default__';
      const beforeTokens = estimateTokens(messages, imageTokenBudget);
      const {
        messages: pruned,
        stats,
        keptImageIdx,
      } = mechanicalPrune(messages, keepLatestImages);
      const afterPruneTokens = estimateTokens(pruned, imageTokenBudget);

      let rebuilt = pruned;
      let summarized = false;
      let finalTokens = afterPruneTokens;
      let summaryMs = 0;

      if (afterPruneTokens >= summarizeThreshold) {
        // Carve out the head slice — everything from the first HumanMessage up
        // to the summary boundary. Whatever sits at or after that boundary is
        // the tail, and survives verbatim.
        const firstHumanIdx = pruned.findIndex((m) => isHumanMessage(m));
        let lastMotionAiIdx = -1;
        for (let i = pruned.length - 1; i >= 0; i--) {
          if (isMotionToolCall(pruned[i])) {
            lastMotionAiIdx = i;
            break;
          }
        }

        // Where the boundary sits. The anchor protects whatever state the NEXT
        // model call depends on, and a session with no motion call anywhere is
        // a real one — capture and narrate, question answering, any read-only
        // interaction — not a degenerate history.
        //
        // 1. With a motion call, the boundary is that call: the motion
        //    AIMessage, its ToolMessage and the injected Before/After composite
        //    (plus anything newer) are held back from the summarizer.
        //
        // 2. With no motion call, the state the next call depends on is the
        //    camera frame. `keepLatestImages` deliberately carries the newest
        //    image-bearing turn through the mechanical strip, so the summarizer
        //    must not then discard what the strip just elected to keep — two
        //    policies in one middleware contradicting each other is how a
        //    capture-and-narrate session reached its narrating call holding a
        //    text recap and no picture. The anchor is therefore the OLDEST
        //    image-bearing turn the strip kept (the newest one at the shipped
        //    `keepLatestImages: 1`), taken from the strip's own answer rather
        //    than recomputed here, so the two can never disagree.
        //
        // 3. Otherwise the boundary is the end of the history: with no frame
        //    and no motion there is genuinely nothing to hold back.
        //
        // Rule 2 applies only while the tail it creates still fits under the
        // summarize threshold, and that condition is load-bearing rather than
        // defensive. Holding back everything from the frame onwards is cheap
        // when the frame is recent — one image block, the same budget the strip
        // already spends — but a session that captures once and then talks puts
        // the frame near the START, and anchoring there hands the summarizer a
        // two-message head while the uncompressed tail keeps growing. Measured
        // at the shipped local profile, that shape crossed the 30000-token hard
        // cap at round 58 and reached 62021 tokens while firing the summarizer
        // on 80 of 120 turns: RC-27's own unbounded-growth defect, reintroduced
        // by the anchor meant to fix a different one. Falling through to rule 3
        // there is what keeps the summarize step making real progress, and it
        // costs the frame only in the shape where no bounded boundary could
        // have kept it.
        //
        // Be precise about WHAT is bounded, because it is not the whole
        // rebuild: the condition is checked on the tail alone. The rebuild is
        // the preserved prefix, plus the summary message, plus that tail, and
        // neither of the first two is counted. So a rebuild lands slightly over
        // the threshold as a matter of course — 21053 against a 21000 threshold
        // for a 200-character opening instruction — and a large enough prefix
        // can carry it past the hard cap outright: a 36000-character first
        // message reaches 30003, and a SystemMessage in `state.messages`
        // reaches 31060. Neither is how this agent runs in production, where
        // the system prompt goes to `createAgent` as `systemPrompt` and the
        // preserved prefix is the operator's opening instruction alone. Counting
        // the prefix into the condition is a one-expression change if that ever
        // stops being true.
        //
        // Orphan safety, in all three cases: a tool call is never summarized
        // away from its result. The injected image HumanMessage always sits
        // AFTER the ToolMessage it was built from, so cutting at it leaves the
        // AIMessage(tool_calls)/ToolMessage pair together in the head; a cut at
        // the end takes the whole tail or none of it.
        const hasMotion = lastMotionAiIdx >= 0;
        const oldestKeptImageIdx =
          keptImageIdx.length > 0 ? keptImageIdx[keptImageIdx.length - 1] : -1;
        const keptFrameTailTokens =
          oldestKeptImageIdx >= 0
            ? estimateTokens(pruned.slice(oldestKeptImageIdx), imageTokenBudget)
            : 0;
        const keptFrameTailFits =
          oldestKeptImageIdx >= 0 && keptFrameTailTokens < summarizeThreshold;
        const boundaryIdx = hasMotion
          ? lastMotionAiIdx
          : keptFrameTailFits
            ? oldestKeptImageIdx
            : pruned.length;
        // Names which of the three rules chose the boundary. Pinned by test:
        // with three branches the label is the only thing that says, from a log
        // alone, which policy is in force on a live run.
        const anchorLabel = hasMotion
          ? 'last-motion'
          : keptFrameTailFits
            ? 'oldest-kept-frame'
            : 'end-of-history (no frame worth holding back)';

        if (firstHumanIdx >= 0 && boundaryIdx > firstHumanIdx + 1) {
          const headSlice = pruned.slice(firstHumanIdx + 1, boundaryIdx);
          const tail = pruned.slice(boundaryIdx);
          const firstHuman = pruned[firstHumanIdx];

          // Deduplicate concurrent in-flight summaries per thread.
          let promise = inflightSummaries.get(threadId);
          if (!promise) {
            promise = runSummary(summaryLlm, summaryPrompt, [firstHuman, ...headSlice]);
            inflightSummaries.set(threadId, promise);
          }
          let summaryText = '';
          const summaryStart = performance.now();
          console.log(
            `[context-pruner] thread=${threadId} threshold crossed ` +
              `(pruned=${afterPruneTokens} ≥ ${summarizeThreshold}); ` +
              `anchor=${anchorLabel}; ` +
              `summarizing head of ${headSlice.length + 1} messages…`
          );
          try {
            summaryText = await promise;
          } catch (err) {
            console.error('[context-pruner] summary call failed:', err);
          } finally {
            inflightSummaries.delete(threadId);
            summaryMs = Math.round(performance.now() - summaryStart);
          }

          if (summaryText) {
            // Append the deterministic pinned state (recent-motion log +
            // calibration) the summarizer is told NOT to reproduce — this is
            // what keeps context-pruner from physically repeating an
            // already-attempted motion after a prune.
            const pinned = formatPinnedState(threadId);
            const summaryBody = pinned
              ? `Summary of prior steps:\n${summaryText}\n\n${pinned}`
              : `Summary of prior steps:\n${summaryText}`;
            // RC-17 (mirrors RC-16's motion-summarization fix): the summary
            // rides as a clearly-marked HumanMessage, NEVER a SystemMessage.
            // It lands at index ≥ 1 of the rebuilt history, and
            // @langchain/anthropic rejects any non-first system message
            // ("System messages are only permitted as the first passed
            // message."). Hoisting to a first-position SystemMessage is no fix
            // either: the lean backend's composed prompt is passed to
            // createAgent as `systemPrompt` (applied at model-call time,
            // outside state.messages), so a state-level SystemMessage would
            // still land behind it, i.e. non-first. A user-role recap is valid
            // at any index on every backend, and consecutive user turns
            // already occur live (ToolMessage → injected composite
            // HumanMessage), so this introduces no new wire shape.
            //
            // Replace-not-accumulate: there is no marker- or role-based
            // detection of a previous summary anywhere in this middleware —
            // folding is purely positional. On a later cycle this message sits
            // at firstHumanIdx + 1, inside the next headSlice
            // (firstHumanIdx + 1 .. boundaryIdx), so it is fed to the
            // summarizer and then discarded when the head is rebuilt around
            // the single new summary. The original first HumanMessage is
            // always preserved in place, so this summary can never become the
            // "first human" anchor itself.
            const summaryMsg = new HumanMessage(`[Context summary]\n${summaryBody}`);
            rebuilt = [
              ...pruned.slice(0, firstHumanIdx + 1),
              summaryMsg,
              ...tail,
            ];
            summarized = true;
            finalTokens = estimateTokens(rebuilt, imageTokenBudget);
          }
        }
      }

      const nothingChanged =
        stats.toolImageDataStripped === 0 &&
        stats.humanImagesStripped === 0 &&
        stats.reasoningStripped === 0 &&
        !summarized;

      // Always log — one line per LLM call so context-pruner activity is
      // visible turn-by-turn even when nothing was pruned.
      console.log(
        `[context-pruner] → LLM thread=${threadId} msgs=${rebuilt.length} ` +
          `tokens ${beforeTokens}→${finalTokens} ` +
          `(cap ${maxContextTokens}, sum@${summarizeThreshold}) ` +
          `tool-data:${stats.toolImageDataStripped} ` +
          `human-images:${stats.humanImagesStripped} ` +
          `reasoning:${stats.reasoningStripped} ` +
          `summarized:${summarized}` +
          (summarized ? ` summary_ms:${summaryMs}` : '')
      );

      if (nothingChanged) return undefined;

      return {
        messages: [new RemoveMessage({ id: REMOVE_ALL_MESSAGES }), ...rebuilt],
      };
    },
  });
}

// Exposed for tests; do not call from app code.
export const __inflightSummariesForTest = inflightSummaries;
