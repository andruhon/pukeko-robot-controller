import { createMiddleware } from 'langchain';
import { HumanMessage, isToolMessage } from '@langchain/core/messages';
import type { MessageContent } from '@langchain/core/messages';
import { MOTION_TOOL_NAMES } from './robotToolNames.js';
import type { LlmProvider } from "../lib/config.js";

interface ImagePayload {
  mimeType?: string;
  data?: string;
  error?: string;
  // Optional human-facing motion label, e.g. "move_forward (steps=2)".
  motion?: string;
}

const MOTION_NAMES: ReadonlySet<string> = new Set(MOTION_TOOL_NAMES);

// thread_id → set of tool_call_ids whose image has already been injected.
// The motion-summarization middleware keeps the latest motion's ToolMessage in
// its retained tail, so without this guard that ToolMessage would be seen again
// on the next turn and its image re-injected — appended *after* the newest
// motion's image, mispairing the current assistant message with a stale frame
// (e.g. a move_forward turn showing the previous turn_right image).
const injectedByThread = new Map<string, Set<string>>();

export interface ImageInjectionOptions {
  // Providers disagree on the vision-block shape they can decode; see
  // `imageBlockFor` for the per-provider mapping and the evidence behind it.
  provider: LlmProvider;
}

// A vision content block the target provider's @langchain converter actually
// decodes. Verified against the installed converters (RC-21):
//   - ollama    → {type:'image_url', image_url:<data-URL string>}. ChatOllama's
//                 convertToOllamaMessages only handles `image_url` blocks
//                 (utils.ts extractBase64FromDataUrl); the LangChain standard
//                 `source_type` block THROWS "Unsupported content type: image".
//   - openai /  → {type:'image_url', image_url:{url:<data-URL>}}. This is the
//     openrouter  native OpenAI shape and is correct on BOTH the Completions API
//                 (gpt-5.5 / gpt-5.6-luna → `image_url`) AND the Responses API
//                 (gpt-*-pro / codex → `input_image`). A raw `source_type`
//                 standard block serialises to an *invalid* `image_url` part on
//                 the Responses path, so we emit the provider-native shape
//                 rather than lean on @langchain/core's (deprecated, internal)
//                 isDataContentBlock auto-conversion.
//   - anthropic → LangChain standard {type:'image', source_type:'base64', ...}.
//     / google    ChatAnthropic (native) and ChatGoogle (→ inlineData) both
//                 decode the standard data content block directly.
function imageBlockFor(provider: LlmProvider, mimeType: string, data: string) {
  const dataUrl = `data:${mimeType};base64,${data}`;
  switch (provider) {
    case 'ollama':
      return { type: 'image_url' as const, image_url: dataUrl };
    case 'openai':
    case 'openrouter':
      return { type: 'image_url' as const, image_url: { url: dataUrl } };
    case 'anthropic':
    case 'google':
    default:
      return {
        type: 'image' as const,
        source_type: 'base64' as const,
        mime_type: mimeType,
        data,
      };
  }
}

export function createFrontendImageInjectionMiddleware(opts: ImageInjectionOptions) {
  return createMiddleware({
    name: 'frontend-image-injection',

    beforeModel: async (state, runtime) => {
      const messages = state.messages || [];
      const threadId = runtime?.configurable?.thread_id ?? '__default__';
      let injectedIds = injectedByThread.get(threadId);
      if (!injectedIds) {
        injectedIds = new Set<string>();
        injectedByThread.set(threadId, injectedIds);
      }

      // Each entry pairs the parsed envelope with the originating tool name so
      // we can prepend a motion label when relevant. Scan forward so injected
      // frames stay in chronological order, and skip any tool_call_id we've
      // already injected (idempotent across the summarizer's retained tail).
      const injected: Array<{ payload: ImagePayload; toolName: string; id: string }> = [];

      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (
          // RC-21 (golden fix): a capture ToolMessage does not always answer to
          // the `ToolMessage` class WE import. `msg instanceof ToolMessage`
          // silently returned false on the real server and no frame was ever
          // injected (the pruner's duck-typed `isToolMessage` saw it fine, hence
          // tool-data:1 / human-images:0 in the dumps) — measured when the robot
          // resolved two @langchain/core copies, which it no longer does. The
          // class check is unreliable here regardless: core answers `instanceof`
          // through a duck test keyed on `Symbol.for('langchain.message')`, so a
          // result rebuilt from the wire without that marker fails it just the
          // same. Use the duck-typed guard the pruner uses.
          isToolMessage(msg) &&
          typeof msg.content === 'string' &&
          (msg.name === 'capture_image' || (msg.name && MOTION_NAMES.has(msg.name)))
        ) {
          const id = msg.tool_call_id;
          if (!id || injectedIds.has(id)) continue;
          try {
            injected.push({
              payload: JSON.parse(msg.content) as ImagePayload,
              toolName: msg.name,
              id,
            });
          } catch {
            // Non-JSON tool result — skip injection.
          }
        }
      }

      if (injected.length === 0) return undefined;

      const newMessages = [...messages];
      for (const { payload, toolName, id } of injected) {
        if (payload.error) {
          // Mark injected so the error note isn't re-emitted on a later turn.
          injectedIds.add(id);
          const label = MOTION_NAMES.has(toolName) ? `Motion (${toolName}) failed` : 'Camera unavailable';
          newMessages.push(new HumanMessage({ content: `${label}: ${payload.error}` }));
          continue;
        }
        if (payload.mimeType && payload.data) {
          // RC-21: mark the tool_call_id as injected ONLY when we actually emit a
          // frame. The original code marked up-front, so a capture ToolMessage
          // that arrived WITHOUT its base64 `data` (dropped upstream — e.g. a
          // pruned/replayed history) both injected nothing AND poisoned the
          // guard, permanently blocking that frame even if the data-bearing
          // result showed up on a later turn. Marking on successful injection
          // keeps the "never re-inject a retained frame" idempotency (a real
          // frame always has data) while letting a later data-bearing sighting
          // recover. (Within this process the context-pruner strips `data` only
          // AFTER this middleware runs, so a data-less sighting means the bytes
          // were absent before FI ever saw them — the RC-21 upstream case.)
          injectedIds.add(id);
          const block = imageBlockFor(opts.provider, payload.mimeType, payload.data);

          const isMotion = MOTION_NAMES.has(toolName);
          const headerText = isMotion
            ? `Before/After frames for ${payload.motion ?? toolName}.`
            : 'Camera frame captured:';

          newMessages.push(
            new HumanMessage({
              content: [
                { type: 'text', text: headerText },
                block,
              ] as MessageContent,
            })
          );
        }
        // else: a capture/motion tool result whose `data` is absent — inject
        // nothing and leave the guard clean so a later data-bearing result for
        // the same tool_call_id can still be injected.
      }

      return { messages: newMessages };
    },
  });
}
