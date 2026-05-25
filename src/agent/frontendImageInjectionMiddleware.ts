import { createMiddleware } from 'langchain';
import { ToolMessage, HumanMessage } from '@langchain/core/messages';
import type { MessageContent } from '@langchain/core/messages';
import { MOTION_TOOL_NAMES } from './robotTools.js';

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
  // ChatOllama only accepts {type:'image_url', image_url: <data-URL>} blocks;
  // ChatAnthropic accepts the LangChain standard {type:'image', source_type, ...}
  // block. Pick the right shape per provider.
  provider: 'ollama' | 'anthropic';
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
          msg instanceof ToolMessage &&
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
        // Mark injected up-front so a retry / re-entry never double-injects,
        // even on the error path below.
        injectedIds.add(id);
        if (payload.error) {
          const label = MOTION_NAMES.has(toolName) ? `Motion (${toolName}) failed` : 'Camera unavailable';
          newMessages.push(new HumanMessage({ content: `${label}: ${payload.error}` }));
          continue;
        }
        if (payload.mimeType && payload.data) {
          const block =
            opts.provider === 'ollama'
              ? {
                  type: 'image_url' as const,
                  image_url: `data:${payload.mimeType};base64,${payload.data}`,
                }
              : {
                  type: 'image' as const,
                  source_type: 'base64' as const,
                  mime_type: payload.mimeType,
                  data: payload.data,
                };

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
      }

      return { messages: newMessages };
    },
  });
}
