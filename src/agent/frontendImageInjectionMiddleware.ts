import { createMiddleware } from 'langchain';
import { ToolMessage, HumanMessage } from '@langchain/core/messages';
import type { MessageContent } from '@langchain/core/messages';
import { MOTION_TOOL_NAMES } from './robotTools.js';

interface ImagePayload {
  mimeType?: string;
  data?: string;
  error?: string;
  // Motion tools include sensor readings around the move.
  distanceBefore?: string;
  distanceAfter?: string;
  // Optional human-facing motion label, e.g. "move_forward (steps=2)".
  motion?: string;
}

const MOTION_NAMES: ReadonlySet<string> = new Set(MOTION_TOOL_NAMES);

export interface ImageInjectionOptions {
  // ChatOllama only accepts {type:'image_url', image_url: <data-URL>} blocks;
  // ChatAnthropic accepts the LangChain standard {type:'image', source_type, ...}
  // block. Pick the right shape per provider.
  provider: 'ollama' | 'anthropic';
}

function formatDistanceDelta(before?: string, after?: string): string | null {
  if (!before && !after) return null;
  const b = before ? `${before} cm` : 'unknown';
  const a = after ? `${after} cm` : 'unknown';
  const nb = before ? Number.parseFloat(before) : Number.NaN;
  const na = after ? Number.parseFloat(after) : Number.NaN;
  let delta = '';
  if (Number.isFinite(nb) && Number.isFinite(na)) {
    const d = na - nb;
    const sign = d >= 0 ? '+' : '';
    delta = ` (Δ ${sign}${d.toFixed(1)} cm)`;
  }
  return `Distance: ${b} → ${a}${delta}`;
}

export function createFrontendImageInjectionMiddleware(opts: ImageInjectionOptions) {
  return createMiddleware({
    name: 'frontend-image-injection',

    beforeModel: async (state) => {
      const messages = state.messages || [];
      // Each entry pairs the parsed envelope with the originating tool name so
      // we can prepend a motion label when relevant.
      const injected: Array<{ payload: ImagePayload; toolName: string }> = [];

      for (let i = messages.length - 1; i >= Math.max(0, messages.length - 5); i--) {
        const msg = messages[i];
        if (
          msg instanceof ToolMessage &&
          typeof msg.content === 'string' &&
          (msg.name === 'capture_image' || (msg.name && MOTION_NAMES.has(msg.name)))
        ) {
          try {
            injected.push({
              payload: JSON.parse(msg.content) as ImagePayload,
              toolName: msg.name,
            });
          } catch {
            // Non-JSON tool result — skip injection.
          }
        }
      }

      if (injected.length === 0) return undefined;

      const newMessages = [...messages];
      for (const { payload, toolName } of injected) {
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
          const distanceLine = formatDistanceDelta(payload.distanceBefore, payload.distanceAfter);
          const headerText = isMotion
            ? `Before/After frames for ${payload.motion ?? toolName}.${distanceLine ? ` ${distanceLine}.` : ''}`
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
