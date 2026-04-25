import { createMiddleware } from 'langchain';
import { ToolMessage, HumanMessage } from '@langchain/core/messages';
import type { MessageContent } from '@langchain/core/messages';

interface ImagePayload {
  mimeType?: string;
  data?: string;
  error?: string;
}

export const frontendImageInjectionMiddleware = createMiddleware({
  name: 'frontend-image-injection',

  beforeModel: async (state) => {
    const messages = state.messages || [];
    const injected: ImagePayload[] = [];

    for (let i = messages.length - 1; i >= Math.max(0, messages.length - 5); i--) {
      const msg = messages[i];
      if (
        msg instanceof ToolMessage &&
        msg.name === 'capture_image' &&
        typeof msg.content === 'string'
      ) {
        try {
          injected.push(JSON.parse(msg.content) as ImagePayload);
        } catch {
          // Non-JSON tool result — skip injection.
        }
      }
    }

    if (injected.length === 0) return undefined;

    const newMessages = [...messages];
    for (const payload of injected) {
      if (payload.error) {
        newMessages.push(new HumanMessage({ content: `Camera unavailable: ${payload.error}` }));
        continue;
      }
      if (payload.mimeType && payload.data) {
        newMessages.push(
          new HumanMessage({
            content: [
              { type: 'text', text: 'Camera frame captured:' },
              {
                type: 'image',
                source_type: 'base64',
                mime_type: payload.mimeType,
                data: payload.data,
              },
            ] as MessageContent,
          })
        );
      }
    }

    return { messages: newMessages };
  },
});
