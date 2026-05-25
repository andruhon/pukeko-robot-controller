import { describe, it, expect } from 'vitest'
import { AIMessage, HumanMessage, ToolMessage, type BaseMessage } from '@langchain/core/messages'
import { createFrontendImageInjectionMiddleware } from '../src/agent/frontendImageInjectionMiddleware.js'

interface HookContainer {
  beforeModel?: unknown
}

function getHook(hook: unknown): (state: unknown, runtime: unknown) => Promise<unknown> {
  if (typeof hook === 'function') return hook as (state: unknown, runtime: unknown) => Promise<unknown>
  if (hook && typeof hook === 'object' && 'hook' in hook && typeof (hook as { hook: unknown }).hook === 'function') {
    return (hook as { hook: (state: unknown, runtime: unknown) => Promise<unknown> }).hook
  }
  throw new Error('Hook not callable')
}

const runtime = { configurable: { thread_id: 'inj-thread' } }

function motionToolMessage(id: string, motion: string, before: string, after: string) {
  return new ToolMessage({
    content: JSON.stringify({ mimeType: 'image/jpeg', data: 'BASE64DATA', motion, distanceBefore: before, distanceAfter: after }),
    tool_call_id: id,
    name: motion.split(' ')[0],
  })
}

function injectedImageCount(messages: BaseMessage[]): number {
  return messages.filter(
    (m) =>
      m instanceof HumanMessage &&
      Array.isArray(m.content) &&
      (m.content as Array<{ type?: string }>).some((b) => b.type === 'image' || b.type === 'image_url')
  ).length
}

describe('frontendImageInjectionMiddleware', () => {
  it('injects exactly one image per tool_call_id and never re-injects a retained ToolMessage', async () => {
    const mw = createFrontendImageInjectionMiddleware({ provider: 'ollama' }) as HookContainer
    const before = getHook(mw.beforeModel)

    const turnRight = motionToolMessage('tc-turn', 'turn_right (steps=3)', '26.1', '23.6')

    // Turn A: the turn_right result arrives -> inject its image once.
    const stateA = { messages: [new HumanMessage('go'), new AIMessage({ content: '', tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-turn' }] }), turnRight] }
    const resA = (await before(stateA, runtime)) as { messages: BaseMessage[] }
    expect(injectedImageCount(resA.messages)).toBe(1)
    const labelA = (resA.messages.at(-1) as HumanMessage).content as Array<{ type?: string; text?: string }>
    expect(labelA[0].text).toContain('turn_right')

    // Turn B: the summarizer retained the SAME turn_right ToolMessage, and a new
    // move_forward result arrived. Only the move_forward image must be injected,
    // and it must be the last message (not a re-injected stale turn_right).
    const moveForward = motionToolMessage('tc-fwd', 'move_forward (steps=3)', '23.5', '18.1')
    const stateB = {
      messages: [
        new HumanMessage('go'),
        turnRight, // retained, already injected
        new AIMessage({ content: '', tool_calls: [{ name: 'move_forward', args: { steps: 3 }, id: 'tc-fwd' }] }),
        moveForward,
      ],
    }
    const resB = (await before(stateB, runtime)) as { messages: BaseMessage[] }
    // turn_right is NOT re-injected; only the new move_forward image is added.
    expect(injectedImageCount(resB.messages)).toBe(1)
    const lastB = resB.messages.at(-1) as HumanMessage
    const labelB = lastB.content as Array<{ type?: string; text?: string }>
    expect(labelB[0].text).toContain('move_forward')
    expect(labelB[0].text).not.toContain('turn_right')
  })
})
