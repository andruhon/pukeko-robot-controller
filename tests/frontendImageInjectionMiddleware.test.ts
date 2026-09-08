import { describe, it, expect } from 'vitest'
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  isHumanMessage,
  type BaseMessage,
} from '@langchain/core/messages'
import { createFrontendImageInjectionMiddleware } from '../src/agent/frontendImageInjectionMiddleware.js'
import { foreignToolMessage, isForeignToThisCore } from './helpers/foreignCoreMessage.js'

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
      isHumanMessage(m) &&
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

// ───────────────────────────────────────────────────────────────────────────
// RC-58 — the once-per-tool_call_id accounting must hold for FOREIGN-copy
// results too
//
// `rc21CrossCoreToolMessage.test.ts` pins that a foreign capture result injects
// at all. What it does not exercise is the bookkeeping this file exists for:
// exactly one image per tool_call_id, and never a re-injection of a result the
// summarizer retained. Both are decided by the same `isToolMessage` scan in
// `src/agent/frontendImageInjectionMiddleware.ts`, so with only native fixtures
// above, that accounting has never been shown to work on the shape the server
// actually receives from gaunt-sloth's pipeline — which is the shape RC-21's
// bug lived in.
// ───────────────────────────────────────────────────────────────────────────
describe('frontendImageInjectionMiddleware — RC-58 foreign-copy motion results', () => {
  function foreignMotionToolMessage(id: string, motion: string, before: string, after: string) {
    return foreignToolMessage({
      content: JSON.stringify({
        mimeType: 'image/jpeg',
        data: 'BASE64DATA',
        motion,
        distanceBefore: before,
        distanceAfter: after,
      }),
      tool_call_id: id,
      name: motion.split(' ')[0],
    })
  }

  it('the fixture really is foreign to the copy this file imports', () => {
    expect(isForeignToThisCore(foreignMotionToolMessage('tc', 'turn_right (steps=3)', '1', '2'))).toBe(true)
    // The control: a message from THIS copy is not foreign.
    expect(isForeignToThisCore(motionToolMessage('tc', 'turn_right (steps=3)', '1', '2'))).toBe(false)
  })

  it('injects exactly one image per foreign-copy result and never re-injects a retained one', async () => {
    const mw = createFrontendImageInjectionMiddleware({ provider: 'ollama' }) as HookContainer
    const before = getHook(mw.beforeModel)

    const turnRight = foreignMotionToolMessage('tc-turn-x', 'turn_right (steps=3)', '26.1', '23.6')

    // Turn A: the foreign turn_right result arrives -> inject its image once.
    const stateA = {
      messages: [
        new HumanMessage('go'),
        new AIMessage({
          content: '',
          tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-turn-x' }],
        }),
        turnRight,
      ],
    }
    const resA = (await before(stateA, runtime)) as { messages: BaseMessage[] }
    expect(injectedImageCount(resA.messages)).toBe(1)
    const labelA = (resA.messages.at(-1) as HumanMessage).content as Array<{ text?: string }>
    expect(labelA[0].text).toContain('turn_right')

    // Turn B: the summarizer retained the SAME foreign turn_right result, and a
    // new foreign move_forward result arrived. Only the move_forward image is
    // injected, and it is last — no stale re-injection.
    const moveForward = foreignMotionToolMessage('tc-fwd-x', 'move_forward (steps=3)', '23.5', '18.1')
    const stateB = {
      messages: [
        new HumanMessage('go'),
        turnRight,
        new AIMessage({
          content: '',
          tool_calls: [{ name: 'move_forward', args: { steps: 3 }, id: 'tc-fwd-x' }],
        }),
        moveForward,
      ],
    }
    const resB = (await before(stateB, runtime)) as { messages: BaseMessage[] }
    expect(injectedImageCount(resB.messages)).toBe(1)
    const labelB = (resB.messages.at(-1) as HumanMessage).content as Array<{ text?: string }>
    expect(labelB[0].text).toContain('move_forward')
    expect(labelB[0].text).not.toContain('turn_right')
  })
})
