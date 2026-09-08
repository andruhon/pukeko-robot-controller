// RC-21 (golden fix) — the vision block must be injected for a capture
// ToolMessage that the class-based check cannot recognise.
//
// WHAT THIS FIXTURE ACTUALLY PINS: marker-absence. `@langchain/core`'s message
// classes answer `instanceof` through `static [Symbol.hasInstance]`, which
// delegates to a duck test keyed on the global-registry symbol
// `Symbol.for('langchain.message')`. So an object that carries the marker passes
// `instanceof` whatever its prototype, and an object that does NOT carry it —
// like the plain-duck one built below — fails `instanceof` while satisfying
// `isToolMessage()`. That is the condition this file pins, and it is the shape a
// message rebuilt from the wire, or built by a core older than the marker,
// arrives in. It says nothing about which copy of core built anything: nothing
// available to this test can distinguish that, and an earlier version of this
// header claiming otherwise was wrong.
//
// WHERE THE COST WAS REAL: when the robot genuinely resolved two @langchain/core
// copies — its own plus one pulled in transitively — a `capture_image` result
// constructed by the other copy was not an instance of the `ToolMessage` class
// this repo imports. The original guard `msg instanceof ToolMessage` silently
// returned false for it, so no frame was ever injected on the real server;
// observability showed tool-data:1 / human-images:0 / imageCount:0, measured
// under those two-copy conditions. The context-pruner, which uses core's
// duck-typed `isToolMessage`, matched the same message fine. This pins the fix:
// FI uses `isToolMessage`, so a message the class check cannot see still
// injects.
//
// This is a MUTATION-CHECKED pin: reverting FI's guard back to
// `msg instanceof ToolMessage` makes the fixture fail the scan and this test's
// "was a vision HumanMessage injected?" assertion fails. (The prior RC-21 repro
// used real ToolMessage instances from THIS copy, so instanceof passed there —
// which is exactly why it could not catch this bug.)
//
// It is also the deliberate exemption in `tests/messageClassInstanceofGuard.test.ts`
// and must NOT be converted to the duck-typed spelling: its `instanceof`
// assertion is the only direct pin on the condition the whole rule exists for,
// and that guard asserts the exemption is still being used.
import { describe, it, expect } from 'vitest'
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  isHumanMessage,
  isToolMessage,
  type BaseMessage,
} from '@langchain/core/messages'
import { createFrontendImageInjectionMiddleware } from '../src/agent/frontendImageInjectionMiddleware.js'

const MIME = 'image/jpeg'
const DATA = 'BASE64DATA_deadbeef'
const DATA_URL = `data:${MIME};base64,${DATA}`

function getBeforeModel(mw: unknown): (s: unknown, r: unknown) => Promise<unknown> {
  const hook = (mw as { beforeModel?: unknown }).beforeModel
  if (typeof hook === 'function') return hook as (s: unknown, r: unknown) => Promise<unknown>
  if (hook && typeof hook === 'object' && 'hook' in hook) {
    return (hook as { hook: (s: unknown, r: unknown) => Promise<unknown> }).hook
  }
  throw new Error('no beforeModel hook')
}

// A ToolMessage-shaped message that is NOT an instance of the `ToolMessage`
// class this repo imports, because it carries no `Symbol.for('langchain.message')`
// marker. Plain-duck construction is what produces that: re-prototyping a real
// message does not, since the marker rides along on the object and `instanceof`
// keeps passing. It satisfies core's duck-typed `isToolMessage()` (has a
// `getType()` returning 'tool') but fails `instanceof ToolMessage`.
function foreignCoreToolMessage(fields: {
  content: string
  tool_call_id: string
  name: string
}): unknown {
  return {
    ...fields,
    getType: () => 'tool',
    _getType: () => 'tool',
  }
}

let nonce = 0

describe('RC-21 (golden) — capture ToolMessage from a foreign @langchain/core copy still injects', () => {
  it('the fixture reproduces the real condition: fails instanceof, passes isToolMessage', () => {
    const foreign = foreignCoreToolMessage({
      content: JSON.stringify({ mimeType: MIME, data: DATA }),
      tool_call_id: 'tc-fixture',
      name: 'capture_image',
    })
    // The exact cross-copy gap the golden bug lived in:
    expect(foreign instanceof ToolMessage).toBe(false)
    expect(isToolMessage(foreign)).toBe(true)
  })

  it('injects a provider-correct vision HumanMessage for a foreign-copy capture result (openai)', async () => {
    const uid = `xcore-${Date.now()}-${nonce++}`
    const fi = createFrontendImageInjectionMiddleware({ provider: 'openai' })
    const before = getBeforeModel(fi)
    const state = {
      messages: [
        new HumanMessage('take a picture, what do you see?'),
        new AIMessage({ content: '', tool_calls: [{ name: 'capture_image', args: {}, id: `tc-${uid}` }] }),
        // The capture result as it really arrives — a foreign-core ToolMessage.
        foreignCoreToolMessage({
          content: JSON.stringify({ mimeType: MIME, data: DATA }),
          tool_call_id: `tc-${uid}`,
          name: 'capture_image',
        }),
      ] as BaseMessage[],
    }

    const res = (await before(state, { configurable: { thread_id: `xcore-${uid}` } })) as {
      messages: BaseMessage[]
    }

    // A vision HumanMessage must have been appended (human-images:1, not 0).
    const injected = res.messages.at(-1)
    expect(injected).toBeDefined()
    expect(isHumanMessage(injected as BaseMessage)).toBe(true)
    const content = (injected as HumanMessage).content as Array<Record<string, unknown>>
    expect(Array.isArray(content)).toBe(true)
    // [ {type:'text', ...}, {type:'image_url', image_url:{url}} ] for openai.
    expect(content[1]).toEqual({ type: 'image_url', image_url: { url: DATA_URL } })
  })

  it('an error-only foreign-copy capture result still surfaces the note (not silently skipped)', async () => {
    const uid = `xcore-err-${Date.now()}-${nonce++}`
    const fi = createFrontendImageInjectionMiddleware({ provider: 'openai' })
    const before = getBeforeModel(fi)
    const state = {
      messages: [
        new HumanMessage('take a picture'),
        new AIMessage({ content: '', tool_calls: [{ name: 'capture_image', args: {}, id: `tc-${uid}` }] }),
        foreignCoreToolMessage({
          content: JSON.stringify({ error: 'Failed to capture frame. Is the camera active?' }),
          tool_call_id: `tc-${uid}`,
          name: 'capture_image',
        }),
      ] as BaseMessage[],
    }

    const res = (await before(state, { configurable: { thread_id: `xcore-err-${uid}` } })) as {
      messages: BaseMessage[]
    }
    const injected = res.messages.at(-1)
    expect(isHumanMessage(injected as BaseMessage)).toBe(true)
    expect((injected as HumanMessage).content).toContain('Camera unavailable')
  })
})
