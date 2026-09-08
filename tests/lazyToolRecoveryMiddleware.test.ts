import { describe, it, expect, vi } from 'vitest'
import {
  AIMessage,
  HumanMessage,
  isHumanMessage,
  type BaseMessage,
} from '@langchain/core/messages'
import { createLazyToolRecoveryMiddleware } from '../src/agent/lazyToolRecoveryMiddleware.js'

interface HookContainer {
  wrapModelCall?: unknown
}

function getWrapModelCall(
  hook: unknown
): (request: unknown, handler: unknown) => Promise<unknown> {
  if (typeof hook === 'function') return hook as never
  if (hook && typeof hook === 'object' && 'hook' in hook && typeof (hook as { hook: unknown }).hook === 'function') {
    return (hook as { hook: (request: unknown, handler: unknown) => Promise<unknown> }).hook
  }
  throw new Error('wrapModelCall hook not callable')
}

const TOOLS = [{ name: 'read_distance' }, { name: 'move_forward' }, { name: 'turn_right' }]

// Build a model whose isolated classifier call returns the given verdict text.
function classifierModel(verdict: string) {
  const invoke = vi.fn(async () => new AIMessage(verdict))
  return { invoke }
}

function lazyReply(text: string, extra: { response_metadata?: Record<string, unknown> } = {}) {
  return new AIMessage({ content: text, ...extra })
}

function toolReply() {
  return new AIMessage({
    content: '',
    tool_calls: [{ name: 'read_distance', args: {}, id: 'tc-recovered' }],
  })
}

function makeRequest(model: { invoke: ReturnType<typeof vi.fn> }, messages: BaseMessage[] = [new HumanMessage('go')]) {
  return { model, messages, tools: TOOLS }
}

describe('lazyToolRecoveryMiddleware', () => {
  it('passes through immediately when the model already called a tool', async () => {
    const model = classifierModel('NO')
    const mw = createLazyToolRecoveryMiddleware() as HookContainer
    const wrap = getWrapModelCall(mw.wrapModelCall)

    const handler = vi.fn(async () => toolReply())
    const out = await wrap(makeRequest(model), handler)

    expect(handler).toHaveBeenCalledTimes(1)
    expect(model.invoke).not.toHaveBeenCalled()
    expect((out as AIMessage).tool_calls?.[0].name).toBe('read_distance')
  })

  it('recovers a lazy reply: classifier says YES, re-prompts via handler', async () => {
    const model = classifierModel('YES\nread_distance')
    const mw = createLazyToolRecoveryMiddleware() as HookContainer
    const wrap = getWrapModelCall(mw.wrapModelCall)

    // First handler call: lazy narration naming a tool. Second: the real call.
    const handler = vi
      .fn()
      .mockResolvedValueOnce(lazyReply('`read_distance` to check current range before moving.'))
      .mockResolvedValueOnce(toolReply())
    const out = await wrap(makeRequest(model), handler)

    expect(model.invoke).toHaveBeenCalledTimes(1) // one classifier round-trip
    expect(handler).toHaveBeenCalledTimes(2) // initial + recovery
    expect((out as AIMessage).tool_calls?.[0].name).toBe('read_distance')

    // The recovery handler call received the lazy reply + a nudge appended.
    const secondCallReq = handler.mock.calls[1][0] as { messages: BaseMessage[] }
    const appended = secondCallReq.messages
    expect(isHumanMessage(appended.at(-1) as BaseMessage)).toBe(true)
    expect((appended.at(-1) as HumanMessage).content).toContain('did not actually call the tool')
  })

  it('does not recover when the classifier says NO', async () => {
    const model = classifierModel('NO')
    const mw = createLazyToolRecoveryMiddleware() as HookContainer
    const wrap = getWrapModelCall(mw.wrapModelCall)

    const handler = vi.fn(async () =>
      lazyReply('I have reached the green marker with read_distance confirming ~5cm. Done.')
    )
    const out = await wrap(makeRequest(model), handler)

    expect(model.invoke).toHaveBeenCalledTimes(1)
    expect(handler).toHaveBeenCalledTimes(1) // no recovery
    expect((out as AIMessage).content).toContain('Done')
  })

  it('skips the classifier entirely when the reply mentions no tool', async () => {
    const model = classifierModel('YES\nread_distance')
    const mw = createLazyToolRecoveryMiddleware() as HookContainer
    const wrap = getWrapModelCall(mw.wrapModelCall)

    const handler = vi.fn(async () => lazyReply('The robot is at the cone. Task complete.'))
    const out = await wrap(makeRequest(model), handler)

    expect(model.invoke).not.toHaveBeenCalled() // cheap gate short-circuits
    expect(handler).toHaveBeenCalledTimes(1)
    expect((out as AIMessage).content).toContain('Task complete')
  })

  it('treats done_reason=length as truncation, not laziness — no recovery', async () => {
    const model = classifierModel('YES\nread_distance')
    const mw = createLazyToolRecoveryMiddleware() as HookContainer
    const wrap = getWrapModelCall(mw.wrapModelCall)

    const handler = vi.fn(async () =>
      lazyReply('I will now call `read_distance` to', {
        response_metadata: { done_reason: 'length' },
      })
    )
    const out = await wrap(makeRequest(model), handler)

    expect(model.invoke).not.toHaveBeenCalled()
    expect(handler).toHaveBeenCalledTimes(1)
    expect((out as AIMessage).response_metadata?.done_reason).toBe('length')
  })

  it('skipClassifier re-prompts on a tool mention without a classifier call', async () => {
    const model = classifierModel('NO') // would say NO, but we skip it
    const mw = createLazyToolRecoveryMiddleware({ skipClassifier: true }) as HookContainer
    const wrap = getWrapModelCall(mw.wrapModelCall)

    const handler = vi
      .fn()
      .mockResolvedValueOnce(lazyReply('Next move: `turn_right` steps=2 to point toward the marker'))
      .mockResolvedValueOnce(toolReply())
    const out = await wrap(makeRequest(model), handler)

    expect(model.invoke).not.toHaveBeenCalled()
    expect(handler).toHaveBeenCalledTimes(2)
    expect((out as AIMessage).tool_calls?.[0].name).toBe('read_distance')
  })

  it('maxRecoveries=0 disables recovery', async () => {
    const model = classifierModel('YES\nread_distance')
    const mw = createLazyToolRecoveryMiddleware({ maxRecoveries: 0 }) as HookContainer
    const wrap = getWrapModelCall(mw.wrapModelCall)

    const handler = vi.fn(async () => lazyReply('`read_distance` next.'))
    const out = await wrap(makeRequest(model), handler)

    expect(model.invoke).not.toHaveBeenCalled()
    expect(handler).toHaveBeenCalledTimes(1)
    expect((out as AIMessage).content).toContain('read_distance')
  })

  it('stops after a single recovery even if the retry is still lazy', async () => {
    const model = classifierModel('YES\nread_distance')
    const mw = createLazyToolRecoveryMiddleware() as HookContainer
    const wrap = getWrapModelCall(mw.wrapModelCall)

    // Both handler calls produce lazy text; recovery is capped at 1.
    const handler = vi.fn(async () => lazyReply('`read_distance` to check the range first.'))
    const out = await wrap(makeRequest(model), handler)

    expect(handler).toHaveBeenCalledTimes(2) // initial + one recovery, then stop
    expect((out as AIMessage).tool_calls?.length ?? 0).toBe(0)
  })

  describe('force mode', () => {
    it('recovers a no-tool reply that names NO tool (no classifier)', async () => {
      const model = classifierModel('NO')
      const mw = createLazyToolRecoveryMiddleware({ force: true }) as HookContainer
      const wrap = getWrapModelCall(mw.wrapModelCall)

      const handler = vi
        .fn()
        .mockResolvedValueOnce(lazyReply('The robot looks centred and the task seems complete.'))
        .mockResolvedValueOnce(toolReply())
      const out = await wrap(makeRequest(model), handler)

      expect(model.invoke).not.toHaveBeenCalled() // force skips the classifier
      expect(handler).toHaveBeenCalledTimes(2)
      expect((out as AIMessage).tool_calls?.[0].name).toBe('read_distance')

      const nudge = (handler.mock.calls[1][0] as { messages: BaseMessage[] }).messages.at(-1)
      expect(isHumanMessage(nudge as BaseMessage)).toBe(true)
      expect((nudge as HumanMessage).content).toContain('finish_task')
    })

    it('recovers an empty no-tool reply', async () => {
      const model = classifierModel('NO')
      const mw = createLazyToolRecoveryMiddleware({ force: true }) as HookContainer
      const wrap = getWrapModelCall(mw.wrapModelCall)

      const handler = vi
        .fn()
        .mockResolvedValueOnce(lazyReply(''))
        .mockResolvedValueOnce(toolReply())
      const out = await wrap(makeRequest(model), handler)

      expect(handler).toHaveBeenCalledTimes(2)
      expect((out as AIMessage).tool_calls?.[0].name).toBe('read_distance')
    })

    it('still leaves a truncated (done_reason=length) reply alone', async () => {
      const model = classifierModel('NO')
      const mw = createLazyToolRecoveryMiddleware({ force: true }) as HookContainer
      const wrap = getWrapModelCall(mw.wrapModelCall)

      const handler = vi.fn(async () =>
        lazyReply('cut off mid-thou', { response_metadata: { done_reason: 'length' } })
      )
      const out = await wrap(makeRequest(model), handler)

      expect(handler).toHaveBeenCalledTimes(1)
      expect((out as AIMessage).response_metadata?.done_reason).toBe('length')
    })

    it('defaults to two attempts in force mode', async () => {
      const model = classifierModel('NO')
      const mw = createLazyToolRecoveryMiddleware({ force: true }) as HookContainer
      const wrap = getWrapModelCall(mw.wrapModelCall)

      // Every reply is no-tool prose; force retries twice (initial + 2).
      const handler = vi.fn(async () => lazyReply('still just talking, no tool'))
      const out = await wrap(makeRequest(model), handler)

      expect(handler).toHaveBeenCalledTimes(3) // initial + 2 recoveries
      expect((out as AIMessage).tool_calls?.length ?? 0).toBe(0)
    })
  })
})
