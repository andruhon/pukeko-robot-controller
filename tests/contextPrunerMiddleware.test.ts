import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  RemoveMessage,
  isHumanMessage,
  isSystemMessage,
  type BaseMessage,
} from '@langchain/core/messages'
import { MemorySaver, messagesStateReducer } from '@langchain/langgraph'
import {
  createContextPrunerMiddleware,
  estimateTokens,
  __inflightSummariesForTest,
} from '../src/agent/contextPrunerMiddleware.js'
import { __resetMotionLogForTest, isMotionToolCall } from '../src/agent/motionLog.js'

interface HookContainer {
  beforeModel?: unknown
  afterModel?: unknown
}

function getHook(hook: unknown): (state: unknown, runtime: unknown) => unknown {
  if (typeof hook === 'function') return hook as (state: unknown, runtime: unknown) => unknown
  if (hook && typeof hook === 'object' && 'hook' in hook && typeof (hook as { hook: unknown }).hook === 'function') {
    return (hook as { hook: (state: unknown, runtime: unknown) => unknown }).hook
  }
  throw new Error('Hook not callable')
}

const SUMMARY_TEXT = 'Robot is south of cone, facing west; turn_right rotates clockwise here.'

function makeStubLlm(summary = SUMMARY_TEXT) {
  const invoke = vi.fn(async () => ({ content: summary }))
  return { invoke } as unknown as Parameters<typeof createContextPrunerMiddleware>[0]['llm'] & {
    invoke: ReturnType<typeof vi.fn>
  }
}

const runtime = { configurable: { thread_id: 'test-thread' } }

function imageBlock() {
  return { type: 'image_url' as const, image_url: 'data:image/jpeg;base64,XXXX' }
}

function motionResultJson(motion: string, dataLen = 100): string {
  return JSON.stringify({
    mimeType: 'image/jpeg',
    data: 'X'.repeat(dataLen),
    motion,
  })
}

beforeEach(() => {
  __inflightSummariesForTest.clear()
  // motionLog is shared module state; reset it so the pinned-state branch
  // exercised below (via afterModel) can't bleed motions into later tests.
  __resetMotionLogForTest()
})

describe('contextPrunerMiddleware — mechanical prune', () => {
  it('strips `data` from every motion ToolMessage unconditionally', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const userMsg = new HumanMessage('go')
    const motionAi = new AIMessage({
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 1 }, id: 'tc-1' }],
    })
    const motionTool = new ToolMessage({
      content: motionResultJson('turn_right (steps=1)'),
      tool_call_id: 'tc-1',
      name: 'turn_right',
    })
    const injected = new HumanMessage({
      content: [{ type: 'text', text: 'Before/After frames for turn_right (steps=1).' }, imageBlock()],
    })

    const result = await before(
      { messages: [userMsg, motionAi, motionTool, injected] },
      runtime
    )
    expect(result).toBeTruthy()
    const updated = (result as { messages: BaseMessage[] }).messages
    // [RemoveMessage, userMsg, motionAi, prunedToolMessage, injected]
    expect(updated[0]).toBeInstanceOf(RemoveMessage)
    const toolOut = updated[3] as ToolMessage
    expect(toolOut).toBeInstanceOf(ToolMessage)
    const parsed = JSON.parse(toolOut.content as string)
    expect(parsed.data).toBeUndefined()
    expect(parsed.motion).toBe('turn_right (steps=1)')
    expect(parsed.mimeType).toBe('image/jpeg')
    expect(parsed.dataDropped).toBe(true)
  })

  it('keeps only the latest N image HumanMessages, defaults N=1', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const userMsg = new HumanMessage('go')
    const img1 = new HumanMessage({
      content: [{ type: 'text', text: 'Frame 1' }, imageBlock()],
    })
    const img2 = new HumanMessage({
      content: [{ type: 'text', text: 'Frame 2' }, imageBlock()],
    })
    const img3 = new HumanMessage({
      content: [{ type: 'text', text: 'Frame 3' }, imageBlock()],
    })

    const result = await before(
      { messages: [userMsg, img1, img2, img3] },
      runtime
    )
    const updated = (result as { messages: BaseMessage[] }).messages
    // RemoveMessage at index 0.
    const pruned1 = updated[2] as HumanMessage
    const pruned2 = updated[3] as HumanMessage
    const keptLatest = updated[4] as HumanMessage

    const hasImage = (m: HumanMessage) =>
      Array.isArray(m.content) &&
      (m.content as Array<{ type?: string }>).some(
        (b) => b.type === 'image' || b.type === 'image_url'
      )
    expect(hasImage(pruned1)).toBe(false)
    expect(hasImage(pruned2)).toBe(false)
    expect(hasImage(keptLatest)).toBe(true)
    // The pruned ones keep their text caption.
    expect((pruned1.content as Array<{ text?: string }>)[0].text).toBe('Frame 1')
    expect((pruned2.content as Array<{ text?: string }>)[0].text).toBe('Frame 2')
  })

  it('keepLatestImages=2 retains the last two image messages', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, keepLatestImages: 2 }) as HookContainer
    const before = getHook(mw.beforeModel)

    const userMsg = new HumanMessage('go')
    const img1 = new HumanMessage({ content: [{ type: 'text', text: 'F1' }, imageBlock()] })
    const img2 = new HumanMessage({ content: [{ type: 'text', text: 'F2' }, imageBlock()] })
    const img3 = new HumanMessage({ content: [{ type: 'text', text: 'F3' }, imageBlock()] })

    const result = await before({ messages: [userMsg, img1, img2, img3] }, runtime)
    const updated = (result as { messages: BaseMessage[] }).messages
    const hasImage = (m: HumanMessage) =>
      Array.isArray(m.content) &&
      (m.content as Array<{ type?: string }>).some(
        (b) => b.type === 'image' || b.type === 'image_url'
      )
    expect(hasImage(updated[2] as HumanMessage)).toBe(false)
    expect(hasImage(updated[3] as HumanMessage)).toBe(true)
    expect(hasImage(updated[4] as HumanMessage)).toBe(true)
  })

  it('strips reasoning_content from all but the last AIMessage', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const userMsg = new HumanMessage('go')
    const aiOld = new AIMessage({
      content: 'first thought',
      additional_kwargs: {
        reasoning_content: 'OLD reasoning',
        other_key: 'kept',
        refusal: null,
        parsed: { shape: 'kept too' },
      },
    })
    const aiMid = new AIMessage({
      content: 'second thought',
      additional_kwargs: { reasoning_content: 'MID reasoning', audio: { id: 'au-1' } },
    })
    const aiLast = new AIMessage({
      content: 'latest thought',
      additional_kwargs: { reasoning_content: 'LATEST reasoning' },
    })

    const result = await before(
      { messages: [userMsg, aiOld, aiMid, aiLast] },
      runtime
    )
    const updated = (result as { messages: BaseMessage[] }).messages
    const out0 = updated[2] as AIMessage
    const out1 = updated[3] as AIMessage
    const out2 = updated[4] as AIMessage
    expect(out0.additional_kwargs?.reasoning_content).toBeUndefined()
    // reasoning_content is the ONLY additional_kwargs key that goes.
    expect(out0.additional_kwargs?.other_key).toBe('kept')
    expect(out0.additional_kwargs?.refusal).toBeNull()
    expect(out0.additional_kwargs?.parsed).toEqual({ shape: 'kept too' })
    expect(Object.keys(out0.additional_kwargs ?? {}).sort()).toEqual([
      'other_key',
      'parsed',
      'refusal',
    ])
    expect(out1.additional_kwargs?.reasoning_content).toBeUndefined()
    expect(out1.additional_kwargs?.audio).toEqual({ id: 'au-1' })
    expect(out2.additional_kwargs?.reasoning_content).toBe('LATEST reasoning')
  })

  it('preserves message ids on rewritten messages', async () => {
    // Rewritten messages MUST keep their original id. Otherwise the
    // add_messages reducer (after RemoveMessage(REMOVE_ALL)) assigns fresh
    // UUIDs every turn, breaking client-side dedup-by-id and causing the
    // AG-UI client to render the same tool call twice.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const userMsg = new HumanMessage({ id: 'h-user', content: 'go' })
    const motionAi = new AIMessage({
      id: 'ai-motion',
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 1 }, id: 'tc-1' }],
      additional_kwargs: { reasoning_content: 'stale reasoning' },
    })
    const motionTool = new ToolMessage({
      id: 'tool-motion',
      content: motionResultJson('turn_right (steps=1)'),
      tool_call_id: 'tc-1',
      name: 'turn_right',
    })
    const oldImg = new HumanMessage({
      id: 'h-img-old',
      content: [{ type: 'text', text: 'Old frame' }, imageBlock()],
    })
    const lastAi = new AIMessage({ id: 'ai-last', content: 'done' })

    const result = await before(
      { messages: [userMsg, motionAi, motionTool, oldImg, lastAi] },
      runtime
    )
    const updated = (result as { messages: BaseMessage[] }).messages
    const ids = updated.filter((m) => !(m instanceof RemoveMessage)).map((m) => m.id)
    // Every rewritten message retains its original id; none are undefined.
    expect(ids).toEqual(['h-user', 'ai-motion', 'tool-motion', 'h-img-old', 'ai-last'])
  })

  it('returns undefined when there is nothing to prune and nothing to summarize', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const result = await before(
      { messages: [new HumanMessage('hi'), new AIMessage('hello')] },
      runtime
    )
    expect(result).toBeUndefined()
    expect(llm.invoke).not.toHaveBeenCalled()
  })
})

describe('contextPrunerMiddleware — threshold summarization', () => {
  it('does not summarize when pruned tokens stay under threshold', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({
      llm,
      maxContextTokens: 30_000,
      summarizeAtFraction: 0.7,
    }) as HookContainer
    const before = getHook(mw.beforeModel)

    const userMsg = new HumanMessage('go')
    const motionAi = new AIMessage({
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 1 }, id: 'tc' }],
    })
    const motionTool = new ToolMessage({
      content: motionResultJson('turn_right (steps=1)', 200),
      tool_call_id: 'tc',
      name: 'turn_right',
    })

    await before({ messages: [userMsg, motionAi, motionTool] }, runtime)
    expect(llm.invoke).not.toHaveBeenCalled()
  })

  it('summarizes synchronously when pruned tokens cross threshold', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({
      llm,
      maxContextTokens: 1000,
      summarizeAtFraction: 0.5, // threshold = 500
      keepLatestImages: 1,
      imageTokenBudget: 50,
    }) as HookContainer
    const before = getHook(mw.beforeModel)

    const userMsg = new HumanMessage('Get the robot to the cone.')
    // Pad the head with a bunch of long-text AIMessages so we cross 500 tokens.
    const filler: BaseMessage[] = []
    for (let i = 0; i < 6; i++) {
      filler.push(new AIMessage('A'.repeat(400)))
    }
    const motionAi = new AIMessage({
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 1 }, id: 'tc' }],
    })
    const motionTool = new ToolMessage({
      content: motionResultJson('turn_right (steps=1)'),
      tool_call_id: 'tc',
      name: 'turn_right',
    })
    const injected = new HumanMessage({
      content: [{ type: 'text', text: 'Before/After.' }, imageBlock()],
    })

    const result = await before(
      { messages: [userMsg, ...filler, motionAi, motionTool, injected] },
      runtime
    )

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const updated = (result as { messages: BaseMessage[] }).messages
    expect(updated[0]).toBeInstanceOf(RemoveMessage)
    // First non-Remove entry is the original user message verbatim.
    expect(updated[1]).toBeInstanceOf(HumanMessage)
    expect((updated[1] as HumanMessage).content).toBe('Get the robot to the cone.')
    // Then the summary as a clearly-marked HumanMessage (RC-17: a SystemMessage
    // here sits at index ≥ 1, which @langchain/anthropic rejects outright).
    expect(updated[2]).toBeInstanceOf(HumanMessage)
    expect(updated[2]).not.toBeInstanceOf(SystemMessage)
    expect((updated[2] as HumanMessage).content).toContain('[Context summary]')
    expect((updated[2] as HumanMessage).content).toContain(SUMMARY_TEXT)
    // Tail is the motion turn (AIMessage + ToolMessage + injected composite).
    expect(updated[3]).toBe(motionAi)
    expect(updated[4]).toBeInstanceOf(ToolMessage)
    expect(updated[5]).toBe(injected)
  })

  it('summarizer sees image-stripped input', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({
      llm,
      maxContextTokens: 1000,
      summarizeAtFraction: 0.5,
      imageTokenBudget: 50,
    }) as HookContainer
    const before = getHook(mw.beforeModel)

    const userMsg = new HumanMessage('Find the cone.')
    const filler: BaseMessage[] = []
    for (let i = 0; i < 6; i++) filler.push(new AIMessage('B'.repeat(400)))
    const motionAi = new AIMessage({
      content: '',
      tool_calls: [{ name: 'move_forward', args: { steps: 2 }, id: 'tc' }],
    })
    const motionTool = new ToolMessage({
      content: motionResultJson('move_forward (steps=2)'),
      tool_call_id: 'tc',
      name: 'move_forward',
    })
    const injected = new HumanMessage({
      content: [{ type: 'text', text: 'frame' }, imageBlock()],
    })

    await before(
      { messages: [userMsg, ...filler, motionAi, motionTool, injected] },
      runtime
    )
    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const sanitizedInput = llm.invoke.mock.calls[0][0] as BaseMessage[]
    for (const m of sanitizedInput) {
      if (Array.isArray(m.content)) {
        for (const block of m.content as Array<{ type?: string }>) {
          expect(block.type === 'image' || block.type === 'image_url').toBe(false)
        }
      }
    }
  })

  it('uses a provided summaryPrompt override', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({
      llm,
      summaryPrompt: 'CUSTOM PRUNER PROMPT',
      maxContextTokens: 1000,
      summarizeAtFraction: 0.5,
    }) as HookContainer
    const before = getHook(mw.beforeModel)

    const userMsg = new HumanMessage('go')
    const filler: BaseMessage[] = []
    for (let i = 0; i < 6; i++) filler.push(new AIMessage('Z'.repeat(400)))
    const motionAi = new AIMessage({
      content: '',
      tool_calls: [{ name: 'turn_left', args: {}, id: 'tc' }],
    })
    const motionTool = new ToolMessage({
      content: motionResultJson('turn_left'),
      tool_call_id: 'tc',
      name: 'turn_left',
    })

    await before(
      { messages: [userMsg, ...filler, motionAi, motionTool] },
      runtime
    )
    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const sanitizedInput = llm.invoke.mock.calls[0][0] as BaseMessage[]
    expect(sanitizedInput[0]).toBeInstanceOf(SystemMessage)
    expect((sanitizedInput[0] as SystemMessage).content).toBe('CUSTOM PRUNER PROMPT')
  })
})

describe('contextPrunerMiddleware — estimateTokens', () => {
  it('counts string-content text via the 4-chars/token heuristic', () => {
    const msg = new HumanMessage('A'.repeat(40))
    // 40 chars / 4 = 10 text tokens + 4 envelope = 14
    expect(estimateTokens([msg], 800)).toBe(14)
  })

  it('charges imageTokenBudget per image block', () => {
    const noImg = new HumanMessage('hi')
    const withImg = new HumanMessage({
      content: [{ type: 'text', text: 'hi' }, { type: 'image_url', image_url: 'data:...' }],
    })
    const noImgTokens = estimateTokens([noImg], 800)
    const withImgTokens = estimateTokens([withImg], 800)
    expect(withImgTokens - noImgTokens).toBe(800)
  })

  it('charges for AIMessage tool_calls', () => {
    const plain = new AIMessage('hello')
    const withTool = new AIMessage({
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'x' }],
    })
    expect(estimateTokens([withTool], 800)).toBeGreaterThan(estimateTokens([plain], 800))
  })
})

// ── RC-17: no mid-history SystemMessage, ever ───────────────────────────────
// @langchain/anthropic throws "System messages are only permitted as the first
// passed message." for a SystemMessage at index ≥ 1 — the PLAT-13 crash, the
// exact defect RC-16 fixed in motion-summarization. Every branch of the
// context-pruner's beforeModel that rebuilds a history must leave no
// SystemMessage past index 0; the summary rides as a marked HumanMessage. The
// two-cycle case additionally proves the marked HumanMessage summary is FOLDED
// (replaced), not accumulated, on a later prune.
describe('contextPrunerMiddleware — RC-17 mid-history SystemMessage fix', () => {
  function systemIndices(messages: BaseMessage[]): number[] {
    return messages
      .map((m, i) => (m instanceof SystemMessage ? i : -1))
      .filter((i) => i >= 0)
  }

  function summaryMessages(messages: BaseMessage[]): HumanMessage[] {
    return messages.filter(
      (m): m is HumanMessage =>
        m instanceof HumanMessage && String(m.content).startsWith('[Context summary]')
    )
  }

  // Force the summarize branch regardless of real token counts: threshold = 1.
  const FORCE_SUMMARIZE = { maxContextTokens: 10, summarizeAtFraction: 0.1 } as const

  // The PLAT-13 crash shape: a read_status tool turn BEFORE the first motion,
  // so the rewrite window (firstHumanIdx+1 .. lastMotionAiIdx) is non-empty.
  function crashShapedHistory() {
    const user = new HumanMessage('Drive the robot to the red cone.')
    const statusAi = new AIMessage({
      content: '',
      tool_calls: [{ name: 'read_status', args: {}, id: 'tc-status' }],
    })
    const statusTool = new ToolMessage({
      content: JSON.stringify({ battery: '7.4V', ok: true }),
      tool_call_id: 'tc-status',
      name: 'read_status',
    })
    const thinking = new AIMessage('Status fine. Turning right to scan.')
    const motionAi = new AIMessage({
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-motion' }],
    })
    const motionTool = new ToolMessage({
      content: JSON.stringify({ mimeType: 'image/jpeg', data: 'B', motion: 'turn_right (steps=3)' }),
      tool_call_id: 'tc-motion',
      name: 'turn_right',
    })
    const composite = new HumanMessage({
      content: [{ type: 'text', text: 'Before/After frames for turn_right (steps=3).' }, imageBlock()],
    })
    const atMotion: BaseMessage[] = [user, statusAi, statusTool, thinking, motionAi]
    return { atMotion, nextTurn: [...atMotion, motionTool, composite] as BaseMessage[] }
  }

  it('summary-applied branch (pinned state present): no SystemMessage at index ≥ 1', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...FORCE_SUMMARIZE }) as HookContainer
    const after = getHook(mw.afterModel)
    const before = getHook(mw.beforeModel)

    const { atMotion, nextTurn } = crashShapedHistory()
    // afterModel observes the just-emitted motion, so formatPinnedState() is
    // non-empty on the following beforeModel.
    await after({ messages: atMotion }, runtime)
    const result = await before({ messages: nextTurn }, runtime)

    expect(result).toBeTruthy()
    const updated = (result as { messages: BaseMessage[] }).messages
    expect(updated[0]).toBeInstanceOf(RemoveMessage)
    const rebuilt = updated.slice(1)
    // The invariant Anthropic enforces.
    expect(systemIndices(rebuilt)).toEqual([])
    // The summary lands as a marked HumanMessage carrying both the LLM summary
    // and the deterministic pinned motion log.
    const summaries = summaryMessages(rebuilt)
    expect(summaries).toHaveLength(1)
    expect(String(summaries[0].content)).toContain(SUMMARY_TEXT)
    expect(String(summaries[0].content)).toContain('Recent motions (newest last):')
  })

  it('summary-applied branch (no pinned state): no SystemMessage at index ≥ 1', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...FORCE_SUMMARIZE }) as HookContainer
    const before = getHook(mw.beforeModel)

    // No afterModel call → motion log empty → formatPinnedState() === '' (the
    // pinned-less branch).
    const { nextTurn } = crashShapedHistory()
    const result = await before({ messages: nextTurn }, runtime)

    expect(result).toBeTruthy()
    const rebuilt = (result as { messages: BaseMessage[] }).messages.slice(1)
    expect(systemIndices(rebuilt)).toEqual([])
    const summaries = summaryMessages(rebuilt)
    expect(summaries).toHaveLength(1)
    expect(String(summaries[0].content)).toContain(SUMMARY_TEXT)
    expect(String(summaries[0].content)).not.toContain('Recent motions')
  })

  it('a pre-existing first-position SystemMessage stays at index 0 only', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...FORCE_SUMMARIZE }) as HookContainer
    const before = getHook(mw.beforeModel)

    const { nextTurn } = crashShapedHistory()
    const withSystem = [new SystemMessage('agent system prompt'), ...nextTurn]
    const result = await before({ messages: withSystem }, runtime)

    expect(result).toBeTruthy()
    const rebuilt = (result as { messages: BaseMessage[] }).messages.slice(1)
    // The leading system message survives in place; no OTHER system message
    // appears anywhere past index 0.
    expect(systemIndices(rebuilt)).toEqual([0])
    expect((rebuilt[0] as SystemMessage).content).toBe('agent system prompt')
    expect(summaryMessages(rebuilt)).toHaveLength(1)
  })

  it('two prune cycles: the summary is folded, not accumulated', async () => {
    // Distinct summary text per call so "no accumulation" is a real content
    // discrimination, not just a count check.
    let n = 0
    const invoke = vi.fn(async () => ({ content: `summary ${++n}: robot scanned then moved.` }))
    const llm = { invoke } as unknown as Parameters<typeof createContextPrunerMiddleware>[0]['llm'] & {
      invoke: ReturnType<typeof vi.fn>
    }
    const mw = createContextPrunerMiddleware({ llm, ...FORCE_SUMMARIZE }) as HookContainer
    const before = getHook(mw.beforeModel)

    // Cycle 1.
    const { nextTurn } = crashShapedHistory()
    const r1 = await before({ messages: nextTurn }, runtime)
    const rebuilt1 = (r1 as { messages: BaseMessage[] }).messages.filter(
      (m) => !(m instanceof RemoveMessage)
    )
    const c1 = summaryMessages(rebuilt1)
    expect(c1).toHaveLength(1)
    expect(String(c1[0].content)).toContain('summary 1')

    // Cycle 2: the model emits a second motion off the rebuilt state; the prior
    // summary (now at firstHumanIdx+1) falls inside the next head slice.
    const motionAi2 = new AIMessage({
      content: '',
      tool_calls: [{ name: 'move_forward', args: { steps: 2 }, id: 'tc-motion-2' }],
    })
    const motionTool2 = new ToolMessage({
      content: JSON.stringify({ mimeType: 'image/jpeg', data: 'C', motion: 'move_forward (steps=2)' }),
      tool_call_id: 'tc-motion-2',
      name: 'move_forward',
    })
    const composite2 = new HumanMessage({
      content: [{ type: 'text', text: 'Before/After frames for move_forward (steps=2).' }, imageBlock()],
    })
    const cycle2Input = [...rebuilt1, motionAi2, motionTool2, composite2]
    const r2 = await before({ messages: cycle2Input }, runtime)
    const rebuilt2 = (r2 as { messages: BaseMessage[] }).messages.filter(
      (m) => !(m instanceof RemoveMessage)
    )

    // No accumulation: exactly ONE summary, carrying cycle-2's text and NOT
    // cycle-1's; and still no SystemMessage at index ≥ 1.
    const c2 = summaryMessages(rebuilt2)
    expect(c2).toHaveLength(1)
    expect(String(c2[0].content)).toContain('summary 2')
    expect(String(c2[0].content)).not.toContain('summary 1')
    expect(systemIndices(rebuilt2)).toEqual([])
    expect(invoke).toHaveBeenCalledTimes(2)
  })

  it('guard branch preserved: motion directly after the first human → no rewrite', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...FORCE_SUMMARIZE }) as HookContainer
    const before = getHook(mw.beforeModel)

    // lastMotionAiIdx === firstHumanIdx + 1 → the summarize window is empty and
    // nothing else needs pruning (plain ToolMessage, no image data) → undefined.
    const messages: BaseMessage[] = [
      new HumanMessage('go'),
      new AIMessage({ content: '', tool_calls: [{ name: 'turn_right', args: { steps: 1 }, id: 'm' }] }),
      new ToolMessage({ content: '{}', tool_call_id: 'm', name: 'turn_right' }),
    ]
    const result = await before({ messages }, runtime)
    expect(result).toBeUndefined()
    expect(llm.invoke).not.toHaveBeenCalled()
  })

  it('empty-summary branch: summary not applied, no SystemMessage introduced', async () => {
    const llm = makeStubLlm('')
    const mw = createContextPrunerMiddleware({ llm, ...FORCE_SUMMARIZE }) as HookContainer
    const before = getHook(mw.beforeModel)

    // The summarizer returns '' → the summary is NOT applied. Mechanical prune
    // still runs (the crash history's ToolMessage carries image data), so a
    // rewrite may be emitted, but it must carry no summary and no mid-history
    // SystemMessage.
    const { nextTurn } = crashShapedHistory()
    const result = await before({ messages: nextTurn }, runtime)

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const rebuilt =
      result === undefined
        ? []
        : (result as { messages: BaseMessage[] }).messages.filter(
            (m) => !(m instanceof RemoveMessage)
          )
    expect(systemIndices(rebuilt)).toEqual([])
    expect(summaryMessages(rebuilt)).toHaveLength(0)
  })
})

// ───────────────────────────────────────────────────────────────────────────
// RC-28 — stripReasoningContent copies the message instead of re-describing it
//
// The defect these pin: the strip used to rebuild each AIMessage from an object
// literal naming five fields, so every field not on that list was silently
// dropped. The first test is the CLASS guard — it asserts on a field the
// production code does not name anywhere, so it stays red under any
// field-enumerating rebuild, however long the enumeration.
// ───────────────────────────────────────────────────────────────────────────
describe('contextPrunerMiddleware — RC-28 reasoning strip preserves the whole message', () => {
  // Read/write a property the production code has never heard of. Cast because
  // no message type declares it — that is exactly the point.
  function stampUnknownField(msg: BaseMessage, value: unknown): void {
    ;(msg as unknown as Record<string, unknown>).field_no_one_enumerated = value
  }
  function readUnknownField(msg: BaseMessage): unknown {
    return (msg as unknown as Record<string, unknown>).field_no_one_enumerated
  }

  it('CLASS GUARD: a field the strip does not name survives the round trip', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const aiOld = new AIMessage({
      id: 'ai-old',
      content: 'thinking',
      additional_kwargs: { reasoning_content: 'OLD reasoning' },
    })
    stampUnknownField(aiOld, { anything: 'at all' })
    const aiLast = new AIMessage({ id: 'ai-last', content: 'done' })

    const result = await before({ messages: [new HumanMessage('go'), aiOld, aiLast] }, runtime)
    const updated = (result as { messages: BaseMessage[] }).messages
    const out = updated[2] as AIMessage

    // The strip fired…
    expect(out.additional_kwargs?.reasoning_content).toBeUndefined()
    // …and it carried across a field nothing in the implementation mentions.
    expect(readUnknownField(out)).toEqual({ anything: 'at all' })
    // The caller still holds the input array; the source must be untouched.
    expect(aiOld.additional_kwargs?.reasoning_content).toBe('OLD reasoning')
    expect(out).not.toBe(aiOld)
  })

  it('preserves response_metadata — the OpenAI Responses API carries the reasoning-item ids a following tool call must be paired with there', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const aiOld = new AIMessage({
      id: 'ai-old',
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 1 }, id: 'tc-1' }],
      additional_kwargs: { reasoning_content: 'OLD reasoning' },
      response_metadata: {
        model_name: 'gpt-5.2',
        output: [{ type: 'reasoning', id: 'rs_abc123' }],
      },
    })
    const aiLast = new AIMessage({ id: 'ai-last', content: 'done' })

    const result = await before({ messages: [new HumanMessage('go'), aiOld, aiLast] }, runtime)
    const updated = (result as { messages: BaseMessage[] }).messages
    const out = updated[2] as AIMessage

    expect(out.additional_kwargs?.reasoning_content).toBeUndefined()
    expect(out.response_metadata).toEqual({
      model_name: 'gpt-5.2',
      output: [{ type: 'reasoning', id: 'rs_abc123' }],
    })
    // The tool call the reasoning item is paired with is still there too.
    expect(out.tool_calls?.map((tc) => tc.id)).toEqual(['tc-1'])
  })

  it('preserves invalid_tool_calls, tool_call_chunks and the message class on an AIMessageChunk', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    // invalid_tool_calls rides on a settled AIMessage. (It cannot be set
    // alongside tool_call_chunks: the chunk constructor derives it from them.)
    const aiInvalid = new AIMessage({
      id: 'ai-invalid',
      content: '',
      invalid_tool_calls: [
        { name: 'walk_forward', args: '{"steps":', id: 'tc-2', error: 'unterminated JSON' },
      ],
      additional_kwargs: { reasoning_content: 'OLD reasoning' },
    })
    // A streamed turn arrives as an AIMessageChunk. isAIMessage() admits it, so
    // it reaches the strip; only AIMessageChunk carries tool_call_chunks.
    const chunk = new AIMessageChunk({
      id: 'ai-chunk',
      content: 'partial',
      tool_call_chunks: [
        { name: 'turn_left', args: '{"steps":1}', id: 'tc-1', index: 0, type: 'tool_call_chunk' },
      ],
      additional_kwargs: { reasoning_content: 'OLD reasoning' },
    })
    const aiLast = new AIMessage({ id: 'ai-last', content: 'done' })

    const result = await before(
      { messages: [new HumanMessage('go'), aiInvalid, chunk, aiLast] },
      runtime
    )
    const updated = (result as { messages: BaseMessage[] }).messages
    const outInvalid = updated[2] as AIMessage
    const outChunk = updated[3] as AIMessageChunk

    expect(outInvalid.additional_kwargs?.reasoning_content).toBeUndefined()
    expect(outInvalid.invalid_tool_calls).toEqual([
      { name: 'walk_forward', args: '{"steps":', id: 'tc-2', error: 'unterminated JSON' },
    ])

    expect(outChunk.additional_kwargs?.reasoning_content).toBeUndefined()
    expect(outChunk.tool_call_chunks).toEqual([
      { name: 'turn_left', args: '{"steps":1}', id: 'tc-1', index: 0, type: 'tool_call_chunk' },
    ])
    // Same class as it arrived as — a rebuild would flatten a chunk into a plain
    // AIMessage. Compared by prototype, never instanceof: this repo resolves two
    // copies of @langchain/core, so instanceof against an imported message class
    // is unreliable here.
    expect(Object.getPrototypeOf(outChunk)).toBe(Object.getPrototypeOf(chunk))
  })

  it('returns the same object reference when there is no reasoning_content, so reasoningStripped counts only changed messages', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
    try {
      const llm = makeStubLlm()
      const mw = createContextPrunerMiddleware({ llm }) as HookContainer
      const before = getHook(mw.beforeModel)

      const aiClean = new AIMessage({ id: 'ai-clean', content: 'no reasoning here' })
      const aiDirty = new AIMessage({
        id: 'ai-dirty',
        content: 'has reasoning',
        additional_kwargs: { reasoning_content: 'OLD reasoning' },
      })
      const aiLast = new AIMessage({ id: 'ai-last', content: 'done' })

      const result = await before(
        { messages: [new HumanMessage('go'), aiClean, aiDirty, aiLast] },
        runtime
      )
      const updated = (result as { messages: BaseMessage[] }).messages

      // Untouched message comes back as the very same object…
      expect(updated[2]).toBe(aiClean)
      // …and the changed one does not.
      expect(updated[3]).not.toBe(aiDirty)

      // The per-call summary line reports exactly one strip. The stat is derived
      // from reference identity, so if the strip ever returned a fresh object
      // for an unchanged message this would read 2.
      const line = logSpy.mock.calls
        .map((c) => String(c[0]))
        .find((s) => s.includes('→ LLM'))
      expect(line).toBeDefined()
      expect(line).toContain('reasoning:1')
    } finally {
      logSpy.mockRestore()
    }
  })

  it('the preserved fields survive the add_messages reducer the rewritten array is fed through', async () => {
    // beforeModel returns RemoveMessage(REMOVE_ALL) + the rebuilt array, and the
    // graph folds that into state through messagesStateReducer. Anything the
    // reducer drops never reaches the model, so the preservation is only real if
    // it holds on the far side of it.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const aiOld = new AIMessage({
      id: 'ai-old',
      content: 'thinking',
      additional_kwargs: { reasoning_content: 'OLD reasoning' },
      response_metadata: { output: [{ type: 'reasoning', id: 'rs_abc123' }] },
    })
    stampUnknownField(aiOld, 'survives')
    const aiLast = new AIMessage({ id: 'ai-last', content: 'done' })

    const result = await before({ messages: [new HumanMessage('go'), aiOld, aiLast] }, runtime)
    const emitted = (result as { messages: BaseMessage[] }).messages
    const reduced = messagesStateReducer([new HumanMessage({ id: 'prior', content: 'old' })], emitted)

    const out = reduced.find((m) => m.id === 'ai-old') as AIMessage
    expect(out).toBeDefined()
    expect(out.additional_kwargs?.reasoning_content).toBeUndefined()
    expect(out.response_metadata).toEqual({ output: [{ type: 'reasoning', id: 'rs_abc123' }] })
    expect(readUnknownField(out)).toBe('survives')
  })

  it('the strip holds through the checkpoint serde — the stripped reasoning is absent from the serialized bytes and the un-named field is present', async () => {
    // Every assertion above reads a property off the returned object. The bytes
    // langgraph checkpoints are produced by a different route: the message's
    // toJSON takes its KEY SET from lc_kwargs and its VALUES from the live
    // instance fields. The strip copies the source's own property descriptors,
    // so the clone shares lc_kwargs with the source BY REFERENCE and that shared
    // object still carries the reasoning. Today the instance field wins and the
    // payload is clean — but that is a property of one library version, not of
    // this module, and nothing else in this suite would notice it changing.
    // Assert on the bytes, so a core that resolved lc_kwargs first (or anyone
    // patching lc_kwargs) is caught here rather than in a checkpoint in
    // production, where the reasoning would be persisted while every
    // property-reading test above still passed.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const aiOld = new AIMessage({
      id: 'ai-old',
      content: 'thinking',
      additional_kwargs: { reasoning_content: 'OLD reasoning' },
      response_metadata: { output: [{ type: 'reasoning', id: 'rs_abc123' }] },
    })
    // A streamed turn reaches the strip as a chunk; it serializes by the same
    // route, and it is the class a literal rebuild would have flattened.
    const chunkOld = new AIMessageChunk({
      id: 'ai-chunk',
      content: 'partial',
      additional_kwargs: { reasoning_content: 'CHUNK reasoning' },
      response_metadata: { output: [{ type: 'reasoning', id: 'rs_chunk_789' }] },
    })
    // Last AI message keeps its reasoning by design, so it carries none here —
    // otherwise it would put the word back in the payload on its own account.
    const aiLast = new AIMessage({ id: 'ai-last', content: 'done' })

    const result = await before(
      { messages: [new HumanMessage('go'), aiOld, chunkOld, aiLast] },
      runtime
    )
    const updated = (result as { messages: BaseMessage[] }).messages

    // `serde` is a public, typed member of BaseCheckpointSaver
    // (`dumpsTyped(data: any): Promise<[string, Uint8Array]>`), so this is the
    // real checkpoint encoder, reached without a cast.
    const saver = new MemorySaver()
    const [encoding, bytes] = await saver.serde.dumpsTyped({ messages: updated })
    expect(encoding).toBe('json')
    const payload = new TextDecoder().decode(bytes)

    // The leak guard.
    expect(payload).not.toContain('OLD reasoning')
    expect(payload).not.toContain('CHUNK reasoning')
    // And the payload still carries fields the strip never names — proof the
    // messages really are in these bytes, and that nothing was lost getting
    // them there.
    expect(payload).toContain('rs_abc123')
    expect(payload).toContain('rs_chunk_789')
  })
})

// ───────────────────────────────────────────────────────────────────────────
// RC-29 — the two image strips copy the message instead of re-describing it
//
// Same defect class as RC-28, two functions earlier in the same file.
// stripToolMessageImageData rebuilt every motion/capture ToolMessage from an
// object literal naming four fields, so `status`, `artifact`,
// `response_metadata` and `additional_kwargs` were dropped. `status` is the
// serious one and the loss was UNCONDITIONAL: step 1 of the prune runs on every
// motion/capture result, not only old ones, so a result carrying
// `status: 'error'` came back undefined and a failed motion was
// indistinguishable from a completed one. pruneImageBlocksInHumanMessage had
// the milder version of the same shape, dropping `additional_kwargs` and
// `response_metadata`.
//
// There is one CLASS GUARD per function: it asserts on a property the
// production code does not name anywhere, so it stays red under any
// field-enumerating rebuild, however long the enumeration. No shared helper was
// extracted for the three strips and none is assumed here — the guard is
// written once per function on purpose.
//
// Every preservation assertion sits beside an assertion that the strip actually
// fired, because a function that simply returned its argument would satisfy the
// preservation half on its own.
// ───────────────────────────────────────────────────────────────────────────

// Read/write a property the production code has never heard of. Cast because no
// message type declares it — that is exactly the point.
function stampUnnamedField(msg: BaseMessage, value: unknown): void {
  ;(msg as unknown as Record<string, unknown>).field_no_one_enumerated = value
}
function readUnnamedField(msg: BaseMessage): unknown {
  return (msg as unknown as Record<string, unknown>).field_no_one_enumerated
}

describe('contextPrunerMiddleware — RC-29 ToolMessage image-data strip preserves the whole message', () => {
  const FRAME_BYTES = 'BASE64FRAMEMARKER'

  function motionResultWithMarker(motion = 'turn_right (steps=1)'): string {
    return JSON.stringify({
      mimeType: 'image/jpeg',
      data: `${FRAME_BYTES}${'X'.repeat(64)}`,
      motion,
    })
  }

  it('CLASS GUARD: a field the strip does not name survives the round trip', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const motionTool = new ToolMessage({
      id: 'tm-1',
      content: motionResultWithMarker(),
      tool_call_id: 'tc-1',
      name: 'turn_right',
    })
    stampUnnamedField(motionTool, { anything: 'at all' })

    const result = await before({ messages: [new HumanMessage('go'), motionTool] }, runtime)
    const updated = (result as { messages: BaseMessage[] }).messages
    const out = updated[2] as ToolMessage

    // The strip fired…
    const parsed = JSON.parse(out.content as string)
    expect(parsed.data).toBeUndefined()
    expect(parsed.dataDropped).toBe(true)
    // …and it carried across a field nothing in the implementation mentions.
    expect(readUnnamedField(out)).toEqual({ anything: 'at all' })
    // Same class it arrived as. Compared by prototype, never instanceof: this
    // repo resolves two copies of @langchain/core, so an instance check against
    // an imported message class is unreliable here.
    expect(Object.getPrototypeOf(out)).toBe(Object.getPrototypeOf(motionTool))
    // The caller still holds the input array; the source must be untouched.
    expect(motionTool.content as string).toContain(FRAME_BYTES)
    expect(out).not.toBe(motionTool)
  })

  it('preserves status — a motion that FAILED must not come back indistinguishable from one that completed', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const motionTool = new ToolMessage({
      id: 'tm-1',
      content: motionResultWithMarker(),
      tool_call_id: 'tc-1',
      name: 'turn_right',
      status: 'error',
      artifact: { frameId: 'FRAMEARTIFACTMARKER', width: 640 },
      response_metadata: { bridge: 'robot-bridge/2', attempt: 2 },
      additional_kwargs: { servo_fault: 'left_hip stalled' },
    })

    const result = await before({ messages: [new HumanMessage('go'), motionTool] }, runtime)
    const updated = (result as { messages: BaseMessage[] }).messages
    const out = updated[2] as ToolMessage

    // The strip fired…
    const parsed = JSON.parse(out.content as string)
    expect(parsed.data).toBeUndefined()
    expect(parsed.dataDropped).toBe(true)
    expect(parsed.motion).toBe('turn_right (steps=1)')

    // …and every field the old literal rebuild forgot is still here. `status`
    // first: this is the one the node is named for.
    expect(out.status).toBe('error')
    expect(out.artifact).toEqual({ frameId: 'FRAMEARTIFACTMARKER', width: 640 })
    expect(out.response_metadata).toEqual({ bridge: 'robot-bridge/2', attempt: 2 })
    expect(out.additional_kwargs).toEqual({ servo_fault: 'left_hip stalled' })
    // The fields the literal did remember are still correct too.
    expect(out.id).toBe('tm-1')
    expect(out.tool_call_id).toBe('tc-1')
    expect(out.name).toBe('turn_right')
  })

  it('returns the same object reference when there is nothing to strip, so toolImageDataStripped counts only changed messages', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
    try {
      const llm = makeStubLlm()
      const mw = createContextPrunerMiddleware({ llm }) as HookContainer
      const before = getHook(mw.beforeModel)

      const withData = new ToolMessage({
        id: 'tm-1',
        content: motionResultWithMarker(),
        tool_call_id: 'tc-1',
        name: 'turn_right',
      })
      // A motion result that carries no frame.
      const noData = new ToolMessage({
        id: 'tm-2',
        content: JSON.stringify({ motion: 'turn_left', ok: true }),
        tool_call_id: 'tc-2',
        name: 'turn_left',
      })
      // Not an image-bearing tool, so its `data` is none of this function's
      // business.
      const nonImageTool = new ToolMessage({
        id: 'tm-3',
        content: JSON.stringify({ data: 'ZZZZ' }),
        tool_call_id: 'tc-3',
        name: 'finish_task',
      })
      // Content that is not JSON at all.
      const notJson = new ToolMessage({
        id: 'tm-4',
        content: 'plain text result',
        tool_call_id: 'tc-4',
        name: 'capture_image',
      })

      const result = await before(
        { messages: [new HumanMessage('go'), withData, noData, nonImageTool, notJson] },
        runtime
      )
      const updated = (result as { messages: BaseMessage[] }).messages

      // The changed one is a new object…
      expect(updated[2]).not.toBe(withData)
      // …and every untouched one comes back as the very same object.
      expect(updated[3]).toBe(noData)
      expect(updated[4]).toBe(nonImageTool)
      expect(updated[5]).toBe(notJson)

      // The per-call summary line reports exactly one strip. The stat is derived
      // from reference identity, so if the strip ever returned a fresh object
      // for an unchanged message this would read 4.
      const line = logSpy.mock.calls.map((c) => String(c[0])).find((s) => s.includes('→ LLM'))
      expect(line).toBeDefined()
      expect(line).toContain('tool-data:1')
    } finally {
      logSpy.mockRestore()
    }
  })

  it('the dropped frame is unreachable by every route — instance content, lc_kwargs and the checkpoint bytes — while status rides into them', async () => {
    // Copying a message from its own property descriptors shares `lc_kwargs`
    // with the source BY REFERENCE, and that bag still holds the original
    // content string. Here — unlike in the reasoning strip — the field being
    // rewritten IS the payload this function exists to free, so a bare
    // descriptor copy would turn a byte-dropping function into a byte-retaining
    // one: the frame would stay reachable for the life of the thread (one per
    // motion), and any serializer resolving values from `lc_kwargs` rather than
    // the live instance field would write the bytes straight back into the
    // checkpoint. The strip therefore replaces `lc_kwargs` too, and this test is
    // what pins that: it checks all three routes rather than only the property.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const motionTool = new ToolMessage({
      id: 'tm-1',
      content: motionResultWithMarker(),
      tool_call_id: 'tc-1',
      name: 'turn_right',
      status: 'error',
      artifact: { frameId: 'FRAMEARTIFACTMARKER' },
    })

    const result = await before({ messages: [new HumanMessage('go'), motionTool] }, runtime)
    const updated = (result as { messages: BaseMessage[] }).messages
    const out = updated[2] as ToolMessage

    expect(out.content as string).not.toContain(FRAME_BYTES)
    expect(JSON.stringify(out.lc_kwargs)).not.toContain(FRAME_BYTES)

    // `serde` is a public, typed member of BaseCheckpointSaver, so this is the
    // real checkpoint encoder, reached without a cast.
    const saver = new MemorySaver()
    const [encoding, bytes] = await saver.serde.dumpsTyped({ messages: updated })
    expect(encoding).toBe('json')
    const payload = new TextDecoder().decode(bytes)

    expect(payload).not.toContain(FRAME_BYTES)
    // And the payload carries the fields the old rebuild dropped — proof the
    // message really is in these bytes, so the assertion above cannot pass on an
    // empty payload.
    expect(payload).toContain('"status":"error"')
    expect(payload).toContain('FRAMEARTIFACTMARKER')
  })
})

describe('contextPrunerMiddleware — RC-29 human image-block prune preserves the whole message', () => {
  const IMAGE_BYTES = 'HUMANIMAGEMARKER'

  function markedImageBlock() {
    return { type: 'image_url' as const, image_url: `data:image/jpeg;base64,${IMAGE_BYTES}` }
  }
  function hasImage(m: BaseMessage): boolean {
    return (
      Array.isArray(m.content) &&
      (m.content as Array<{ type?: string }>).some(
        (b) => b.type === 'image' || b.type === 'image_url'
      )
    )
  }

  it('CLASS GUARD: a field the prune does not name survives the round trip', async () => {
    const llm = makeStubLlm()
    // Default keepLatestImages = 1, so the older of the two is pruned.
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const older = new HumanMessage({
      id: 'h-old',
      content: [{ type: 'text', text: 'Frame 1' }, markedImageBlock()],
    })
    stampUnnamedField(older, { anything: 'at all' })
    const newer = new HumanMessage({
      id: 'h-new',
      content: [{ type: 'text', text: 'Frame 2' }, markedImageBlock()],
    })

    const result = await before({ messages: [new HumanMessage('go'), older, newer] }, runtime)
    const updated = (result as { messages: BaseMessage[] }).messages
    const out = updated[2] as HumanMessage

    // The prune fired…
    expect(hasImage(out)).toBe(false)
    expect((out.content as Array<{ text?: string }>)[0].text).toBe('Frame 1')
    // …and it carried across a field nothing in the implementation mentions.
    expect(readUnnamedField(out)).toEqual({ anything: 'at all' })
    // Same class it arrived as — by prototype, never instanceof (dual
    // @langchain/core in this repo).
    expect(Object.getPrototypeOf(out)).toBe(Object.getPrototypeOf(older))
    // Source untouched, and the message that was NOT pruned comes back as the
    // very same object — mechanicalPrune counts strips by reference identity.
    expect(hasImage(older)).toBe(true)
    expect(out).not.toBe(older)
    expect(updated[3]).toBe(newer)
  })

  it('preserves additional_kwargs and response_metadata on an aged-out image turn', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    const older = new HumanMessage({
      id: 'h-old',
      name: 'operator',
      content: [{ type: 'text', text: 'Frame 1' }, markedImageBlock()],
      additional_kwargs: { capture_ts: 1717, source: 'front_cam' },
      response_metadata: { bridge: 'robot-bridge/2' },
    })
    const newer = new HumanMessage({
      id: 'h-new',
      content: [{ type: 'text', text: 'Frame 2' }, markedImageBlock()],
    })

    const result = await before({ messages: [new HumanMessage('go'), older, newer] }, runtime)
    const updated = (result as { messages: BaseMessage[] }).messages
    const out = updated[2] as HumanMessage

    // The prune fired…
    expect(hasImage(out)).toBe(false)
    // …and the two fields the old literal rebuild dropped are still here.
    expect(out.additional_kwargs).toEqual({ capture_ts: 1717, source: 'front_cam' })
    expect(out.response_metadata).toEqual({ bridge: 'robot-bridge/2' })
    // The fields the literal did remember are still correct too.
    expect(out.id).toBe('h-old')
    expect(out.name).toBe('operator')
  })

  it('keeps the caption-less fallback and still preserves the rest of the message', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)

    // Nothing survives the filter, so the "[image dropped]" placeholder stands
    // in for the slot. That branch builds its own content array, which is
    // exactly where a rebuild is most tempting.
    const older = new HumanMessage({
      id: 'h-old',
      content: [markedImageBlock()],
      additional_kwargs: { source: 'front_cam' },
    })
    stampUnnamedField(older, 'survives the fallback branch too')
    const newer = new HumanMessage({
      id: 'h-new',
      content: [{ type: 'text', text: 'Frame 2' }, markedImageBlock()],
    })

    const result = await before({ messages: [new HumanMessage('go'), older, newer] }, runtime)
    const updated = (result as { messages: BaseMessage[] }).messages
    const out = updated[2] as HumanMessage

    expect(hasImage(out)).toBe(false)
    expect(out.content).toEqual([{ type: 'text', text: '[image dropped]' }])
    expect(readUnnamedField(out)).toBe('survives the fallback branch too')
    expect(out.additional_kwargs).toEqual({ source: 'front_cam' })
  })

  it('the dropped image bytes are unreachable by every route, and humanImagesStripped counts only changed messages', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
    try {
      const llm = makeStubLlm()
      const mw = createContextPrunerMiddleware({ llm }) as HookContainer
      const before = getHook(mw.beforeModel)

      const plain = new HumanMessage({ id: 'h-plain', content: 'no images here' })
      const older = new HumanMessage({
        id: 'h-old',
        content: [{ type: 'text', text: 'Frame 1' }, markedImageBlock()],
        response_metadata: { bridge: 'HUMANMETAMARKER' },
      })
      // The newest image message keeps its blocks by design, so it would put
      // IMAGE_BYTES back into the payload on its own account — it carries an
      // unmarked block instead.
      const newer = new HumanMessage({
        id: 'h-new',
        content: [
          { type: 'text', text: 'Frame 2' },
          { type: 'image_url', image_url: 'data:image/jpeg;base64,KEPTFRAME' },
        ],
      })

      const result = await before({ messages: [plain, older, newer] }, runtime)
      const updated = (result as { messages: BaseMessage[] }).messages
      const out = updated[2] as HumanMessage

      expect(JSON.stringify(out.content)).not.toContain(IMAGE_BYTES)
      expect(JSON.stringify(out.lc_kwargs)).not.toContain(IMAGE_BYTES)

      const saver = new MemorySaver()
      const [encoding, bytes] = await saver.serde.dumpsTyped({ messages: updated })
      expect(encoding).toBe('json')
      const payload = new TextDecoder().decode(bytes)

      expect(payload).not.toContain(IMAGE_BYTES)
      // Proof the messages really are in these bytes: the field the old rebuild
      // dropped is present, and so is the frame that was deliberately kept.
      expect(payload).toContain('HUMANMETAMARKER')
      expect(payload).toContain('KEPTFRAME')

      // Exactly one human message changed.
      const line = logSpy.mock.calls.map((c) => String(c[0])).find((s) => s.includes('→ LLM'))
      expect(line).toBeDefined()
      expect(line).toContain('human-images:1')
    } finally {
      logSpy.mockRestore()
    }
  })

  it('an image-free content array comes back by reference on the summary path, where dropAllImageBlocks reaches this same function', async () => {
    // Both strips are also reached from dropAllImageBlocks, which sanitizes the
    // head slice fed to the summarizer. In mechanicalPrune the "nothing changed"
    // early return is protected by selection — only messages that HAVE an image
    // block are ever passed in — so this path is the only place the contract is
    // observable, and without it a prune that always cloned would go unnoticed.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({
      llm,
      maxContextTokens: 1000,
      summarizeAtFraction: 0.5,
      imageTokenBudget: 50,
    }) as HookContainer
    const before = getHook(mw.beforeModel)

    const userMsg = new HumanMessage('Find the cone.')
    const textArray = new HumanMessage({
      id: 'h-text',
      content: [{ type: 'text', text: 'C'.repeat(400) }],
    })
    const filler: BaseMessage[] = []
    for (let i = 0; i < 6; i++) filler.push(new AIMessage('B'.repeat(400)))
    const motionAi = new AIMessage({
      content: '',
      tool_calls: [{ name: 'move_forward', args: { steps: 2 }, id: 'tc' }],
    })
    const motionTool = new ToolMessage({
      content: motionResultJson('move_forward (steps=2)'),
      tool_call_id: 'tc',
      name: 'move_forward',
    })

    await before({ messages: [userMsg, textArray, ...filler, motionAi, motionTool] }, runtime)
    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const sanitized = llm.invoke.mock.calls[0][0] as BaseMessage[]
    expect(sanitized.find((m) => m.id === 'h-text')).toBe(textArray)
  })
})

// ───────────────────────────────────────────────────────────────────────────
// RC-27 — a session with no motion call must still prune
//
// The summarize path was guarded on `lastMotionAiIdx > firstHumanIdx + 1`. In a
// session that never issues a motion call — capture and narrate, question
// answering, any read-only interaction — `lastMotionAiIdx` never leaves its -1
// sentinel, so the guard could never hold: the middleware summarized nothing,
// ever, and context grew to the hard cap with nothing saying why.
//
// The trap, and the reason these tests are shaped the way they are: the broken
// behaviour satisfies any test that only checks pruning is CORRECT when it
// happens. Every summarization test above supplies a motion call, so all of them
// passed while the no-motion case was dead. The test that matters is therefore
// that a motion-free history over the threshold DOES summarize.
//
// The motion anchor is pinned in the same block, because the obvious wrong way
// to make the first test pass is to drop the anchor and summarize up to the end
// unconditionally — which would let the summarizer eat the state the most recent
// motion turn depends on. Nothing else in this file would report that.
//
// A motion-free session has state to protect too, and the first build of this
// fix did not: it summarized to the end of the history, so the camera frame
// `keepLatestImages` had just carried through the mechanical strip was thrown
// away by the summarizer one step later, and the narrating call was asked to
// describe a picture it no longer held. The tests below therefore pin BOTH
// halves — that a motion-free session summarizes at all, and that the frame the
// strip kept survives it — because a boundary that satisfies one half while
// breaking the other passes as easily as the original defect did.
//
// Message classes are compared by reference or by `getType()`, never
// `instanceof` and never by prototype identity: this repo resolves two copies
// of @langchain/core (see the RC-29 header), so both of those silently
// misjudge a message minted under the other copy.
// ───────────────────────────────────────────────────────────────────────────
describe('contextPrunerMiddleware — RC-27 a session with no motion call still prunes', () => {
  // Threshold = 500 estimated tokens; image blocks charged cheaply so the
  // histories below cross on their text, which is what the fixtures control.
  const OVER_THRESHOLD = {
    maxContextTokens: 1000,
    summarizeAtFraction: 0.5,
    imageTokenBudget: 50,
  } as const

  function captureResultJson(dataLen = 400): string {
    return JSON.stringify({ mimeType: 'image/jpeg', data: 'X'.repeat(dataLen), captured: true })
  }

  function summaryMessages(messages: BaseMessage[]): HumanMessage[] {
    return messages.filter(
      (m): m is HumanMessage =>
        isHumanMessage(m) && String(m.content).startsWith('[Context summary]')
    )
  }

  function hasImage(m: BaseMessage): boolean {
    return (
      Array.isArray(m.content) &&
      (m.content as { type?: string }[]).some(
        (b) => b && (b.type === 'image' || b.type === 'image_url')
      )
    )
  }

  // `getType()` is the serialised-field form. A prototype comparison against
  // RemoveMessage is `instanceof` by another name and carries the same
  // dual-copy failure mode the repo constraint exists for: this repo resolves
  // two copies of @langchain/core, and the check would silently evaluate false
  // for a message minted under the other one.
  function withoutRemoveMessages(result: unknown): BaseMessage[] {
    return (result as { messages: BaseMessage[] }).messages.filter(
      (m) => m.getType() !== 'remove'
    )
  }

  // A capture-and-narrate session: the model looks, describes what it sees, and
  // looks again. Not one motion call anywhere — this is a normal way to use the
  // robot, not a degenerate history.
  function narrationHistory(cycles = 5): { user: HumanMessage; messages: BaseMessage[] } {
    const user = new HumanMessage({
      id: 'h-user',
      content: 'Look around and tell me what you can see.',
    })
    const messages: BaseMessage[] = [user]
    for (let i = 0; i < cycles; i++) {
      messages.push(
        new AIMessage({
          id: `ai-capture-${i}`,
          content: '',
          tool_calls: [{ name: 'capture_image', args: {}, id: `tc-cap-${i}` }],
        })
      )
      messages.push(
        new ToolMessage({
          id: `tm-cap-${i}`,
          content: captureResultJson(),
          tool_call_id: `tc-cap-${i}`,
          name: 'capture_image',
        })
      )
      messages.push(
        new HumanMessage({
          id: `h-frame-${i}`,
          content: [{ type: 'text', text: `Frame ${i}.` }, imageBlock()],
        })
      )
      messages.push(new AIMessage({ id: `ai-narrate-${i}`, content: 'N'.repeat(400) }))
    }
    return { user, messages }
  }

  it('THE DEFECT: a motion-free history over the threshold IS summarized', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    const { user, messages } = narrationHistory()
    // No message in this history carries a motion tool call — the condition the
    // old guard could not survive.
    expect(messages.some((m) => isMotionToolCall(m))).toBe(false)

    const result = await before({ messages }, runtime)

    // It summarized at all. This is the assertion the old code failed.
    expect(llm.invoke).toHaveBeenCalledTimes(1)
    expect(result).toBeTruthy()

    const rebuilt = withoutRemoveMessages(result)
    // The first human message survives verbatim, by reference, and the history
    // between it and the anchor is replaced by the single summary. The anchor
    // is the newest image-bearing turn — the one `keepLatestImages: 1` kept —
    // so that turn and everything after it ride in the tail.
    expect(rebuilt).toHaveLength(4)
    expect(rebuilt[0]).toBe(user)
    expect(rebuilt.map((m) => m.id)).toEqual(['h-user', undefined, 'h-frame-4', 'ai-narrate-4'])
    const summaries = summaryMessages(rebuilt)
    expect(summaries).toHaveLength(1)
    expect(summaries[0]).toBe(rebuilt[1])
    expect(String(summaries[0].content)).toContain(SUMMARY_TEXT)
    // The summary rides as a HumanMessage (RC-17): a SystemMessage at index ≥ 1
    // is rejected outright by @langchain/anthropic.
    expect(isHumanMessage(summaries[0])).toBe(true)
    expect(rebuilt.some((m) => isSystemMessage(m))).toBe(false)
    // No motion ran, so there is no pinned motion log to append.
    expect(String(summaries[0].content)).not.toContain('Recent motions')
  })

  it('the history between the first human message and the kept frame is what the summarizer is given', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    const { user, messages } = narrationHistory()
    await before({ messages }, runtime)

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const sent = llm.invoke.mock.calls[0][0] as BaseMessage[]
    // [system prompt, first human, ...head slice, "Write the summary now."]
    expect(isSystemMessage(sent[0])).toBe(true)
    expect(sent[1].content).toBe(user.content)
    const sentIds = sent.map((m) => m.id)
    // Everything older than the kept frame goes to the summarizer…
    expect(sentIds).toContain('ai-capture-0')
    expect(sentIds).toContain('ai-narrate-3')
    expect(sentIds).toContain('h-frame-3')
    // …and the kept frame's own turn is held back from it, because that frame
    // is the state the next model call is about to be asked to describe. The
    // AIMessage/ToolMessage pair that produced it goes to the summarizer
    // TOGETHER, so nothing is orphaned by cutting here.
    expect(sentIds).toContain('ai-capture-4')
    expect(sentIds).toContain('tm-cap-4')
    expect(sentIds).not.toContain('h-frame-4')
    expect(sentIds).not.toContain('ai-narrate-4')
  })

  it('the ordinary threshold still governs: a motion-free history under it is not summarized', async () => {
    // Discriminates the fix from "always summarize when there is no motion".
    // The history has real mechanical pruning to do (an aged-out image), so a
    // rewrite IS emitted — it just carries no summary.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    const messages: BaseMessage[] = [
      new HumanMessage({ id: 'h-user', content: 'What do you see?' }),
      new HumanMessage({ id: 'h-old', content: [{ type: 'text', text: 'Frame 0.' }, imageBlock()] }),
      new HumanMessage({ id: 'h-new', content: [{ type: 'text', text: 'Frame 1.' }, imageBlock()] }),
      new AIMessage({ id: 'ai-1', content: 'A cone, about half a metre ahead.' }),
    ]
    expect(messages.some((m) => isMotionToolCall(m))).toBe(false)

    const result = await before({ messages }, runtime)

    expect(llm.invoke).not.toHaveBeenCalled()
    const rebuilt = withoutRemoveMessages(result)
    expect(summaryMessages(rebuilt)).toHaveLength(0)
    expect(rebuilt.map((m) => m.id)).toEqual(['h-user', 'h-old', 'h-new', 'ai-1'])
  })

  it('the boundary is the first HUMAN message, not index 0: a leading system message survives in place', async () => {
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    const system = new SystemMessage({ id: 'sys', content: 'agent system prompt' })
    const { user, messages } = narrationHistory()
    const result = await before({ messages: [system, ...messages] }, runtime)

    const rebuilt = withoutRemoveMessages(result)
    expect(rebuilt).toHaveLength(5)
    expect(rebuilt[0]).toBe(system)
    expect(rebuilt[1]).toBe(user)
    expect(rebuilt.map((m) => m.id)).toEqual([
      'sys',
      'h-user',
      undefined,
      'h-frame-4',
      'ai-narrate-4',
    ])
    expect(summaryMessages(rebuilt)).toHaveLength(1)
    // The Anthropic invariant: no SystemMessage past index 0.
    expect(rebuilt.slice(1).some((m) => isSystemMessage(m))).toBe(false)
  })

  it('two no-motion prune cycles: the summary is folded, not accumulated', async () => {
    let n = 0
    const invoke = vi.fn(async () => ({ content: `summary ${++n}: the robot described the room.` }))
    const llm = { invoke } as unknown as Parameters<typeof createContextPrunerMiddleware>[0]['llm'] & {
      invoke: ReturnType<typeof vi.fn>
    }
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    const { user, messages } = narrationHistory()
    const cycle1 = withoutRemoveMessages(await before({ messages }, runtime))
    expect(String(summaryMessages(cycle1)[0].content)).toContain('summary 1')

    // The session carries on narrating off the rebuilt state, still motion-free.
    const { messages: more } = narrationHistory()
    const cycle2Input = [...cycle1, ...more.slice(1)]
    const cycle2 = withoutRemoveMessages(await before({ messages: cycle2Input }, runtime))

    expect(invoke).toHaveBeenCalledTimes(2)
    expect(cycle2).toHaveLength(4)
    expect(cycle2[0]).toBe(user)
    expect(cycle2.map((m) => m.id)).toEqual(['h-user', undefined, 'h-frame-4', 'ai-narrate-4'])
    const folded = summaryMessages(cycle2)
    expect(folded).toHaveLength(1)
    expect(String(folded[0].content)).toContain('summary 2')
    expect(String(folded[0].content)).not.toContain('summary 1')
  })

  it('MOTION ANCHOR PINNED: with a motion call, the summary still stops at the last motion turn', async () => {
    // The guard against the wrong fix. If the no-motion branch were implemented
    // by summarizing to the end of the history unconditionally, this reads as
    // two messages instead of eight and the motion state the tail protects is
    // gone. Two motions, so the anchor is provably the LAST one.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    const user = new HumanMessage({ id: 'h-user', content: 'Drive to the red cone.' })
    const filler: BaseMessage[] = []
    for (let i = 0; i < 4; i++) filler.push(new AIMessage({ id: `ai-fill-${i}`, content: 'F'.repeat(400) }))
    const earlyMotionAi = new AIMessage({
      id: 'ai-motion-early',
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 1 }, id: 'tc-early' }],
    })
    const earlyMotionTool = new ToolMessage({
      id: 'tm-motion-early',
      content: motionResultJson('turn_right (steps=1)', 200),
      tool_call_id: 'tc-early',
      name: 'turn_right',
    })
    const earlyComposite = new HumanMessage({
      id: 'h-early-frame',
      content: [{ type: 'text', text: 'Before/After frames for turn_right (steps=1).' }, imageBlock()],
    })
    const lastMotionAi = new AIMessage({
      id: 'ai-motion-last',
      content: '',
      tool_calls: [{ name: 'move_forward', args: { steps: 2 }, id: 'tc-last' }],
    })
    const lastMotionTool = new ToolMessage({
      id: 'tm-motion-last',
      content: motionResultJson('move_forward (steps=2)', 200),
      tool_call_id: 'tc-last',
      name: 'move_forward',
    })
    const lastComposite = new HumanMessage({
      id: 'h-last-frame',
      content: [{ type: 'text', text: 'Before/After frames for move_forward (steps=2).' }, imageBlock()],
    })
    // Non-motion turns AFTER the anchor: these must ride in the tail too, so the
    // anchor cannot be confused with "the last message".
    const laterAi = new AIMessage({ id: 'ai-later', content: 'Closer now. Checking the view.' })
    const laterCaptureAi = new AIMessage({
      id: 'ai-later-capture',
      content: '',
      tool_calls: [{ name: 'capture_image', args: {}, id: 'tc-later-cap' }],
    })
    const laterCaptureTool = new ToolMessage({
      id: 'tm-later-capture',
      content: captureResultJson(200),
      tool_call_id: 'tc-later-cap',
      name: 'capture_image',
    })

    const result = await before(
      {
        messages: [
          user,
          ...filler,
          earlyMotionAi,
          earlyMotionTool,
          earlyComposite,
          lastMotionAi,
          lastMotionTool,
          lastComposite,
          laterAi,
          laterCaptureAi,
          laterCaptureTool,
        ],
      },
      runtime
    )

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const rebuilt = withoutRemoveMessages(result)
    expect(rebuilt.map((m) => m.id)).toEqual([
      'h-user',
      undefined, // the summary message, which carries no id
      'ai-motion-last',
      'tm-motion-last',
      'h-last-frame',
      'ai-later',
      'ai-later-capture',
      'tm-later-capture',
    ])
    // The tail is the original objects, not re-described ones — except the two
    // image-bearing ToolMessages, which are copies by design because the
    // mechanical strip drops their base64 frame.
    expect(rebuilt[0]).toBe(user)
    expect(rebuilt[2]).toBe(lastMotionAi)
    expect(rebuilt[4]).toBe(lastComposite)
    expect(rebuilt[5]).toBe(laterAi)
    expect(rebuilt[6]).toBe(laterCaptureAi)
    expect(JSON.parse(rebuilt[7].content as string)).toEqual({
      mimeType: 'image/jpeg',
      captured: true,
      dataDropped: true,
    })
    // Everything before the anchor was eaten, including the earlier motion turn…
    expect(rebuilt).not.toContain(earlyMotionAi)
    // …and it went to the summarizer, which is where it was supposed to go.
    const sentIds = (llm.invoke.mock.calls[0][0] as BaseMessage[]).map((m) => m.id)
    expect(sentIds).toContain('ai-motion-early')
    expect(sentIds).not.toContain('ai-motion-last')
  })

  // ─────────────────────────────────────────────────────────────────────────
  // The frame the mechanical strip kept must survive the summarizer.
  // ─────────────────────────────────────────────────────────────────────────

  it('THE FRAME SURVIVES: a motion-free capture-and-narrate history keeps the image block the strip elected to keep', async () => {
    // The defect this pins: `keepLatestImages: 1` deliberately carries the
    // newest image-bearing turn through the mechanical strip, and a summarizer
    // that then cuts to the end of the history throws away exactly what the
    // strip just decided to preserve. The narrating call is then asked to
    // describe a picture it is no longer holding.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    const { messages } = narrationHistory()
    const newestFrame = messages.find((m) => m.id === 'h-frame-4') as HumanMessage
    expect(hasImage(newestFrame)).toBe(true)

    const result = await before({ messages }, runtime)
    expect(llm.invoke).toHaveBeenCalledTimes(1)

    const rebuilt = withoutRemoveMessages(result)
    const kept = rebuilt.find((m) => m.id === 'h-frame-4')
    // Present, the same object, and still carrying its image.
    expect(kept).toBe(newestFrame)
    expect(hasImage(kept as BaseMessage)).toBe(true)
    // Exactly one image block survives — the one the strip kept, not an older
    // one that should have been stripped to text.
    expect(rebuilt.filter(hasImage).map((m) => m.id)).toEqual(['h-frame-4'])
    // And it really did summarize; this is not the old never-prune behaviour
    // passing by accident.
    expect(summaryMessages(rebuilt)).toHaveLength(1)
  })

  it('the anchor is the OLDEST kept frame, not the newest: keepLatestImages 2 holds back both', async () => {
    // At the shipped `keepLatestImages: 1` the oldest kept frame and the newest
    // are the same message, so that wording is untestable there. With two kept
    // frames they differ, and anchoring on the newest would summarize away a
    // frame the strip had just elected to keep — the same defect one frame in.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({
      llm,
      ...OVER_THRESHOLD,
      keepLatestImages: 2,
    }) as HookContainer
    const before = getHook(mw.beforeModel)

    const { messages } = narrationHistory()
    const result = await before({ messages }, runtime)

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const rebuilt = withoutRemoveMessages(result)
    expect(rebuilt.map((m) => m.id)).toEqual([
      'h-user',
      undefined,
      'h-frame-3',
      'ai-narrate-3',
      'ai-capture-4',
      'tm-cap-4',
      'h-frame-4',
      'ai-narrate-4',
    ])
    // Both kept frames still carry their image blocks.
    expect(rebuilt.filter(hasImage).map((m) => m.id)).toEqual(['h-frame-3', 'h-frame-4'])
    // The older frames were stripped to text and summarized away.
    const sentIds = (llm.invoke.mock.calls[0][0] as BaseMessage[]).map((m) => m.id)
    expect(sentIds).toContain('h-frame-2')
    expect(sentIds).not.toContain('h-frame-3')
  })

  it('BOUNDEDNESS: a session that captures once and then only talks stays under the hard cap', async () => {
    // The reason the kept-frame anchor is conditional. Holding back everything
    // from the frame onwards is cheap when the frame is recent, but this shape
    // puts it near the START: anchoring there leaves a two-message head and an
    // uncompressed tail that grows without limit, which is RC-27's own defect
    // reintroduced by the anchor meant to fix a different one. Measured at the
    // shipped local profile, the unconditional form crossed the 30000-token
    // hard cap by round 58 and reached 62021 tokens while calling the
    // summarizer on 80 of 120 turns.
    //
    // Written as a loop over the real rewrite, not as a boundary assertion,
    // because what has to hold is a property of the SESSION.
    const invoke = vi.fn(async () => ({ content: SUMMARY_TEXT }))
    const llm = { invoke } as unknown as Parameters<typeof createContextPrunerMiddleware>[0]['llm']
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    let state: BaseMessage[] = [
      new HumanMessage({ id: 'h-user', content: 'Look once, then answer my questions.' }),
      new AIMessage({
        id: 'ai-capture-0',
        content: '',
        tool_calls: [{ name: 'capture_image', args: {}, id: 'tc-cap-0' }],
      }),
      new ToolMessage({
        id: 'tm-cap-0',
        content: captureResultJson(),
        tool_call_id: 'tc-cap-0',
        name: 'capture_image',
      }),
      new HumanMessage({
        id: 'h-frame-0',
        content: [{ type: 'text', text: 'Frame 0.' }, imageBlock()],
      }),
    ]
    let peak = 0
    const ROUNDS = 60
    for (let round = 0; round < ROUNDS; round++) {
      const result = await before({ messages: state }, runtime)
      const body = result ? withoutRemoveMessages(result) : state
      peak = Math.max(peak, estimateTokens(body, OVER_THRESHOLD.imageTokenBudget))
      state = [
        ...body,
        new AIMessage({ id: `ai-answer-${round}`, content: 'A'.repeat(400) }),
        new HumanMessage({ id: `h-q-${round}`, content: 'And to the left of that?' }),
      ]
    }
    // Never over the configured hard cap…
    expect(peak).toBeLessThan(OVER_THRESHOLD.maxContextTokens)
    // …and it is not achieving that by summarizing on every single turn, which
    // is what an anchor that never makes progress looks like from the outside.
    expect(invoke.mock.calls.length).toBeLessThan(ROUNDS / 2)
    expect(invoke.mock.calls.length).toBeGreaterThan(0)
  })

  it('the kept-frame anchor gives way when its tail would not fit under the threshold', async () => {
    // The single-turn view of the loop above: an early frame with a long text
    // history after it falls through to the end-of-history boundary, because a
    // tail that is itself over the threshold compresses nothing.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    const user = new HumanMessage({ id: 'h-user', content: 'Look once, then answer.' })
    const messages: BaseMessage[] = [
      user,
      new AIMessage({
        id: 'ai-capture-0',
        content: '',
        tool_calls: [{ name: 'capture_image', args: {}, id: 'tc-cap-0' }],
      }),
      new ToolMessage({
        id: 'tm-cap-0',
        content: captureResultJson(),
        tool_call_id: 'tc-cap-0',
        name: 'capture_image',
      }),
      new HumanMessage({
        id: 'h-frame-0',
        content: [{ type: 'text', text: 'Frame 0.' }, imageBlock()],
      }),
    ]
    for (let i = 0; i < 12; i++) {
      messages.push(new AIMessage({ id: `ai-answer-${i}`, content: 'A'.repeat(400) }))
    }

    const result = await before({ messages }, runtime)

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const rebuilt = withoutRemoveMessages(result)
    // Rule 3: the whole history after the first human message is compressed.
    expect(rebuilt).toHaveLength(2)
    expect(rebuilt[0]).toBe(user)
    expect(summaryMessages(rebuilt)).toHaveLength(1)
  })

  it('the log names which of the three rules chose the boundary', async () => {
    // The `anchor=` field is the only thing that says, from a live log alone,
    // which policy is in force — so each label is pinned rather than left to
    // drift against the branch it describes.
    async function anchorLabelFor(messages: BaseMessage[], opts = {}): Promise<string> {
      const llm = makeStubLlm()
      const mw = createContextPrunerMiddleware({
        llm,
        ...OVER_THRESHOLD,
        ...opts,
      }) as HookContainer
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {})
      try {
        await getHook(mw.beforeModel)({ messages }, runtime)
        const line = spy.mock.calls
          .map((c) => String(c[0]))
          .find((l) => l.includes('anchor='))
        return line ? (line.match(/anchor=([^;]*)/)?.[1] ?? '') : ''
      } finally {
        spy.mockRestore()
      }
    }

    // Rule 2 — a kept frame near the end of a motion-free history.
    expect(await anchorLabelFor(narrationHistory().messages)).toBe('oldest-kept-frame')

    // Rule 3 — a motion-free history with no image anywhere.
    const textOnly: BaseMessage[] = [new HumanMessage({ id: 'h-user', content: 'Explain.' })]
    for (let i = 0; i < 12; i++) {
      textOnly.push(new AIMessage({ id: `ai-${i}`, content: 'A'.repeat(400) }))
    }
    expect(await anchorLabelFor(textOnly)).toBe('end-of-history (no frame worth holding back)')

    // Rule 1 — a motion call takes precedence over both.
    const withMotion: BaseMessage[] = [new HumanMessage({ id: 'h-user', content: 'Drive on.' })]
    for (let i = 0; i < 12; i++) {
      withMotion.push(new AIMessage({ id: `ai-${i}`, content: 'A'.repeat(400) }))
    }
    withMotion.push(
      new AIMessage({
        id: 'ai-motion',
        content: '',
        tool_calls: [{ name: 'move_forward', args: { steps: 1 }, id: 'tc-mv' }],
      }),
      new ToolMessage({
        id: 'tm-motion',
        content: motionResultJson('move_forward (steps=1)', 200),
        tool_call_id: 'tc-mv',
        name: 'move_forward',
      }),
      new HumanMessage({
        id: 'h-motion-frame',
        content: [{ type: 'text', text: 'Before/After frames.' }, imageBlock()],
      })
    )
    expect(await anchorLabelFor(withMotion)).toBe('last-motion')
  })
})

// ───────────────────────────────────────────────────────────────────────────
// RC-56: a failed motion tool leaves no frame, and the motion anchor
// summarizes the last one away.
//
// frontendImageInjectionMiddleware injects a Before/After composite only when
// the tool result carries `mimeType` + `data`. On `payload.error` it injects a
// TEXT-ONLY note instead, and on a result whose `data` is absent it injects
// nothing at all. Either way the newest image-bearing turn is then OLDER than
// the last motion AIMessage.
//
// RC-27's rule 1 gave the motion anchor unconditional precedence, so the
// boundary sat at that motion message and the kept frame — the one the
// mechanical strip had just elected to preserve on the very same pass — fell in
// the head and was summarized away. The model was handed a motion failure to
// recover from and no picture to recover with.
//
// The fix takes the EARLIER of the two anchors, under the same tail-fits
// condition rule 2 already ships under. That condition is not defensive: RC-27
// measured that an anchor free to sit arbitrarily early reintroduces unbounded
// growth at 2x the token cap. Conditioning the min means it can only ever move
// the boundary to a point whose tail is already proven under the threshold.
//
// Message classes are compared by reference or `getType()`, never `instanceof`
// and never by prototype identity — this repo resolves two copies of
// @langchain/core.
// ───────────────────────────────────────────────────────────────────────────
describe('contextPrunerMiddleware — RC-56 a failed motion keeps the frame the strip kept', () => {
  const OVER_THRESHOLD = {
    maxContextTokens: 1000,
    summarizeAtFraction: 0.5,
    imageTokenBudget: 50,
  } as const

  function captureResultJson(dataLen = 400): string {
    return JSON.stringify({ mimeType: 'image/jpeg', data: 'X'.repeat(dataLen), captured: true })
  }

  function withoutRemoveMessages(result: unknown): BaseMessage[] {
    return (result as { messages: BaseMessage[] }).messages.filter((m) => m.getType() !== 'remove')
  }

  function summaryMessages(messages: BaseMessage[]): HumanMessage[] {
    return messages.filter(
      (m): m is HumanMessage =>
        isHumanMessage(m) && String(m.content).startsWith('[Context summary]')
    )
  }

  function hasImage(m: BaseMessage): boolean {
    return (
      Array.isArray(m.content) &&
      (m.content as { type?: string }[]).some(
        (b) => b && (b.type === 'image' || b.type === 'image_url')
      )
    )
  }

  // Orphan safety as an invariant rather than an argument: every ToolMessage in
  // the rebuilt array must still have the AIMessage that called it. Acceptance
  // asks for this on EVERY branch, so it is a helper applied to each fixture
  // rather than a claim made once about the shape of the code.
  function orphanToolMessageIds(messages: BaseMessage[]): string[] {
    const callIds = new Set<string>()
    for (const m of messages) {
      const tcs = (m as AIMessage).tool_calls
      if (Array.isArray(tcs)) for (const tc of tcs) if (tc.id) callIds.add(tc.id)
    }
    return messages
      .filter((m) => m.getType() === 'tool')
      .filter((m) => {
        const id = (m as ToolMessage).tool_call_id
        return !id || !callIds.has(id)
      })
      .map((m) => String(m.id ?? '(no id)'))
  }

  // A motion session that has been producing frames normally. `cycles` complete
  // motion turns, each ending in the injected Before/After composite.
  function motionCycles(cycles: number): BaseMessage[] {
    const out: BaseMessage[] = []
    for (let i = 0; i < cycles; i++) {
      out.push(
        new AIMessage({
          id: `ai-motion-${i}`,
          content: '',
          tool_calls: [{ name: 'move_forward', args: { steps: 2 }, id: `tc-mv-${i}` }],
        })
      )
      out.push(
        new ToolMessage({
          id: `tm-motion-${i}`,
          content: motionResultJson('move_forward (steps=2)', 400),
          tool_call_id: `tc-mv-${i}`,
          name: 'move_forward',
        })
      )
      out.push(
        new HumanMessage({
          id: `h-frame-${i}`,
          content: [{ type: 'text', text: 'Before/After frames.' }, imageBlock()],
        })
      )
      out.push(new AIMessage({ id: `ai-narrate-${i}`, content: 'N'.repeat(400) }))
    }
    return out
  }

  // …and then the newest motion FAILS. This is what the real pipeline leaves
  // behind: the AIMessage, an error-status ToolMessage carrying no image, and
  // the text-only note frontendImageInjectionMiddleware pushes on `error`.
  function failedMotionHistory(cycles = 5): {
    user: HumanMessage
    frame: HumanMessage
    messages: BaseMessage[]
  } {
    const user = new HumanMessage({ id: 'h-user', content: 'Drive to the red cone.' })
    const cyclesMsgs = motionCycles(cycles)
    const messages: BaseMessage[] = [
      user,
      ...cyclesMsgs,
      new AIMessage({
        id: 'ai-motion-failed',
        content: '',
        tool_calls: [{ name: 'move_forward', args: { steps: 2 }, id: 'tc-failed' }],
      }),
      new ToolMessage({
        id: 'tm-motion-failed',
        content: JSON.stringify({ error: 'robot unreachable' }),
        tool_call_id: 'tc-failed',
        name: 'move_forward',
        status: 'error',
      }),
      new HumanMessage({
        id: 'h-fail-note',
        content: 'Motion (move_forward) failed: robot unreachable',
      }),
    ]
    const frame = messages.find((m) => m.id === `h-frame-${cycles - 1}`) as HumanMessage
    return { user, frame, messages }
  }

  it('THE DEFECT: a failed newest motion still delivers the kept frame to the recovery call', async () => {
    // Red on RC-27's code: the boundary sits unconditionally at
    // `ai-motion-failed`, so `h-frame-4` — which the mechanical strip kept on
    // this very pass — lands in the head and is summarized away. The model is
    // asked to recover from a motion failure with no picture at all.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    const { user, frame, messages } = failedMotionHistory()
    // The premise: the newest image-bearing turn really is older than the last
    // motion call. Without this the fixture would not exercise the branch.
    const frameIdx = messages.indexOf(frame)
    const motionIdx = messages.findIndex((m) => m.id === 'ai-motion-failed')
    expect(frameIdx).toBeGreaterThan(-1)
    expect(frameIdx).toBeLessThan(motionIdx)
    expect(hasImage(frame)).toBe(true)

    const result = await before({ messages }, runtime)

    // It did summarize — this is not the under-threshold path passing by
    // accident.
    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const rebuilt = withoutRemoveMessages(result)
    expect(summaryMessages(rebuilt)).toHaveLength(1)

    // THE ASSERTION: the frame survives, as the same object, still carrying its
    // image block.
    const kept = rebuilt.find((m) => m.id === frame.id)
    expect(kept).toBe(frame)
    expect(hasImage(kept as BaseMessage)).toBe(true)
    expect(rebuilt.filter(hasImage).map((m) => m.id)).toEqual(['h-frame-4'])

    // The failed motion turn is still in the tail — the recovery call needs the
    // failure as well as the picture.
    expect(rebuilt.map((m) => m.id)).toEqual([
      'h-user',
      undefined,
      'h-frame-4',
      'ai-narrate-4',
      'ai-motion-failed',
      'tm-motion-failed',
      'h-fail-note',
    ])
    expect(rebuilt[0]).toBe(user)
    // The error status rides through, so the model can tell a failed motion
    // from a completed one.
    expect((rebuilt.find((m) => m.id === 'tm-motion-failed') as ToolMessage).status).toBe('error')
    expect(orphanToolMessageIds(rebuilt)).toEqual([])
  })

  it('the same holds for a motion result whose image data never arrived', async () => {
    // The second route to "newest frame older than the last motion", and it is
    // not an error case: frontendImageInjectionMiddleware injects nothing at all
    // when a motion result arrives without its base64 `data`, deliberately
    // leaving the guard clean (RC-21). No error note either, so the tail here is
    // shorter than the failed-motion one — a fixture that would pass on a fix
    // keyed to `status === 'error'` and must not.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    const user = new HumanMessage({ id: 'h-user', content: 'Drive to the red cone.' })
    const messages: BaseMessage[] = [
      user,
      ...motionCycles(5),
      new AIMessage({
        id: 'ai-motion-last',
        content: '',
        tool_calls: [{ name: 'turn_left', args: { steps: 1 }, id: 'tc-last' }],
      }),
      new ToolMessage({
        id: 'tm-motion-last',
        content: JSON.stringify({ mimeType: 'image/jpeg', motion: 'turn_left (steps=1)' }),
        tool_call_id: 'tc-last',
        name: 'turn_left',
      }),
    ]

    const result = await before({ messages }, runtime)

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const rebuilt = withoutRemoveMessages(result)
    expect(rebuilt.map((m) => m.id)).toEqual([
      'h-user',
      undefined,
      'h-frame-4',
      'ai-narrate-4',
      'ai-motion-last',
      'tm-motion-last',
    ])
    expect(rebuilt.filter(hasImage).map((m) => m.id)).toEqual(['h-frame-4'])
    expect(orphanToolMessageIds(rebuilt)).toEqual([])
  })

  it('UNCHANGED: a successful newest motion is byte-identical at the shipped keepLatestImages=1', async () => {
    // The successful path is where the injected composite sits AFTER the motion
    // message, so the earlier of the two anchors IS the motion message and the
    // min is a no-op. Pinned field by field — ids, object identity, image
    // blocks, tool-call pairing and what the summarizer was given.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    const user = new HumanMessage({ id: 'h-user', content: 'Drive to the red cone.' })
    const cycles = motionCycles(6)
    const messages: BaseMessage[] = [user, ...cycles]

    const result = await before({ messages }, runtime)

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const rebuilt = withoutRemoveMessages(result)
    expect(rebuilt.map((m) => m.id)).toEqual([
      'h-user',
      undefined,
      'ai-motion-5',
      'tm-motion-5',
      'h-frame-5',
      'ai-narrate-5',
    ])
    // Object identity, not just ids: the tail is the original messages, except
    // the motion ToolMessage, which the mechanical strip copies to drop its
    // base64 frame.
    expect(rebuilt[0]).toBe(user)
    expect(rebuilt[2]).toBe(messages.find((m) => m.id === 'ai-motion-5'))
    expect(rebuilt[4]).toBe(messages.find((m) => m.id === 'h-frame-5'))
    expect(rebuilt[5]).toBe(messages.find((m) => m.id === 'ai-narrate-5'))
    expect(JSON.parse(rebuilt[3].content as string)).toEqual({
      mimeType: 'image/jpeg',
      motion: 'move_forward (steps=2)',
      dataDropped: true,
    })
    expect(rebuilt.filter(hasImage).map((m) => m.id)).toEqual(['h-frame-5'])
    expect(orphanToolMessageIds(rebuilt)).toEqual([])
    // And the summarizer was given exactly the head, up to but excluding the
    // anchor.
    const sentIds = (llm.invoke.mock.calls[0][0] as BaseMessage[]).map((m) => m.id)
    expect(sentIds).toContain('ai-motion-4')
    expect(sentIds).toContain('h-frame-4')
    expect(sentIds).not.toContain('ai-motion-5')
  })

  it('keepLatestImages=2 on a SUCCESSFUL motion session now holds back both kept frames', async () => {
    // A deliberate, measured behaviour change, disclosed rather than hidden.
    // At the shipped N=1 the oldest kept frame is the newest one and always
    // sits after the last motion, so the min never fires. At N>=2 the OLDER of
    // the two kept frames sits before the last motion, and RC-27's code
    // summarized it away — the strip kept a frame and the summarizer discarded
    // it, on a perfectly healthy session. That is the same contradiction RC-27
    // named in its own rule 2 ("the summarizer must not discard what the strip
    // just elected to keep"), one frame in; making the anchor the earlier of the
    // two resolves it consistently rather than only on the motion-free path.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({
      llm,
      ...OVER_THRESHOLD,
      keepLatestImages: 2,
    }) as HookContainer
    const before = getHook(mw.beforeModel)

    const messages: BaseMessage[] = [
      new HumanMessage({ id: 'h-user', content: 'Drive to the red cone.' }),
      ...motionCycles(6),
    ]

    const result = await before({ messages }, runtime)

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const rebuilt = withoutRemoveMessages(result)
    // Both frames the strip kept are still here, with their images.
    expect(rebuilt.filter(hasImage).map((m) => m.id)).toEqual(['h-frame-4', 'h-frame-5'])
    expect(rebuilt.map((m) => m.id)).toEqual([
      'h-user',
      undefined,
      'h-frame-4',
      'ai-narrate-4',
      'ai-motion-5',
      'tm-motion-5',
      'h-frame-5',
      'ai-narrate-5',
    ])
    expect(orphanToolMessageIds(rebuilt)).toEqual([])
  })

  it('the min gives way when the kept frame is early: an old frame does not drag the boundary back', async () => {
    // The boundedness condition, seen in one call. A frame near the START of a
    // long history has a tail that is itself over the threshold, so anchoring
    // there would compress nothing. The boundary falls back to the last motion
    // message — RC-27's rule 1 — and the frame is spent. This is the case where
    // no bounded boundary could have kept it.
    const llm = makeStubLlm()
    const mw = createContextPrunerMiddleware({ llm, ...OVER_THRESHOLD }) as HookContainer
    const before = getHook(mw.beforeModel)

    const user = new HumanMessage({ id: 'h-user', content: 'Look once, then drive.' })
    const messages: BaseMessage[] = [
      user,
      new AIMessage({
        id: 'ai-capture-0',
        content: '',
        tool_calls: [{ name: 'capture_image', args: {}, id: 'tc-cap-0' }],
      }),
      new ToolMessage({
        id: 'tm-cap-0',
        content: captureResultJson(),
        tool_call_id: 'tc-cap-0',
        name: 'capture_image',
      }),
      new HumanMessage({
        id: 'h-frame-0',
        content: [{ type: 'text', text: 'Frame 0.' }, imageBlock()],
      }),
    ]
    for (let i = 0; i < 12; i++) {
      messages.push(new AIMessage({ id: `ai-answer-${i}`, content: 'A'.repeat(400) }))
    }
    messages.push(
      new AIMessage({
        id: 'ai-motion-failed',
        content: '',
        tool_calls: [{ name: 'move_forward', args: { steps: 2 }, id: 'tc-failed' }],
      }),
      new ToolMessage({
        id: 'tm-motion-failed',
        content: JSON.stringify({ error: 'robot unreachable' }),
        tool_call_id: 'tc-failed',
        name: 'move_forward',
        status: 'error',
      })
    )

    const result = await before({ messages }, runtime)

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const rebuilt = withoutRemoveMessages(result)
    expect(rebuilt.map((m) => m.id)).toEqual([
      'h-user',
      undefined,
      'ai-motion-failed',
      'tm-motion-failed',
    ])
    expect(rebuilt.filter(hasImage)).toHaveLength(0)
    expect(orphanToolMessageIds(rebuilt)).toEqual([])
  })

  it('the log names the fourth rule when the kept frame precedes the last motion', async () => {
    // `anchor=` is the only thing that says from a live log which policy is in
    // force. With the min there are four outcomes, not three, and the existing
    // three-label test cannot see the new one: its rule-1 fixture puts the frame
    // AFTER the motion, so it takes the same branch it always did.
    async function anchorLabelFor(messages: BaseMessage[], opts = {}): Promise<string> {
      const llm = makeStubLlm()
      const mw = createContextPrunerMiddleware({
        llm,
        ...OVER_THRESHOLD,
        ...opts,
      }) as HookContainer
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {})
      try {
        await getHook(mw.beforeModel)({ messages }, runtime)
        const line = spy.mock.calls.map((c) => String(c[0])).find((l) => l.includes('anchor='))
        return line ? (line.match(/anchor=([^;]*)/)?.[1] ?? '') : ''
      } finally {
        spy.mockRestore()
      }
    }

    // The new branch: motion present, but the kept frame is older than it.
    expect(await anchorLabelFor(failedMotionHistory().messages)).toBe(
      'kept-frame-before-last-motion'
    )
    // Still rule 1 when the frame sits after the motion — the successful path.
    expect(
      await anchorLabelFor([
        new HumanMessage({ id: 'h-user', content: 'Drive on.' }),
        ...motionCycles(6),
      ])
    ).toBe('last-motion')
  })

  // The session property, at the SHIPPED production profile rather than the
  // scaled-down one the rest of this block uses, because the number that
  // matters is the one the robot actually runs against. Written as loops over
  // the real rewrite, fed back, exactly as RC-27's boundedness test is: the
  // failure mode being guarded is a feedback effect, which a single-call
  // boundary assertion cannot see.
  const PRUNER_LOCAL = {
    maxContextTokens: 30_000,
    summarizeAtFraction: 0.7,
    keepLatestImages: 1,
    imageTokenBudget: 800,
  } as const

  // One round of a driving session. `motionFails` decides whether the tool
  // result carries a frame (and therefore whether a composite is injected),
  // which is the whole variable this node turns on.
  function driveRound(round: number, motionFails: boolean): BaseMessage[] {
    const out: BaseMessage[] = [
      new AIMessage({ id: `ai-think-${round}`, content: 'A'.repeat(4000) }),
      new AIMessage({
        id: `ai-motion-${round}`,
        content: '',
        tool_calls: [{ name: 'move_forward', args: { steps: 2 }, id: `tc-mv-${round}` }],
      }),
    ]
    if (motionFails) {
      out.push(
        new ToolMessage({
          id: `tm-motion-${round}`,
          content: JSON.stringify({ error: 'robot unreachable' }),
          tool_call_id: `tc-mv-${round}`,
          name: 'move_forward',
          status: 'error',
        }),
        // Text-only: what frontendImageInjectionMiddleware pushes on `error`.
        new HumanMessage({
          id: `h-fail-note-${round}`,
          content: 'Motion (move_forward) failed: robot unreachable',
        })
      )
    } else {
      out.push(
        new ToolMessage({
          id: `tm-motion-${round}`,
          content: motionResultJson('move_forward (steps=2)', 400),
          tool_call_id: `tc-mv-${round}`,
          name: 'move_forward',
        }),
        new HumanMessage({
          id: `h-frame-${round}`,
          content: [{ type: 'text', text: 'Before/After frames.' }, imageBlock()],
        })
      )
    }
    return out
  }

  async function runSession(
    fails: (round: number) => boolean,
    rounds = 60
  ): Promise<{ peak: number; summaries: number; framesHeld: number }> {
    const invoke = vi.fn(async () => ({ content: SUMMARY_TEXT }))
    const llm = { invoke } as unknown as Parameters<typeof createContextPrunerMiddleware>[0]['llm']
    const mw = createContextPrunerMiddleware({ llm, ...PRUNER_LOCAL }) as HookContainer
    const before = getHook(mw.beforeModel)

    let state: BaseMessage[] = [
      new HumanMessage({ id: 'h-user', content: 'Drive to the red cone.' }),
    ]
    let peak = 0
    let framesHeld = 0
    for (let round = 0; round < rounds; round++) {
      const result = await before({ messages: state }, runtime)
      const body = result ? withoutRemoveMessages(result) : state
      peak = Math.max(peak, estimateTokens(body, PRUNER_LOCAL.imageTokenBudget))
      if (body.some(hasImage)) framesHeld++
      // Orphan safety is a per-round invariant, not a property of one fixture.
      expect(orphanToolMessageIds(body)).toEqual([])
      state = [...body, ...driveRound(round, fails(round))]
    }
    return { peak, summaries: invoke.mock.calls.length, framesHeld }
  }

  it('BOUNDEDNESS: a driving session with intermittent motion failures stays under the hard cap', async () => {
    // The shape this node is FOR: the robot is driving and producing frames,
    // and every third motion fails. On RC-27's code the boundary sat on the
    // failed motion and the frame from the previous turn was summarized away.
    const ROUNDS = 60
    const { peak, summaries, framesHeld } = await runSession((r) => r % 3 === 2, ROUNDS)

    // Never over the configured hard cap…
    expect(peak).toBeLessThan(PRUNER_LOCAL.maxContextTokens)
    // …and not achieved by summarizing on every turn, which is what an anchor
    // that never makes progress looks like from the outside.
    expect(summaries).toBeLessThan(ROUNDS / 2)
    expect(summaries).toBeGreaterThan(0)
    // The point of the change: the frame is genuinely carried through the
    // session rather than merely bounded away.
    expect(framesHeld).toBeGreaterThan(ROUNDS * 0.9)
  })

  it('BOUNDEDNESS: a session whose motions ALWAYS fail is bounded, with no frame to keep', async () => {
    // The degenerate end of the range. Every motion fails, so no composite is
    // ever injected and this session never holds a frame at all — there is
    // nothing for the frame anchor to protect, and the boundary is the last
    // motion turn on every round, exactly as on RC-27's code. Bounded is the
    // only property available here, and it is the one asserted; framesHeld is
    // pinned at 0 so a future change that started injecting frames on the error
    // path would show up here rather than silently altering what this test
    // covers.
    const ROUNDS = 60
    const { peak, summaries, framesHeld } = await runSession(() => true, ROUNDS)

    expect(peak).toBeLessThan(PRUNER_LOCAL.maxContextTokens)
    expect(summaries).toBeLessThan(ROUNDS / 2)
    expect(summaries).toBeGreaterThan(0)
    expect(framesHeld).toBe(0)
  })

  it('BOUNDEDNESS: an all-successful driving session is unchanged and bounded', async () => {
    // The control for the two above: the path this node must not disturb.
    const ROUNDS = 60
    const { peak, summaries, framesHeld } = await runSession(() => false, ROUNDS)

    expect(peak).toBeLessThan(PRUNER_LOCAL.maxContextTokens)
    expect(summaries).toBeLessThan(ROUNDS / 2)
    expect(summaries).toBeGreaterThan(0)
    expect(framesHeld).toBeGreaterThan(ROUNDS * 0.9)
  })
})
