import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  type BaseMessage,
} from '@langchain/core/messages'
import {
  createMotionSummarizationMiddleware,
  stripImageBlocks,
  stripUnpairedToolCalls,
  buildSummarizationMessages,
  __pendingSummariesForTest,
  __motionLogForTest,
} from '../src/agent/motionSummarizationMiddleware.js'

import {
  isAIMessage,
  isHumanMessage,
  isSystemMessage,
  isToolMessage,
} from '@langchain/core/messages'
import {
  foreignAIMessage,
  foreignCoreMessage,
  foreignHumanMessage,
  isForeignToThisCore,
} from './helpers/foreignCoreMessage.js'

// Collect every tool_use id an AIMessage carries, from BOTH representations:
// the generic `.tool_calls` array AND Anthropic-native `tool_use` content
// blocks. Anthropic requires a tool_result for each, whichever shape it took.
function collectToolUseIds(m: AIMessage): string[] {
  const ids: string[] = []
  for (const tc of m.tool_calls ?? []) if (tc.id) ids.push(tc.id)
  if (Array.isArray(m.content)) {
    for (const block of m.content as Array<{ type?: string; id?: string }>) {
      if (block && block.type === 'tool_use' && typeof block.id === 'string') ids.push(block.id)
    }
  }
  return ids
}

// Asserts the Anthropic pairing invariant: every AIMessage tool_use id (from
// either representation) is immediately followed by a ToolMessage carrying that
// id. This is exactly what INVALID_TOOL_RESULTS enforces ("tool_use ids without
// tool_result blocks immediately after").
function assertNoUnpairedToolUse(messages: BaseMessage[]) {
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i]
    if (!isAIMessage(m)) continue
    const ids = collectToolUseIds(m as AIMessage)
    if (ids.length === 0) continue
    // The results must be the ToolMessages immediately following this AIMessage.
    const followingIds = new Set<string>()
    for (let j = i + 1; j < messages.length; j++) {
      const n = messages[j]
      if (isToolMessage(n)) {
        const id = (n as ToolMessage).tool_call_id
        if (id) followingIds.add(id)
        continue
      }
      break
    }
    for (const id of ids) {
      expect(followingIds.has(id)).toBe(true)
    }
  }
}

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

const SUMMARY_TEXT = 'User wanted to find the red cone. Robot turned right twice, then drove forward.'

function makeStubLlm(summary = SUMMARY_TEXT) {
  const invoke = vi.fn(async () => ({ content: summary }))
  // Cast around BaseChatModel because we only need .invoke for this test.
  return { invoke } as unknown as Parameters<typeof createMotionSummarizationMiddleware>[0]['llm'] & {
    invoke: ReturnType<typeof vi.fn>
  }
}

const runtime = { configurable: { thread_id: 'test-thread' } }

function imageBlock() {
  return { type: 'image' as const, source_type: 'base64' as const, mime_type: 'image/jpeg', data: 'XXXX' }
}

beforeEach(() => {
  __pendingSummariesForTest.clear()
  __motionLogForTest.clear()
})

describe('motionSummarizationMiddleware', () => {
  it('afterModel kicks off summarization only for motion tool calls', async () => {
    const llm = makeStubLlm()
    const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
    const after = getHook(mw.afterModel)

    const userMsg = new HumanMessage('Get the robot to the red cone.')
    const aiPlain = new AIMessage('I will start by looking around.')
    await after({ messages: [userMsg, aiPlain] }, runtime)
    expect(llm.invoke).not.toHaveBeenCalled()

    const aiMotion = new AIMessage({
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-1' }],
    })
    await after({ messages: [userMsg, aiPlain, aiMotion] }, runtime)
    // Give the in-flight Promise a tick.
    await new Promise((r) => setTimeout(r, 0))
    expect(llm.invoke).toHaveBeenCalledTimes(1)
  })

  it('beforeModel awaits the summary and replaces the middle of history', async () => {
    const llm = makeStubLlm()
    const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
    const after = getHook(mw.afterModel)
    const before = getHook(mw.beforeModel)

    const userMsg = new HumanMessage('Get the robot to the red cone.')
    const noise: BaseMessage[] = [
      new AIMessage('Calibrating.'),
      new AIMessage({ content: '', tool_calls: [{ name: 'capture_image', args: {}, id: 'tc-0' }] }),
      new ToolMessage({ content: JSON.stringify({ mimeType: 'image/jpeg', data: 'A' }), tool_call_id: 'tc-0', name: 'capture_image' }),
      // simulate the image-injection middleware adding a HumanMessage with image blocks
      new HumanMessage({ content: [{ type: 'text', text: 'Camera frame captured:' }, imageBlock()] }),
      new AIMessage('Face appears to be at the bottom.'),
    ]
    const motionAi = new AIMessage({
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-motion' }],
    })
    const motionTool = new ToolMessage({
      content: JSON.stringify({ mimeType: 'image/jpeg', data: 'COMPOSITE', motion: 'turn_right (steps=3)' }),
      tool_call_id: 'tc-motion',
      name: 'turn_right',
    })
    const compositeInjected = new HumanMessage({
      content: [
        { type: 'text', text: 'Before/After frames for turn_right (steps=3). Distance: 25.0 cm → 27.0 cm.' },
        imageBlock(),
      ],
    })

    // afterModel sees the assistant's just-emitted motion tool call as the
    // last message — the tool result and the injected composite arrive later.
    const messagesAfter = [userMsg, ...noise, motionAi]
    await after({ messages: messagesAfter }, runtime)
    // beforeModel runs on the next turn, with the tool result + composite appended.
    const messagesBefore = [...messagesAfter, motionTool, compositeInjected]
    const result = await before({ messages: messagesBefore }, runtime)

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    expect(result).toBeTruthy()
    const updated = (result as { messages: BaseMessage[] }).messages
    // First entry is the REMOVE_ALL_MESSAGES marker.
    expect(updated[0].getType()).toBe('remove')
    // Next entry is the original user message verbatim.
    expect(isHumanMessage(updated[1])).toBe(true)
    expect((updated[1] as HumanMessage).content).toBe('Get the robot to the red cone.')
    // Then the summary as a clearly-marked HumanMessage (RC-16: a SystemMessage
    // here sits at index ≥ 1, which @langchain/anthropic rejects outright).
    expect(isHumanMessage(updated[2])).toBe(true)
    expect(isSystemMessage(updated[2])).toBe(false)
    expect((updated[2] as HumanMessage).content).toContain('[Motion summary]')
    expect((updated[2] as HumanMessage).content).toContain(SUMMARY_TEXT)
    // Then the most recent motion turn (AIMessage with motion tool call, ToolMessage, composite HumanMessage).
    expect(updated[3]).toBe(motionAi)
    expect(updated[4]).toBe(motionTool)
    expect(updated[5]).toBe(compositeInjected)
    expect(updated).toHaveLength(6)
  })

  it('keeps the very first user message verbatim in the LLM input and strips image blocks', async () => {
    const llm = makeStubLlm()
    const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
    const after = getHook(mw.afterModel)

    const userMsg = new HumanMessage('Find the red cone.')
    const imageHuman = new HumanMessage({
      content: [
        { type: 'text', text: 'Camera frame captured:' },
        imageBlock(),
      ],
    })
    const aiMotion = new AIMessage({
      content: '',
      tool_calls: [{ name: 'move_forward', args: { steps: 2 }, id: 'tc' }],
    })

    await after({ messages: [userMsg, imageHuman, aiMotion] }, runtime)
    await new Promise((r) => setTimeout(r, 0))

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const sanitizedInput = llm.invoke.mock.calls[0][0] as BaseMessage[]
    // First entry is our summarization system prompt; the user prompt should be intact among the rest.
    const userInSanitized = sanitizedInput.find(
      (m) => isHumanMessage(m) && m.content === 'Find the red cone.'
    )
    expect(userInSanitized).toBeDefined()
    // No image blocks remain in any sanitized message.
    for (const m of sanitizedInput) {
      if (Array.isArray(m.content)) {
        for (const block of m.content as Array<{ type?: string }>) {
          expect(block.type === 'image' || block.type === 'image_url').toBe(false)
        }
      }
    }
  })

  it('uses a provided summaryPrompt override as the first system message', async () => {
    const llm = makeStubLlm()
    const mw = createMotionSummarizationMiddleware({
      llm,
      summaryPrompt: 'CUSTOM SUMMARY PROMPT',
    }) as HookContainer
    const after = getHook(mw.afterModel)

    const userMsg = new HumanMessage('Find the red cone.')
    const aiMotion = new AIMessage({
      content: '',
      tool_calls: [{ name: 'move_forward', args: { steps: 2 }, id: 'tc' }],
    })

    await after({ messages: [userMsg, aiMotion] }, runtime)
    await new Promise((r) => setTimeout(r, 0))

    expect(llm.invoke).toHaveBeenCalledTimes(1)
    const sanitizedInput = llm.invoke.mock.calls[0][0] as BaseMessage[]
    expect(isSystemMessage(sanitizedInput[0])).toBe(true)
    expect((sanitizedInput[0] as SystemMessage).content).toBe('CUSTOM SUMMARY PROMPT')
  })

  it('logs motions, marking the previous done and the newest pending', async () => {
    const llm = makeStubLlm()
    const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
    const after = getHook(mw.afterModel)

    const userMsg = new HumanMessage('go')
    const tr = new AIMessage({ content: '', tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'a' }] })
    await after({ messages: [userMsg, tr] }, runtime)
    const fwd = new AIMessage({ content: '', tool_calls: [{ name: 'move_forward', args: { steps: 2 }, id: 'b' }] })
    await after({ messages: [userMsg, tr, fwd] }, runtime)

    expect(__motionLogForTest.get('test-thread')).toEqual([
      { label: 'turn_right (steps=3)', pending: false },
      { label: 'move_forward (steps=2)', pending: true },
    ])
  })

  it('appends a deterministic recent-motions list to the summary, newest pending', async () => {
    const llm = makeStubLlm()
    const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
    const after = getHook(mw.afterModel)
    const before = getHook(mw.beforeModel)

    const userMsg = new HumanMessage('Get the robot to the red cone.')
    const noise: BaseMessage[] = [
      new AIMessage('Looking.'),
      new AIMessage({ content: '', tool_calls: [{ name: 'capture_image', args: {}, id: 'tc-0' }] }),
      new ToolMessage({ content: JSON.stringify({ mimeType: 'image/jpeg', data: 'A' }), tool_call_id: 'tc-0', name: 'capture_image' }),
      new HumanMessage({ content: [{ type: 'text', text: 'frame' }, imageBlock()] }),
    ]
    const motionAi = new AIMessage({ content: '', tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-m' }] })
    const motionTool = new ToolMessage({
      content: JSON.stringify({ mimeType: 'image/jpeg', data: 'C', motion: 'turn_right (steps=3)' }),
      tool_call_id: 'tc-m',
      name: 'turn_right',
    })
    const composite = new HumanMessage({
      content: [{ type: 'text', text: 'Before/After frames for turn_right (steps=3).' }, imageBlock()],
    })

    const messagesAfter = [userMsg, ...noise, motionAi]
    await after({ messages: messagesAfter }, runtime)
    const result = await before({ messages: [...messagesAfter, motionTool, composite] }, runtime)

    const updated = (result as { messages: BaseMessage[] }).messages
    const summaryMsg = updated[2] as HumanMessage
    expect(isHumanMessage(summaryMsg)).toBe(true)
    expect(summaryMsg.content).toContain('Recent motions (newest last):')
    expect(summaryMsg.content).toContain('turn_right (steps=3) (pending')
  })

  it('beforeModel is a no-op when no summary is pending', async () => {
    const llm = makeStubLlm()
    const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
    const before = getHook(mw.beforeModel)
    const result = await before({ messages: [new HumanMessage('hi')] }, runtime)
    expect(result).toBeUndefined()
    expect(llm.invoke).not.toHaveBeenCalled()
  })

  // ── RC-9: INVALID_TOOL_RESULTS pairing fix ────────────────────────────────
  describe('RC-9 tool-call pairing (INVALID_TOOL_RESULTS)', () => {
    it('strips a trailing tool_use that has no matching tool_result', () => {
      // Exact INVALID_TOOL_RESULTS shape: the just-emitted motion tool call is
      // the last message and its result does not exist yet.
      const history: BaseMessage[] = [
        new HumanMessage('Get the robot to the red cone.'),
        new AIMessage({ content: '', tool_calls: [{ name: 'capture_image', args: {}, id: 'tc-0' }] }),
        new ToolMessage({ content: JSON.stringify({ ok: true }), tool_call_id: 'tc-0', name: 'capture_image' }),
        new AIMessage('Face is at the bottom.'),
        new AIMessage({ content: '', tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-motion' }] }),
      ]

      const cleaned = stripUnpairedToolCalls(history)

      // The paired capture_image call + its result survive; the trailing
      // unpaired motion call is gone (and its now-empty AIMessage dropped).
      assertNoUnpairedToolUse(cleaned)
      const hasMotionCall = cleaned.some(
        (m) => isAIMessage(m) && ((m as AIMessage).tool_calls ?? []).some((tc) => tc.id === 'tc-motion')
      )
      expect(hasMotionCall).toBe(false)
      const hasCaptureCall = cleaned.some(
        (m) => isAIMessage(m) && ((m as AIMessage).tool_calls ?? []).some((tc) => tc.id === 'tc-0')
      )
      expect(hasCaptureCall).toBe(true)
      // The empty AIMessage (content '' + only the stripped call) is removed.
      expect(cleaned).toHaveLength(4)
    })

    it('drops an orphan tool_result whose tool_use is absent', () => {
      const history: BaseMessage[] = [
        new HumanMessage('go'),
        new ToolMessage({ content: '{}', tool_call_id: 'ghost', name: 'turn_right' }),
      ]
      const cleaned = stripUnpairedToolCalls(history)
      assertNoUnpairedToolUse(cleaned)
      expect(cleaned.some((m) => isToolMessage(m))).toBe(false)
      expect(cleaned).toHaveLength(1)
    })

    it('passes a fully-paired history through unchanged (same instances)', () => {
      const history: BaseMessage[] = [
        new HumanMessage('go'),
        new AIMessage({ content: '', tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'a' }] }),
        new ToolMessage({ content: '{}', tool_call_id: 'a', name: 'turn_right' }),
        new AIMessage('Done turning.'),
      ]
      const cleaned = stripUnpairedToolCalls(history)
      assertNoUnpairedToolUse(cleaned)
      expect(cleaned).toHaveLength(history.length)
      // Untouched messages keep their original instances.
      cleaned.forEach((m, i) => expect(m).toBe(history[i]))
    })

    it('afterModel sends the summarizer an Anthropic-valid (fully paired) history', async () => {
      const llm = makeStubLlm()
      const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
      const after = getHook(mw.afterModel)

      const userMsg = new HumanMessage('Get the robot to the red cone.')
      const capAi = new AIMessage({ content: '', tool_calls: [{ name: 'capture_image', args: {}, id: 'tc-0' }] })
      const capTool = new ToolMessage({ content: JSON.stringify({ ok: true }), tool_call_id: 'tc-0', name: 'capture_image' })
      const motionAi = new AIMessage({ content: '', tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-motion' }] })

      // afterModel sees the just-emitted motion tool call as the LAST message —
      // its tool result does not exist yet (the INVALID_TOOL_RESULTS trigger).
      await after({ messages: [userMsg, capAi, capTool, motionAi] }, runtime)
      await new Promise((r) => setTimeout(r, 0))

      expect(llm.invoke).toHaveBeenCalledTimes(1)
      const sentToLlm = llm.invoke.mock.calls[0][0] as BaseMessage[]
      // The real payload the LLM would receive must have no unpaired tool_use.
      assertNoUnpairedToolUse(sentToLlm)
      // And the unpaired motion call specifically must be gone from that payload.
      const stillHasMotion = sentToLlm.some(
        (m) => isAIMessage(m) && ((m as AIMessage).tool_calls ?? []).some((tc) => tc.id === 'tc-motion')
      )
      expect(stillHasMotion).toBe(false)
    })

    // ── Anthropic-native content-block tool_use shape ──────────────────────
    // This is the ONLY shape that triggers the bug in production: the tool_use
    // lives in the AIMessage's `content` array, not (only) in `.tool_calls`.
    it('strips an unpaired trailing tool_use carried as a content block', () => {
      const history: BaseMessage[] = [
        new HumanMessage('Get the robot to the red cone.'),
        // capture_image emitted as a native content-block tool_use (id only in content).
        new AIMessage({
          content: [{ type: 'tool_use', id: 'blk-0', name: 'capture_image', input: {} }] as unknown as AIMessage['content'],
        }),
        new ToolMessage({ content: JSON.stringify({ ok: true }), tool_call_id: 'blk-0', name: 'capture_image' }),
        // trailing motion tool_use as a content block, NO result yet.
        new AIMessage({
          content: [
            { type: 'text', text: 'Turning now.' },
            { type: 'tool_use', id: 'blk-motion', name: 'turn_right', input: { steps: 3 } },
          ] as unknown as AIMessage['content'],
        }),
      ]

      const cleaned = stripUnpairedToolCalls(history)
      assertNoUnpairedToolUse(cleaned)
      // The paired capture_image block + its result survive.
      const stillHasCapture = cleaned.some(
        (m) => isAIMessage(m) && Array.isArray(m.content) &&
          (m.content as Array<{ id?: string }>).some((b) => b.id === 'blk-0')
      )
      expect(stillHasCapture).toBe(true)
      expect(cleaned.some((m) => isToolMessage(m) && (m as ToolMessage).tool_call_id === 'blk-0')).toBe(true)
      // The unpaired motion block is gone; its message keeps only the text.
      const stillHasMotion = cleaned.some(
        (m) => isAIMessage(m) && Array.isArray(m.content) &&
          (m.content as Array<{ id?: string }>).some((b) => b.id === 'blk-motion')
      )
      expect(stillHasMotion).toBe(false)
      const textSurvived = cleaned.some(
        (m) => isAIMessage(m) && Array.isArray(m.content) &&
          (m.content as Array<{ type?: string; text?: string }>).some((b) => b.type === 'text' && b.text === 'Turning now.')
      )
      expect(textSurvived).toBe(true)
    })

    it('passes a fully-paired content-block tool_use history through unchanged', () => {
      const history: BaseMessage[] = [
        new HumanMessage('go'),
        new AIMessage({
          content: [{ type: 'tool_use', id: 'blk-a', name: 'turn_right', input: { steps: 3 } }] as unknown as AIMessage['content'],
        }),
        new ToolMessage({ content: '{}', tool_call_id: 'blk-a', name: 'turn_right' }),
        new AIMessage('Done turning.'),
      ]
      const cleaned = stripUnpairedToolCalls(history)
      assertNoUnpairedToolUse(cleaned)
      expect(cleaned).toHaveLength(history.length)
      // Nothing stripped → original instances preserved, incl. the paired
      // content-block tool_use whose result must be retained.
      cleaned.forEach((m, i) => expect(m).toBe(history[i]))
    })

    it('buildSummarizationMessages wraps a paired history with system + human nudge', () => {
      const history: BaseMessage[] = [
        new HumanMessage('go'),
        new AIMessage({ content: '', tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-motion' }] }),
      ]
      const built = buildSummarizationMessages(history, 'PROMPT')
      expect(isSystemMessage(built[0])).toBe(true)
      expect((built[0] as SystemMessage).content).toBe('PROMPT')
      expect(isHumanMessage(built[built.length - 1])).toBe(true)
      expect((built[built.length - 1] as HumanMessage).content).toBe('Write the summary now.')
      // The unpaired motion call between the wrappers has been stripped.
      assertNoUnpairedToolUse(built)
      const hasMotion = built.some(
        (m) => isAIMessage(m) && ((m as AIMessage).tool_calls ?? []).some((tc) => tc.id === 'tc-motion')
      )
      expect(hasMotion).toBe(false)
    })
  })

  // ── RC-16: no mid-history SystemMessage, ever ─────────────────────────────
  // @langchain/anthropic throws "System messages are only permitted as the
  // first passed message." for a SystemMessage at index ≥ 1 — the PLAT-13
  // crash. Every branch of beforeModel must therefore rebuild a history with
  // no SystemMessage past index 0; the summary rides as a marked HumanMessage.
  describe('RC-16 mid-history SystemMessage fix', () => {
    function systemIndices(messages: BaseMessage[]): number[] {
      return messages
        .map((m, i) => (isSystemMessage(m) ? i : -1))
        .filter((i) => i >= 0)
    }

    // The PLAT-13 crash shape: a read_status tool turn BEFORE the first motion,
    // so the rewrite window (firstHumanIdx+1 .. lastMotionIdx) is non-empty.
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
      const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
      const after = getHook(mw.afterModel)
      const before = getHook(mw.beforeModel)

      const { atMotion, nextTurn } = crashShapedHistory()
      await after({ messages: atMotion }, runtime)
      const result = await before({ messages: nextTurn }, runtime)

      expect(result).toBeTruthy()
      const updated = (result as { messages: BaseMessage[] }).messages
      expect(updated[0].getType()).toBe('remove')
      const rebuilt = updated.slice(1)
      // The invariant Anthropic enforces.
      expect(systemIndices(rebuilt)).toEqual([])
      // The summary content still lands, as a marked HumanMessage, with the
      // pinned state (afterModel logged the motion, so it is present here).
      const summaryMsg = rebuilt.find(
        (m) => isHumanMessage(m) && String(m.content).startsWith('[Motion summary]')
      )
      expect(summaryMsg).toBeDefined()
      expect((summaryMsg as HumanMessage).content).toContain(SUMMARY_TEXT)
      expect((summaryMsg as HumanMessage).content).toContain('Recent motions (newest last):')
    })

    it('summary-applied branch (no pinned state): no SystemMessage at index ≥ 1', async () => {
      const llm = makeStubLlm()
      const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
      const before = getHook(mw.beforeModel)

      // Seed the pending summary directly (no afterModel), so the motion log is
      // empty and formatPinnedState() returns '' — the pinned-less branch.
      __pendingSummariesForTest.set('test-thread', Promise.resolve(SUMMARY_TEXT))
      const { nextTurn } = crashShapedHistory()
      const result = await before({ messages: nextTurn }, runtime)

      expect(result).toBeTruthy()
      const rebuilt = (result as { messages: BaseMessage[] }).messages.slice(1)
      expect(systemIndices(rebuilt)).toEqual([])
      const summaryMsg = rebuilt.find(
        (m) => isHumanMessage(m) && String(m.content).startsWith('[Motion summary]')
      )
      expect(summaryMsg).toBeDefined()
      expect((summaryMsg as HumanMessage).content).toContain(SUMMARY_TEXT)
    })

    it('a pre-existing first-position SystemMessage stays at index 0 only', async () => {
      const llm = makeStubLlm()
      const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
      const before = getHook(mw.beforeModel)

      __pendingSummariesForTest.set('test-thread', Promise.resolve(SUMMARY_TEXT))
      const { nextTurn } = crashShapedHistory()
      const withSystem = [new SystemMessage('agent system prompt'), ...nextTurn]
      const result = await before({ messages: withSystem }, runtime)

      expect(result).toBeTruthy()
      const rebuilt = (result as { messages: BaseMessage[] }).messages.slice(1)
      // The leading system message survives in place; no OTHER system message
      // appears anywhere past index 0.
      expect(systemIndices(rebuilt)).toEqual([0])
      expect((rebuilt[0] as SystemMessage).content).toBe('agent system prompt')
    })

    it('guard branch preserved: motion directly after the first human → no rewrite', async () => {
      const llm = makeStubLlm()
      const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
      const before = getHook(mw.beforeModel)

      // lastMotionIdx === firstHumanIdx + 1 → the existing guard must bail.
      __pendingSummariesForTest.set('test-thread', Promise.resolve(SUMMARY_TEXT))
      const messages: BaseMessage[] = [
        new HumanMessage('go'),
        new AIMessage({ content: '', tool_calls: [{ name: 'turn_right', args: { steps: 1 }, id: 'm' }] }),
        new ToolMessage({ content: '{}', tool_call_id: 'm', name: 'turn_right' }),
      ]
      const result = await before({ messages }, runtime)
      expect(result).toBeUndefined()
    })

    it('empty-summary branch: no rewrite, no SystemMessage introduced', async () => {
      const llm = makeStubLlm()
      const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
      const before = getHook(mw.beforeModel)

      __pendingSummariesForTest.set('test-thread', Promise.resolve(''))
      const { nextTurn } = crashShapedHistory()
      const result = await before({ messages: nextTurn }, runtime)
      expect(result).toBeUndefined()
    })
  })
})

// ───────────────────────────────────────────────────────────────────────────
// RC-58 — the summarizer must recognise a FOREIGN-copy HumanMessage
//
// Every fixture above is built with `new HumanMessage(...)` from the copy this
// file imports, so `instanceof` and `getType()` agree on all of them. The real
// history does not come from here — it arrives through gaunt-sloth's AG-UI
// pipeline, and RC-21 is where a message from that side failed `instanceof` and
// was silently invisible to the guard meant to see it.
//
// `firstHumanIdx` is where that would bite here. It is the anchor for the whole
// rewrite: if the user's opening turn is not recognised as a HumanMessage the
// index is -1, beforeModel bails, and the summary is never applied — no crash,
// no warning, just a history that grows forever. Reverting `isHumanMessage` in
// `src/agent/motionSummarizationMiddleware.ts` to `instanceof` turns this red
// while every native-fixture case above stays green.
// ───────────────────────────────────────────────────────────────────────────
describe('motionSummarizationMiddleware — RC-58 a foreign-copy first HumanMessage still anchors the rewrite', () => {
  // The PLAT-13 crash shape again, but with the opening user turn arriving from
  // the other core copy — the one thing that differs from the cases above.
  function foreignAnchoredHistory() {
    const user = foreignHumanMessage({ id: 'u-foreign', content: 'Drive the robot to the cone.' })
    const statusAi = new AIMessage({
      content: '',
      tool_calls: [{ name: 'read_status', args: {}, id: 'tc-status' }],
    })
    const statusTool = new ToolMessage({
      content: JSON.stringify({ battery: '7.4V', ok: true }),
      tool_call_id: 'tc-status',
      name: 'read_status',
    })
    const motionAi = new AIMessage({
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-motion' }],
    })
    const motionTool = new ToolMessage({
      content: JSON.stringify({ motion: 'turn_right (steps=3)' }),
      tool_call_id: 'tc-motion',
      name: 'turn_right',
    })
    return {
      atMotion: [user, statusAi, statusTool, motionAi],
      nextTurn: [user, statusAi, statusTool, motionAi, motionTool],
      user,
    }
  }

  it('the fixture really is foreign to the copy this file imports', () => {
    expect(isForeignToThisCore(foreignHumanMessage({ content: 'hi' }))).toBe(true)
    // The control: a message from THIS copy is not foreign, so the check above
    // cannot be passing for everything.
    expect(isForeignToThisCore(new HumanMessage('hi'))).toBe(false)
  })

  it('applies the summary when the opening user turn came from the other copy', async () => {
    const llm = makeStubLlm()
    const mw = createMotionSummarizationMiddleware({ llm }) as HookContainer
    const after = getHook(mw.afterModel)
    const before = getHook(mw.beforeModel)

    const { atMotion, nextTurn } = foreignAnchoredHistory()
    await after({ messages: atMotion }, runtime)
    const result = await before({ messages: nextTurn }, runtime)

    // The rewrite happened at all — this is the assertion that goes to
    // `undefined` the moment `firstHumanIdx` stops finding the foreign turn.
    expect(result).toBeTruthy()
    const updated = (result as { messages: BaseMessage[] }).messages
    expect(updated[0].getType()).toBe('remove')
    const rebuilt = updated.slice(1)

    // The foreign opening turn is kept verbatim as the head of the rebuild.
    expect(rebuilt[0].content).toBe('Drive the robot to the cone.')

    // ...and the summary rides as a marked HumanMessage, never a SystemMessage.
    const summaryMsg = rebuilt.find(
      (m) => isHumanMessage(m) && String(m.content).startsWith('[Motion summary]')
    )
    expect(summaryMsg).toBeDefined()
    expect(String((summaryMsg as BaseMessage).content)).toContain(SUMMARY_TEXT)
    expect(rebuilt.filter((m) => isSystemMessage(m))).toEqual([])
  })
})

// ───────────────────────────────────────────────────────────────────────────
// RC-30 — stripImageBlocks copies the message instead of re-describing it
//
// Same defect class as RC-28 and RC-29 in contextPrunerMiddleware, and the
// fourth instance of it. The strip used to rebuild each of the four message
// types from an object literal naming at most three fields, so `id`, `status`,
// `artifact`, `response_metadata` and `additional_kwargs` were dropped from
// every message it touched, and a streamed AIMessageChunk was flattened into a
// plain AIMessage.
//
// The severity is genuinely lower than its siblings', and saying so is part of
// the record: this output feeds the summarizer's own LLM call and nothing else,
// so a dropped field never reaches conversation state or a checkpoint. The trap
// is that the loss is silent and open-ended — every field added upstream later
// goes the same way.
//
// There is ONE CLASS GUARD PER MESSAGE TYPE, each asserting on a property the
// production code does not name anywhere, so each stays red under any
// field-enumerating rebuild of that type however long the enumeration. Four
// guards rather than one parameterised sweep, so putting the literal back for a
// single type reds exactly one of them.
//
// Every preservation assertion sits beside an assertion that the strip actually
// fired: a function that simply returned its argument would satisfy the
// preservation half on its own.
//
// The guards call `stripImageBlocks` directly rather than reaching it through
// `buildSummarizationMessages`, and that is load-bearing: they pin ONE stage, so
// a red here names this function and nothing else. When they were written the
// next stage, `stripUnpairedToolCalls`, still rebuilt an AIMessage from a
// literal whenever it stripped a call, so routing them through the pipeline
// would have tested two functions and reported the result as one. RC-64 closed
// that rebuild; the guards that go through the whole pipeline sit in the RC-64
// block below. These stay exactly as they are — they are the surviving control.
// ───────────────────────────────────────────────────────────────────────────
describe('motionSummarizationMiddleware — RC-30 the image strip preserves the whole message', () => {
  const FRAME_BYTES = 'SUMMARIZERFRAMEMARKER'

  // Read/write a property the production code has never heard of. Cast because
  // no message type declares it — that is exactly the point.
  function stampUnnamedField(msg: BaseMessage, value: unknown): void {
    ;(msg as unknown as Record<string, unknown>).field_no_one_enumerated = value
  }
  function readUnnamedField(msg: BaseMessage): unknown {
    return (msg as unknown as Record<string, unknown>).field_no_one_enumerated
  }

  function markedImageBlock() {
    return {
      type: 'image' as const,
      source_type: 'base64' as const,
      mime_type: 'image/jpeg',
      data: `${FRAME_BYTES}${'X'.repeat(64)}`,
    }
  }
  function hasImage(content: unknown): boolean {
    return (
      Array.isArray(content) &&
      (content as Array<{ type?: string }>).some(
        (b) => b.type === 'image' || b.type === 'image_url'
      )
    )
  }

  it('CLASS GUARD (AIMessage): a field the strip does not name survives, and a chunk stays a chunk', () => {
    const ai = new AIMessage({
      id: 'ai-1',
      name: 'pilot',
      content: [{ type: 'text', text: 'Looking at the frame.' }, markedImageBlock()],
      tool_calls: [{ name: 'turn_right', args: { steps: 1 }, id: 'tc-1' }],
      additional_kwargs: { reasoning_content: 'thought about it' },
      response_metadata: { model_name: 'gpt-5.2', output: [{ type: 'reasoning', id: 'rs_abc123' }] },
    })
    stampUnnamedField(ai, { anything: 'at all' })

    const out = stripImageBlocks(ai) as AIMessage

    // The strip fired…
    expect(hasImage(out.content)).toBe(false)
    expect(out.content).toEqual([{ type: 'text', text: 'Looking at the frame.' }])
    // …and it carried across a field nothing in the implementation mentions.
    expect(readUnnamedField(out)).toEqual({ anything: 'at all' })
    // …along with every field the old literal rebuild forgot.
    expect(out.id).toBe('ai-1')
    expect(out.response_metadata).toEqual({
      model_name: 'gpt-5.2',
      output: [{ type: 'reasoning', id: 'rs_abc123' }],
    })
    expect(out.additional_kwargs).toEqual({ reasoning_content: 'thought about it' })
    // The fields the literal did remember are still correct too.
    expect(out.name).toBe('pilot')
    expect(out.tool_calls?.map((tc) => tc.id)).toEqual(['tc-1'])
    // The caller still holds the input array; the source must be untouched.
    expect(hasImage(ai.content)).toBe(true)
    expect(out).not.toBe(ai)

    // A streamed turn arrives as an AIMessageChunk, which isAIMessage admits and
    // a literal rebuild would flatten into a plain AIMessage. Compared by
    // prototype, never instanceof: a message rebuilt from the wire carries no
    // `Symbol.for('langchain.message')` marker, so a class check fails it while
    // `getType()` answers correctly (RC-21/RC-58).
    const chunk = new AIMessageChunk({
      id: 'ai-chunk',
      content: [{ type: 'text', text: 'partial' }, markedImageBlock()],
      tool_call_chunks: [
        { name: 'turn_left', args: '{"steps":1}', id: 'tc-2', index: 0, type: 'tool_call_chunk' },
      ],
    })
    const outChunk = stripImageBlocks(chunk) as AIMessageChunk
    expect(hasImage(outChunk.content)).toBe(false)
    expect(Object.getPrototypeOf(outChunk)).toBe(Object.getPrototypeOf(chunk))
    expect(outChunk.tool_call_chunks).toEqual([
      { name: 'turn_left', args: '{"steps":1}', id: 'tc-2', index: 0, type: 'tool_call_chunk' },
    ])
  })

  it('CLASS GUARD (HumanMessage): a field the strip does not name survives the injected camera turn', () => {
    const human = new HumanMessage({
      id: 'h-1',
      name: 'operator',
      content: [{ type: 'text', text: 'Camera frame captured:' }, markedImageBlock()],
      additional_kwargs: { capture_ts: 1717, source: 'front_cam' },
      response_metadata: { bridge: 'robot-bridge/2' },
    })
    stampUnnamedField(human, { anything: 'at all' })

    const out = stripImageBlocks(human) as HumanMessage

    expect(hasImage(out.content)).toBe(false)
    expect(out.content).toEqual([{ type: 'text', text: 'Camera frame captured:' }])
    expect(readUnnamedField(out)).toEqual({ anything: 'at all' })
    expect(out.id).toBe('h-1')
    expect(out.additional_kwargs).toEqual({ capture_ts: 1717, source: 'front_cam' })
    expect(out.response_metadata).toEqual({ bridge: 'robot-bridge/2' })
    expect(out.name).toBe('operator')
    expect(Object.getPrototypeOf(out)).toBe(Object.getPrototypeOf(human))
    expect(hasImage(human.content)).toBe(true)
    expect(out).not.toBe(human)
  })

  it('CLASS GUARD (ToolMessage): a field the strip does not name survives, and a FAILED motion stays distinguishable', () => {
    const tool = new ToolMessage({
      id: 'tm-1',
      content: [{ type: 'text', text: 'turn_right (steps=1)' }, markedImageBlock()],
      tool_call_id: 'tc-1',
      name: 'turn_right',
      status: 'error',
      artifact: { frameId: 'FRAMEARTIFACTMARKER', width: 640 },
      response_metadata: { bridge: 'robot-bridge/2', attempt: 2 },
      additional_kwargs: { servo_fault: 'left_hip stalled' },
    })
    stampUnnamedField(tool, { anything: 'at all' })

    const out = stripImageBlocks(tool) as ToolMessage

    // The strip fired, and a ToolMessage still carries its survivors as a JSON
    // string — the transformation is unchanged, only the way it is carried.
    expect(out.content).toBe(JSON.stringify([{ type: 'text', text: 'turn_right (steps=1)' }]))
    expect(readUnnamedField(out)).toEqual({ anything: 'at all' })
    // `status` first: a motion that FAILED must not come back indistinguishable
    // from one that completed.
    expect(out.status).toBe('error')
    expect(out.artifact).toEqual({ frameId: 'FRAMEARTIFACTMARKER', width: 640 })
    expect(out.response_metadata).toEqual({ bridge: 'robot-bridge/2', attempt: 2 })
    expect(out.additional_kwargs).toEqual({ servo_fault: 'left_hip stalled' })
    expect(out.id).toBe('tm-1')
    expect(out.tool_call_id).toBe('tc-1')
    expect(out.name).toBe('turn_right')
    expect(Object.getPrototypeOf(out)).toBe(Object.getPrototypeOf(tool))
    expect(hasImage(tool.content)).toBe(true)
    expect(out).not.toBe(tool)
  })

  it('CLASS GUARD (SystemMessage): a field the strip does not name survives', () => {
    const system = new SystemMessage({
      id: 's-1',
      name: 'harness',
      content: [{ type: 'text', text: 'You control a biped robot.' }, markedImageBlock()],
      additional_kwargs: { profile: 'local-ollama' },
      response_metadata: { composed_by: 'lean-backend' },
    })
    stampUnnamedField(system, { anything: 'at all' })

    const out = stripImageBlocks(system) as SystemMessage

    expect(hasImage(out.content)).toBe(false)
    expect(out.content).toEqual([{ type: 'text', text: 'You control a biped robot.' }])
    expect(readUnnamedField(out)).toEqual({ anything: 'at all' })
    expect(out.id).toBe('s-1')
    expect(out.additional_kwargs).toEqual({ profile: 'local-ollama' })
    expect(out.response_metadata).toEqual({ composed_by: 'lean-backend' })
    expect(out.name).toBe('harness')
    expect(Object.getPrototypeOf(out)).toBe(Object.getPrototypeOf(system))
    expect(hasImage(system.content)).toBe(true)
    expect(out).not.toBe(system)
  })

  it('keeps the caption-less fallback and still preserves the rest of the message', () => {
    // Nothing survives the filter, so the '[image omitted]' placeholder stands in
    // for the slot. That branch builds its own content, which is exactly where a
    // rebuild is most tempting.
    const human = new HumanMessage({
      id: 'h-bare',
      content: [markedImageBlock()],
      additional_kwargs: { source: 'front_cam' },
    })
    stampUnnamedField(human, 'survives the fallback branch too')

    const out = stripImageBlocks(human) as HumanMessage

    expect(out.content).toBe('[image omitted]')
    expect(readUnnamedField(out)).toBe('survives the fallback branch too')
    expect(out.additional_kwargs).toEqual({ source: 'front_cam' })
    expect(out.id).toBe('h-bare')
  })

  it('the dropped frame is unreachable through the copy — content AND the shared lc_kwargs bag', () => {
    // A copy taken from the source's own property descriptors shares `lc_kwargs`
    // with the source BY REFERENCE, and that bag still holds the original
    // content — the frame this function exists to drop. The strip replaces it,
    // and this is the assertion that reds if that line goes: every property
    // assertion above reads the live field and would stay green.
    const human = new HumanMessage({
      id: 'h-1',
      content: [{ type: 'text', text: 'Camera frame captured:' }, markedImageBlock()],
      response_metadata: { bridge: 'HUMANMETAMARKER' },
    })

    const out = stripImageBlocks(human)

    expect(JSON.stringify(out.content)).not.toContain(FRAME_BYTES)
    expect(JSON.stringify(out.lc_kwargs)).not.toContain(FRAME_BYTES)
    // The bag still carries what the strip did not name, so the assertion above
    // cannot be passing on an emptied or absent `lc_kwargs`.
    expect(JSON.stringify(out.lc_kwargs)).toContain('HUMANMETAMARKER')
    // The source is untouched: a different bag, still holding the frame.
    expect(JSON.stringify(human.lc_kwargs)).toContain(FRAME_BYTES)
    expect(out.lc_kwargs).not.toBe(human.lc_kwargs)
  })

  it('returns the same object reference when there is nothing to strip', () => {
    const plainString = new HumanMessage({ id: 'h-str', content: 'no blocks at all' })
    const noImages = new AIMessage({
      id: 'ai-txt',
      content: [{ type: 'text', text: 'text only' }],
    })
    // Not one of the four types the strip transforms. It passes through
    // untouched, images and all — deliberate, and unchanged by RC-30.
    const other = foreignCoreMessage('remove', {
      id: 'rm-1',
      content: [{ type: 'text', text: 'gone' }, markedImageBlock()],
    })

    expect(stripImageBlocks(plainString)).toBe(plainString)
    expect(stripImageBlocks(noImages)).toBe(noImages)
    expect(stripImageBlocks(other)).toBe(other)
  })

  it('a message from the OTHER core copy is stripped, and comes back still foreign', () => {
    // The duck-typed gate is what admits it at all: reverting any of the four
    // checks to `instanceof` would return this message unstripped, and its frame
    // would ride into the summarizer's payload. Copying rather than rebuilding is
    // what keeps it foreign — a literal rebuild would re-mint it under THIS
    // copy's class, quietly changing what the rest of the pipeline holds.
    const foreign = foreignHumanMessage({
      id: 'h-foreign',
      content: [{ type: 'text', text: 'Camera frame captured:' }, markedImageBlock()],
      response_metadata: { bridge: 'robot-bridge/2' },
    })
    expect(isForeignToThisCore(foreign)).toBe(true)

    const out = stripImageBlocks(foreign)

    expect(hasImage(out.content)).toBe(false)
    expect(out.content).toEqual([{ type: 'text', text: 'Camera frame captured:' }])
    expect(isForeignToThisCore(out)).toBe(true)
    expect(out.id).toBe('h-foreign')
    expect(out.response_metadata).toEqual({ bridge: 'robot-bridge/2' })
  })

  it('the preservation holds through buildSummarizationMessages on a fully paired history', () => {
    // The wiring check. The history is paired ON PURPOSE, so this case pins the
    // first stage's wiring alone: `stripUnpairedToolCalls` runs straight after
    // the strip, and when this was written it rebuilt an AIMessage from a literal
    // whenever it dropped a call, so an unpaired fixture would have lost fields
    // for a reason outside this function. RC-64 closed that rebuild; the unpaired
    // shape is covered end to end in the RC-64 block below.
    const user = new HumanMessage({ id: 'u-1', content: 'Find the red cone.' })
    const imageHuman = new HumanMessage({
      id: 'h-1',
      content: [{ type: 'text', text: 'Camera frame captured:' }, markedImageBlock()],
      response_metadata: { bridge: 'robot-bridge/2' },
    })
    stampUnnamedField(imageHuman, { anything: 'at all' })
    const motionAi = new AIMessage({
      id: 'ai-1',
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 1 }, id: 'tc-1' }],
    })
    const motionTool = new ToolMessage({
      id: 'tm-1',
      content: JSON.stringify({ motion: 'turn_right (steps=1)' }),
      tool_call_id: 'tc-1',
      name: 'turn_right',
      status: 'error',
    })

    const built = buildSummarizationMessages(
      [user, imageHuman, motionAi, motionTool],
      'SUMMARY PROMPT'
    )

    // No image block reaches the summarizer…
    for (const m of built) {
      expect(hasImage(m.content)).toBe(false)
    }
    // …and the camera turn arrives with everything it came with.
    const out = built.find((m) => m.id === 'h-1') as HumanMessage
    expect(out).toBeDefined()
    expect(readUnnamedField(out)).toEqual({ anything: 'at all' })
    expect(out.response_metadata).toEqual({ bridge: 'robot-bridge/2' })
    // The paired motion pair survived intact, so the fixture really did take the
    // pass-through branch of stripUnpairedToolCalls.
    expect(built.find((m) => m.id === 'ai-1')).toBe(motionAi)
    expect(built.find((m) => m.id === 'tm-1')).toBe(motionTool)
  })
})

// ───────────────────────────────────────────────────────────────────────────
// RC-64 — stripUnpairedToolCalls copies too, so the WHOLE pipeline preserves
//
// The fifth instance of the class RC-28, RC-29 and RC-30 closed, and the one
// downstream of RC-30's fix: `buildSummarizationMessages` runs `stripImageBlocks`
// and then `stripUnpairedToolCalls`, so until this one was closed "every field
// survives stripImageBlocks" was true while "every field survives the pipeline"
// was not. A reader checking the function RC-30 names would have concluded the
// pipeline was safe.
//
// These guards therefore go THROUGH `buildSummarizationMessages`, which RC-30's
// could not. Each fixture is shaped so every stage that has a path for its type
// fires on it: an image block for the first stage, and for the AIMessage an
// unpaired call the second stage must strip — the shape afterModel actually
// hands over, since it fires on the just-emitted motion call before its tool
// has run. Every preservation assertion sits beside an assertion that the stage
// fired, so a stage turned into the identity cannot pass the preservation half
// for free.
//
// One guard per message type, each asserting on a property the production code
// does not name anywhere, so a literal put back in EITHER stage reds it. The
// second stage rebuilds only AIMessages today; for the other three types it is
// a pass-through, and the guard is what would catch a rebuild being added
// there. RC-30's direct-call guards above are the surviving control and are
// unchanged.
// ───────────────────────────────────────────────────────────────────────────
describe('motionSummarizationMiddleware — RC-64 the whole summarizer pipeline preserves the message', () => {
  const FRAME_BYTES = 'PIPELINEFRAMEMARKER'
  const UNPAIRED = 'tc-unpaired-motion'
  const PAIRED = 'tc-paired-motion'
  const PROMPT = 'SUMMARY PROMPT'

  function stampUnnamedField(msg: BaseMessage, value: unknown): void {
    ;(msg as unknown as Record<string, unknown>).field_no_one_enumerated = value
  }
  function readUnnamedField(msg: BaseMessage): unknown {
    return (msg as unknown as Record<string, unknown>).field_no_one_enumerated
  }
  function markedImageBlock() {
    return {
      type: 'image' as const,
      source_type: 'base64' as const,
      mime_type: 'image/jpeg',
      data: `${FRAME_BYTES}${'X'.repeat(64)}`,
    }
  }
  function hasImage(content: unknown): boolean {
    return (
      Array.isArray(content) &&
      (content as Array<{ type?: string }>).some(
        (b) => b.type === 'image' || b.type === 'image_url'
      )
    )
  }
  function opener() {
    return new HumanMessage({ id: 'u-1', content: 'Find the red cone.' })
  }

  it('CLASS GUARD (AIMessage): the just-emitted motion turn keeps every field through both stages', () => {
    // The real trigger shape: the motion call whose ToolMessage does not exist
    // yet, so the second stage strips the call; the image block makes the first
    // stage fire on the same message. Text survives, so the message is copied
    // rather than dropped.
    const motionAi = new AIMessage({
      id: 'ai-1',
      name: 'pilot',
      content: [{ type: 'text', text: 'Turning right now.' }, markedImageBlock()],
      tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: UNPAIRED }],
      additional_kwargs: { reasoning_content: 'thought about it' },
      response_metadata: { model_name: 'gpt-5.2', output: [{ type: 'reasoning', id: 'rs_abc123' }] },
      usage_metadata: { input_tokens: 11, output_tokens: 7, total_tokens: 18 },
    })
    stampUnnamedField(motionAi, { anything: 'at all' })

    const built = buildSummarizationMessages([opener(), motionAi], PROMPT)
    const out = built.find((m) => m.id === 'ai-1') as AIMessage
    expect(out).toBeDefined()

    // Both stages fired on this message…
    expect(hasImage(out.content)).toBe(false)
    expect(out.content).toEqual([{ type: 'text', text: 'Turning right now.' }])
    expect(collectToolUseIds(out)).toEqual([])
    // …and it still carries a field nothing in the implementation names.
    expect(readUnnamedField(out)).toEqual({ anything: 'at all' })
    // …along with every field the five-field literal forgot.
    expect(out.response_metadata).toEqual({
      model_name: 'gpt-5.2',
      output: [{ type: 'reasoning', id: 'rs_abc123' }],
    })
    expect(out.usage_metadata).toEqual({ input_tokens: 11, output_tokens: 7, total_tokens: 18 })
    expect(out.invalid_tool_calls).toEqual([])
    // The fields the literal did remember are still correct too.
    expect(out.id).toBe('ai-1')
    expect(out.name).toBe('pilot')
    expect(out.additional_kwargs).toEqual({ reasoning_content: 'thought about it' })
    expect(Object.getPrototypeOf(out)).toBe(Object.getPrototypeOf(motionAi))
    // The caller still holds the input; the source must be untouched.
    expect(hasImage(motionAi.content)).toBe(true)
    expect(collectToolUseIds(motionAi)).toEqual([UNPAIRED])
    expect(out).not.toBe(motionAi)
  })

  it('CLASS GUARD (AIMessage, streamed): the strip clears the unpaired call from tool_call_chunks too, while a PAIRED chunk survives', () => {
    // A streamed turn arrives as an AIMessageChunk, which isAIMessage admits and
    // a literal rebuild flattened into a plain AIMessage. Compared by prototype
    // against the input, never against a class.
    //
    // `tool_call_chunks` is the THIRD representation of a tool call, and the
    // strip has to reach it: `toJSON()`, `toDict()` and `concat()` all resolve
    // the call from there or from the `lc_kwargs` bag, so a strip that clears
    // only `.tool_calls` puts the unpaired call straight back into anything
    // that serializes the summarizer input (RC-65).
    //
    // TWO chunks, one resolved and one not, because the assertion has to
    // DISCRIMINATE: a guard that only checked the unpaired one would be passed
    // by a function that stripped the field wholesale.
    const chunk = new AIMessageChunk({
      id: 'ai-chunk',
      content: [{ type: 'text', text: 'partial' }, markedImageBlock()],
      tool_call_chunks: [
        { name: 'turn_left', args: '{"steps":1}', id: PAIRED, index: 0, type: 'tool_call_chunk' },
        { name: 'turn_right', args: '{"steps":2}', id: UNPAIRED, index: 1, type: 'tool_call_chunk' },
      ],
      usage_metadata: { input_tokens: 3, output_tokens: 2, total_tokens: 5 },
    })
    stampUnnamedField(chunk, { anything: 'at all' })
    const pairedResult = new ToolMessage({
      id: 'tm-paired',
      content: 'turn_left done',
      tool_call_id: PAIRED,
      name: 'turn_left',
    })
    // The chunk's own constructor collapsed both chunks into live tool calls
    // keeping their ids, so the second stage has something to strip and
    // something to keep.
    expect(collectToolUseIds(chunk)).toEqual([PAIRED, UNPAIRED])
    expect(chunk.tool_call_chunks?.map((c) => c.id)).toEqual([PAIRED, UNPAIRED])

    const built = buildSummarizationMessages([opener(), chunk, pairedResult], PROMPT)
    const out = built.find((m) => m.id === 'ai-chunk') as AIMessageChunk
    expect(out).toBeDefined()

    expect(hasImage(out.content)).toBe(false)
    // The paired call survives in `.tool_calls`; the unpaired one is gone.
    expect(collectToolUseIds(out)).toEqual([PAIRED])
    expect(readUnnamedField(out)).toEqual({ anything: 'at all' })
    expect(Object.getPrototypeOf(out)).toBe(Object.getPrototypeOf(chunk))

    // The third representation is filtered on the same ids, and the guard
    // discriminates: the paired chunk is still here, the unpaired one is not.
    expect(out.tool_call_chunks).toEqual([
      { name: 'turn_left', args: '{"steps":1}', id: PAIRED, index: 0, type: 'tool_call_chunk' },
    ])

    // The unpaired call is unreachable through the copy in each representation
    // this function filters — the live field, the shared `lc_kwargs` bag, and
    // both serialized forms that resolve from it — while the paired call is
    // still reachable in every one of them, so none of these can be passing on
    // an emptied or absent field.
    for (const [what, value] of [
      ['tool_call_chunks', out.tool_call_chunks],
      ['lc_kwargs', out.lc_kwargs],
      ['toJSON()', out.toJSON()],
      ['toDict()', out.toDict()],
    ] as Array<[string, unknown]>) {
      expect(JSON.stringify(value), `${what} must not carry the unpaired call`).not.toContain(
        UNPAIRED
      )
      expect(JSON.stringify(value), `${what} must still carry the paired call`).toContain(PAIRED)
    }

    // The node's stated forward risk: `concat()` rebuilds `.tool_calls` from
    // `tool_call_chunks`, so an unstripped chunk array resurrects the unpaired
    // call in a merged message even though the copy's `.tool_calls` looked clean.
    const merged = out.concat(new AIMessageChunk({ content: '' }))
    expect(JSON.stringify(merged.tool_calls)).not.toContain(UNPAIRED)
    expect(JSON.stringify(merged.tool_call_chunks)).not.toContain(UNPAIRED)
    expect(JSON.stringify(merged.tool_calls)).toContain(PAIRED)

    // Every field the function does not rewrite is still carried across whole.
    expect(out.usage_metadata).toEqual({ input_tokens: 3, output_tokens: 2, total_tokens: 5 })
    expect(out).not.toBe(chunk)

    // The caller still holds the input array; the source must be untouched, in
    // the live field and in its own bag.
    expect(chunk.tool_call_chunks?.map((c) => c.id)).toEqual([PAIRED, UNPAIRED])
    expect(JSON.stringify(chunk.lc_kwargs)).toContain(UNPAIRED)
    expect(out.lc_kwargs).not.toBe(chunk.lc_kwargs)

    // The paired call's own result survived too, so the fixture really did take
    // the keep branch rather than dropping the pair wholesale.
    expect(built.find((m) => m.id === 'tm-paired')).toBe(pairedResult)
  })

  it('a chunk is stripped when `.tool_calls` is empty and only the chunk and invalid arrays disagree', () => {
    // When a streamed chunk's `args` are unparseable the constructor routes the
    // call to `invalid_tool_calls` and leaves `.tool_calls` EMPTY. So
    // `keptCalls.length === calls.length` holds and the content is unchanged,
    // and the unpaired call is carried ONLY by the third and fourth
    // representations — the case that decides whether the early "nothing
    // changed" return has to consider them. Without that, this message comes
    // back as the same instance with the unpaired call still on it.
    //
    // The title says "the chunk and invalid arrays" rather than naming one of
    // them alone because this fixture disagrees in BOTH: the constructor writes
    // the malformed call to `invalid_tool_calls` as well as keeping the chunk.
    // The RC-67 describe block below covers the invalid array disagreeing ALONE.
    const chunk = new AIMessageChunk({
      id: 'ai-chunk',
      content: [{ type: 'text', text: 'partial' }],
      tool_call_chunks: [
        {
          name: 'turn_left',
          args: 'not json at all',
          id: UNPAIRED,
          index: 0,
          type: 'tool_call_chunk',
        },
      ],
    })
    // The premise of the fixture, asserted rather than assumed: nothing for the
    // `.tool_calls` filter to do.
    expect(chunk.tool_calls).toEqual([])
    expect(chunk.tool_call_chunks?.map((c) => c.id)).toEqual([UNPAIRED])

    const [out] = stripUnpairedToolCalls([chunk]) as AIMessageChunk[]

    // A copy was taken, which only happens if the chunk disagreement counted as
    // a change at all.
    expect(out).not.toBe(chunk)
    expect(JSON.stringify(out.tool_call_chunks)).not.toContain(UNPAIRED)
    expect(JSON.stringify(out.lc_kwargs.tool_call_chunks)).not.toContain(UNPAIRED)
    // Text content is what keeps the message alive through the drop check.
    expect(out.content).toEqual([{ type: 'text', text: 'partial' }])
    // The source is untouched.
    expect(chunk.tool_call_chunks?.map((c) => c.id)).toEqual([UNPAIRED])

    // RC-67 FLIPPED THIS. It stood as the pin on a known residual — the
    // unpaired malformed call kept its id in `invalid_tool_calls`, the FOURTH
    // representation — and RC-67 filters that field on the same `resolvedIds`
    // set, so the id is now gone from the live field and from the bag that
    // feeds both serialized forms.
    expect(JSON.stringify(out.invalid_tool_calls)).not.toContain(UNPAIRED)
    expect(JSON.stringify(out.lc_kwargs.invalid_tool_calls)).not.toContain(UNPAIRED)
    expect(JSON.stringify(out.toJSON())).not.toContain(UNPAIRED)
    expect(JSON.stringify(out.toDict())).not.toContain(UNPAIRED)
    // The source is untouched in the fourth representation too.
    expect(JSON.stringify(chunk.invalid_tool_calls)).toContain(UNPAIRED)
  })

  it('a plain AIMessage does not ACQUIRE a tool_call_chunks it never had', () => {
    // `tool_call_chunks` exists only on AIMessageChunk, so the strip writes it
    // to the copy and to the `lc_kwargs` bag only when the source actually
    // carried it. Unguarded, a plain AIMessage would come back with an OWN
    // `tool_call_chunks` property whose value is `undefined` — which changes
    // what the message serializes to and what a later `concat()` or converter
    // sees on a shape that never had the field.
    //
    // Asserted on OWN-PROPERTY PRESENCE, never on the value: `undefined` is
    // exactly what an unguarded assignment writes, so `toBeUndefined()` would
    // pass whether or not the guard is there and could not fail.
    const ai = new AIMessage({
      id: 'ai-1',
      content: [{ type: 'text', text: 'Turning now.' }],
      tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: UNPAIRED }],
      response_metadata: { bridge: 'AIMETAMARKER' },
    })
    const hasOwn = (o: object, k: string) => Object.prototype.hasOwnProperty.call(o, k)
    expect(hasOwn(ai, 'tool_call_chunks')).toBe(false)

    const [out] = stripUnpairedToolCalls([ai]) as AIMessage[]

    // The fixture must take the COPY path, or the assignment never runs and the
    // assertions below would hold for the wrong reason.
    expect(out).not.toBe(ai)
    expect(collectToolUseIds(out)).toEqual([])

    // Neither as an own property nor anywhere on the prototype chain.
    expect(hasOwn(out, 'tool_call_chunks')).toBe(false)
    expect('tool_call_chunks' in out).toBe(false)
    // And the bag is not given the key either — it carries its own conditional.
    expect(hasOwn(out.lc_kwargs, 'tool_call_chunks')).toBe(false)
    // The keys the strip DOES rewrite are present, so none of the above is
    // passing on an emptied or unbuilt copy.
    expect(hasOwn(out.lc_kwargs, 'tool_calls')).toBe(true)
    expect(hasOwn(out.lc_kwargs, 'content')).toBe(true)
    expect(JSON.stringify(out.lc_kwargs)).toContain('AIMETAMARKER')
  })

  it('CLASS GUARD (HumanMessage): the injected camera turn keeps every field through the pipeline', () => {
    const camera = new HumanMessage({
      id: 'h-1',
      name: 'operator',
      content: [{ type: 'text', text: 'Camera frame captured:' }, markedImageBlock()],
      additional_kwargs: { capture_ts: 1717, source: 'front_cam' },
      response_metadata: { bridge: 'robot-bridge/2' },
    })
    stampUnnamedField(camera, { anything: 'at all' })

    const built = buildSummarizationMessages([opener(), camera], PROMPT)
    const out = built.find((m) => m.id === 'h-1') as HumanMessage
    expect(out).toBeDefined()

    expect(hasImage(out.content)).toBe(false)
    expect(out.content).toEqual([{ type: 'text', text: 'Camera frame captured:' }])
    expect(readUnnamedField(out)).toEqual({ anything: 'at all' })
    expect(out.id).toBe('h-1')
    expect(out.additional_kwargs).toEqual({ capture_ts: 1717, source: 'front_cam' })
    expect(out.response_metadata).toEqual({ bridge: 'robot-bridge/2' })
    expect(out.name).toBe('operator')
    expect(Object.getPrototypeOf(out)).toBe(Object.getPrototypeOf(camera))
    expect(hasImage(camera.content)).toBe(true)
    expect(out).not.toBe(camera)
  })

  it('CLASS GUARD (ToolMessage): a paired FAILED motion result keeps every field through the pipeline', () => {
    // Paired on purpose: the second stage keeps a ToolMessage only while its
    // tool_use survives, so this is the branch that carries one to the output.
    const motionAi = new AIMessage({
      id: 'ai-1',
      content: '',
      tool_calls: [{ name: 'turn_right', args: { steps: 1 }, id: 'tc-1' }],
    })
    const tool = new ToolMessage({
      id: 'tm-1',
      content: [{ type: 'text', text: 'turn_right (steps=1)' }, markedImageBlock()],
      tool_call_id: 'tc-1',
      name: 'turn_right',
      status: 'error',
      artifact: { frameId: 'FRAMEARTIFACTMARKER', width: 640 },
      response_metadata: { bridge: 'robot-bridge/2', attempt: 2 },
      additional_kwargs: { servo_fault: 'left_hip stalled' },
    })
    stampUnnamedField(tool, { anything: 'at all' })

    const built = buildSummarizationMessages([opener(), motionAi, tool], PROMPT)
    const out = built.find((m) => m.id === 'tm-1') as ToolMessage
    expect(out).toBeDefined()

    // The first stage fired: survivors carried as a JSON string, as before.
    expect(out.content).toBe(JSON.stringify([{ type: 'text', text: 'turn_right (steps=1)' }]))
    expect(readUnnamedField(out)).toEqual({ anything: 'at all' })
    // `status` first: a motion that FAILED must not come back indistinguishable
    // from one that completed.
    expect(out.status).toBe('error')
    expect(out.artifact).toEqual({ frameId: 'FRAMEARTIFACTMARKER', width: 640 })
    expect(out.response_metadata).toEqual({ bridge: 'robot-bridge/2', attempt: 2 })
    expect(out.additional_kwargs).toEqual({ servo_fault: 'left_hip stalled' })
    expect(out.id).toBe('tm-1')
    expect(out.tool_call_id).toBe('tc-1')
    expect(out.name).toBe('turn_right')
    expect(Object.getPrototypeOf(out)).toBe(Object.getPrototypeOf(tool))
    expect(hasImage(tool.content)).toBe(true)
    expect(out).not.toBe(tool)
    // Its pair survived as the same instance, so the fixture took the keep
    // branch of the second stage and not the drop.
    expect(built.find((m) => m.id === 'ai-1')).toBe(motionAi)
  })

  it('CLASS GUARD (SystemMessage): a leading system turn keeps every field through the pipeline', () => {
    const system = new SystemMessage({
      id: 's-1',
      name: 'harness',
      content: [{ type: 'text', text: 'You control a biped robot.' }, markedImageBlock()],
      additional_kwargs: { profile: 'local-ollama' },
      response_metadata: { composed_by: 'lean-backend' },
    })
    stampUnnamedField(system, { anything: 'at all' })

    const built = buildSummarizationMessages([system, opener()], PROMPT)
    // The pipeline prepends its own SystemMessage, so find by id, not position.
    const out = built.find((m) => m.id === 's-1') as SystemMessage
    expect(out).toBeDefined()

    expect(hasImage(out.content)).toBe(false)
    expect(out.content).toEqual([{ type: 'text', text: 'You control a biped robot.' }])
    expect(readUnnamedField(out)).toEqual({ anything: 'at all' })
    expect(out.id).toBe('s-1')
    expect(out.additional_kwargs).toEqual({ profile: 'local-ollama' })
    expect(out.response_metadata).toEqual({ composed_by: 'lean-backend' })
    expect(out.name).toBe('harness')
    expect(Object.getPrototypeOf(out)).toBe(Object.getPrototypeOf(system))
    expect(hasImage(system.content)).toBe(true)
    expect(out).not.toBe(system)
  })

  it('the stripped call is unreachable through the copy — tool_calls, content AND the shared lc_kwargs bag', () => {
    // A copy taken from the source's own property descriptors shares `lc_kwargs`
    // with the source BY REFERENCE, and that bag still holds the unpaired call in
    // both representations. The strip replaces it, and this is the assertion
    // that reds if that line goes: every property assertion above reads the live
    // field and would stay green.
    const ai = new AIMessage({
      id: 'ai-1',
      content: [
        { type: 'text', text: 'Turning now.' },
        { type: 'tool_use', id: UNPAIRED, name: 'turn_right', input: { steps: 3 } },
      ] as unknown as AIMessage['content'],
      tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: UNPAIRED }],
      response_metadata: { bridge: 'AIMETAMARKER' },
    })

    const [out] = stripUnpairedToolCalls([ai]) as AIMessage[]

    expect(JSON.stringify(out.tool_calls)).not.toContain(UNPAIRED)
    expect(JSON.stringify(out.content)).not.toContain(UNPAIRED)
    expect(JSON.stringify(out.lc_kwargs)).not.toContain(UNPAIRED)
    // The bag still carries what the strip did not name, so the assertion above
    // cannot be passing on an emptied or absent `lc_kwargs`.
    expect(JSON.stringify(out.lc_kwargs)).toContain('AIMETAMARKER')
    // The source is untouched: a different bag, still holding the call.
    expect(JSON.stringify(ai.lc_kwargs)).toContain(UNPAIRED)
    expect(out.lc_kwargs).not.toBe(ai.lc_kwargs)
  })

  it('a message from the OTHER core copy is stripped, and comes back still foreign', () => {
    // The duck-typed gate is what admits it at all; copying rather than
    // rebuilding is what keeps it foreign — a literal rebuild would re-mint it
    // under THIS copy's class, quietly changing what the rest of the pipeline
    // holds.
    const foreign = foreignAIMessage({
      id: 'ai-foreign',
      content: [{ type: 'text', text: 'Turning now.' }],
      tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: UNPAIRED }],
      response_metadata: { bridge: 'robot-bridge/2' },
    })
    expect(isForeignToThisCore(foreign)).toBe(true)

    const [out] = stripUnpairedToolCalls([foreign])

    expect(collectToolUseIds(out as AIMessage)).toEqual([])
    expect(out.content).toEqual([{ type: 'text', text: 'Turning now.' }])
    expect(isForeignToThisCore(out)).toBe(true)
    expect(out.id).toBe('ai-foreign')
    expect(out.response_metadata).toEqual({ bridge: 'robot-bridge/2' })
  })
})

describe('motionSummarizationMiddleware — RC-67 invalid_tool_calls, the fourth representation', () => {
  const MALFORMED = 'tc-malformed-motion'
  const hasOwn = (o: object, k: string) => Object.prototype.hasOwnProperty.call(o, k)

  // A settled AIMessage carrying `invalid_tool_calls` DIRECTLY, with no
  // `tool_call_chunks` at all — the shape a rebuilt/settled message takes, and
  // the one where the invalid array is the ONLY representation that disagrees.
  function settledWithInvalid(id: string, content: unknown) {
    return new AIMessage({
      id: 'ai-settled',
      content: content as never,
      invalid_tool_calls: [
        { name: 'turn_left', args: 'not json at all', id, error: 'Malformed args.' },
      ],
    })
  }

  it('an UNPAIRED malformed call is stripped when the invalid array is the ONLY representation that disagrees', () => {
    // No `tool_call_chunks`, `.tool_calls` empty, content unchanged. Every other
    // representation agrees, so this message reaches the strip only because
    // `invalidChanged` joins the early-return guard. Without that it comes back
    // as the same instance with the unpaired malformed call still on it — which
    // is the half-stripped state RC-65 closed for the chunk array.
    const ai = settledWithInvalid(MALFORMED, [{ type: 'text', text: 'partial' }])
    // Premises, asserted rather than assumed.
    expect(ai.tool_calls).toEqual([])
    expect(hasOwn(ai, 'tool_call_chunks')).toBe(false)
    expect(ai.invalid_tool_calls?.map((c) => c.id)).toEqual([MALFORMED])

    const [out] = stripUnpairedToolCalls([ai]) as AIMessage[]

    // A copy was taken, which only happens if the invalid-array disagreement
    // counted as a change at all.
    expect(out).not.toBe(ai)
    expect(out.invalid_tool_calls).toEqual([])
    expect(JSON.stringify(out.invalid_tool_calls)).not.toContain(MALFORMED)
    expect(JSON.stringify(out.lc_kwargs.invalid_tool_calls)).not.toContain(MALFORMED)
    // Both serialized forms resolve from that bag, so they must be clean too.
    expect(JSON.stringify(out.toJSON())).not.toContain(MALFORMED)
    expect(JSON.stringify(out.toDict())).not.toContain(MALFORMED)
    // Text content is what keeps the message alive through the drop check.
    expect(out.content).toEqual([{ type: 'text', text: 'partial' }])
    // The caller still holds the input; the source must be untouched.
    expect(ai.invalid_tool_calls?.map((c) => c.id)).toEqual([MALFORMED])
    expect(JSON.stringify(ai.lc_kwargs)).toContain(MALFORMED)
    expect(out.lc_kwargs).not.toBe(ai.lc_kwargs)
  })

  it('DISCRIMINATES: a RESOLVED malformed call keeps its invalid_tool_calls entry', () => {
    // The guard must discriminate, not just pass: the filter is keyed on
    // `resolvedIds`, so a malformed call whose `tool_result` is present is
    // KEPT. Without this, "strip everything" would satisfy the test above.
    // Two malformed calls: one resolved, one genuinely unpaired — so the
    // message still changes and the copy path still runs. With only the
    // resolved one this would pass by identity and prove nothing about the
    // filter.
    const ai = new AIMessage({
      id: 'ai-settled',
      content: [{ type: 'text', text: 'partial' }],
      invalid_tool_calls: [
        { name: 'turn_left', args: 'not json at all', id: MALFORMED, error: 'Malformed args.' },
        {
          name: 'turn_right',
          args: 'also not json',
          id: 'tc-malformed-unpaired',
          error: 'Malformed args.',
        },
      ],
    })
    const result = new ToolMessage({ id: 'tm-mal', content: 'turned', tool_call_id: MALFORMED })

    const out = stripUnpairedToolCalls([ai, result])
    const outAi = out.find((m) => m.id === 'ai-settled') as AIMessage

    expect(outAi).toBeDefined()
    // The resolved one survives in the live field AND in the bag; the unpaired
    // one is gone from both.
    expect(outAi.invalid_tool_calls?.map((c) => c.id)).toEqual([MALFORMED])
    expect(JSON.stringify(outAi.lc_kwargs.invalid_tool_calls)).toContain(MALFORMED)
    expect(JSON.stringify(outAi.lc_kwargs.invalid_tool_calls)).not.toContain(
      'tc-malformed-unpaired'
    )
  })

  it('a WELL-FORMED resolved call is untouched by the fourth-representation filter', () => {
    // The discrimination that matters most: the new filter must not disturb the
    // ordinary paired path. Returned BY IDENTITY, with its `tool_result`.
    const ai = new AIMessage({
      id: 'ai-ok',
      content: [{ type: 'text', text: 'Turning now.' }],
      tool_calls: [{ name: 'turn_left', args: { steps: 1 }, id: 'tc-ok' }],
    })
    const result = new ToolMessage({ id: 'tm-ok', content: 'turned', tool_call_id: 'tc-ok' })

    const out = stripUnpairedToolCalls([ai, result])

    expect(out).toHaveLength(2)
    expect(out[0]).toBe(ai)
    expect(out[1]).toBe(result)
  })

  it('DELIBERATE: a PAIRED malformed call keeps its entry and STILL loses its tool_result', () => {
    // RC-67 resolution (b), and this asserts it rather than leaving it to a
    // comment. Measured: no input converter of any provider this repo
    // constructs reads `invalid_tool_calls`, so a malformed call puts no
    // `tool_use` on the wire — and KEEPING its `tool_result` would emit an
    // orphan `tool_result`, which is the INVALID_TOOL_RESULTS shape itself.
    //
    // Mechanically: `.tool_calls` is empty, so the message takes the identity
    // return and records nothing into `keptCallIds`; the ToolMessage arm then
    // drops the result. Reproduced with this fixture before the change and
    // unchanged by it.
    const chunk = new AIMessageChunk({
      id: 'ai-chunk',
      content: [{ type: 'text', text: 'partial' }],
      tool_call_chunks: [
        {
          name: 'turn_left',
          args: 'not json at all',
          id: MALFORMED,
          index: 0,
          type: 'tool_call_chunk',
        },
      ],
    })
    const result = new ToolMessage({ id: 'tm-mal', content: 'turned', tool_call_id: MALFORMED })
    expect(chunk.tool_calls).toEqual([])
    expect(chunk.invalid_tool_calls?.map((c) => c.id)).toEqual([MALFORMED])

    const out = stripUnpairedToolCalls([chunk, result])

    // The AI message comes back BY IDENTITY — nothing in any of the four
    // representations disagreed, because the malformed call's id is resolved.
    expect(out).toHaveLength(1)
    expect(out[0]).toBe(chunk)
    expect(JSON.stringify((out[0] as AIMessageChunk).invalid_tool_calls)).toContain(MALFORMED)
    // And the `tool_result` is dropped. This is the deliberate half.
    expect(out.find((m) => (m as ToolMessage).tool_call_id === MALFORMED)).toBeUndefined()
  })

  it('an AI message left with nothing but a stripped malformed call is dropped entirely', () => {
    // The new branch `invalidChanged` opens: with EMPTY content and no valid
    // calls, this message used to leave by the identity return (carrying the
    // unpaired malformed call), and now falls through to the drop check — where
    // an AIMessage with no content and no surviving call is dropped, because an
    // empty AIMessage is itself invalid for Anthropic.
    const ai = settledWithInvalid(MALFORMED, '')
    expect(ai.tool_calls).toEqual([])

    const out = stripUnpairedToolCalls([ai])

    expect(out).toEqual([])
  })

  it('an ID-LESS invalid call is dropped by the same predicate', () => {
    // A call with no id is one no `tool_result` can ever pair with, so it is
    // dropped on the same terms as an id-less chunk. Asserted because the
    // `c.id != null` half of the predicate is otherwise unpinned.
    const ai = new AIMessage({
      id: 'ai-settled',
      content: [{ type: 'text', text: 'partial' }],
      invalid_tool_calls: [
        { name: 'turn_left', args: 'not json at all', id: undefined, error: 'Malformed args.' },
      ],
    })
    expect(ai.invalid_tool_calls).toHaveLength(1)

    const [out] = stripUnpairedToolCalls([ai]) as AIMessage[]

    expect(out).not.toBe(ai)
    expect(out.invalid_tool_calls).toEqual([])
    expect(out.lc_kwargs.invalid_tool_calls).toEqual([])
  })

  it('a message that never carried invalid_tool_calls does not ACQUIRE one', () => {
    // `invalid_tool_calls` is written to the copy and to the bag only when the
    // source actually carried it. Every message the constructor builds has the
    // field (it defaults to `[]`), so the shape that can expose an unguarded
    // write is one rebuilt from the wire — which this repo reads duck-typed
    // precisely because a class check would reject it (RC-21, RC-58).
    //
    // Asserted on OWN-PROPERTY PRESENCE, never on the value: an unguarded
    // assignment writes the VALUE `undefined`, and reading an ABSENT property
    // also returns `undefined`, so `toBeUndefined()` would pass either way and
    // could not fail.
    const wire = {
      _getType: () => 'ai' as const,
      id: 'ai-wire',
      content: [{ type: 'text', text: 'Turning now.' }],
      tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-unpaired-wire' }],
      lc_kwargs: { content: [{ type: 'text', text: 'Turning now.' }], tool_calls: [] },
    }
    expect(hasOwn(wire, 'invalid_tool_calls')).toBe(false)

    const [out] = stripUnpairedToolCalls([wire as unknown as BaseMessage]) as AIMessage[]

    // The fixture must take the COPY path, or the assignment never runs and the
    // assertions below would hold for the wrong reason.
    expect(out).not.toBe(wire)
    expect(out.tool_calls).toEqual([])

    // Neither as an own property nor anywhere on the prototype chain.
    expect(hasOwn(out, 'invalid_tool_calls')).toBe(false)
    expect('invalid_tool_calls' in out).toBe(false)
    // And the bag is not given the key either — it carries its own conditional.
    expect(hasOwn(out.lc_kwargs, 'invalid_tool_calls')).toBe(false)
    // The keys the strip DOES rewrite are present, so none of the above is
    // passing on an emptied or unbuilt copy.
    expect(hasOwn(out.lc_kwargs, 'tool_calls')).toBe(true)
    expect(hasOwn(out.lc_kwargs, 'content')).toBe(true)
  })
})
