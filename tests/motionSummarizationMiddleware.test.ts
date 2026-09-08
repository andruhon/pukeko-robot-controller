import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  type BaseMessage,
} from '@langchain/core/messages'
import {
  createMotionSummarizationMiddleware,
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
import { foreignHumanMessage, isForeignToThisCore } from './helpers/foreignCoreMessage.js'

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
