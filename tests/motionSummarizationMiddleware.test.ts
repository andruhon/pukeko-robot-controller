import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  RemoveMessage,
  type BaseMessage,
} from '@langchain/core/messages'
import {
  createMotionSummarizationMiddleware,
  __pendingSummariesForTest,
  __motionLogForTest,
} from '../src/agent/motionSummarizationMiddleware.js'

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
    expect(updated[0]).toBeInstanceOf(RemoveMessage)
    // Next entry is the original user message verbatim.
    expect(updated[1]).toBeInstanceOf(HumanMessage)
    expect((updated[1] as HumanMessage).content).toBe('Get the robot to the red cone.')
    // Then the summary as a SystemMessage.
    expect(updated[2]).toBeInstanceOf(SystemMessage)
    expect((updated[2] as SystemMessage).content).toContain(SUMMARY_TEXT)
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
      (m) => m instanceof HumanMessage && m.content === 'Find the red cone.'
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
    expect(sanitizedInput[0]).toBeInstanceOf(SystemMessage)
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
    const summaryMsg = updated[2] as SystemMessage
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
})
