// RC-24: the summarization sub-calls send a transcript and NO `tools`, so they
// must not carry the agent model's baked-in tool-only request params.
//
// Measured live against gpt-4.1-mini (Chat Completions) with the exact model
// `server/createLlm.ts` builds — `new ChatOpenAI({ modelKwargs: {
// parallel_tool_calls: false, tool_choice: 'auto' } })` — and the exact message
// shape `runSummary` assembles:
//   * with the baked-in params → 400 "Invalid value for 'tool_choice':
//     'tool_choice' is only allowed when 'tools' are specified."
//   * with only `tool_choice` removed → 400 "Invalid value for
//     'parallel_tool_calls': 'parallel_tool_calls' is only allowed when 'tools'
//     are specified."  (so removing tool_choice alone is NOT enough)
//   * with both removed → 200, even though the transcript still contains
//     assistant tool_calls and role:"tool" results. Sending tool-protocol
//     content without a `tools` array is fine; sending the *params* is not.
// Anthropic accepted all three, so the defect is OpenAI-shaped (the robot
// default profile) — but the fix is provider-agnostic.
import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  isAIMessage,
  isToolMessage,
  type BaseMessage,
} from '@langchain/core/messages'
import { ChatOpenAI } from '@langchain/openai'
import { ChatAnthropic } from '@langchain/anthropic'
import { ChatOpenRouter } from '@langchain/openrouter'
import { ChatOllama } from '@langchain/ollama'
import { ChatGoogle } from '@langchain/google/node'
import { createContextPrunerMiddleware } from '../src/agent/contextPrunerMiddleware.js'
import {
  createMotionSummarizationMiddleware,
  __pendingSummariesForTest,
} from '../src/agent/motionSummarizationMiddleware.js'
import { createLazyToolRecoveryMiddleware } from '../src/agent/lazyToolRecoveryMiddleware.js'
import { toolFreeModel } from '../src/agent/toolFreeModel.js'
import { __resetMotionLogForTest } from '../src/agent/motionLog.js'
import { ScriptedRobotChatModel } from '../server/test-support/scriptedRobotModel.js'

type AnyLlm = Parameters<typeof createContextPrunerMiddleware>[0]['llm']

interface HookContainer {
  beforeModel?: unknown
  afterModel?: unknown
  wrapModelCall?: unknown
}

// Duck-typed, not `instanceof RemoveMessage`: a message rebuilt from the wire
// carries no `Symbol.for('langchain.message')` marker, so a class check fails it
// while `getType()` answers correctly, and this filter would silently become a
// no-op.
function isRemoveMessage(m: BaseMessage): boolean {
  return (m as { getType?: () => string }).getType?.() === 'remove'
}

function getHook(hook: unknown): (state: unknown, runtime: unknown) => unknown {
  if (typeof hook === 'function') return hook as (state: unknown, runtime: unknown) => unknown
  if (hook && typeof hook === 'object' && 'hook' in hook && typeof (hook as { hook: unknown }).hook === 'function') {
    return (hook as { hook: (state: unknown, runtime: unknown) => unknown }).hook
  }
  throw new Error('Hook not callable')
}

const runtime = { configurable: { thread_id: 'test-thread' } }

// ── The oracle ──────────────────────────────────────────────────────────────
// Encodes the two 400s measured above. Given the request params a sub-call
// would put on the wire, returns the provider's rejection reason, or null when
// the request is well-formed.
export function openAiRejection(params: Record<string, unknown>): string | null {
  const hasTools = Array.isArray(params.tools) && params.tools.length > 0
  if (!hasTools && params.tool_choice !== undefined) {
    return "'tool_choice' is only allowed when 'tools' are specified"
  }
  if (!hasTools && params.parallel_tool_calls !== undefined) {
    return "'parallel_tool_calls' is only allowed when 'tools' are specified"
  }
  return null
}

// ── A stand-in for a provider chat model ────────────────────────────────────
// Mirrors the two things `toolFreeModel` relies on and every real provider
// model has: `lc_kwargs` (Serializable's record of the constructor fields) and
// a constructor that accepts them back. `modelKwargs` is what a ChatOpenAI /
// ChatOpenRouter spreads into the request body.
interface FakeCall {
  modelKwargs: Record<string, unknown>
  invocationKwargs: Record<string, unknown>
  messages: BaseMessage[]
}

class FakeToolBoundModel {
  static calls: FakeCall[] = []
  lc_kwargs: Record<string, unknown>
  modelKwargs: Record<string, unknown>
  invocationKwargs: Record<string, unknown>
  // What a successful sub-call returns. A constructor field rather than a
  // static, so it rides through `lc_kwargs` on a rebuild exactly like every
  // other field; the summarizer sub-calls want summary prose, the lazy-tool
  // classifier wants a YES/NO verdict.
  reply: string

  constructor(fields: Record<string, unknown>) {
    this.lc_kwargs = fields
    this.modelKwargs = (fields.modelKwargs as Record<string, unknown>) ?? {}
    this.invocationKwargs = (fields.invocationKwargs as Record<string, unknown>) ?? {}
    this.reply = (fields.reply as string) ?? 'Robot is south of the cone, facing west.'
  }

  // The request body a real OpenAI-shaped provider would send: the tool-less
  // sub-call contributes no `tools`, and modelKwargs is merged in last.
  requestParams(): Record<string, unknown> {
    return { model: this.lc_kwargs.model, ...this.modelKwargs }
  }

  async invoke(messages: BaseMessage[]) {
    FakeToolBoundModel.calls.push({
      modelKwargs: this.modelKwargs,
      invocationKwargs: this.invocationKwargs,
      messages,
    })
    // Behave like the real endpoint: reject the request the same way OpenAI
    // did in the live measurement above, so a caller that leaves the tool-only
    // params on a tool-less call gets the same swallowed failure it gets in
    // production, not a convenient success.
    const rejection = openAiRejection(this.requestParams())
    if (rejection) {
      throw new Error(`400 Invalid value: ${rejection}.`)
    }
    return { content: this.reply }
  }
}

// Exactly what server/createLlm.ts builds for the `openai` provider.
function agentModel(): AnyLlm {
  return new FakeToolBoundModel({
    model: 'gpt-5.5',
    modelKwargs: { parallel_tool_calls: false, tool_choice: 'auto' },
  }) as unknown as AnyLlm
}

function lastCall(): FakeCall {
  const call = FakeToolBoundModel.calls.at(-1)
  if (!call) throw new Error('the summarizer sub-call was never made')
  return call
}

// The request the sub-call actually put on the wire.
function lastCallRequestParams(): Record<string, unknown> {
  return { model: 'gpt-5.5', ...lastCall().modelKwargs }
}

const FORCE_SUMMARIZE = { maxContextTokens: 10, summarizeAtFraction: 0.1 } as const

function imageBlock() {
  return { type: 'image_url' as const, image_url: { url: 'data:image/jpeg;base64,XXXX' } }
}

// A history whose summarize window (firstHumanIdx+1 .. lastMotionAiIdx) is
// non-empty, so the pruner actually reaches the summarization sub-call.
function historyReadyToSummarize(): BaseMessage[] {
  return [
    new HumanMessage('Drive the robot to the red cone.'),
    new AIMessage({ content: '', tool_calls: [{ name: 'read_status', args: {}, id: 'tc-status' }] }),
    new ToolMessage({ content: JSON.stringify({ battery: '7.4V' }), tool_call_id: 'tc-status', name: 'read_status' }),
    new AIMessage('Status fine. Turning right to scan.'),
    new AIMessage({ content: '', tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-motion' }] }),
    new ToolMessage({
      content: JSON.stringify({ mimeType: 'image/jpeg', data: 'B'.repeat(40), motion: 'turn_right (steps=3)' }),
      tool_call_id: 'tc-motion',
      name: 'turn_right',
    }),
    new HumanMessage({ content: [{ type: 'text', text: 'Before/After frames.' }, imageBlock()] }),
  ]
}

beforeEach(() => {
  FakeToolBoundModel.calls = []
  __pendingSummariesForTest.clear()
  __resetMotionLogForTest()
})

// ── toolFreeModel, directly ─────────────────────────────────────────────────
describe('toolFreeModel', () => {
  it('strips BOTH tool_choice and parallel_tool_calls from modelKwargs', () => {
    const agent = new FakeToolBoundModel({
      model: 'gpt-5.5',
      modelKwargs: { parallel_tool_calls: false, tool_choice: 'auto' },
    })
    const free = toolFreeModel(agent as unknown as AnyLlm) as unknown as FakeToolBoundModel

    expect(openAiRejection(agent.requestParams())).toBe(
      "'tool_choice' is only allowed when 'tools' are specified"
    )
    expect(openAiRejection(free.requestParams())).toBeNull()
    expect(free.modelKwargs).toEqual({})
  })

  it('keeps every non-tool modelKwarg', () => {
    const agent = new FakeToolBoundModel({
      model: 'gpt-5.5',
      modelKwargs: { tool_choice: 'auto', parallel_tool_calls: false, seed: 42, service_tier: 'flex' },
    })
    const free = toolFreeModel(agent as unknown as AnyLlm) as unknown as FakeToolBoundModel
    expect(free.modelKwargs).toEqual({ seed: 42, service_tier: 'flex' })
    expect(free.lc_kwargs.model).toBe('gpt-5.5')
  })

  it('strips a baked-in tools array too — defensive breadth, no production path writes one', () => {
    // `tools` is in TOOL_ONLY_REQUEST_KEYS for breadth, not because anything in
    // this repo puts it there: server/createLlm.ts bakes in only tool_choice
    // and parallel_tool_calls. The entry is covered here so it stays a real,
    // falsifiable part of the helper's contract instead of decoration — but
    // this test speaks for the helper, NOT for any shipped call site.
    const agent = new FakeToolBoundModel({
      model: 'gpt-5.5',
      modelKwargs: { tools: [{ type: 'function', function: { name: 'move_forward' } }], seed: 7 },
      invocationKwargs: { tools: [{ name: 'move_forward' }] },
    })
    const free = toolFreeModel(agent as unknown as AnyLlm) as unknown as FakeToolBoundModel
    expect(free).not.toBe(agent as unknown as AnyLlm)
    expect(free.modelKwargs).toEqual({ seed: 7 })
    expect(free.invocationKwargs).toEqual({})
    // The original bag is untouched.
    expect(agent.modelKwargs.tools).toBeDefined()
  })

  it('strips the Anthropic spelling too (invocationKwargs.tool_choice)', () => {
    const agent = new FakeToolBoundModel({
      model: 'claude-x',
      invocationKwargs: { tool_choice: { type: 'auto', disable_parallel_tool_use: true } },
    })
    const free = toolFreeModel(agent as unknown as AnyLlm) as unknown as FakeToolBoundModel
    expect(free.invocationKwargs).toEqual({})
  })

  it('returns the SAME instance when there is nothing to strip', () => {
    // The Ollama / Google path: no tool params baked in, so no needless rebuild.
    const agent = new FakeToolBoundModel({ model: 'gemma4:12b' })
    expect(toolFreeModel(agent as unknown as AnyLlm)).toBe(agent as unknown as AnyLlm)
  })

  it('passes through a model with no lc_kwargs untouched', () => {
    const stub = { invoke: vi.fn() } as unknown as AnyLlm
    expect(toolFreeModel(stub)).toBe(stub)
  })

  it('falls back to the original model when the rebuild throws', () => {
    class Unconstructable {
      lc_kwargs = { modelKwargs: { tool_choice: 'auto' } }
      constructor() {
        if (arguments.length > 0) throw new Error('cannot rebuild')
      }
    }
    const agent = new Unconstructable() as unknown as AnyLlm
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    expect(toolFreeModel(agent)).toBe(agent)
    expect(warn).toHaveBeenCalled()
    warn.mockRestore()
  })
})

// ── The rebuild, against the REAL provider classes ──────────────────────────
// Everything above uses `FakeToolBoundModel`, which does `this.lc_kwargs =
// fields` and therefore round-trips BY CONSTRUCTION — it cannot fail the way a
// real class can. These cases run the rebuild through the actual `@langchain/*`
// constructors `server/createLlm.ts` uses, with the exact fields it passes.
//
// The failure mode being guarded is NOT the try/catch one: it is a future
// `@langchain/*` bump whose constructor SUCCEEDS while silently dropping or
// renaming a field. Nothing throws, no warning prints, and the summary model
// comes back quietly missing its endpoint or its key. So these assert field
// SURVIVAL, not just the absence of the tool params.
//
// Hermetic and key-free: dummy API keys, and the only method read is
// `invocationParams({})` — the pure function that assembles the request body.
// NOTHING HERE MAY CALL `.invoke()` OR `.stream()`. That is the one route to a
// network call, and on a machine with LangSmith tracing configured it would
// also spend a key. Construction alone opens no socket.

// `invocationParams` is public on all of these but its option type is
// per-provider, so the call goes through one narrow structural cast.
function requestParamsOf(model: unknown): Record<string, unknown> {
  return (
    model as { invocationParams: (o: Record<string, never>) => Record<string, unknown> }
  ).invocationParams({})
}

describe('toolFreeModel against the real provider classes (offline, no key spend)', () => {
  it('ChatOpenAI: both params go, and model / apiKey / temperature / baseURL survive the rebuild', () => {
    const agent = new ChatOpenAI({
      model: 'gpt-4.1-mini',
      apiKey: 'sk-not-a-real-key',
      temperature: 0.3,
      modelKwargs: { parallel_tool_calls: false, tool_choice: 'auto' },
      configuration: { baseURL: 'https://example.invalid/v1' },
    })
    // Before: the request the summarizer would have sent is rejectable.
    expect(openAiRejection(requestParamsOf(agent))).toBe(
      "'tool_choice' is only allowed when 'tools' are specified"
    )

    const free = toolFreeModel(agent)
    expect(free).not.toBe(agent)
    expect(free).toBeInstanceOf(ChatOpenAI)

    const params = requestParamsOf(free)
    expect(openAiRejection(params)).toBeNull()
    // Value, not key presence: ChatOpenAI's param object carries plenty of keys
    // set to `undefined`, and JSON.stringify drops those before they reach the
    // wire. A defined value is what OpenAI rejects, so that is what we assert.
    expect(params.tool_choice).toBeUndefined()
    expect(params.parallel_tool_calls).toBeUndefined()

    // Field survival. `clientConfig.baseURL` is the effective endpoint — the
    // one that reaches the wire — so it is what a silent drop would break.
    expect(params.model).toBe('gpt-4.1-mini')
    expect(free.model).toBe('gpt-4.1-mini')
    expect(free.temperature).toBe(0.3)
    // Compared as a boolean, not by value. The dummy key must be the one that
    // survived — truthiness would not do, because a rebuild that DROPPED the
    // field falls back to process.env.OPENAI_API_KEY and would look fine. But a
    // value comparison would print that ambient key into the failure output on
    // any machine that has one, so the assertion is reduced to a boolean first.
    expect(free.apiKey === 'sk-not-a-real-key').toBe(true)
    expect(free.clientConfig.baseURL).toBe('https://example.invalid/v1')

    // The shared agent model is never mutated — other threads still get theirs.
    expect(requestParamsOf(agent).tool_choice).toBe('auto')
  })

  it('ChatAnthropic: the invocationKwargs spelling goes, and model / apiKey / max_tokens survive', () => {
    const agent = new ChatAnthropic({
      model: 'claude-sonnet-4-5',
      apiKey: 'sk-ant-not-a-real-key',
      invocationKwargs: { tool_choice: { type: 'auto', disable_parallel_tool_use: true } },
    })
    expect(requestParamsOf(agent).tool_choice).toBeDefined()

    const free = toolFreeModel(agent)
    expect(free).not.toBe(agent)
    expect(free).toBeInstanceOf(ChatAnthropic)

    const params = requestParamsOf(free)
    expect(params.tool_choice).toBeUndefined()
    expect(params.model).toBe('claude-sonnet-4-5')
    expect(params.max_tokens).toBe(requestParamsOf(agent).max_tokens)
    // Boolean, not by value — see the ChatOpenAI case above.
    expect(free.apiKey === 'sk-ant-not-a-real-key').toBe(true)

    expect(requestParamsOf(agent).tool_choice).toBeDefined()
  })

  it('ChatOpenRouter: both params go, and model / apiKey / the default baseURL survive', () => {
    const agent = new ChatOpenRouter({
      model: 'z-ai/glm-4.6',
      apiKey: 'or-not-a-real-key',
      modelKwargs: { parallel_tool_calls: false, tool_choice: 'auto' },
    })
    expect(openAiRejection(requestParamsOf(agent))).toBe(
      "'tool_choice' is only allowed when 'tools' are specified"
    )

    const free = toolFreeModel(agent)
    expect(free).not.toBe(agent)
    expect(free).toBeInstanceOf(ChatOpenRouter)
    expect(openAiRejection(requestParamsOf(free))).toBeNull()
    expect(requestParamsOf(free).model).toBe('z-ai/glm-4.6')
    // Boolean, not by value — see the ChatOpenAI case above.
    expect(free.apiKey === 'or-not-a-real-key').toBe(true)
    // The default endpoint is applied by the constructor, not passed in — a
    // rebuild that lost it would still look fine on `model` alone.
    expect(free.baseURL).toBe('https://openrouter.ai/api/v1')
  })

  it('ChatOpenRouter: a custom baseURL survives the rebuild', () => {
    const agent = new ChatOpenRouter({
      model: 'z-ai/glm-4.6',
      apiKey: 'or-not-a-real-key',
      modelKwargs: { parallel_tool_calls: false, tool_choice: 'auto' },
      baseURL: 'https://example.invalid/openrouter/v1',
    })
    const free = toolFreeModel(agent)
    expect(free).not.toBe(agent)
    expect(free.baseURL).toBe('https://example.invalid/openrouter/v1')
    expect(openAiRejection(requestParamsOf(free))).toBeNull()
  })

  it('ChatOllama and ChatGoogle bake in nothing to strip, so they come back as the SAME instance', () => {
    // Not a constructor-name comparison: `ChatGoogle` from the node entrypoint
    // rebuilds as `ChatGoogleNode`, so identity is the only honest assertion.
    const ollama = new ChatOllama({ baseUrl: 'http://localhost:11434', model: 'gemma4:12b' })
    expect(toolFreeModel(ollama)).toBe(ollama)

    const google = new ChatGoogle({
      model: 'gemini-2.5-flash',
      apiKey: 'g-not-a-real-key',
      platformType: 'gai',
    })
    expect(toolFreeModel(google)).toBe(google)
  })

  it('ScriptedRobotChatModel comes back as the SAME instance — the e2e seam is untouched', () => {
    // The PUKEKO_FAKE_LLM path (server/createLlm.ts). The `pnpm run e2e` gate
    // covers this end to end, but only when a browser and three live services are
    // available; pinning it here keeps the property under the unit suite too.
    const scripted = new ScriptedRobotChatModel()
    expect(toolFreeModel(scripted)).toBe(scripted)
  })
})

// ── The sub-calls the middlewares actually make ─────────────────────────────
describe('context-pruner summarization sub-call', () => {
  it('sends no tool_choice and no parallel_tool_calls (the request would 400 otherwise)', async () => {
    const mw = createContextPrunerMiddleware({ llm: agentModel(), ...FORCE_SUMMARIZE }) as HookContainer
    const before = getHook(mw.beforeModel)

    await before({ messages: historyReadyToSummarize() }, runtime)

    expect(FakeToolBoundModel.calls).toHaveLength(1)
    const params = lastCallRequestParams()
    expect(openAiRejection(params)).toBeNull()
    expect(params).not.toHaveProperty('tool_choice')
    expect(params).not.toHaveProperty('parallel_tool_calls')
  })

  it('the summary is actually applied — pruning happens on an OpenAI-shaped model', async () => {
    // The consequence of the defect: the sub-call threw, the catch swallowed it,
    // summaryText stayed '', and no summary was ever folded in. Assert the
    // observable end state, not just the request shape.
    const mw = createContextPrunerMiddleware({ llm: agentModel(), ...FORCE_SUMMARIZE }) as HookContainer
    const result = (await getHook(mw.beforeModel)({ messages: historyReadyToSummarize() }, runtime)) as {
      messages: BaseMessage[]
    }
    const rebuilt = result.messages.filter((m) => !isRemoveMessage(m))
    const summaries = rebuilt.filter((m) => String(m.content).startsWith('[Context summary]'))
    expect(summaries).toHaveLength(1)
    expect(rebuilt.length).toBeLessThan(historyReadyToSummarize().length)
  })
})

describe('motion-summarization summarization sub-call', () => {
  it('sends no tool_choice and no parallel_tool_calls', async () => {
    const mw = createMotionSummarizationMiddleware({ llm: agentModel() }) as HookContainer
    const after = getHook(mw.afterModel)

    await after(
      {
        messages: [
          new HumanMessage('Drive the robot to the red cone.'),
          new AIMessage({ content: '', tool_calls: [{ name: 'turn_right', args: { steps: 3 }, id: 'tc-motion' }] }),
        ],
      },
      runtime
    )
    await new Promise((r) => setTimeout(r, 0))

    expect(FakeToolBoundModel.calls).toHaveLength(1)
    expect(openAiRejection(lastCallRequestParams())).toBeNull()
  })
})

// The third tool-less sub-call on the agent model. `req.model` is the model
// BEFORE the framework binds tools (the tool list arrives separately as
// `req.tools`), so the classifier request carries the same baked-in params and
// the same swallowed 400 — after which the recovery net silently never engages.
describe('lazy-tool-recovery classifier sub-call', () => {
  it('sends no tool_choice and no parallel_tool_calls, so the classifier is not silently rejected', async () => {
    const model = new FakeToolBoundModel({
      model: 'gpt-5.5',
      modelKwargs: { parallel_tool_calls: false, tool_choice: 'auto' },
      reply: 'YES\nread_distance',
    })
    const mw = createLazyToolRecoveryMiddleware() as HookContainer
    const wrap = getHook(mw.wrapModelCall)

    const handler = vi
      .fn()
      .mockResolvedValueOnce(new AIMessage('`read_distance` to check the range before moving.'))
      .mockResolvedValueOnce(
        new AIMessage({
          content: '',
          tool_calls: [{ name: 'read_distance', args: {}, id: 'tc-recovered' }],
        })
      )
    const request = {
      model,
      messages: [new HumanMessage('Drive to the cone.')],
      tools: [{ name: 'read_distance' }, { name: 'move_forward' }],
    }

    const out = (await wrap(request, handler)) as AIMessage

    // The classifier request itself: the throwing fake reproduces the measured
    // 400, so reaching a verdict at all proves the params were gone.
    expect(FakeToolBoundModel.calls).toHaveLength(1)
    expect(openAiRejection(lastCallRequestParams())).toBeNull()

    // And the observable consequence — before the fix the 400 was swallowed by
    // the classifier's catch, the verdict defaulted to not-lazy, and the net
    // never re-prompted.
    expect(handler).toHaveBeenCalledTimes(2)
    expect(out.tool_calls?.[0].name).toBe('read_distance')
  })
})

// ── Regression guard: call→result pairing across the summary boundary ───────
// This is characterization, not a bug fix: the pruner's slicing was measured
// (1296 beforeModel invocations over 432 distinct histories x 3 cycles x both
// summary outcomes) never to introduce an orphan, because the tail always
// starts at the assistant message that issued the last motion call. The guard
// pins that invariant, which only now becomes reachable — before the sub-call
// fix the summarize path could never complete on an OpenAI profile.
//
// Sensitivity, verified by mutation: changing the tail to
// `pruned.slice(lastMotionAiIdx + 1)` — exactly the "parenting tool_call
// dropped, its result kept" shape — turns the two-cycle case below red with
// `orphan-result tc-motion(turn_right)`.
interface PairIssue {
  kind: string
  detail: string
}

function pairIssues(messages: BaseMessage[]): PairIssue[] {
  const issues: PairIssue[] = []
  const called = new Set<string>()
  const answered = new Set<string>()
  for (const m of messages) {
    if (isRemoveMessage(m)) continue
    if (isAIMessage(m)) {
      for (const tc of (m as AIMessage).tool_calls ?? []) if (tc.id) called.add(tc.id)
    } else if (isToolMessage(m)) {
      const id = (m as ToolMessage).tool_call_id
      if (!called.has(id)) issues.push({ kind: 'orphan-result', detail: `${id}(${m.name})` })
      else answered.add(id)
    }
  }
  for (const id of called) {
    if (!answered.has(id)) issues.push({ kind: 'dangling-call', detail: id })
  }
  return issues
}

function motionTurn(n: number): BaseMessage[] {
  const id = `tc-motion-${n}`
  return [
    new AIMessage({ content: '', tool_calls: [{ name: 'move_forward', args: { steps: 2 }, id }] }),
    new ToolMessage({
      content: JSON.stringify({ mimeType: 'image/jpeg', data: 'C'.repeat(40), motion: 'move_forward (steps=2)' }),
      tool_call_id: id,
      name: 'move_forward',
    }),
    new HumanMessage({ content: [{ type: 'text', text: 'Before/After frames.' }, imageBlock()] }),
  ]
}

describe('context-pruner: no orphaned tool results across the summary boundary', () => {
  it('one cycle leaves every tool result paired with its call', async () => {
    const mw = createContextPrunerMiddleware({ llm: agentModel(), ...FORCE_SUMMARIZE }) as HookContainer
    const result = (await getHook(mw.beforeModel)({ messages: historyReadyToSummarize() }, runtime)) as {
      messages: BaseMessage[]
    }
    expect(pairIssues(result.messages)).toEqual([])
  })

  it('two cycles do not accumulate orphans — state carried across the boundary stays paired', async () => {
    const mw = createContextPrunerMiddleware({ llm: agentModel(), ...FORCE_SUMMARIZE }) as HookContainer
    const before = getHook(mw.beforeModel)

    // Cycle 1.
    const r1 = (await before({ messages: historyReadyToSummarize() }, runtime)) as { messages: BaseMessage[] }
    const rebuilt1 = r1.messages.filter((m) => !isRemoveMessage(m))
    expect(pairIssues(rebuilt1)).toEqual([])

    // Cycle 2: the model does one more motion turn off the rebuilt state, and
    // the prior summary now sits inside the next head slice.
    const r2 = (await before({ messages: [...rebuilt1, ...motionTurn(2)] }, runtime)) as { messages: BaseMessage[] }
    const rebuilt2 = r2.messages.filter((m) => !isRemoveMessage(m))
    expect(pairIssues(rebuilt2)).toEqual([])

    // And the summarize path really did run on both cycles — otherwise this
    // asserts pairing on a history that was never rebuilt.
    expect(FakeToolBoundModel.calls).toHaveLength(2)
    expect(rebuilt2.filter((m) => String(m.content).startsWith('[Context summary]'))).toHaveLength(1)

    // Cycle 3, to catch an orphan that needs two boundaries to surface.
    const r3 = (await before({ messages: [...rebuilt2, ...motionTurn(3)] }, runtime)) as { messages: BaseMessage[] }
    expect(pairIssues(r3.messages.filter((m) => !isRemoveMessage(m)))).toEqual([])
  })
})
