import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { ChatOllama } from '@langchain/ollama'
import { contextWindowWarning, loadConfig } from '../server/loadConfig.js'
import { createLlm } from '../server/createLlm.js'
import type { PukekoProfile } from '../src/lib/config.js'
import exampleConfig from '../pukeko.config.example.js'

/**
 * RC-62 — the context window the model is given and the budget the pruner spends
 * are one number, and a disagreement is reported rather than silent.
 *
 * The defect this pins is not a wrong constant. `context-pruner` sized history
 * against `maxContextTokens: 30000` while nothing sent `num_ctx`, so ollama served
 * 4096 and dropped the overflow FROM THE HEAD — the opening instruction, the
 * framing, and the pruner's own summary, which exists precisely so the early
 * history survives in compressed form. Neither component reported anything,
 * because neither knew the other's number.
 *
 * Two kinds of assertion here, and they answer different questions:
 *
 *   - `contextWindowWarning` is a pure function of the profile, so the branch
 *     tests below are exact. But a pure function nobody calls is the "knob wired
 *     to nothing" shape `src/lib/config.ts` warns about — so the loadConfig group
 *     asserts it is actually installed at the call site, and reds if the call is
 *     removed while every branch test still passes.
 *   - The request-body assertions follow `tests/createLlm.test.ts` (RC-50):
 *     `invocationParams()` is what the client spreads into `client.chat(...)`, so
 *     it is one `JSON.stringify` from the wire, and reading it also proves the
 *     camelCase-to-snake_case mapping. A field on the instance would prove only
 *     that the constructor stored something.
 *
 * **The control that makes this set discriminating is the SILENT case.** A
 * warning that fires on everything passes every positive test here; the profiles
 * that agree must produce no warning at all, and that is asserted three times —
 * on a hand-built profile, on both shipped local profiles, and end-to-end through
 * `loadConfig`.
 */

const ENV_KEYS = [
  'PUKEKO_PROFILE',
  'PUKEKO_FAKE_LLM',
  'LLM_PROVIDER',
  'OLLAMA_MODEL',
  'OLLAMA_BASE_URL',
] as const

beforeEach(() => {
  for (const k of ENV_KEYS) delete process.env[k]
})

afterEach(() => {
  for (const k of ENV_KEYS) delete process.env[k]
  vi.restoreAllMocks()
})

/** A local profile in the shape the example config ships, minus the window. */
function localProfile(overrides: Partial<PukekoProfile> = {}): PukekoProfile {
  return {
    llm: { provider: 'ollama', model: 'gemma4:12b' },
    middleware: ['frontend-images', 'context-pruner', 'observability', 'lazy-tool-recovery'],
    contextPruner: {
      maxContextTokens: 30_000,
      summarizeAtFraction: 0.7,
      keepLatestImages: 1,
      imageTokenBudget: 800,
    },
    ...overrides,
  }
}

describe('RC-62 — contextWindowWarning', () => {
  it('reports a profile that sends no num_ctx, naming the server default it will get', () => {
    // The exact shape this repo shipped: `gemma-default` had no `llm.ollama`
    // block at all, so the request body was `{"model":…,"options":{}}` and the
    // window was whatever ollama chose. The warning has to fire here — an
    // absent key is the common way to disagree, not an edge case — and it has
    // to name the absence rather than pretend a number was configured.
    const warning = contextWindowWarning('gemma-default', localProfile())

    expect(warning).toContain(`profile 'gemma-default' sends no llm.ollama.numCtx`)
    expect(warning).toContain('4096')
    expect(warning).toContain('maxContextTokens=30000')
    // Both numbers AND the consequence: 30000 − 4096 tokens the pruner elected
    // to keep can be thrown away by the server.
    expect(warning).toContain('Up to 25904 tokens')
    expect(warning).toContain('HEAD')
  })

  it('reports a numCtx that is set but too small, and does not blame an absent key', () => {
    // The other way to disagree, and it has a different fix — so it gets a
    // different sentence rather than a paraphrase covering both. Naming the
    // server default here would be a false statement: this profile does send a
    // window, it is just too narrow.
    const warning = contextWindowWarning(
      'gemma-tuned',
      localProfile({ llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: 8192 } } })
    )

    expect(warning).toContain('sets llm.ollama.numCtx=8192')
    expect(warning).toContain('maxContextTokens=30000')
    expect(warning).toContain('Up to 21808 tokens')
    expect(warning).not.toContain('sends no llm.ollama.numCtx')
    expect(warning).not.toContain('4096')
  })

  it('SAYS NOTHING when the window covers the budget — the control', () => {
    // The assertion the rest of this file exists to protect. Every positive
    // case above passes just as well under a check that warns unconditionally;
    // only this one can tell the two apart.
    expect(
      contextWindowWarning(
        'gemma-default',
        localProfile({
          llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: 32768 } },
        })
      )
    ).toBeNull()
  })

  it('treats an exactly-equal window as agreement, and one token less as disagreement', () => {
    // Pins the direction and the boundary of the comparison. A `>` where `>=`
    // belongs would warn about a profile that is exactly right, and the
    // once-per-start warning would then be noise a reader learns to ignore.
    const exact = contextWindowWarning(
      'exact',
      localProfile({
        llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: 30_000 } },
      })
    )
    expect(exact).toBeNull()

    const short = contextWindowWarning(
      'short',
      localProfile({
        llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: 29_999 } },
      })
    )
    expect(short).toContain('Up to 1 tokens')
  })

  it('says nothing about a hosted profile, which has no num_ctx to compare', () => {
    // 130000 against a provider that takes no window setting is not a
    // disagreement — it is a profile this check has no opinion about. Warning
    // here would fire on every hosted profile in the example config.
    expect(
      contextWindowWarning('anthropic', {
        llm: { provider: 'anthropic', model: 'claude-sonnet-4-6', cache: true },
        middleware: ['frontend-images', 'context-pruner', 'observability'],
        contextPruner: { maxContextTokens: 130_000, summarizeAtFraction: 0.7 },
      })
    ).toBeNull()
  })

  it('says nothing when context-pruner is not in the stack', () => {
    // With the pruner absent, `contextPruner` on the profile is inert: nothing
    // is sizing history against 30000, so there is no second opinion for the
    // window to disagree with.
    expect(
      contextWindowWarning('no-pruner', localProfile({ middleware: ['frontend-images'] }))
    ).toBeNull()
  })

  it('reports a profile that names no middleware, because the default stack includes the pruner', () => {
    // `buildMiddleware` and the fallback profile both default to
    // ['frontend-images', 'context-pruner'], so an unset `middleware` means the
    // pruner IS in force. A check that read an unset list as "no pruner" would
    // stay silent on exactly the configs nobody has tuned.
    const warning = contextWindowWarning('bare', localProfile({ middleware: undefined }))
    expect(warning).toContain('maxContextTokens=30000')
  })

  it('reports against the pruner default when the profile sets no maxContextTokens', () => {
    // The budget in force is the middleware's own default (30000) when the
    // profile names none, so that is the number to compare. The check imports
    // that default from the middleware rather than re-typing it: a second copy
    // would be one more pair of components disagreeing about how much context
    // exists, which is the shape it was written to report.
    const warning = contextWindowWarning('bare', localProfile({ contextPruner: undefined }))
    expect(warning).toContain('maxContextTokens=30000')
    expect(warning).toContain('Up to 25904 tokens')
  })
})

describe('RC-62 — the check is installed in loadConfig', () => {
  let tmpDir: string

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), 'pukeko-rc62-'))
  })

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true })
  })

  function writeConfig(profile: Record<string, unknown>): void {
    writeFileSync(
      join(tmpDir, 'pukeko.config.json'),
      JSON.stringify({ defaultProfile: 'local', profiles: { local: profile } })
    )
  }

  it('warns once, at config load, on a profile whose window is below the budget', async () => {
    // A pure function nobody calls reports nothing. This is the assertion that
    // reds if the call is deleted from `loadConfig` while every branch test
    // above still passes.
    writeConfig({
      llm: { provider: 'ollama', model: 'gemma4:12b' },
      contextPruner: { maxContextTokens: 30_000 },
    })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})

    const resolved = await loadConfig(tmpDir)

    expect(resolved.profileName).toBe('local')
    // Once, not once per turn: config load happens once per server start, which
    // is why this needs no dedupe bookkeeping of its own.
    expect(warn).toHaveBeenCalledTimes(1)
    const message = String(warn.mock.calls[0]?.[0])
    expect(message).toContain('maxContextTokens=30000')
    expect(message).toContain('4096')
  })

  it('says nothing at config load when the profile agrees — the end-to-end control', async () => {
    // The same route as the test above, one key different. If the warning ever
    // starts firing on everything, this is where it shows up on the real code
    // path rather than on the pure function alone.
    writeConfig({
      llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: 32768 } },
      contextPruner: { maxContextTokens: 30_000 },
    })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})

    await loadConfig(tmpDir)

    expect(warn).not.toHaveBeenCalled()
  })
})

/**
 * The `/api/chat` body ChatOllama would send, minus the messages — the same
 * helper `tests/createLlm.test.ts` uses, and for the same reason: the client
 * serialises this object, and `JSON.stringify` drops every `undefined`, so
 * whatever survives is precisely what ollama receives.
 */
function wireBody(llm: unknown): unknown {
  return JSON.parse(JSON.stringify((llm as ChatOllama).invocationParams()))
}

/**
 * A profile with the SAMPLING options removed, leaving `numCtx` in place.
 *
 * `numCtx` sits in the same `llm.ollama` bag as the sampling knobs but is not one
 * of them: it changes what the model can SEE, not how it draws tokens. So it
 * belongs on the equal side of the A/B comparison, along with everything outside
 * that bag.
 */
function withoutSamplingOptions(profile: PukekoProfile): PukekoProfile {
  return {
    ...profile,
    llm: { ...profile.llm, ollama: { numCtx: profile.llm.ollama?.numCtx } },
  }
}

describe('RC-62 — the shipped local profiles', () => {
  const local = exampleConfig.profiles

  it('gemma-default asks ollama for a window that covers the pruner budget', () => {
    // Whole-body equality, written out by hand rather than derived from the
    // profile: a change in how production builds the request cannot move the
    // expectation with it. `num_ctx` present and ≥ 30000 is the acceptance.
    const { provider, llm } = createLlm(local['gemma-default'].llm)

    expect(provider).toBe('ollama')
    expect(wireBody(llm)).toEqual({
      model: 'gemma4:12b',
      options: { num_ctx: 32768 },
    })
    expect(local['gemma-default'].contextPruner?.maxContextTokens).toBe(30_000)
  })

  it('gemma-tuned sends the same window alongside its sampling options', () => {
    const { llm } = createLlm(local['gemma-tuned'].llm)

    expect(wireBody(llm)).toEqual({
      model: 'gemma4:12b',
      think: false,
      options: {
        num_ctx: 32768,
        temperature: 0.6,
        top_k: 64,
        top_p: 0.95,
        repeat_penalty: 1.15,
        repeat_last_n: 512,
      },
    })
  })

  it('neither local profile warns — the shipped control', () => {
    expect(contextWindowWarning('gemma-default', local['gemma-default'])).toBeNull()
    expect(contextWindowWarning('gemma-tuned', local['gemma-tuned'])).toBeNull()
  })

  it('the two local profiles differ ONLY in their sampling options', () => {
    // RC-50 shipped `gemma-tuned` as a controlled A/B partner for
    // `gemma-default`: run one against the other and the only variable is how
    // the model samples. Setting a window on one and not the other would
    // destroy that silently — the difference would look like a sampling knob
    // and be the loudest change in the experiment. One deep comparison pins it,
    // and also catches a model, middleware list or pruner setting that moves in
    // one profile alone.
    expect(withoutSamplingOptions(local['gemma-tuned'])).toEqual(
      withoutSamplingOptions(local['gemma-default'])
    )
  })
})
