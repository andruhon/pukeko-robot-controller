import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs'
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
 * **The window the profile needs is the budget PLUS the system prompt.** The
 * pruner budgets `state.messages`; the prompt goes to `createAgent` as
 * `systemPrompt` and is applied outside that array, so it was in nobody's
 * budget. Comparing the window against `maxContextTokens` alone let the check
 * bless the configuration its own remedy produced — it told the reader to raise
 * `numCtx` to 30000, and then had no opinion about the truncating window that
 * produced. That is why the headroom is READ FROM THE PROFILE'S OWN PROMPT FILE
 * rather than written down as a constant: `systemPromptPath` lets a profile
 * point anywhere, and a constant would be right for the shipped file and wrong
 * for every override — the same defect one layer down. The custom-path cell
 * below is what makes that distinction testable; a constant passes every other
 * cell in this file.
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

/**
 * A prompt root these branch tests own, holding a prompt of a length chosen by
 * hand: 8000 characters is 2000 tokens at the pruner's ceil(chars/4), so the
 * required window is a round 32000 against the 30000 budget.
 *
 * Deliberately NOT the repo's real `system-prompt.md`. That file is prose people
 * edit, and pinning exact expectations to its current length would make every
 * cell here fail on an unrelated wording change. The shipped-profile group below
 * is the one place the live file is read, and it says so.
 */
const PROMPT_CHARS = 8000
const PROMPT_TOKENS = 2000
const BUDGET = 30_000
const REQUIRED = BUDGET + PROMPT_TOKENS

describe('RC-62 — contextWindowWarning', () => {
  let promptRoot: string

  beforeEach(() => {
    promptRoot = mkdtempSync(join(tmpdir(), 'pukeko-rc62-prompt-'))
    writeFileSync(join(promptRoot, 'system-prompt.md'), 'x'.repeat(PROMPT_CHARS))
  })

  afterEach(() => {
    rmSync(promptRoot, { recursive: true, force: true })
  })

  /** The check as production calls it, but rooted at this test's prompt dir. */
  function warn(name: string, profile: PukekoProfile): string | null {
    return contextWindowWarning(name, profile, promptRoot)
  }

  it('reports a profile that sends no num_ctx, naming the server default it will get', () => {
    // The exact shape this repo shipped: `gemma-default` had no `llm.ollama`
    // block at all, so the request body was `{"model":…,"options":{}}` and the
    // window was whatever ollama chose. The warning has to fire here — an
    // absent key is the common way to disagree, not an edge case — and it has
    // to name the absence rather than pretend a number was configured.
    const warning = warn('gemma-default', localProfile())

    expect(warning).toContain(`profile 'gemma-default' sends no llm.ollama.numCtx`)
    expect(warning).toContain('4096')
    expect(warning).toContain('maxContextTokens=30000')
    // The requirement it names is the budget plus the prompt, not the budget.
    expect(warning).toContain(`needs a window of at least ${REQUIRED} tokens`)
    expect(warning).toContain(`adds about ${PROMPT_TOKENS} on top of it`)
    expect(warning).toContain('HEAD')
    // The remedy has to name the window that actually works. Naming 30000 here
    // is the defect: a reader who does exactly that still truncates, and this
    // check then goes silent on the result.
    expect(warning).toContain(`Raise llm.ollama.numCtx to at least ${REQUIRED}`)
    // Anchored on the remedy verb, not on the bare number. `maxContextTokens=30000`
    // appears legitimately in the requirement clause, so an unanchored
    // `not.toContain('at least 30000')` would pass today by an accident of
    // wording and red on a correct implementation that rephrased that clause.
    // The defect lived in the remedy, so that is what this pins.
    expect(warning).not.toMatch(/Raise llm\.ollama\.numCtx to at least 30000/)
    // On this path there is no config file, so the two keys the remedy names
    // may not exist to edit; the one lever that path has must appear as a
    // remedy and not only in the condition clause.
    expect(warning).toContain('OLLAMA_CONTEXT_LENGTH on the ollama server is the lever')
  })

  it('does not promise an upper bound on what is discarded, because none is enforced', () => {
    // `maxContextTokens` is read in exactly three places
    // (contextPrunerMiddleware.ts: the option default, the summarize threshold,
    // and a log line) and is never a cap the pruner enforces — that file's own
    // notes record rebuilds reaching 30003 and 31060 against it. So "up to N
    // tokens can be discarded" was a false ceiling, and swapping in a corrected
    // N would keep the falsehood and change only the number.
    const warning = warn('gemma-default', localProfile())

    expect(warning).not.toMatch(/Up to \d+ tokens/)
    expect(warning).toContain('not bounded by the gap between these numbers')
    expect(warning).toContain('summarize threshold rather than a ceiling')
  })

  it('reports a numCtx that is set but too small, and does not blame an absent key', () => {
    // The other way to disagree, and it has a different fix — so it gets a
    // different sentence rather than a paraphrase covering both. Naming the
    // server default here would be a false statement: this profile does send a
    // window, it is just too narrow.
    const warning = warn(
      'gemma-tuned',
      localProfile({ llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: 8192 } } })
    )

    expect(warning).toContain('sets llm.ollama.numCtx=8192')
    expect(warning).toContain('maxContextTokens=30000')
    expect(warning).toContain(`Raise llm.ollama.numCtx to at least ${REQUIRED}`)
    // The other direction of the fix, and it must subtract the prompt too:
    // 8192 − 2000. A budget lowered to the raw window would truncate again.
    expect(warning).toContain('lower contextPruner.maxContextTokens to 6192 or less')
    expect(warning).not.toContain('sends no llm.ollama.numCtx')
    expect(warning).not.toContain('4096')
    // This profile HAS a config file with the key in it, and an explicit
    // num_ctx in the request is not the server default. Offering the env var
    // here would be a remedy for a condition that does not hold.
    expect(warning).not.toContain('OLLAMA_CONTEXT_LENGTH')
  })

  it('SAYS NOTHING when the window covers the budget AND the prompt — the control', () => {
    // The assertion the rest of this file exists to protect. Every positive
    // case above passes just as well under a check that warns unconditionally;
    // only this one can tell the two apart.
    expect(
      warn(
        'gemma-default',
        localProfile({
          llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: 32768 } },
        })
      )
    ).toBeNull()
  })

  it('WARNS on a window exactly equal to maxContextTokens — the configuration the old remedy produced', () => {
    // The acceptance for this fix, and it is red on the previous code, which
    // tested `window >= budget` and returned null here.
    //
    // 30000 is not an arbitrary number: it is precisely what the warning used
    // to tell the reader to set. Following the remedy exactly landed on a
    // window with no room for the ~2000-token prompt that is sent outside the
    // pruned history — so the run still truncated, and the check that sent them
    // there had nothing further to say about it.
    const warning = warn(
      'followed-the-old-remedy',
      localProfile({
        llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: BUDGET } },
      })
    )

    expect(warning).toContain(`sets llm.ollama.numCtx=${BUDGET}`)
    expect(warning).toContain(`needs a window of at least ${REQUIRED} tokens`)
  })

  it('treats a window equal to budget-plus-prompt as agreement, and one token less as disagreement', () => {
    // Pins the direction and the boundary of the comparison. A `>` where `>=`
    // belongs would warn about a profile that is exactly right, and the
    // once-per-start warning would then be noise a reader learns to ignore.
    const exact = warn(
      'exact',
      localProfile({
        llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: REQUIRED } },
      })
    )
    expect(exact).toBeNull()

    const short = warn(
      'short',
      localProfile({
        llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: REQUIRED - 1 } },
      })
    )
    expect(short).toContain(`sets llm.ollama.numCtx=${REQUIRED - 1}`)
  })

  it('sizes the headroom from the profile OWN prompt file, not from a constant', () => {
    // The cell that decides the design. A hardcoded headroom — even one that is
    // exactly right for the shipped prompt — passes every other assertion in
    // this file and fails here, because `systemPromptPath` lets a profile point
    // at a file of any size. That is the same class of error this node is
    // fixing: a number that is true for one configuration stated as if it were
    // true for all of them.
    writeFileSync(join(promptRoot, 'big-prompt.md'), 'y'.repeat(40_000))

    const warning = warn('big-prompt', localProfile({
      systemPromptPath: 'big-prompt.md',
      llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: 32768 } },
    }))

    // 40000 chars is 10000 tokens, so this profile needs 40000 — and the 32768
    // that is comfortably enough for the default prompt is not enough here.
    expect(warning).toContain('needs a window of at least 40000 tokens')
    expect(warning).toContain('adds about 10000 on top of it')
    expect(warning).toContain('big-prompt.md')

    // The same window, same budget, default prompt: silent. Both halves are
    // needed — a check that warned on everything would pass the assertion above.
    expect(
      warn('default-prompt', localProfile({
        llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: 32768 } },
      }))
    ).toBeNull()
  })

  it('does not turn an unreadable prompt file into a startup failure, and says so instead of quoting a number', () => {
    // A missing prompt must not crash config load, so the headroom degrades to
    // zero and the comparison falls back to the budget alone. The text then has
    // to admit that rather than print "adds about 0 tokens", because the real
    // overhead is not zero — it is unknown, and above whatever is counted.
    rmSync(join(promptRoot, 'system-prompt.md'))

    const warning = warn('no-prompt-file', localProfile())

    expect(warning).toContain('no prompt file could be read at system-prompt.md')
    expect(warning).toContain(`needs a window of at least ${BUDGET} tokens`)
    expect(warning).not.toContain('adds about')

    // And with the window over the budget it is silent — same as the check did
    // before the prompt was counted, which is all it can honestly claim here.
    expect(
      warn('no-prompt-file', localProfile({
        llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: BUDGET } },
      }))
    ).toBeNull()
  })

  it('offers no negative budget when the prompt alone overruns the window', () => {
    // `window − promptTokens` is the second remedy, and it is only a remedy
    // while it is positive. With a prompt longer than the whole window there is
    // no budget to lower to, and printing "lower maxContextTokens to −1808" is
    // the kind of sentence this node exists to keep out of the log.
    writeFileSync(join(promptRoot, 'huge-prompt.md'), 'z'.repeat(40_000))

    const warning = warn('huge-prompt', localProfile({ systemPromptPath: 'huge-prompt.md' }))

    // Omitting the clause is the whole assertion: there is no negative number to
    // print if the sentence offering it is not built. A broader "no hyphen
    // followed by a digit anywhere in the message" tripwire was tried and
    // dropped — it would red on any future wording containing a version string
    // or a range, which is not what this cell is about.
    expect(warning).toContain('needs a window of at least 40000 tokens')
    expect(warning).not.toContain('lower contextPruner.maxContextTokens')
  })

  it('says nothing about a hosted profile, which has no num_ctx to compare', () => {
    // 130000 against a provider that takes no window setting is not a
    // disagreement — it is a profile this check has no opinion about. Warning
    // here would fire on every hosted profile in the example config.
    // Also a control the headroom arithmetic must not reach: this returns
    // before the prompt is ever sized, so it must survive every mutation to
    // that arithmetic.
    expect(
      warn('anthropic', {
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
    // The second control that returns before the headroom arithmetic, and so
    // must survive every mutation to it.
    expect(warn('no-pruner', localProfile({ middleware: ['frontend-images'] }))).toBeNull()
  })

  it('reports a profile that names no middleware, because the default stack includes the pruner', () => {
    // `buildMiddleware` and the fallback profile both default to
    // ['frontend-images', 'context-pruner'], so an unset `middleware` means the
    // pruner IS in force. A check that read an unset list as "no pruner" would
    // stay silent on exactly the configs nobody has tuned.
    const warning = warn('bare', localProfile({ middleware: undefined }))
    expect(warning).toContain('maxContextTokens=30000')
  })

  it('reports against the pruner default when the profile sets no maxContextTokens', () => {
    // The budget in force is the middleware's own default (30000) when the
    // profile names none, so that is the number to compare. The check imports
    // that default from the middleware rather than re-typing it: a second copy
    // would be one more pair of components disagreeing about how much context
    // exists, which is the shape it was written to report.
    const warning = warn('bare', localProfile({ contextPruner: undefined }))
    expect(warning).toContain('maxContextTokens=30000')
    expect(warning).toContain(`needs a window of at least ${REQUIRED} tokens`)
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

  it('sizes the prompt against the loader own root, so a real prompt file moves the threshold', async () => {
    // Both cells above run in a tmpdir with no prompt file, where the headroom
    // is zero and the threshold is the bare budget — so they would pass
    // unchanged if the prompt were never read at all. This is the end-to-end
    // cell that puts a prompt file on the path `loadConfig` resolves from, and
    // watches a window that used to be enough stop being enough.
    writeFileSync(join(tmpDir, 'system-prompt.md'), 'x'.repeat(8000))
    writeConfig({
      llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: 30_000 } },
      contextPruner: { maxContextTokens: 30_000 },
    })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})

    await loadConfig(tmpDir)

    expect(warn).toHaveBeenCalledTimes(1)
    expect(String(warn.mock.calls[0]?.[0])).toContain('needs a window of at least 32000 tokens')
  })

  it('says nothing at config load once the window covers the prompt too', async () => {
    // The paired control for the cell above: same prompt file, same budget, a
    // window raised to what the warning actually asks for. Without this, a
    // check that had started firing on every ollama profile would still pass.
    writeFileSync(join(tmpDir, 'system-prompt.md'), 'x'.repeat(8000))
    writeConfig({
      llm: { provider: 'ollama', model: 'gemma4:12b', ollama: { numCtx: 32_000 } },
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
    // No explicit root: these run at the repo root, against the live
    // `system-prompt.md`, which is the configuration a contributor actually
    // starts the server in. The cell below says what to do when this one fires.
    const guidance =
      'the shipped numCtx no longer covers maxContextTokens plus the live system-prompt.md — ' +
      'see the margin assertion in the next cell for the numbers and the remedy'
    expect(contextWindowWarning('gemma-default', local['gemma-default']), guidance).toBeNull()
    expect(contextWindowWarning('gemma-tuned', local['gemma-tuned']), guidance).toBeNull()
  })

  it('measures a shipped profile against the prompt file it is POINTED AT, not the default one', () => {
    // Every other cell in this group reads the real `system-prompt.md`, so none
    // of them can tell a derived headroom from a constant that happens to equal
    // that file's token count — substituting the literal 1887 into the check
    // leaves this whole group green (measured). This is the cell that tells them
    // apart at the repo root: the shipped profile, the shipped window,
    // redirected at a much larger real file.
    //
    // `AGENTS.md` stands in for an oversized prompt. Its size is asserted FIRST,
    // so that if it ever shrinks below the shipped margin this cell fails saying
    // so, rather than passing while testing nothing.
    const bigTokens = Math.ceil(readFileSync('AGENTS.md', 'utf8').length / 4)
    expect(
      bigTokens,
      'AGENTS.md no longer overruns the shipped window, so this cell would pass without ' +
        'discriminating a derived headroom from a constant. Point it at a larger tracked file.'
    ).toBeGreaterThan(32_768 - 30_000)

    const warning = contextWindowWarning('gemma-default-big-prompt', {
      ...local['gemma-default'],
      systemPromptPath: 'AGENTS.md',
    })

    expect(warning).toContain(`needs a window of at least ${30_000 + bigTokens} tokens`)
    expect(warning).toContain('AGENTS.md')
  })

  it('the shipped window still covers the live system-prompt.md, and names the slack when it stops', () => {
    // The one cell in this file that reads the repo's real prompt file, and the
    // one that will fire on an unrelated change: `system-prompt.md` is prose,
    // edited for behavioural reasons by people who are not thinking about
    // context budgets. A bare `expected null` in that PR is close to
    // undiagnosable, so this states the cause and the remedy in the failure
    // message itself.
    //
    // The token estimate is written out here rather than imported, so that a
    // change to the pruner's estimator cannot move this expectation with it.
    const promptChars = readFileSync('system-prompt.md', 'utf8').length
    const promptTokens = Math.ceil(promptChars / 4)
    const required = 30_000 + promptTokens
    const shipped = 32_768

    expect(local['gemma-default'].llm.ollama?.numCtx).toBe(shipped)
    expect(local['gemma-tuned'].llm.ollama?.numCtx).toBe(shipped)
    expect(local['gemma-default'].contextPruner?.maxContextTokens).toBe(30_000)

    expect(
      shipped - required,
      `system-prompt.md has grown to ${promptChars} characters (~${promptTokens} tokens), so a ` +
        `local profile now needs a window of ${required}. The shipped numCtx of ${shipped} no ` +
        `longer covers it, and the server would silently truncate from the head. Raise numCtx on ` +
        `both local profiles in pukeko.config.example.ts. Do NOT lower the pruner budget to fit ` +
        `and do NOT relax this assertion — the margin is what makes the shipped profiles correct.`
    ).toBeGreaterThanOrEqual(0)
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
