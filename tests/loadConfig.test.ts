import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { loadConfig } from '../server/loadConfig.js'

let tmpDir: string
let warnSpy: ReturnType<typeof vi.spyOn>

/** Every `console.warn` this spec's `loadConfig` call produced, in order. */
function warnings(): string[] {
  return warnSpy.mock.calls.map((call: unknown[]) => String(call[0]))
}
const ENV_KEYS = [
  'PUKEKO_PROFILE',
  'LLM_PROVIDER',
  'OLLAMA_MODEL',
  'OLLAMA_BASE_URL',
  'ANTHROPIC_MODEL',
  'ROBOT_HOST',
  'ROBOT_PRESET',
  'PUKEKO_DUMP_DIR',
  'PUKEKO_VERBOSE',
] as const

// RC-63. Most profiles in this file are ollama with no `numCtx`, written before
// RC-62's window check existed and kept deliberately minimal — so nearly every
// spec here now emits a context-window warning that is perfectly true and has
// nothing to do with what the spec is testing. It is CAPTURED rather than
// silenced: discarding it would leave a future spec that wants to assert on
// stderr looking at a mock with no history, and leaving it on stderr trains a
// reader to skim past the one run where it matters. `warnings()` above is what
// a spec reads to make an assertion about it, and the PUKEKO_PROFILE spec below
// does exactly that.
//
// It is deliberately NOT fixed by giving these profiles a window. Their subject
// is profile selection and env overrides; a `numCtx` on each would be noise in
// the fixture, and the no-config path's own window is asserted in
// `tests/contextWindowAgreement.test.ts`, where the check is the subject.
beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), 'pukeko-cfg-'))
  for (const k of ENV_KEYS) delete process.env[k]
  warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
})

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true })
  for (const k of ENV_KEYS) delete process.env[k]
  vi.restoreAllMocks()
})

describe('loadConfig', () => {
  it('falls back to gemma4 when no config file is present', async () => {
    // RC-63: 12b, not 31b. The fallback is a floor — chosen for the weakest
    // machine that should still work, since the user who reaches it configured
    // nothing — and it now matches the model AGENTS.md and the example config
    // both name. The reasoning lives beside `FALLBACK_PROFILE`; the window that
    // comes with the model is asserted in `tests/contextWindowAgreement.test.ts`.
    const resolved = await loadConfig(tmpDir)
    expect(resolved.configPath).toBeNull()
    expect(resolved.profile.llm.provider).toBe('ollama')
    expect(resolved.profile.llm.model).toBe('gemma4:12b')
  })

  it('RC-16: the no-config fallback profile uses context-pruner, never motion-summary', async () => {
    // RC-9/RC-12 moved every named profile off the deprecated motion-summary
    // middleware; the fallback must match, or a fresh checkout on Anthropic
    // routes through the middleware and dies on its history rewrite.
    const resolved = await loadConfig(tmpDir)
    expect(resolved.configPath).toBeNull()
    expect(resolved.profile.middleware).toEqual(['frontend-images', 'context-pruner'])
    expect(resolved.profile.middleware).not.toContain('motion-summary')
  })

  it('reads pukeko.config.json', async () => {
    writeFileSync(
      join(tmpDir, 'pukeko.config.json'),
      JSON.stringify({
        defaultProfile: 'a',
        profiles: {
          a: { llm: { provider: 'ollama', model: 'a-model' } },
          b: { llm: { provider: 'anthropic', model: 'b-model' } },
        },
      })
    )
    const resolved = await loadConfig(tmpDir)
    expect(resolved.profileName).toBe('a')
    expect(resolved.profile.llm.model).toBe('a-model')
  })

  it('selects profile via PUKEKO_PROFILE', async () => {
    writeFileSync(
      join(tmpDir, 'pukeko.config.json'),
      JSON.stringify({
        defaultProfile: 'a',
        profiles: {
          a: { llm: { provider: 'ollama', model: 'a-model' } },
          b: { llm: { provider: 'anthropic', model: 'b-model' } },
        },
      })
    )
    process.env.PUKEKO_PROFILE = 'b'
    const resolved = await loadConfig(tmpDir)
    expect(resolved.profileName).toBe('b')
    expect(resolved.profile.llm.provider).toBe('anthropic')
  })

  it('applies env-var overrides on top of profile', async () => {
    writeFileSync(
      join(tmpDir, 'pukeko.config.json'),
      JSON.stringify({
        defaultProfile: 'a',
        profiles: { a: { llm: { provider: 'ollama', model: 'a-model' } } },
      })
    )
    process.env.OLLAMA_MODEL = 'gemma-override'
    process.env.ROBOT_HOST = '10.0.0.1'
    const resolved = await loadConfig(tmpDir)
    expect(resolved.profile.llm.model).toBe('gemma-override')
    expect(resolved.profile.robot?.host).toBe('10.0.0.1')
  })

  it('applies ROBOT_PRESET as an env override (RC-1)', async () => {
    writeFileSync(
      join(tmpDir, 'pukeko.config.json'),
      JSON.stringify({
        defaultProfile: 'a',
        profiles: { a: { llm: { provider: 'ollama', model: 'a-model' } } },
      })
    )
    process.env.ROBOT_PRESET = 'ACEBOTT-QD021'
    const resolved = await loadConfig(tmpDir)
    expect(resolved.profile.robot?.preset).toBe('ACEBOTT-QD021')
  })

  it('PUKEKO_VERBOSE=1 flips observability on with defaults', async () => {
    writeFileSync(
      join(tmpDir, 'pukeko.config.json'),
      JSON.stringify({
        defaultProfile: 'a',
        profiles: { a: { llm: { provider: 'ollama', model: 'm' } } },
      })
    )
    process.env.PUKEKO_VERBOSE = '1'
    process.env.PUKEKO_DUMP_DIR = './my-logs'
    const resolved = await loadConfig(tmpDir)
    expect(resolved.profile.observability?.verbose).toBe(true)
    expect(resolved.profile.observability?.dumpDir).toBe('./my-logs')
    expect(resolved.profile.observability?.dumpImages).toBe(true)
  })

  it('warns and falls back when PUKEKO_PROFILE is unknown', async () => {
    writeFileSync(
      join(tmpDir, 'pukeko.config.json'),
      JSON.stringify({
        defaultProfile: 'a',
        profiles: { a: { llm: { provider: 'ollama', model: 'm' } } },
      })
    )
    process.env.PUKEKO_PROFILE = 'does-not-exist'
    const resolved = await loadConfig(tmpDir)
    expect(resolved.profileName).toBe('a')
    // The spec is named for a warning it never read. Matched among all the
    // calls rather than as the only one: this profile is ollama with no window,
    // so RC-62's check speaks here too, and pinning a call COUNT would make an
    // unrelated warning fail a spec about profile selection.
    expect(warnings()).toEqual(
      expect.arrayContaining([
        expect.stringContaining("PUKEKO_PROFILE='does-not-exist' not found; falling back to 'a'"),
      ])
    )
  })

  it('rejects config without a profiles object', async () => {
    writeFileSync(
      join(tmpDir, 'pukeko.config.json'),
      JSON.stringify({ defaultProfile: 'a' })
    )
    await expect(loadConfig(tmpDir)).rejects.toThrow(/missing 'profiles'/)
  })
})
