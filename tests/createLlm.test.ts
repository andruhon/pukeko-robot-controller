import { describe, it, expect } from 'vitest'
import type { ChatOllama } from '@langchain/ollama'
import { createLlm } from '../server/createLlm.js'
import type { OllamaGenerationOptions } from '../src/lib/config.js'

/**
 * RC-50 — a profile can set ollama generation options, and they reach the request.
 *
 * **Why these assertions read the request body and not the model's fields.** A field
 * on the instance only proves the constructor stored something; it does not prove
 * ollama is ever told. `ChatOllama.invocationParams()` is the object the client
 * spreads into `client.chat({ ...params, messages, stream })`, so it is one
 * `JSON.stringify` away from the bytes on the wire — and reading it also proves the
 * camelCase-to-snake_case mapping happens, which a field assertion cannot see. A
 * knob that ChatOllama accepts but never sends (see the note on `stop` in
 * `OllamaGenerationOptions`) fails here and passes a field check.
 *
 * The expected bodies below are written out by hand rather than derived from
 * `createLlm` or from a second `ChatOllama`, so a change in how production builds
 * the model cannot move the expectation with it.
 */

/**
 * The `/api/chat` body ChatOllama would send, minus the messages. Round-tripping
 * through JSON is deliberate and is what makes the default assertion exact: the
 * client serialises this object, and `JSON.stringify` drops every `undefined`, so
 * whatever survives is precisely what ollama receives.
 */
function wireBody(llm: unknown): unknown {
  return JSON.parse(JSON.stringify((llm as ChatOllama).invocationParams()))
}

describe('createLlm — ollama generation options (RC-50)', () => {
  it('sends model and nothing else when the profile sets no generation options', () => {
    // The control that matters most. Every existing profile is on this path and
    // every smoke observation recorded so far was taken here, so a stray default
    // added to the ollama branch would silently invalidate all of it. An empty
    // `options` object is what today's two-field construction produces: ollama
    // applies its own defaults for everything absent.
    const { provider, llm } = createLlm({ provider: 'ollama', model: 'gemma4:31b' })

    expect(provider).toBe('ollama')
    expect(wireBody(llm)).toEqual({ model: 'gemma4:31b', options: {} })
    expect((llm as ChatOllama).baseUrl).toBe('http://localhost:11434')
  })

  it('sends nothing extra when only baseUrl is set', () => {
    const { llm } = createLlm({
      provider: 'ollama',
      model: 'gemma4:12b',
      baseUrl: 'http://ollama.lan:11434',
    })

    expect((llm as ChatOllama).baseUrl).toBe('http://ollama.lan:11434')
    expect(wireBody(llm)).toEqual({ model: 'gemma4:12b', options: {} })
  })

  it('forwards every option the interface declares, under its ollama name', () => {
    // Whole-body equality over the full surface: a knob that never reaches the
    // request shows up as a missing key, and a knob nobody asked for shows up as
    // an extra one. Asserting the options one at a time would let a dead one hide.
    const { llm } = createLlm({
      provider: 'ollama',
      model: 'gemma4:12b',
      ollama: {
        think: false,
        numCtx: 16384,
        numPredict: 512,
        temperature: 0.6,
        topK: 40,
        topP: 0.9,
        repeatPenalty: 1.15,
        repeatLastN: 512,
        presencePenalty: 0.4,
        frequencyPenalty: 0.3,
        seed: 7,
      },
    })

    expect(wireBody(llm)).toEqual({
      model: 'gemma4:12b',
      think: false,
      options: {
        num_ctx: 16384,
        num_predict: 512,
        temperature: 0.6,
        top_k: 40,
        top_p: 0.9,
        repeat_penalty: 1.15,
        repeat_last_n: 512,
        presence_penalty: 0.4,
        frequency_penalty: 0.3,
        seed: 7,
      },
    })
  })

  it('puts think on the request as a top-level field, on and off', () => {
    // `think` is not a sampling option: ollama takes it beside `model`, not inside
    // `options`. Both settings are pinned because `false` is the one the node needs
    // reachable and `true` is what proves the `false` case is not just an absent key.
    const off = createLlm({ provider: 'ollama', model: 'gemma4:12b', ollama: { think: false } })
    expect(wireBody(off.llm)).toEqual({ model: 'gemma4:12b', think: false, options: {} })

    const on = createLlm({ provider: 'ollama', model: 'gemma4:12b', ollama: { think: true } })
    expect(wireBody(on.llm)).toEqual({ model: 'gemma4:12b', think: true, options: {} })
  })

  it('never lets the options bag shadow the model or the endpoint', () => {
    // `pukeko.config.json` is parsed at runtime and never type-checked, so a key
    // the interface does not declare can still arrive here — hence the cast, which
    // reproduces what a hand-written JSON profile can actually deliver. Identity
    // fields must win, or a stray key would silently redirect the request.
    const { llm } = createLlm({
      provider: 'ollama',
      model: 'gemma4:31b',
      baseUrl: 'http://localhost:11434',
      ollama: {
        temperature: 0.5,
        model: 'some-other-model',
        baseUrl: 'http://elsewhere.invalid:9999',
      } as OllamaGenerationOptions,
    })

    expect((llm as ChatOllama).baseUrl).toBe('http://localhost:11434')
    expect(wireBody(llm)).toEqual({ model: 'gemma4:31b', options: { temperature: 0.5 } })
  })
})
