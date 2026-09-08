// RC-58 — message fixtures that do NOT come from the `@langchain/core` copy this
// repo imports.
//
// Why these exist. The robot can resolve more than one `@langchain/core` at
// runtime: its own, plus whatever the `@gaunt-sloth/*` dependencies pull in. A
// message built inside gaunt-sloth's AG-UI pipeline is therefore not necessarily
// an instance of the class this repo imports, and RC-21 is where that cost a real
// bug — `msg instanceof ToolMessage` returned false for every capture result on
// the live server, no frame was ever injected, and the dumps read
// `tool-data:1 / human-images:0 / imageCount:0`. Production is duck-typed as a
// result (`isToolMessage`, `isHumanMessage`, `getType()`), and that constraint is
// non-negotiable.
//
// A suite that builds every fixture with `new HumanMessage(...)` cannot see that
// condition: the prototypes match, so `instanceof` and `getType()` agree on every
// message and the two spellings are indistinguishable. These builders are what
// make them distinguishable again — the assertions they feed go red the moment a
// production check is reverted to `instanceof`.
//
// What "foreign" means here, precisely. The shape below is message-SHAPED but
// carries neither this copy's prototype chain nor the global
// `Symbol.for('langchain.message')` marker that `BaseMessage.isInstance` looks
// for. It therefore fails `instanceof` against any class in this copy while
// satisfying every `getType()`-based predicate. That is the shape RC-21 actually
// met on the wire, and it is the same construction
// `tests/rc21CrossCoreToolMessage.test.ts` uses for its control.
//
// Note that re-prototyping a real message does NOT produce a foreign fixture on
// the installed core: the message classes define
// `static [Symbol.hasInstance](obj) { return this.isInstance(obj) }`, and
// `isInstance` is itself a duck-type test keyed on that global symbol, so a clone
// that keeps the marker still passes `instanceof` whatever its prototype. Omitting
// the marker is what makes a fixture foreign, not swapping the prototype.

import type { BaseMessage } from '@langchain/core/messages'

/** The fields a foreign message carries, mirroring a real one closely enough for
 *  the middlewares to treat it as an ordinary history entry. */
export interface ForeignMessageFields {
  content: unknown
  id?: string
  name?: string
  tool_call_id?: string
  tool_calls?: unknown[]
  additional_kwargs?: Record<string, unknown>
  response_metadata?: Record<string, unknown>
}

/**
 * A message-shaped object from a foreign `@langchain/core`: it answers
 * `getType()` / `_getType()` but is an instance of none of this copy's classes.
 *
 * Returned as `BaseMessage` because that is what the middlewares' signatures ask
 * for and what the real cross-copy value is at runtime — the whole point being
 * that the static type cannot tell the two apart, and only the runtime check can.
 */
export function foreignCoreMessage(
  type: 'human' | 'ai' | 'system' | 'tool' | 'remove',
  fields: ForeignMessageFields
): BaseMessage {
  const message = {
    ...fields,
    additional_kwargs: fields.additional_kwargs ?? {},
    response_metadata: fields.response_metadata ?? {},
    lc_kwargs: { ...fields },
    getType: () => type,
    _getType: () => type,
  }
  return message as unknown as BaseMessage
}

/**
 * Anti-vacuity check for the builders above: is this value really foreign to the
 * `@langchain/core` copy we import?
 *
 * It answers without naming a message class, so it stays on the right side of
 * the repo constraint, and it is sharper than `instanceof` would be anyway — it
 * names the two facts `instanceof` is downstream of. A fixture that quietly grew
 * this copy's prototype, or the global marker, would keep passing every
 * `getType()` assertion while no longer testing anything, and this is what
 * catches that.
 */
export function isForeignToThisCore(message: unknown): boolean {
  return (
    typeof message === 'object' &&
    message !== null &&
    !(Symbol.for('langchain.message') in message) &&
    Object.getPrototypeOf(message) === Object.prototype
  )
}

export function foreignHumanMessage(fields: ForeignMessageFields): BaseMessage {
  return foreignCoreMessage('human', fields)
}

export function foreignAIMessage(fields: ForeignMessageFields): BaseMessage {
  return foreignCoreMessage('ai', fields)
}

export function foreignSystemMessage(fields: ForeignMessageFields): BaseMessage {
  return foreignCoreMessage('system', fields)
}

export function foreignToolMessage(fields: ForeignMessageFields): BaseMessage {
  return foreignCoreMessage('tool', fields)
}
