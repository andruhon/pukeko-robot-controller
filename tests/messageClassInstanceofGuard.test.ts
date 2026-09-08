import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import ts from 'typescript';

/**
 * RC-58 — nothing in this repo may test a message's type with `instanceof`, or
 * by comparing a prototype against a message class's.
 *
 * The rule, and the honest reason for it. A LangChain message's class is not a
 * reliable answer to "what kind of message is this?" here — but not, today, for
 * the reason this rule was originally written down. On the core installed now
 * the robot resolves ONE `@langchain/core`, and every message class defines
 * `static [Symbol.hasInstance]` delegating to a duck-type test keyed on the
 * global-registry symbol `Symbol.for('langchain.message')`. `instanceof` is
 * therefore itself duck-typed, and is not currently blind across two v1.2.x
 * copies.
 *
 * The ban survives on what that rests on. It is a transitive dependency's
 * implementation detail: no lockfile of ours pins it, no test of ours guards it,
 * and it was absent in 1.2.1 and 1.2.3 and arrived at an unreviewed patch bump —
 * so it can leave the same way. A message rebuilt from the wire without the
 * marker, or built by a pre-marker copy, still fails `instanceof` while
 * answering `getType()` correctly. RC-21 is where that cost a real bug —
 * `msg instanceof ToolMessage` returned false for every capture result on the
 * live server, no frame was ever injected, and the dumps read
 * `tool-data:1 / human-images:0 / imageCount:0` — measured when two copies
 * genuinely coexisted. The duck-typed forms (`isToolMessage`, `isHumanMessage`,
 * `getType()`) answer correctly whatever built the message.
 *
 * Why a spec and not a review habit. The constraint has been restated in four
 * separate node texts and re-caught by a human reviewer twice, and RC-58 still
 * found it violated in six files at once — because a test written next to an
 * `instanceof` one copies the style, and a suite built entirely from native
 * fixtures passes either way. Nothing about a violation is visible in a green
 * run. This is the thing that looks.
 *
 * What else it bans, and why it has to. Comparing a prototype against a message
 * class's own — `Object.getPrototypeOf(m) === HumanMessage.prototype` — asks the
 * banned question in a second spelling, and it is blind in exactly the way
 * `instanceof` was before the marker existed: it reads the prototype chain and
 * nothing else, so no `Symbol.hasInstance` can rescue it. RC-58's acceptance
 * names it alongside `instanceof`, so the detector below flags it in either
 * operand position, against a `getPrototypeOf` call or any other expression.
 *
 * What it does NOT ban. Comparing two messages' prototypes to EACH OTHER
 * (`Object.getPrototypeOf(out) === Object.getPrototypeOf(input)`) is safe and
 * load-bearing: production clones messages prototype-first precisely so a
 * foreign message survives a rewrite as itself, and three specs in
 * `tests/contextPrunerMiddleware.test.ts` pin that. Those name no class, so the
 * detector leaves them alone, as it leaves alone the fixture helper's
 * `Object.getPrototypeOf(message) === Object.prototype` — `Object` is not a
 * message class. `instanceof` against a non-message class — `ChatOpenAI`,
 * `Error` — is likewise none of this spec's business.
 *
 * Where it stops. The detector matches a message class named directly, so an
 * aliased import (`import { ToolMessage as TM }`), a namespace import
 * (`msgs.HumanMessage`) and element access (`HumanMessage['prototype']`) all get
 * past it. That bound is deliberate: writing one of those to test a message type
 * is evading a stated rule knowingly, and a guard is a gate against drift, not a
 * sandbox against an author who means it.
 *
 * The one exemption. `tests/rc21CrossCoreToolMessage.test.ts` uses `instanceof`
 * deliberately, as the control that pins the hazard itself: it asserts a
 * message-shaped object with no `Symbol.for('langchain.message')` marker fails
 * `instanceof ToolMessage` while passing `isToolMessage`. Removing that would
 * delete the only direct pin on the condition this whole rule exists for, so it
 * is exempt by name — and the spec asserts the exemption is still being USED, so
 * a silent rewrite of the control cannot quietly retire it.
 *
 * The file list comes from `git ls-files`, as in `gatedSourceCoverage.test.ts`,
 * so build output can never trip it and every tracked file is covered. The price
 * is that this spec fails in a detached copy of the source, and the failure does
 * not say "no git" — so read this before chasing it. Extract the tree somewhere
 * that is itself inside a git repo (a `git archive` unpacked under
 * `_worktrees/<NODE>/handoff/`, the path this project prescribes, sits inside the
 * takahē repo) and `git ls-files -z` exits 0 and returns an EMPTY list, because
 * the extraction's own paths are ignored there. Nothing errors; the anti-vacuity
 * case below fires instead, as `expected 0 to be greater than 60`.
 *
 * Which is precisely why that case exists, and it is not decoration. The offender
 * assertion is a check that a list is empty, so it passes perfectly by scanning
 * nothing — the most dangerous way this spec could fail, since it fails green.
 * The anti-vacuity case is the only thing standing between a broken file list and
 * a guard that reports success forever. Do not weaken it to make an extraction
 * pass; give the extraction a real tracked set instead.
 *
 * Detection is done on TypeScript's own AST rather than by regex, which is what
 * keeps the ~15 mentions of `instanceof` in this repo's explanatory comments —
 * including the ones in this header — from being read as violations. Vue SFCs
 * are not a TypeScript program, so they get a plain textual scan instead; see
 * the last case for what that costs.
 */

const REPO_ROOT = `${process.cwd().replace(/\\/g, '/').replace(/\/$/, '')}/`;
if (!existsSync(`${REPO_ROOT}tsconfig.json`)) {
  throw new Error(`RC-58 instanceof guard: ${REPO_ROOT} is not the repo root (no tsconfig.json)`);
}

/** Every message class `@langchain/core/messages` exports, chunks included. */
const MESSAGE_CLASSES: ReadonlySet<string> = new Set([
  'BaseMessage',
  'BaseMessageChunk',
  'AIMessage',
  'AIMessageChunk',
  'HumanMessage',
  'HumanMessageChunk',
  'SystemMessage',
  'SystemMessageChunk',
  'ToolMessage',
  'ToolMessageChunk',
  'ChatMessage',
  'ChatMessageChunk',
  'FunctionMessage',
  'FunctionMessageChunk',
  'RemoveMessage',
]);

/** The deliberate control — see the header. */
const EXEMPT = 'tests/rc21CrossCoreToolMessage.test.ts';

const PARSEABLE = /\.(?:m|c)?[jt]sx?$/;

interface Violation {
  file: string;
  line: number;
  text: string;
}

/** Operators that ask "is this the same prototype?" and so carry the hazard. */
const EQUALITY_OPERATORS: ReadonlySet<ts.SyntaxKind> = new Set([
  ts.SyntaxKind.EqualsEqualsEqualsToken,
  ts.SyntaxKind.ExclamationEqualsEqualsToken,
  ts.SyntaxKind.EqualsEqualsToken,
  ts.SyntaxKind.ExclamationEqualsToken,
]);

/** Matchers that are an equality comparison written as an assertion. */
const EQUALITY_MATCHERS: ReadonlySet<string> = new Set(['toBe', 'toEqual', 'toStrictEqual']);

/** True for `HumanMessage.prototype` and its siblings — and only for those. */
function isMessageClassPrototype(node: ts.Node): boolean {
  return (
    ts.isPropertyAccessExpression(node) &&
    node.name.text === 'prototype' &&
    ts.isIdentifier(node.expression) &&
    MESSAGE_CLASSES.has(node.expression.text)
  );
}

/**
 * Every banned spelling in one source text:
 *
 *   - `x instanceof MessageClass` and `expect(x).toBeInstanceOf(MessageClass)`.
 *     The same hazard twice — vitest's matcher runs the `instanceof` operator.
 *   - a comparison against `MessageClass.prototype` in either operand position,
 *     written with an operator or with an equality matcher. The class's own
 *     prototype is the thing `instanceof` used to read, so this asks the banned
 *     question in a form no `Symbol.hasInstance` can answer.
 *
 * `getPrototypeOf(a) === getPrototypeOf(b)` names no class and is not matched;
 * neither is `=== Object.prototype`. See the header.
 */
function violationsIn(file: string, source: string): Violation[] {
  const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true);
  const found: Violation[] = [];

  const record = (node: ts.Node) => {
    const { line } = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile));
    found.push({ file, line: line + 1, text: node.getText(sourceFile).replace(/\s+/g, ' ').slice(0, 100) });
  };

  const visit = (node: ts.Node): void => {
    if (
      ts.isBinaryExpression(node) &&
      node.operatorToken.kind === ts.SyntaxKind.InstanceOfKeyword &&
      ts.isIdentifier(node.right) &&
      MESSAGE_CLASSES.has(node.right.text)
    ) {
      record(node);
    }
    // Either side: `getPrototypeOf(m) === HumanMessage.prototype` and the
    // reverse read identically to the engine, so a detector that only looked
    // right would be a spelling away from useless.
    if (
      ts.isBinaryExpression(node) &&
      EQUALITY_OPERATORS.has(node.operatorToken.kind) &&
      (isMessageClassPrototype(node.left) || isMessageClassPrototype(node.right))
    ) {
      record(node);
    }
    if (
      ts.isCallExpression(node) &&
      ts.isPropertyAccessExpression(node.expression) &&
      node.arguments.length === 1 &&
      ((node.expression.name.text === 'toBeInstanceOf' &&
        ts.isIdentifier(node.arguments[0]) &&
        MESSAGE_CLASSES.has((node.arguments[0] as ts.Identifier).text)) ||
        (EQUALITY_MATCHERS.has(node.expression.name.text) &&
          isMessageClassPrototype(node.arguments[0])))
    ) {
      record(node);
    }
    ts.forEachChild(node, visit);
  };

  visit(sourceFile);
  return found;
}

function trackedFiles(): string[] {
  // -z: NUL-separated, so a path containing a newline cannot split one entry in two.
  const out = execFileSync('git', ['ls-files', '-z'], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  });
  return out.split(String.fromCharCode(0)).filter(Boolean);
}

function scannedFiles(): string[] {
  return trackedFiles().filter((file) => PARSEABLE.test(file));
}

function allViolations(): Violation[] {
  return scannedFiles().flatMap((file) =>
    violationsIn(file, readFileSync(`${REPO_ROOT}${file}`, 'utf8'))
  );
}

describe('RC-58 no `instanceof` or prototype comparison against a LangChain message class', () => {
  it('detects every banned spelling and nothing else', () => {
    // The discriminating control. Run the real detector over a source that holds
    // one of each banned spelling plus eight near-misses, so a detector that
    // matched everything — or nothing — cannot look like success. The expected
    // list is exact by line number on purpose: a length check or a `toContain`
    // would let an over-matching detector swallow a near-miss silently.
    const sample = [
      // Banned — lines 1-7:
      "const a = msg instanceof ToolMessage",
      "expect(x).toBeInstanceOf(HumanMessage)",
      "expect(x).not.toBeInstanceOf(SystemMessage)",
      "const p = Object.getPrototypeOf(m) === HumanMessage.prototype",
      "const q = ToolMessage.prototype === Object.getPrototypeOf(m)",
      "const r = proto !== AIMessage.prototype",
      "expect(Object.getPrototypeOf(m)).toBe(SystemMessage.prototype)",
      // Near-misses that must NOT be flagged:
      "const b = llm instanceof ChatOpenAI",
      "const c = err instanceof Error",
      "expect(free).toBeInstanceOf(ChatAnthropic)",
      "const d = Object.getPrototypeOf(out) === Object.getPrototypeOf(input)",
      // The two shapes that really exist in this repo, so the widening is proved
      // to leave them alone here and not only by the suite staying green:
      "const f = Object.getPrototypeOf(message) === Object.prototype",
      "expect(Object.getPrototypeOf(out)).toBe(Object.getPrototypeOf(motionTool))",
      // Naming a class's prototype is not asking the banned question:
      "const g = Object.assign({}, HumanMessage.prototype)",
      "// a comment mentioning msg instanceof ToolMessage and HumanMessage",
      "const e = 'a string saying x instanceof AIMessage'",
    ].join('\n');

    const hits = violationsIn('sample.ts', sample);
    expect(hits.map((h) => h.line)).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it('scans a plausible number of tracked files, and the control is one of them', () => {
    // Anti-vacuity. A failed `git ls-files` or a wrong cwd yields an empty list,
    // and the offender assertion below would then pass by scanning nothing.
    const scanned = scannedFiles();
    expect(scanned.length).toBeGreaterThan(60);
    expect(scanned).toContain(EXEMPT);
    expect(scanned).toContain('src/agent/contextPrunerMiddleware.ts');
  });

  it('the exemption is still earning its place', () => {
    // If the control file stopped using `instanceof`, this spec would be
    // exempting a file for nothing — and, far worse, the hazard would no longer
    // be pinned anywhere. Assert the exemption is USED, so it cannot be
    // retired by accident.
    const control = violationsIn(EXEMPT, readFileSync(`${REPO_ROOT}${EXEMPT}`, 'utf8'));
    expect(control.length).toBeGreaterThan(0);
  });

  it('no tracked file outside the control tests a message type by class', () => {
    const offenders = allViolations().filter((v) => v.file !== EXEMPT);
    expect(
      offenders.map((v) => `${v.file}:${v.line}  ${v.text}`),
      'Use the duck-typed check instead — isToolMessage / isHumanMessage / getType(). ' +
        "A message's class answers this reliably only while @langchain/core keeps " +
        'wiring `Symbol.hasInstance` to its own duck test — a detail nothing here ' +
        'pins, which arrived at an unreviewed patch bump and can leave the same way. ' +
        'A message rebuilt from the wire without the marker fails both spellings ' +
        'today, and that is the shape RC-21 met. See tests/helpers/foreignCoreMessage.ts ' +
        'for a fixture that proves it.'
    ).toEqual([]);
  });

  it('no Vue SFC tests a message type by class either', () => {
    // The AST detector needs a TypeScript program and an SFC is not one, so the
    // .vue files get a textual scan for the same spellings. It covers the
    // prototype form too: a coarse net narrower than the precise one is a hole
    // in the shape of a file extension. It looks for the hazard itself rather
    // than for an import of the module, because an import is only a proxy: a
    // component that one day writes `import type { BaseMessage }` for a prop
    // signature is harmless, and a gate that reds on it would be training people
    // to weaken this file.
    //
    // The cost of scanning text is that a mention inside an SFC comment would
    // count. That is a loud, obvious failure with an obvious fix, and it is the
    // right side of the trade for files that hold no message logic today.
    const sfcs = trackedFiles().filter((file) => file.endsWith('.vue'));
    expect(sfcs.length).toBeGreaterThan(0);

    const classAlternatives = [...MESSAGE_CLASSES].join('|');
    const banned = new RegExp(
      `instanceof\\s+(?:${classAlternatives})\\b` +
        `|toBeInstanceOf\\(\\s*(?:${classAlternatives})\\s*\\)` +
        `|(?:${classAlternatives})\\s*\\.\\s*prototype\\b`
    );
    const offenders = sfcs.filter((file) =>
      banned.test(readFileSync(`${REPO_ROOT}${file}`, 'utf8'))
    );
    expect(offenders).toEqual([]);

    // The pattern can match — proved here, so an empty offender list is evidence
    // of absence rather than of a regex that never fires.
    expect(banned.test('if (m instanceof HumanMessage) {}')).toBe(true);
    expect(banned.test('expect(x).toBeInstanceOf(ToolMessage)')).toBe(true);
    expect(banned.test('if (p === SystemMessage.prototype) {}')).toBe(true);
    expect(banned.test('if (llm instanceof ChatOpenAI) {}')).toBe(false);
    expect(banned.test('if (Object.getPrototypeOf(m) === Object.prototype) {}')).toBe(false);
    expect(banned.test('if (getPrototypeOf(a) === getPrototypeOf(b)) {}')).toBe(false);
  });
});
