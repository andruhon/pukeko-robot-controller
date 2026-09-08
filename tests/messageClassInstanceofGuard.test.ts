import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import ts from 'typescript';

/**
 * RC-58 — nothing in this repo may test a message's type with `instanceof`.
 *
 * The rule. A LangChain message's class is not a reliable answer to "what kind of
 * message is this?" here, because the robot can resolve more than one
 * `@langchain/core` at runtime: its own, plus whatever the `@gaunt-sloth/*`
 * dependencies pull in. A message built inside gaunt-sloth's AG-UI pipeline is
 * not an instance of the class this repo imports, and RC-21 is where that cost a
 * real bug — `msg instanceof ToolMessage` returned false for every capture result
 * on the live server, no frame was ever injected, and the dumps read
 * `tool-data:1 / human-images:0 / imageCount:0`. The duck-typed forms
 * (`isToolMessage`, `isHumanMessage`, `getType()`) answer correctly whatever copy
 * the message came from.
 *
 * Why a spec and not a review habit. The constraint has been restated in four
 * separate node texts and re-caught by a human reviewer twice, and RC-58 still
 * found it violated in six files at once — because a test written next to an
 * `instanceof` one copies the style, and a suite built entirely from native
 * fixtures passes either way. Nothing about a violation is visible in a green
 * run. This is the thing that looks.
 *
 * What it does NOT ban. Comparing two messages' prototypes to EACH OTHER
 * (`Object.getPrototypeOf(out) === Object.getPrototypeOf(input)`) is safe and
 * load-bearing: production clones messages prototype-first precisely so a
 * foreign message survives a rewrite as itself, and several specs pin that.
 * Those name no class, so the detector below leaves them alone. `instanceof`
 * against a non-message class — `ChatOpenAI`, `Error` — is likewise none of this
 * spec's business.
 *
 * The one exemption. `tests/rc21CrossCoreToolMessage.test.ts` uses `instanceof`
 * deliberately, as the control that pins the hazard itself: it asserts a
 * foreign-copy message fails `instanceof ToolMessage` while passing
 * `isToolMessage`. Removing that would delete the only direct pin on the
 * condition this whole rule exists for, so it is exempt by name — and the spec
 * asserts the exemption is still being USED, so a silent rewrite of the control
 * cannot quietly retire it.
 *
 * The file list comes from `git ls-files`, as in `gatedSourceCoverage.test.ts`,
 * so build output can never trip it and every tracked file is covered. As with
 * that spec, this one therefore fails inside a `git archive` extraction for lack
 * of a `.git` — that is the price of reading the tracked set, not a defect.
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

/**
 * Every `x instanceof MessageClass` and `expect(x).toBeInstanceOf(MessageClass)`
 * in one source text. Both spellings are the same hazard: vitest's matcher runs
 * the `instanceof` operator.
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
    if (
      ts.isCallExpression(node) &&
      ts.isPropertyAccessExpression(node.expression) &&
      node.expression.name.text === 'toBeInstanceOf' &&
      node.arguments.length === 1 &&
      ts.isIdentifier(node.arguments[0]) &&
      MESSAGE_CLASSES.has((node.arguments[0] as ts.Identifier).text)
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

describe('RC-58 no `instanceof` against a LangChain message class', () => {
  it('detects both banned spellings and nothing else', () => {
    // The discriminating control. Run the real detector over a source that holds
    // one of each banned spelling plus four near-misses, so a detector that
    // matched everything — or nothing — cannot look like success.
    const sample = [
      "const a = msg instanceof ToolMessage",
      "expect(x).toBeInstanceOf(HumanMessage)",
      "expect(x).not.toBeInstanceOf(SystemMessage)",
      // Near-misses that must NOT be flagged:
      "const b = llm instanceof ChatOpenAI",
      "const c = err instanceof Error",
      "expect(free).toBeInstanceOf(ChatAnthropic)",
      "const d = Object.getPrototypeOf(out) === Object.getPrototypeOf(input)",
      "// a comment mentioning msg instanceof ToolMessage and HumanMessage",
      "const e = 'a string saying x instanceof AIMessage'",
    ].join('\n');

    const hits = violationsIn('sample.ts', sample);
    expect(hits.map((h) => h.line)).toEqual([1, 2, 3]);
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

  it('no tracked file outside the control tests a message type with `instanceof`', () => {
    const offenders = allViolations().filter((v) => v.file !== EXEMPT);
    expect(
      offenders.map((v) => `${v.file}:${v.line}  ${v.text}`),
      'Use the duck-typed check instead — isToolMessage / isHumanMessage / getType(). ' +
        'The robot can resolve two @langchain/core copies, so a message built by ' +
        "gaunt-sloth's pipeline is not an instance of the class this repo imports " +
        '(RC-21). See tests/helpers/foreignCoreMessage.ts for a fixture that proves it.'
    ).toEqual([]);
  });

  it('no Vue SFC tests a message type with `instanceof` either', () => {
    // The AST detector needs a TypeScript program and an SFC is not one, so the
    // .vue files get a textual scan for the same two spellings. It looks for the
    // hazard itself rather than for an import of the module, because an import
    // is only a proxy: a component that one day writes `import type
    // { BaseMessage }` for a prop signature is harmless, and a gate that reds on
    // it would be training people to weaken this file.
    //
    // The cost of scanning text is that a mention inside an SFC comment would
    // count. That is a loud, obvious failure with an obvious fix, and it is the
    // right side of the trade for files that hold no message logic today.
    const sfcs = trackedFiles().filter((file) => file.endsWith('.vue'));
    expect(sfcs.length).toBeGreaterThan(0);

    const classAlternatives = [...MESSAGE_CLASSES].join('|');
    const banned = new RegExp(
      `instanceof\\s+(?:${classAlternatives})\\b|toBeInstanceOf\\(\\s*(?:${classAlternatives})\\s*\\)`
    );
    const offenders = sfcs.filter((file) =>
      banned.test(readFileSync(`${REPO_ROOT}${file}`, 'utf8'))
    );
    expect(offenders).toEqual([]);

    // The pattern can match — proved here, so an empty offender list is evidence
    // of absence rather than of a regex that never fires.
    expect(banned.test('if (m instanceof HumanMessage) {}')).toBe(true);
    expect(banned.test('expect(x).toBeInstanceOf(ToolMessage)')).toBe(true);
    expect(banned.test('if (llm instanceof ChatOpenAI) {}')).toBe(false);
  });
});
