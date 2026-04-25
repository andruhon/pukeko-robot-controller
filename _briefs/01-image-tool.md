# Webcam → Vision via `capture_image` Tool

> **Status: implemented (2026-04-26).**
> Shipped via:
> - `@gaunt-sloth/core@0.0.4` — `metadata.client === true` interrupt-wrapping in `extractAndFlattenTools`, `streamWithEventsResume`, `GraphInterrupt` handling in `streamWithEvents`.
> - `@gaunt-sloth/api@0.0.3` — `forwardedProps.command.resume` routing in `apiAgUiModule`, `express.json` body limit raised to 5 MB.
> - `@galvanized-pukeko/vue-ui@0.0.4` — `clientTools` / `clientToolHandlers` props on `ChatInterface`, `chatService.resumeWithCommand`, `onRunStart` callback.
> - This repo: `server/index.ts` boots gaunt-sloth as a library; `src/agent/captureImageTool.ts` and `src/agent/frontendImageInjectionMiddleware.ts` carry the robot-specific tool and image-injection middleware; `src/App.vue` wires the handler to `WebcamPanel`; `.gsloth.system.md` carries the system prompt.
>
> Verified end-to-end in browser: agent calls `capture_image`, receives the live webcam frame, and accurately describes the scene.
>
> History compaction for accumulated frames remains future work.

## Context

The robot controller has a webcam feed and a chat to a vision-capable agent (`claude-sonnet-4-5` via gaunt-sloth). Goal: let the agent **iteratively perceive** — issue a movement, then *decide on its own* to call a `capture_image` tool, look at the result, and repeat. Auto-attaching every frame is wrong because (a) the agent shouldn't always pay vision tokens, and (b) iterative tasks need on-demand perception, not synchronous human attachments.

Planned shape: `capture_image` is a **frontend-fulfilled tool** suspended via LangGraph `interrupt()` and resumed via `Command({ resume })` carried over AG-UI's `forwardedProps.command` convention. The browser captures a frame on resume, the resume value flows in as the tool's return → becomes a `ToolMessage`, and a server middleware translates it to a `HumanMessage` with multimodal blocks before the next model invocation.

History bloat from accumulated frames is acknowledged but **out of scope** — the user has signalled custom compaction (collapsing past frames, leaving the latest) is a separate future track.

### Repos under our control (all checked out under `_refs/`)

1. `pukeko-robot-controller` — this app
2. `_refs/gaunt-sloth-assistant` — agent server (AG-UI)
3. `_refs/galvanized-pukeko-ai-ui` — vue-ui (`ChatInterface`, `chatService`)
4. `_refs/ag-ui` — AG-UI protocol (TS SDK at `sdks/typescript/`). Reference only — no fork needed for v1.
5. `_refs/langchainjs` — LangChain JS. Reference only — no fork needed for v1; the existing `binaryContentInjectionMiddleware` already proves the LangChain Anthropic image-block shape works end-to-end.

### Findings from exploration

- **AG-UI resume convention** is `forwardedProps.command = { resume: <value>, interruptEvent: { toolCallId, runId, … } }` (per CopilotKit/Mastra; see `_refs/ag-ui/integrations/mastra/typescript/src/mastra.ts:100-160`). `RunAgentParameters` (the input to `HttpAgent.runAgent`) carries `forwardedProps` through to `RunAgentInput.forwardedProps` at `_refs/ag-ui/sdks/typescript/packages/client/src/agent/agent.ts:378`.
- **LangGraph JS** ships `interrupt()` and `Command` (`node_modules/@langchain/langgraph/dist/interrupt.d.ts`). `interrupt(value)` pauses the graph (throws `GraphInterrupt` internally), persisting state via the checkpointer; resume by invoking the graph with `new Command({ resume })`. The `interrupt(...)` expression then returns the resume value to the calling node — perfect for a tool body that yields a placeholder and resumes with the real result.
- **gaunt-sloth `GthLangChainAgent`** at `_refs/gaunt-sloth-assistant/packages/core/src/core/GthLangChainAgent.ts` runs all tools server-side via `createAgent` from `langchain` (line 136). `streamWithEvents` (line 333) yields tool events as the agent streams; there's no API yet for resuming a paused graph with a `Command`. We must add one.
- **Checkpointing.** `apiAgUiModule.ts:85-87` already creates a `MemorySaver` and passes a `thread_id` from the AG-UI `threadId` (line 125). That suffices for resume on the same process — exactly the local-only deployment model gaunt-sloth's API targets (line 63's warning).
- **AG-UI `ToolMessage.content`** is `string` only (`_refs/ag-ui/sdks/typescript/packages/core/src/types.ts:138-145`). LangChain JS `ToolMessage.content` likewise lacks robust multimodal-block support across providers (the existing `binaryContentInjectionMiddleware.ts:15` comment confirms). The image must travel as a **string** (a JSON envelope), and a server middleware translates that into a `HumanMessage` with image blocks before the next model call. The pattern is already established by `binaryContentInjectionMiddleware` (`_refs/gaunt-sloth-assistant/packages/tools/src/middleware/binaryContentInjectionMiddleware.ts`) — just keyed on a different tool name.
- **Canonical LangChain Anthropic image block** (proven working, used by binary middleware lines 75-87):
  ```ts
  { type: 'image', source_type: 'base64', mime_type: 'image/jpeg', data: '<base64>' }
  ```

## End-to-end flow

1. **Client declares the tool.** `chatService.sendMessage` includes `capture_image` (no parameters) in its outbound `RunAgentInput.tools`.
2. **Agent decides to call it.** Model emits `tool_calls: [{ name: 'capture_image', args: {} }]`. LangGraph routes to the tool node.
3. **Server suspends via `interrupt`.** The server-registered `capture_image` is a stub whose body is `return await interrupt({ name: 'capture_image', toolCallId })`. LangGraph throws `GraphInterrupt`; `streamWithEvents` emits `tool_start` / `tool_args` / `tool_end` for the call, **does not emit `tool_result`**, and the run finishes (`RUN_FINISHED`) with the tool call hanging. State is persisted in `MemorySaver` keyed by `threadId`.
4. **Client fulfills.** Browser sees `TOOL_CALL_END` for `capture_image`, captures a frame from `WebcamPanel.captureFrame()`, parses the dataURL, builds a JSON envelope `{ mimeType, data }` (or an error envelope `{ error: '…' }`), and submits a new run carrying:
   ```ts
   forwardedProps: { command: { resume: <envelope-string>, interruptEvent: { toolCallId, runId } } }
   ```
   No new `messages` are appended — same `threadId`, no `messages` mutation; the resume command is the entire payload.
5. **Server resumes.** `apiAgUiModule.ts` reads `forwardedProps.command`. If `resume` is present, it invokes the agent with `new Command({ resume })` instead of feeding messages. LangGraph wakes the paused tool node; `interrupt(...)` returns the resume value, the tool's body returns it, LangGraph wraps it in a `ToolMessage` keyed to the original `toolCallId`.
6. **Image injection middleware.** `frontendImageInjectionMiddleware.beforeModel` detects a fresh `ToolMessage` whose name is `capture_image` and whose content parses as `{ mimeType, data }`. It appends a `HumanMessage` with `[ {type:'text',text:'Camera frame:'}, {type:'image',source_type:'base64',mime_type,data} ]`. (Error envelopes pass through with a text-only "camera unavailable" `HumanMessage`.)
7. **Model sees the image.** The next model call receives the image as a content block. Anthropic vision answers; agent continues iteratively (may issue movement tools, then call `capture_image` again).

## Plan

### 1. Server (gaunt-sloth)

**A. Register `capture_image` as an interrupt-suspended tool.**

Two sub-options for declaration:
- **A1** Accept the tool list from `req.body.tools` in `apiAgUiModule.ts` and dynamically register it on each request.
- **A2** Static declaration in `.gsloth.config.json` with metadata `{ client: true }` and the `pukeko-robot-controller`'s config opts in.

Prefer **A2** for v1 — simpler, no per-request agent rebuild, composes with gaunt-sloth's existing config-driven tool system.

**File:** `_refs/gaunt-sloth-assistant/packages/core/src/core/GthLangChainAgent.ts`
- At tool-flattening (line 78, `extractAndFlattenTools`) detect tools tagged `client: true` in their metadata and replace their `func` with a stub that calls `interrupt({ name: <toolName>, toolCallId })` from `@langchain/langgraph`. The stub body:
  ```ts
  const value = await interrupt({ name, toolCallId: getCurrentToolCallId() });
  return typeof value === 'string' ? value : JSON.stringify(value);
  ```
- Maintain a `clientToolNames: Set<string>` on the agent for diagnostic logging only (LangGraph handles the actual suspend/resume; we don't need per-stream filtering).
- In `streamWithEvents` (line 333) — when the underlying `agent.streamEvents` surfaces a `GraphInterrupt` chunk (or no `tool_result` follows a `tool_end` for a client tool), end the iterator gracefully. Most likely no code change needed beyond catching `GraphInterrupt` if it bubbles out.
- **Add a resume entry point.** New method `streamWithEventsResume(resumeValue: unknown, runConfig)` that invokes the underlying LangGraph agent via `this.agent!.streamEvents(new Command({ resume: resumeValue }), runConfig, …)` and emits the same event stream as `streamWithEvents`.

**File:** `_refs/gaunt-sloth-assistant/packages/api/src/modules/apiAgUiModule.ts`
- Lift `express.json` body limit (line 60): `express.json({ limit: '5mb' })`.
- Read `forwardedProps` from `req.body` (line 93). If `forwardedProps?.command?.resume !== undefined`:
  - Skip `convertMessage` mapping.
  - Skip the `buildSystemMessages` prepend (this is a resume of an existing thread; system messages are already in the checkpoint).
  - Call `agent.streamWithEventsResume(forwardedProps.command.resume, runConfig)` instead of `agent.streamWithEvents(...)`.
  - The rest of the SSE event loop is identical (same event types).
- Otherwise, current path (messages → `streamWithEvents`) unchanged.

**B. New middleware: `frontendImageInjectionMiddleware`.**

**File (new):** `_refs/gaunt-sloth-assistant/packages/tools/src/middleware/frontendImageInjectionMiddleware.ts`

Modeled on `binaryContentInjectionMiddleware.ts` but:
- Trigger: `ToolMessage` whose `name === 'capture_image'` (or any name in a config-supplied set) AND whose `content` parses as JSON `{ mimeType: string, data: string }`.
- Action (`beforeModel`): append a `HumanMessage` with `[ { type:'text', text:'Camera frame captured at <time>:' }, { type:'image', source_type:'base64', mime_type: <mimeType>, data: <data> } ]`.
- Error envelope (`{ error: '…' }`): append a text-only `HumanMessage` describing the failure (`"Camera unavailable: <error>"`) so the agent can react gracefully.
- Auto-register when any client tool is configured (analogous to the binary middleware auto-register at `packages/tools/src/builtInToolsConfig.ts`).

**Tests:** new spec covering:
- Valid envelope → injected `HumanMessage` with correct image block.
- Error envelope → text-only `HumanMessage`.
- Invalid JSON content → pass-through, no injection (defensive).
- Mixed history (server-executed tool result earlier + capture_image at end) → only the capture_image is translated.

### 2. vue-ui (`chatService` + `ChatInterface`)

**File:** `_refs/galvanized-pukeko-ai-ui/packages/galvanized-pukeko-vue-ui/src/services/chatService.ts`

- Allow callers to declare tools. Extend `sendMessage`:
  ```ts
  sendMessage(text: string, callbacks: ChatCallbacks, opts?: { tools?: Tool[] }): Promise<void>
  ```
  Pass through to `agent.runAgent({ tools }, subscriber)` — `RunAgentParameters` already accepts `tools` (`_refs/ag-ui/sdks/typescript/packages/client/src/agent/types.ts:47-49`).
- Add a sibling `resumeWithCommand(resumeValue: unknown, interruptEvent: { toolCallId: string; runId?: string }, callbacks: ChatCallbacks, opts?: { tools?: Tool[] }): Promise<void>` (do not modify the existing `submitToolResult` — `show_a2ui_surface` depends on its current `UserMessage`-adding behavior). The new method:
  ```ts
  await agent.runAgent(
    { tools: opts?.tools, forwardedProps: { command: { resume: resumeValue, interruptEvent } } },
    subscriber
  )
  ```
  Note: it does not call `agent.addMessage(...)`. The resume is the only payload.

**File:** `_refs/galvanized-pukeko-ai-ui/packages/galvanized-pukeko-vue-ui/src/components/ChatInterface.vue`

- Add optional props:
  ```ts
  defineProps<{
    a2ui?: …,
    clientTools?: Tool[],                                                        // declarations
    clientToolHandlers?: Record<string, (args: unknown, ctx: { toolCallId: string }) => Promise<string> | string>,
  }>()
  ```
- In `sendMessage` / `sendFormMessage` (lines 144, 181), pass `{ tools: props.clientTools }` to `chatService.sendMessage`.
- Stash the latest `runId` from `onTextMessageStartEvent` (or any event that carries it; if not exposed, capture it from the AG-UI subscriber) into a closure, so `resumeWithCommand` can include it in `interruptEvent`.
- In `onToolCallEnd` of `createStreamCallbacks` (lines 80-82): if `toolCallName` matches a key in `clientToolHandlers`:
  1. Parse `toolCallBuffer` (or use `toolCallArgs`) into `args`.
  2. `await handler(args, { toolCallId })` to obtain the result string.
  3. Call `chatService.resumeWithCommand(result, { toolCallId, runId }, createStreamCallbacks(), { tools: props.clientTools })`. Reuse the same callback pattern so streamed text from the resumed run goes through `messages.value`.
- Pre-existing hardcoded `show_a2ui_surface` branch stays; ensure the generic branch is gated on `if (toolCallName !== 'show_a2ui_surface')` (or runs only when a handler is registered) so behaviors don't double-fire.

**Tests:**
- `chatService.sendMessage` forwards `tools` to `runAgent`; `resumeWithCommand` invokes `runAgent` with the right `forwardedProps.command` shape and does not add a message.
- `ChatInterface`: when a tool call for a registered handler arrives, the handler is invoked with parsed args and `resumeWithCommand` is called with the right `runId` + `toolCallId` + result. Regression: `show_a2ui_surface` still works.

### 3. robot-controller

**`.gsloth.config.json`** — declare `capture_image` with `metadata: { client: true }`. Add a system-prompt hint that the agent can request a camera snapshot via `capture_image` whenever it needs to see the scene; the result will then appear as the next user-side message containing the image.

**`src/components/WebcamPanel.vue`** — extend `captureFrame` to accept `maxWidth = 512` and downscale via canvas; drop quality to ~0.7. Default-arg keeps existing tests/callers working.

**`src/App.vue`** — wire the tool:
```ts
const clientTools: Tool[] = [{
  name: 'capture_image',
  description: 'Capture the current frame from the robot-controller camera. Use this whenever you need to visually verify the scene or robot state.',
  parameters: { type: 'object', properties: {}, additionalProperties: false },
}]
const clientToolHandlers = {
  capture_image: async () => {
    const dataUrl = webcamRef.value?.captureFrame()
    if (!dataUrl) return JSON.stringify({ error: 'camera unavailable' })
    const m = dataUrl.match(/^data:([^;]+);base64,(.+)$/)
    if (!m) return JSON.stringify({ error: 'invalid frame' })
    return JSON.stringify({ mimeType: m[1], data: m[2] })
  },
}
```
- Pass to `<ChatInterface :client-tools="clientTools" :client-tool-handlers="clientToolHandlers" />`.

**Tests:** unit test for the handler — null camera → error envelope; real dataURL → correct envelope shape.

## Critical files

- `/home/parents/Documents/robots/pukeko-robot-controller/_refs/gaunt-sloth-assistant/packages/api/src/modules/apiAgUiModule.ts` (line 60: body limit; line 93: parse `forwardedProps`; route handler: branch on resume)
- `/home/parents/Documents/robots/pukeko-robot-controller/_refs/gaunt-sloth-assistant/packages/core/src/core/GthLangChainAgent.ts` (lines 70-90 tool flatten — wrap `client: true` tools with `interrupt()`; new `streamWithEventsResume` method)
- `/home/parents/Documents/robots/pukeko-robot-controller/_refs/gaunt-sloth-assistant/packages/tools/src/middleware/frontendImageInjectionMiddleware.ts` (NEW — sibling of `binaryContentInjectionMiddleware.ts`)
- `/home/parents/Documents/robots/pukeko-robot-controller/_refs/gaunt-sloth-assistant/packages/tools/src/builtInToolsConfig.ts` (auto-register the new middleware)
- `/home/parents/Documents/robots/pukeko-robot-controller/_refs/galvanized-pukeko-ai-ui/packages/galvanized-pukeko-vue-ui/src/services/chatService.ts` (extend `sendMessage`; add `resumeWithCommand`)
- `/home/parents/Documents/robots/pukeko-robot-controller/_refs/galvanized-pukeko-ai-ui/packages/galvanized-pukeko-vue-ui/src/components/ChatInterface.vue` (new props `clientTools`, `clientToolHandlers`; tool-result wiring in `onToolCallEnd`; track `runId`)
- `/home/parents/Documents/robots/pukeko-robot-controller/.gsloth.config.json` (declare `capture_image` with `client: true`; system-prompt hint)
- `/home/parents/Documents/robots/pukeko-robot-controller/src/components/WebcamPanel.vue` (add `maxWidth` arg)
- `/home/parents/Documents/robots/pukeko-robot-controller/src/App.vue` (pass tool + handler to ChatInterface)
- Reference (do not edit): `_refs/gaunt-sloth-assistant/packages/tools/src/middleware/binaryContentInjectionMiddleware.ts`, `_refs/ag-ui/sdks/typescript/packages/core/src/types.ts:135,201,222`, `_refs/ag-ui/integrations/mastra/typescript/src/mastra.ts:100-160` (resume convention), `node_modules/@langchain/langgraph/dist/interrupt.d.ts` (interrupt/Command API)

## Sequencing

Each step is independently verifiable; older client/server combos still work after each.

1. **Server: lift body limit** to 5 MB. One-line change. Smoke-test existing chat unchanged.
2. **Server: add `frontendImageInjectionMiddleware`** + auto-register hook. Unit-test in isolation: feed it a fake state with a `ToolMessage(name='capture_image', content=JSON envelope)` and assert a `HumanMessage` with the right blocks is appended. Error envelope → text `HumanMessage`. With no such message, state passes through. Existing chat unchanged.
3. **Server: interrupt-wrap `client: true` tools in `GthLangChainAgent`** + add `streamWithEventsResume`. Unit-test: invoke the agent with a configured client tool, model emits a tool call → confirm `tool_start/args/end` are yielded but no `tool_result`, and the iterator ends gracefully. Then invoke `streamWithEventsResume('test')` against the same `thread_id` → confirm a fresh `ToolMessage` containing `'test'` enters state and the agent continues. Existing chat unchanged when no client tools configured.
4. **Server: `apiAgUiModule.ts` resume branch.** Read `forwardedProps.command` from request body. If `resume` is present, route to `streamWithEventsResume`. Integration-test against a fake LangGraph agent.
5. **vue-ui: extend `chatService`** (`sendMessage` accepts `tools`; new `resumeWithCommand`). Unit-test with mocked `HttpAgent`. Old call sites unchanged.
6. **vue-ui: ChatInterface client-tool plumbing.** New props, tool-handler invocation in callbacks, `resumeWithCommand` invocation, `runId` tracking. Unit-test with a mocked handler. Old usage (no props) works as before.
7. **robot-controller: `WebcamPanel.captureFrame(maxWidth)` downscale.** Default-arg backwards compat.
8. **robot-controller: `.gsloth.config.json` + `App.vue` wiring.** Declare `capture_image` server-side and `clientTools`/`clientToolHandlers` client-side. Manual end-to-end (below).

## Verification

**Unit tests:**
- `_refs/gaunt-sloth-assistant`: `frontendImageInjectionMiddleware.spec.ts`; `GthLangChainAgent` interrupt + resume spec; `apiAgUiModule` resume-branch spec.
- `_refs/galvanized-pukeko-ai-ui/packages/galvanized-pukeko-vue-ui`: `chatService.spec.ts` (`tools` forwarding, `resumeWithCommand`); `ChatInterface.spec.ts` (handler invocation + resume call).
- `pukeko-robot-controller`: handler unit test (existing 26 tests + new).

**End-to-end manual** (no robot hardware required):
1. Start gaunt-sloth API server (per existing README's `npm run server`). Requires `ANTHROPIC_API_KEY`.
2. `npm run dev:ag-ui` → `http://localhost:5173`. Allow camera; confirm WebcamPanel preview.
3. Type **"Look at the camera and describe what you see."** → Send.
   - Expect: agent stream emits a tool-call badge for `capture_image`, then text describing the scene.
   - DevTools network: two POSTs to `/agents/.../run`. First body has `messages: [...]` and `tools: [{ name: 'capture_image', … }]`; SSE response has `TOOL_CALL_*` for `capture_image` and `RUN_FINISHED` with no `TOOL_CALL_RESULT`. Second POST body has **no new messages** (same `threadId`) and `forwardedProps: { command: { resume: '<envelope-string>', interruptEvent: { toolCallId, runId } } }`; SSE response is the agent's text answer.
4. Ask **"Now is anything different?"** without moving anything → expect the agent to call `capture_image` again, see the same scene, answer accordingly.
5. **Iterative test:** start the robot stub, ask **"Move forward and check what's in front."** → expect: tool calls for movement (existing robot tools) → `capture_image` → text answer. Validates the decision-loop.
6. **Camera unavailable path:** stop camera (revoke permission). Ask vision question → handler returns `{ error: 'camera unavailable' }` → middleware injects text-only `HumanMessage` → agent responds gracefully (e.g. "I cannot see the camera right now.").
7. Server logs: confirm no exceptions, run finishes cleanly, request body sizes well under 5 MB.

## Risks / open questions

- **`createAgent` interoperability with `interrupt`.** The version of `langchain` used by gaunt-sloth must surface `GraphInterrupt` cleanly when a tool body calls `interrupt()`. Verify with a smoke test before wiring the rest. If problems surface, fallback is to bypass `createAgent` and build a small custom ReAct loop that supports interrupts — significant scope creep, flag early.
- **`runId` capture in vue-ui.** `interruptEvent.runId` is required by the resume convention. If the AG-UI subscriber doesn't surface `runId` directly, derive it from the `RUN_STARTED` event (look in `_refs/ag-ui/sdks/typescript/packages/client/src/` for the typed event; if the existing chatService callbacks don't expose it, extend them).
- **MemorySaver lifetime.** `apiAgUiModule.ts` creates a single `MemorySaver` for the process (line 85). The `threadId` keys persist across the server's lifetime — fine for local dev, but a server restart between the suspend and resume rounds drops state. Document this; out of scope to fix.
- **Single-pause-point safety.** `interrupt()` may be called multiple times for multiple tool calls in one model turn. LangGraph handles this sequentially (per docs). Verify behavior when the model issues two `capture_image` calls in one turn (rare but possible). Likely fine because each interrupt is keyed to its own `toolCallId`.
- **Error envelope handling.** Settled: middleware injects text-only `HumanMessage` for error envelopes. Document this in the middleware spec.
- **Request size.** Even one frame is ~30–60 KB; multi-turn tasks with several `capture_image` calls + history will grow. Within the 5 MB body limit for typical sessions; long-running iterative tasks will need the user's planned compaction.
- **Tool name collision.** If gaunt-sloth ever ships a built-in `capture_image`, our `client: true` flag must take precedence. Document precedence at the wrapping site in `extractAndFlattenTools`.
- **A2UI interaction.** Existing `show_a2ui_surface` is hardcoded in `ChatInterface.vue`. The generic `clientToolHandlers` branch must skip names already handled by A2UI. Add a regression test.
- **History growth across runs.** Out of scope (user-acknowledged future work) but worth flagging for the implementer to leave structured tool-call/tool-result IDs intact so a future compaction pass can drop image envelopes from prior turns surgically.