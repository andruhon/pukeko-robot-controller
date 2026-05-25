# AGENTS.md — Pukeko Robot Controller

A Vue web UI + AG-UI backend that lets an LLM drive an Acebott biped robot in front of a webcam. Tuned to work well with local Gemma 4 via Ollama; also runs against Claude (Anthropic) and the robot stub.

## Run

```sh
# robot stub (port 8080)
npm run stub

# AG-UI backend (port 3000)
ROBOT_HOST=localhost:8080 npm run server

# web UI (port 5173)
npm run dev:ag-ui
```

Real robot: skip `npm run stub` and connect to the robot's Wi-Fi AP. Set `ROBOT_HOST=192.168.4.1` (default) or whatever IP it picks up.

## Architecture

- **`server/index.ts`** — boots `@gaunt-sloth/api` AG-UI server. Wires LLM, tools, and middleware.
- **`src/agent/robotTools.ts`** — declares the agent's tools.
  - **Motion tools** (`move_forward`, `move_backward`, `turn_left`, `turn_right`) are **client-fulfilled** (`metadata: { client: true }`). The browser orchestrates the whole capture+move+capture+compose sequence — see below.
  - **Sensor & control tools** (`read_distance`, `read_status`, `stop`) stay server-side and hit the robot's HTTP API directly.
  - **`capture_image`** is the original standalone client-fulfilled camera tool, kept for mid-sequence alignment checks.
- **`src/agent/frontendImageInjectionMiddleware.ts`** — converts the JSON envelope `{ mimeType, data, ... }` that comes back from `capture_image` / motion tools into a provider-shaped `HumanMessage` content block (Ollama: `image_url`; Anthropic: `image`). For motion envelopes it also stamps the sensor delta line ("Distance: 25.4 cm → 27.1 cm (Δ +1.7 cm)").
- **`src/agent/motionSummarizationMiddleware.ts`** — concurrent-with-motion history compression. `afterModel` detects a motion tool call in the assistant message and immediately fires a summarization LLM call (uses the same `llm`). `beforeModel` on the next turn awaits the summary and rewrites the message history to keep only: the original user prompt verbatim, a single `SystemMessage` summary, and the most recent motion turn cluster.
- **`src/App.vue` + `src/components/WebcamPanel.vue`** — Vue UI. Camera capture happens in the hidden `<canvas>` in `WebcamPanel`. The motion-tool handlers in `App.vue` fetch the robot directly (`http://${VITE_ROBOT_HOST}/forward?steps=N`), capture Before/After frames, and compose them via `WebcamPanel.composeBeforeAfter`. **Requires CORS on the robot** — both `robot-stub/robotStub.ts` and `acebot-biped/for-agents/Biped_Robot_Web.py` send `Access-Control-Allow-Origin: *`.

## Conventions worth knowing

- Default model: Gemma 4 via Ollama (`OLLAMA_MODEL=gemma4:31b`). Optimized for small models — the motion-tools auto-capture + summarization combo means the LLM doesn't have to remember to bracket motions with `capture_image` and the context window stays trim.
- Parallel tool calls are **disabled** on Anthropic (`disable_parallel_tool_use: true` in `server/createLlm.ts`) because client-fulfilled tools rely on a LangGraph interrupt and parallel batching breaks the resume ordering.
- The agent's behavioural prompt lives in `system-prompt.md` at the repo root (the operating mindset, calibration sequence, distance-sensor caveats). Update there, not in code, when changing how the agent thinks. It's loaded via gaunt-sloth's configurable `projectGuidelines` slot (set in `server/index.ts`), overridable per profile with `systemPromptPath`.
- The motion-summarization prompt lives in `summarization-prompt.md` at the repo root (what each compressed history must carry forward — objective, calibration outcomes, action log, findings). Overridable per profile with `summaryPromptPath`. `src/agent/motionSummarizationMiddleware.ts` keeps an identical `DEFAULT_SUMMARY_PROMPT` as a fallback for when the file is absent — keep the two in sync.
- Briefs in `_briefs/` describe planned/completed work. `01-*` documented the `capture_image` interrupt/resume flow; `02-*` documented the motion-tool auto-capture and summarization work that this AGENTS.md describes.

## Tests

`npm test` — 40+ vitest cases covering the robot stub HTTP behaviour, the webcam panel, and the motion-summarization middleware. The before/after canvas composition is **not** covered by unit tests (jsdom has no canvas) — verify it in the browser instead.

## Smoke-testing in the browser

Always exercise a motion tool in a real Chrome tab before declaring UI changes done. Type-check + unit-test passes are not sufficient for the camera + canvas path. Open `http://localhost:5173`, allow camera permission, and ask the agent to e.g. "turn right one cycle". The tool-call badge should expand to show a single composite image with **Before / After** labels and a divider, plus the sensor delta line.
