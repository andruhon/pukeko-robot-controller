# Pukeko Robot Controller

AI-driven control of an Acebott biped robot via the Pukeko web UI and `@gaunt-sloth/api` backend.

A webcam is mounted above a table, pointing down at the robot. The user issues high-level goals
(e.g. "move the robot to the center of the table") and the LLM agent iterates autonomously:
observe (capture a camera frame), reason about the robot's position, act (issue locomotion commands
and/or read the ultrasonic sensor), then repeat until the goal is met.

## Prerequisites

- Node.js >= 22
- An LLM backend — either:
  - A local [Ollama](https://ollama.com) server running a vision-capable, tool-supporting model (default: `qwen3-vl:8b`), or
  - An `ANTHROPIC_API_KEY` for Claude
- A webcam accessible to the browser

## Choosing an LLM

The backend picks its model from environment variables when `npm run server` starts.

| Var | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `anthropic` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Used when provider is `ollama` |
| `OLLAMA_MODEL` | `gemma4:31b` | Used when provider is `ollama`. Must support tool calling and ideally vision. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Used when provider is `anthropic` |
| `ROBOT_HOST` | `192.168.4.1` | Robot HTTP API host |

Ollama happy path:

```sh
ollama pull qwen3-vl:8b
npm run server   # picks up LLM_PROVIDER=ollama implicitly
```

Anthropic opt-in:

```sh
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-... npm run server
```

## Install

```sh
npm install
```

`@galvanized-pukeko/vue-ui` and `@gaunt-sloth/*` are pulled from npm. During
local development against unpublished versions of those packages, this repo
expects a [Verdaccio](https://verdaccio.org) registry at `http://localhost:4873`
— the gitignored `.npmrc` at the repo root routes those scopes there:

```
@gaunt-sloth:registry=http://localhost:4873
@galvanized-pukeko:registry=http://localhost:4873
```

If you only want to consume the published versions on the public registry,
delete `.npmrc` before `npm install`. Verdaccio container/auth setup is
documented in [`gaunt-sloth-assistant/CONTRIBUTING.md`](https://github.com/Galvanized-Pukeko/gaunt-sloth-assistant/blob/main/CONTRIBUTING.md#local-development-registry-optional).

## Running with the robot stub (development / no hardware)

Start three processes in separate terminals:

```sh
# Terminal 1 — robot stub (mimics Biped_Robot_Web.py on port 8080)
npm run stub

# Terminal 2 — AG-UI backend (port 3000), pointing tools at the stub
ROBOT_HOST=localhost:8080 npm run server

# Terminal 3 — web UI (port 5173)
npm run dev:ag-ui
```

Open http://localhost:5173 in a browser, allow camera access, and chat with the agent.
The stub logs every command it receives and exposes `GET /status` for inspection.

## Running with the real robot

The Acebott biped robot creates its own Wi-Fi AP. Connect your machine's Wi-Fi interface
to the robot's network (default IP `192.168.4.1`). Use a wired ethernet connection for
internet access to reach the Anthropic API.

Start two processes in separate terminals:

```sh
# Terminal 1 — AG-UI backend (port 3000), using the real robot's default IP
npm run server

# Terminal 2 — web UI (port 5173)
npm run dev:ag-ui
```

If the robot is on a different IP, override it:

```sh
ROBOT_HOST=192.168.4.1 npm run server
```

## Scripts

| Script | Description |
|---|---|
| `npm run dev` | Vite dev server (UI only, no backend) |
| `npm run dev:ag-ui` | Vite dev server pre-configured to connect to AG-UI on port 3000 |
| `npm run server` | Start `@gaunt-sloth/api` AG-UI backend on port 3000 |
| `npm run stub` | Start robot HTTP stub on port 8080 |
| `npm test` | Run all unit tests |
| `npm run build` | Type-check and build for production |

## Tests

```sh
npm test
```

36 tests across robot stub (HTTP API behaviour, command mapping, sensor simulation)
and web client (WebcamPanel component, useRobotChat composable).

## Project structure

```
├── src/                    Vue 3 frontend
│   ├── main.ts             Entry: loads config, mounts app
│   ├── App.vue             Layout: camera feed + chat side-by-side
│   ├── components/
│   │   └── WebcamPanel.vue getUserMedia camera with captureFrame()
│   └── composables/
│       └── useRobotChat.ts AG-UI streaming + capture_image UI tool
├── robot-stub/             Test double for Biped_Robot_Web.py
│   ├── robotStub.ts        Express app: /control, /status, /reset
│   └── index.ts            Server entry (port 8080)
├── tests/                  Unit tests (Vitest)
├── .gsloth.config.json     AG-UI server config (LLM, robot tools, CORS)
├── vite.config.ts          Vite build + aliases
└── vitest.config.ts        Test runner config
```

## Motion tools

The four motion tools — `move_forward`, `move_backward`, `turn_left`, `turn_right` — are
**client-fulfilled** and auto-instrument every call. When the agent invokes one, the browser
handler runs the following sequence and returns a single tool result:

1. `read_distance` → ultrasonic reading **before**
2. capture **Before** frame from the webcam
3. send the motion request to the robot (`GET /forward?steps=N`, etc.)
4. `read_distance` → ultrasonic reading **after**
5. capture **After** frame from the webcam
6. compose the two frames into one image via `<canvas>` — labels "Before" / "After", a divider in the middle, both halves aligned to a shared baseline
7. return `{ mimeType, data, motion, distanceBefore, distanceAfter }` as the tool result

The agent therefore sees a single composite image plus the sensor delta on every move, with no
need to call `capture_image` or `read_distance` around motions explicitly. `capture_image`,
`read_distance`, `read_status`, and `stop` remain available for mid-sequence checks and
emergencies.

### Concurrent history summarization

While the browser is doing the capture+move+capture+compose round-trip, the server fires a
summarization LLM call in parallel (see `src/agent/motionSummarizationMiddleware.ts`). The
summary keeps the user's original prompt verbatim, drops every prior image content block, and
condenses what the agent has been doing. Before the next model turn, the middleware splices the
summary into the message history in place of the older scratch turns — keeping the context tight
for small local models.

### Robot HTTP API

The robot (firmware `acebot-biped/for-agents/Biped_Robot_Web.py`) and the stub both expose:

| Path | Method | Description |
|---|---|---|
| `/forward?steps=N` | GET | Walk N forward cycles (default 1, max 10). Blocks until done. |
| `/backward?steps=N` | GET | As above, reversed. |
| `/turn_left?steps=N` | GET | Rotate left in place. |
| `/turn_right?steps=N` | GET | Rotate right in place. |
| `/stop` | GET | Halt servos. |
| `/distance` | GET | Plain-text ultrasonic reading in cm (`-1.0` on failure). |
| `/status` | GET | JSON heartbeat. |
| `OPTIONS *` | OPTIONS | CORS preflight (204). |

All responses send `Access-Control-Allow-Origin: *` so the browser-side motion-tool handlers
can fetch the robot directly. If you fork the firmware, keep the CORS headers — the UI depends
on them.
