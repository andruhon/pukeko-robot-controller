# Pukeko Robot Controller Project

## Concept

AI-driven control of an Acebott biped robot via the Pukeko web UI and `@gaunt-sloth/api` backend.

A webcam is mounted above a table, pointing down at the robot. The user issues high-level goals
(e.g. "move the robot to the center of the table") and the LLM agent iterates autonomously:
observe (capture a camera frame), reason about the robot's position, act (issue locomotion commands
and/or read the ultrasonic sensor), then repeat until the goal is met.

### Components

1. **Robot** — Acebott ESP32 biped running MicroPython. Hosts its own Wi-Fi AP and an HTTP
   control API (`/control?var=robot&val=N`). See `_refs/Biped_Robot_Web.py` (Available in https://github.com/andruhon/acebot-biped-robot-qd021-mpremote/blob/main/fixed/lesson7/Biped_Robot_Web.py).
2. **Pukeko web client** — Vue 3 UI served by the Pukeko web-client package. Provides:
   - Chat interface for user commands
   - Webcam view (browser `getUserMedia`) exposed as a **UI tool** (AG-UI) so the backend agent can
     request a snapshot at any time
3. **`@gaunt-sloth/api` backend** — Express + AG-UI server running LangChain/LangGraph agent
   with the following tools:
   - `capture_image` — UI tool: requests a camera frame from the browser and returns it to the model
   - `move_forward` — HTTP call to robot (`val=1`)
   - `move_backward` — HTTP call to robot (`val=2`)
   - `turn_left` — HTTP call to robot (`val=3`)
   - `turn_right` — HTTP call to robot (`val=4`)
   - `stop` — HTTP call to robot (`val=8`)
   - `read_distance` — HTTP call to robot ultrasonic sensor, returns distance in cm

### Robot HTTP API (from `Biped_Robot_Web.py`)

| val | Action        | Notes                              |
|-----|---------------|------------------------------------|
| 1   | Forward       | Single step, then auto-stops       |
| 2   | Backward      | Single step, then auto-stops       |
| 3   | Turn left     | Single step, then auto-stops       |
| 4   | Turn right    | Single step, then auto-stops       |
| 8   | Stop          | Halt all motion                    |
| 10  | Sprint        |                                    |
| 11  | Left kick     |                                    |
| 12  | Right kick    |                                    |
| 13  | Left tilt     |                                    |
| 14  | Right tilt    |                                    |
| 15  | Left stamp    |                                    |
| 16  | Dance         |                                    |
| 17  | Avoid         | Autonomous obstacle avoidance loop |
| 18  | Follow        | Autonomous follow mode             |
| 19  | Left ankles   |                                    |
| 20  | Right stamp   |                                    |
| 21  | Right ankles  |                                    |

Ultrasonic sensor: Trig=GPIO13, Echo=GPIO14 (no existing HTTP endpoint — needs adding or proxying).

### LLM Providers

- **Claude Sonnet 4.6** (via Anthropic) — primary, cloud-based vision model
- **Qwen3-VL** (local) — alternative, locally-hosted vision model (OpenAI-compatible API)

Both must support vision (image input in tool results) for the observe-act loop to work.

### Agent Loop

```
User goal → Agent
  ┌─────────────────────────────┐
  │ 1. capture_image            │
  │ 2. Reason about position    │
  │ 3. Issue locomotion command │
  │ 4. Optionally read_distance │
  │ 5. Goal met? → respond      │
  │    Not met?  → goto 1       │
  └─────────────────────────────┘
```

## Dependency: Pukeko (`galvanized-pukeko-ai-ui`)

- **Location:** `/home/parents/Documents/_pukeko`
- **Type:** npm monorepo (Node 22, TypeScript, ESM)
- **Packages:**
  - `galvanized-pukeko-vue-ui` — Vue 3 UI component library (composables, components, services)
  - `galvanized-pukeko-web-client` — Web client host that serves the Vue UI build, owns `config.json` and Playwright e2e tests
  - `galvanized-pukeko-agent-adk` — Java Spring Boot ADK agent (backend)
  - `gaunt-sloth-assistant` — TypeScript CLI tool (bundled copy)
- **Key deps:** `@gaunt-sloth/api`, `@langchain/openai`, `@modelcontextprotocol/sdk`, `express`, `vue`, `ws`, `zod`
- **Dev stack:** Vite 7, Vitest, Playwright, ESLint + oxlint, Prettier
- **Scripts of note:**
  - `npm run web` — dev server for web client
  - `npm run start-gth-ag-ui` — start Gaunt Sloth AG-UI server
  - `npm run it-adk` / `it-gth-ag-ui` — integration tests

## Dependency: `@gaunt-sloth/api` (from gaunt-sloth-assistant)

- **Location:** `/home/parents/Documents/_gaunt-sloth/gaunt-sloth-assistant/packages/api`
- **Repo:** `gaunt-sloth-workspace` — npm monorepo (Node >=22, TypeScript 5, ESM)
- **Packages in workspace:**
  - `@gaunt-sloth/core` — core agent logic, config, LLM utils, middleware
  - `@gaunt-sloth/tools` — tool abstractions (filesystem toolkit, etc.)
  - `@gaunt-sloth/api` — API server: Express-based, exposes AG-UI + A2A + MCP endpoints
  - `@gaunt-sloth/review` — code review module
  - `gaunt-sloth-assistant` — CLI entrypoint (ask, review, pr, chat, code commands)
- **`@gaunt-sloth/api` exports:**
  - `apiAgUiModule` — Express router implementing the AG-UI protocol (SSE streaming, LangChain agent invocation)
  - `interactiveSessionModule` — interactive chat/code session factory
  - `A2AClientWrapper` / `A2AAgentTool` — Agent-to-Agent protocol support
  - `OAuthClientProviderImpl` — OAuth for MCP servers
  - `mcpUtils` — MCP tool resolution utilities
  - `createResolvers` — resolver factory for tools/MCP
- **Key deps:** `@ag-ui/core`, `@ag-ui/encoder`, `@a2a-js/sdk`, `@langchain/mcp-adapters`, `@modelcontextprotocol/sdk`, `express`
- **LLM providers (via @gaunt-sloth/core):** Anthropic, Vertex AI, Google AI Studio, Groq, DeepSeek, OpenAI-compatible, xAI
- **Architecture:** LangChain/LangGraph 0.3 based; middleware pattern for hooks (beforeModel, afterModel, wrapModelCall, wrapToolCall, etc.)
