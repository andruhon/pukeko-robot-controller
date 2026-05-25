# Pukeko Robot Controller

An educational, browser-based controller that lets a large language model drive a physical
device through an **observe → reason → act** loop. A webcam watches the device; the operator
gives a high-level goal ("move to the centre of the table") and the agent iterates
autonomously — capture a frame, reason about what it sees, issue a command, repeat — until the
goal is met.

The reference target is an Acebott ESP32 STEM biped robot, but that is only an example.
Because motion commands are fulfilled in the browser against a small HTTP control API, the same
loop can drive a different contraption by adapting the client-side tools and the agent's system
prompt. The LLM, the web UI, and the control loop are device-agnostic.

> ⚠️ **Educational use only.** The agent autonomously actuates real hardware and can make
> mistakes. Run it only in a controlled environment under adult supervision.
> See [SECURITY.md](./SECURITY.md).

## How it works

- **Frontend** — Vue 3 + the Pukeko web UI: renders the webcam feed and chat, and fulfils the
  motion and camera tools directly in the browser.
- **Backend** — a [`@gaunt-sloth/api`](https://www.npmjs.com/package/@gaunt-sloth/api) AG-UI
  server that runs the agent, wiring the LLM, tools, and middleware.
- **Device** — any host that exposes a small HTTP control API. A bundled stub provides one so
  you can run the loop without hardware.

**This project does not include device firmware.** Building it is part of the exercise: expose
your contraption's actions (motors, servos, sensors) over a simple HTTP/REST interface and the
agent can drive it. As a worked example, firmware for the Acebott biped that exposes its
controls as a REST API is published separately at
[andruhon/acebot-biped-robot-qd021-mpremote](https://github.com/andruhon/acebot-biped-robot-qd021-mpremote/tree/main/for-agents).

Architecture and internals are documented in [AGENTS.md](./AGENTS.md).

## Prerequisites

- Node.js ≥ 22
- An LLM backend: a local [Ollama](https://ollama.com) server with a vision- and tool-capable
  model, **or** an `ANTHROPIC_API_KEY` for Claude
- A webcam accessible to the browser

## Quick start (no hardware)

The bundled stub mimics the device's HTTP API, so the full loop runs without a robot:

```sh
npm install
npm run stub                              # device stub on :8080
ROBOT_HOST=localhost:8080 npm run server  # AG-UI backend on :3000
npm run dev:ag-ui                         # web UI on :5173
```

Open <http://localhost:5173>, allow camera access, and give the agent a goal.

## Running against hardware

Flash your device with firmware that exposes its actions over HTTP (the project ships none —
see above), then point the backend at the device's host and start the backend and UI:

```sh
ROBOT_HOST=<device-ip> npm run server
npm run dev:ag-ui
```

The example Acebott biped serves its own Wi-Fi access point at `192.168.4.1`, which is the
default `ROBOT_HOST`.

## Configuration

- **LLM / profile** — `pukeko.config.ts` (copy from `pukeko.config.example.ts`). Select a
  profile with `PUKEKO_PROFILE=<name>`, and override individual fields with env vars
  (`LLM_PROVIDER`, `OLLAMA_MODEL`, `ANTHROPIC_MODEL`, `ROBOT_HOST`, …).
- **Agent behaviour** — `system-prompt.md`: the operating mindset, the device description, and
  sensor guidance.
- **Driving a different device** — provide firmware that exposes your device's actions over
  HTTP, adapt the client tool definitions and handlers in `src/App.vue` to those endpoints, and
  rewrite `system-prompt.md` to describe the device.

## Scripts

| Script | Description |
|---|---|
| `npm run dev:ag-ui` | Web UI (connects to the backend on :3000) |
| `npm run server` | AG-UI backend on :3000 |
| `npm run stub` | Device HTTP stub on :8080 |
| `npm test` | Unit tests |
| `npm run build` | Type-check and production build |

## Contributing & license

- [CONTRIBUTING.md](./CONTRIBUTING.md) — dev setup, local development registry, PR checklist
- [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- [SECURITY.md](./SECURITY.md) — safety disclaimer & vulnerability reporting
- Licensed under the [MIT License](./LICENSE).
