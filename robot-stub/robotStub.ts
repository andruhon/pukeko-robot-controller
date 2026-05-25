import express, { type Express, type Request, type Response } from 'express'

/**
 * Test stub matching the agent build of for-agents/Biped_Robot_Web.py.
 * Endpoints are named after the action they perform; movement endpoints
 * accept an optional ?steps=N (default 1, capped at 10).
 */
export const MAX_STEPS = 10

export const MOVEMENT_ENDPOINTS = ['/forward', '/backward', '/turn_left', '/turn_right'] as const
export type MovementPath = (typeof MOVEMENT_ENDPOINTS)[number]

export const TRICK_ENDPOINTS = [
  '/sprint',
  '/dance',
  '/avoid',
  '/follow',
  '/kick_left',
  '/kick_right',
  '/tilt_left',
  '/tilt_right',
  '/stamp_left',
  '/stamp_right',
  '/ankles_left',
  '/ankles_right',
] as const
export type TrickPath = (typeof TRICK_ENDPOINTS)[number]

export interface RobotState {
  lastCommand: string | null
  lastSteps: number | null
  lastCommandAtMs: number | null
  lastDistanceCm: number | null
  bootMs: number
  commandHistory: Array<{ name: string; steps: number; timestamp: number }>
  simulatedDistance: number
}

export function createRobotState(): RobotState {
  return {
    lastCommand: null,
    lastSteps: null,
    lastCommandAtMs: null,
    lastDistanceCm: null,
    bootMs: Date.now(),
    commandHistory: [],
    simulatedDistance: 25.0,
  }
}

function clampSteps(raw: unknown): number {
  if (typeof raw !== 'string') return 1
  const n = parseInt(raw, 10)
  if (isNaN(n) || n < 1) return 1
  return Math.min(n, MAX_STEPS)
}

function recordCommand(state: RobotState, name: string, steps: number) {
  state.lastCommand = name
  state.lastSteps = steps
  state.lastCommandAtMs = Date.now() - state.bootMs
  state.commandHistory.push({ name, steps, timestamp: Date.now() })
}

export function createRobotStubApp(state?: RobotState): { app: Express; state: RobotState } {
  const robotState = state ?? createRobotState()
  const app = express()

  // CORS so the browser-side motion tool handler can call /distance, /forward,
  // etc. directly. The real robot firmware (Biped_Robot_Web.py) does the same.
  app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
    if (req.method === 'OPTIONS') {
      res.status(204).end()
      return
    }
    next()
  })

  app.get('/', (_req, res) => {
    res.type('text/plain').send('Acebott biped robot - agent API.\n')
  })

  app.get('/status', (_req, res) => {
    res.json({
      uptimeMs: Date.now() - robotState.bootMs,
      lastCommand: robotState.lastCommand,
      lastSteps: robotState.lastSteps,
      lastCommandAtMs: robotState.lastCommandAtMs,
      lastDistanceCm: robotState.lastDistanceCm,
    })
  })

  app.get('/distance', (_req, res) => {
    const noise = (Math.random() - 0.5) * 2
    const distance = Math.max(2, robotState.simulatedDistance + noise)
    robotState.lastDistanceCm = distance
    res.type('text/plain').send(distance.toFixed(1))
  })

  for (const path of MOVEMENT_ENDPOINTS) {
    app.get(path, (req: Request, res: Response) => {
      const name = path.slice(1)
      const steps = clampSteps(req.query.steps)
      recordCommand(robotState, name, steps)
      res.json({ action: name, steps })
    })
  }

  for (const path of TRICK_ENDPOINTS) {
    app.get(path, (_req, res) => {
      const name = path.slice(1)
      recordCommand(robotState, name, 1)
      res.json({ action: name })
    })
  }

  // Test helpers (not on the real robot).
  app.post('/reset', (_req, res) => {
    robotState.lastCommand = null
    robotState.lastSteps = null
    robotState.lastCommandAtMs = null
    robotState.lastDistanceCm = null
    robotState.bootMs = Date.now()
    robotState.commandHistory = []
    robotState.simulatedDistance = 25.0
    res.status(200).json({ reset: true })
  })

  return { app, state: robotState }
}
