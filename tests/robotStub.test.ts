import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest'
import {
  createRobotStubApp,
  createRobotState,
  MAX_STEPS,
  MOVEMENT_ENDPOINTS,
  TRICK_ENDPOINTS,
  type RobotState,
} from '../robot-stub/robotStub.js'
import type { Server } from 'node:http'

let server: Server
let state: RobotState
let baseUrl: string

beforeAll(async () => {
  const result = createRobotStubApp()
  state = result.state

  await new Promise<void>((resolve) => {
    server = result.app.listen(0, () => {
      const addr = server.address()
      if (addr && typeof addr !== 'string') {
        baseUrl = `http://localhost:${addr.port}`
      }
      resolve()
    })
  })
})

afterAll(async () => {
  await new Promise<void>((resolve) => {
    server.close(() => resolve())
  })
})

beforeEach(async () => {
  await fetch(`${baseUrl}/reset`, { method: 'POST' })
})

describe('Movement endpoints', () => {
  it('forward without steps defaults to 1 cycle', async () => {
    const res = await fetch(`${baseUrl}/forward`)
    expect(res.status).toBe(200)
    const body = await res.json()
    expect(body).toEqual({ action: 'forward', steps: 1 })
    expect(state.lastCommand).toBe('forward')
    expect(state.lastSteps).toBe(1)
  })

  it('forward with explicit steps', async () => {
    const res = await fetch(`${baseUrl}/forward?steps=4`)
    const body = await res.json()
    expect(body).toEqual({ action: 'forward', steps: 4 })
    expect(state.lastSteps).toBe(4)
  })

  it('clamps steps above MAX_STEPS', async () => {
    const res = await fetch(`${baseUrl}/forward?steps=999`)
    const body = await res.json()
    expect(body.steps).toBe(MAX_STEPS)
  })

  it('clamps steps below 1', async () => {
    const res = await fetch(`${baseUrl}/forward?steps=0`)
    const body = await res.json()
    expect(body.steps).toBe(1)
  })

  it('treats non-integer steps as 1', async () => {
    const res = await fetch(`${baseUrl}/forward?steps=abc`)
    const body = await res.json()
    expect(body.steps).toBe(1)
  })

  it.each(MOVEMENT_ENDPOINTS)('records last command for %s', async (path) => {
    const res = await fetch(`${baseUrl}${path}?steps=2`)
    expect(res.status).toBe(200)
    expect(state.lastCommand).toBe(path.slice(1))
    expect(state.lastSteps).toBe(2)
  })
})

describe('Distance endpoint', () => {
  it('returns numeric string with one decimal', async () => {
    const res = await fetch(`${baseUrl}/distance`)
    expect(res.status).toBe(200)
    const body = await res.text()
    expect(body).toMatch(/^\d+\.\d$/)
    const distance = parseFloat(body)
    expect(distance).toBeGreaterThan(20)
    expect(distance).toBeLessThan(30)
  })

  it('updates lastDistanceCm in status', async () => {
    await fetch(`${baseUrl}/distance`)
    expect(state.lastDistanceCm).not.toBeNull()
  })
})

describe('Status endpoint', () => {
  it('returns full heartbeat shape', async () => {
    await fetch(`${baseUrl}/forward?steps=3`)
    const res = await fetch(`${baseUrl}/status`)
    const data = await res.json()
    expect(data.lastCommand).toBe('forward')
    expect(data.lastSteps).toBe(3)
    expect(typeof data.uptimeMs).toBe('number')
    expect(typeof data.lastCommandAtMs).toBe('number')
  })

  it('returns null fields before any commands', async () => {
    const res = await fetch(`${baseUrl}/status`)
    const data = await res.json()
    expect(data.lastCommand).toBeNull()
    expect(data.lastSteps).toBeNull()
    expect(data.lastCommandAtMs).toBeNull()
    expect(data.lastDistanceCm).toBeNull()
  })
})

describe('Trick endpoints', () => {
  it.each(TRICK_ENDPOINTS)('records single-cycle trick %s', async (path) => {
    const res = await fetch(`${baseUrl}${path}`)
    expect(res.status).toBe(200)
    expect(await res.json()).toEqual({ action: path.slice(1) })
    expect(state.lastCommand).toBe(path.slice(1))
    expect(state.lastSteps).toBe(1)
  })
})

describe('Command history', () => {
  it('records commands in order', async () => {
    await fetch(`${baseUrl}/forward`)
    await fetch(`${baseUrl}/turn_left?steps=8`)
    await fetch(`${baseUrl}/forward?steps=2`)
    await fetch(`${baseUrl}/backward`)

    expect(state.commandHistory).toHaveLength(4)
    expect(state.commandHistory[0].name).toBe('forward')
    expect(state.commandHistory[1]).toMatchObject({ name: 'turn_left', steps: 8 })
    expect(state.commandHistory[2]).toMatchObject({ name: 'forward', steps: 2 })
    expect(state.commandHistory[3].name).toBe('backward')
  })
})

describe('Unknown endpoints', () => {
  it('returns 404', async () => {
    const res = await fetch(`${baseUrl}/control?var=robot&val=1`)
    expect(res.status).toBe(404)
  })
})

describe('createRobotState', () => {
  it('creates clean initial state', () => {
    const fresh = createRobotState()
    expect(fresh.lastCommand).toBeNull()
    expect(fresh.lastSteps).toBeNull()
    expect(fresh.lastCommandAtMs).toBeNull()
    expect(fresh.lastDistanceCm).toBeNull()
    expect(fresh.commandHistory).toHaveLength(0)
    expect(fresh.simulatedDistance).toBe(25.0)
  })
})
