import { createRobotStubApp } from './robotStub.js'

const port = parseInt(process.env.ROBOT_STUB_PORT ?? '8080', 10)
const { app, state } = createRobotStubApp()

const server = app.listen(port, () => {
  console.log(`Robot stub server running on http://localhost:${port}`)
  console.log('Endpoints:')
  console.log(`  GET /forward[?steps=N]       - Walk forward (default 1, max 10)`)
  console.log(`  GET /backward[?steps=N]      - Walk backward`)
  console.log(`  GET /turn_left[?steps=N]     - Rotate left in place`)
  console.log(`  GET /turn_right[?steps=N]    - Rotate right in place`)
  console.log(`  GET /stop                    - Freeze servos`)
  console.log(`  GET /distance                - Ultrasonic reading in cm`)
  console.log(`  GET /status                  - JSON heartbeat`)
  console.log(`  POST /reset                  - Reset stub state (test only)`)
})

process.on('SIGTERM', () => {
  console.log(`\nShutting down robot stub (processed ${state.commandHistory.length} commands)`)
  server.close()
})
