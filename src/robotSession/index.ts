// RobotSession barrel (RC-7). App.vue and tests import from here.
export { RobotSession } from './RobotSession.js';
export type { RobotSessionOptions, ClientToolHandler } from './RobotSession.js';
export {
  runRecipe,
  frameToEnvelope,
  coerceSteps,
  namedCameraFailure,
  type RobotCapabilities,
  type BrowserCapabilities,
  type CallScopedCapabilities,
} from './interpreter.js';
export {
  REAL_ROBOT_WORLD_ID,
  SIMULATED_WORLD_ID,
  WORLDS,
  DEFAULT_ROBOT_HOST,
  DEFAULT_EMULATOR_HOST,
  CAPTURE_PATH,
  resolveHost,
  hostForWorld,
  captureUrlForWorld,
  createWorldCapabilities,
  createWorldSession,
  type WorldId,
  type WorldOption,
  type WorldHosts,
  type WebcamPanelLike,
  type WorldCapabilitiesDeps,
  type WorldSessionOptions,
} from './worlds.js';
