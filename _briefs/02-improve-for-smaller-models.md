The program has proven to work very well with local gemma 4. We need a number of improvements to make it more reliable.

## New motion tools process

When one of the motion tools move_forward, move_backward, turn_left, turn_right is called,
the following sequence should happen:

In parallel:

1. Launch summarization of the prompt. Make sure that the initial user prompt is clearly retained, make sure that none of the previous photographs is included, make sure from the summary it is clear what was happening prior to the command call. 

2. At the same time, call the tool, which should do the following: 
- Register the state before
  - Read the sensor reading before
  - Capture the camera snapshot before
- Send the command request to the robot
- Register the state after
  - Read the sensor reading after
  - Capture the camera snapshot after
- Combine the photograph into one image before on the left, after on the right, "Before" label above the before photo, "After" label above the after photo. There should be a distinct spacing or a line between before and after.

## References

We have full source code of the following dependencies:

### Internal tools under our control, which we can modify
../gaunt-sloth-assistant
../galvanized-pukeko-ai-ui

## 3rd party tools we cannot modify
../langchainjs
../langgraphjs
../ag-ui

We can't modify these, but we can create a PR in dire circumstances.
You may want to update to newer versions of these libraries.

## Documentation

- Update README.md if necessary.
- Create a slim AGENTS.md with the most important information, don't let it get too long.
- Create CLAUDE.md pointing to AGENTS.md

## Testing

Smoke test in the browser, the camera is on, robot is ON. Don't skip testing in browser.
