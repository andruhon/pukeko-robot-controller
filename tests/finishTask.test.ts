import { describe, it, expect } from 'vitest'
import { isToolMessage } from '@langchain/core/messages'
import type { StructuredToolInterface } from '@langchain/core/tools'
import { createRobotTools } from '../src/agent/robotTools.js'

function getFinishTool(): StructuredToolInterface {
  const tools = createRobotTools('localhost:8080')
  const t = tools.find((x) => x.name === 'finish_task')
  if (!t) throw new Error('finish_task tool not found')
  return t
}

// Invoke the tool the way ToolNode does: with a ToolCall (so the result is a
// named ToolMessage) plus a config carrying the thread_id.
function invokeFinish(
  tool: StructuredToolInterface,
  status: string,
  summary: string,
  id = 'tc-finish'
) {
  return tool.invoke(
    { name: 'finish_task', args: { status, summary }, id, type: 'tool_call' },
    { configurable: { thread_id: 't1' } }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ) as Promise<any>
}

describe('finish_task tool', () => {
  it('is registered as a server tool', () => {
    const tools = createRobotTools('localhost:8080')
    expect(tools.map((t) => t.name)).toContain('finish_task')
  })

  it('is returnDirect so createAgent routes straight to END after it runs', () => {
    const tool = getFinishTool()
    // returnDirect on the StructuredTool is what the agent router keys off of.
    expect((tool as unknown as { returnDirect?: boolean }).returnDirect).toBe(true)
  })

  it('returns a named FINISH ToolMessage (router matches on the name → END)', async () => {
    const tool = getFinishTool()
    const result = await invokeFinish(tool, 'success', 'reached the cone')
    expect(isToolMessage(result)).toBe(true)
    expect(result.name).toBe('finish_task')
    expect(result.content).toBe('FINISH[success]: reached the cone')
  })

  it('encodes the status in the FINISH payload', async () => {
    const tool = getFinishTool()
    const failed = await invokeFinish(tool, 'failed', 'tried 5 approaches, stuck')
    expect(failed.content).toBe('FINISH[failed]: tried 5 approaches, stuck')
    const needInput = await invokeFinish(tool, 'need_input', 'where is it?')
    expect(needInput.content).toBe('FINISH[need_input]: where is it?')
  })
})
