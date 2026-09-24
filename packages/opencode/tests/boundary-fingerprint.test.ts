import assert from "node:assert/strict"
import test from "node:test"
import { boundaryRangeHash } from "../lib/boundary/fingerprint"
import type { WithParts } from "../lib/state"

function toolMessage(messageID: string, sessionID: string, inputID: string): WithParts {
    return {
        info: {
            id: messageID,
            sessionID,
            role: "assistant",
            time: { created: 1 },
        } as WithParts["info"],
        parts: [
            {
                id: `${messageID}-part`,
                messageID,
                sessionID,
                type: "tool",
                tool: "read",
                callID: "call-1",
                state: {
                    status: "completed",
                    input: { id: inputID },
                    output: "result",
                    title: "read",
                    metadata: {},
                    time: { start: 1, end: 2 },
                },
            } as WithParts["parts"][number],
        ],
    }
}

test("boundary fingerprint ignores transport IDs after a fork", () => {
    const source = toolMessage("source-message", "source-session", "customer-a")
    const fork = toolMessage("fork-message", "fork-session", "customer-a")

    assert.equal(boundaryRangeHash([source]), boundaryRangeHash([fork]))
})

test("boundary fingerprint retains semantic payload fields named id", () => {
    const first = toolMessage("message", "session", "customer-a")
    const second = toolMessage("message", "session", "customer-b")

    assert.notEqual(boundaryRangeHash([first]), boundaryRangeHash([second]))
})

test("prefix identity ignores usage and completion metadata but detects actual source revisions", () => {
    const original = toolMessage("message", "session", "customer-a")
    const accounting = structuredClone(original)
    Object.assign(accounting.info, {
        cost: 3,
        tokens: { total: 32_000, input: 30_000, output: 2_000, reasoning: 5_000 },
        finish: "stop",
        time: { created: 1, completed: 10 },
    })
    Object.assign(accounting.parts[0], { time: { start: 1, end: 10 } })
    const state = (accounting.parts[0] as any).state
    state.time = { start: 2, end: 10 }
    assert.equal(boundaryRangeHash([original]), boundaryRangeHash([accounting]))

    state.output = "new result"
    assert.notEqual(boundaryRangeHash([original]), boundaryRangeHash([accounting]))
})

test("reasoning timing cannot invalidate the prefix, but reasoning content still can", () => {
    const original = toolMessage("message", "session", "customer-a")
    original.parts.push({
        id: "reasoning-part",
        messageID: original.info.id,
        sessionID: original.info.sessionID,
        type: "reasoning",
        text: "Important conclusion",
    } as WithParts["parts"][number])
    const updated = structuredClone(original)
    Object.assign(updated.parts[1], { time: { start: 3, end: 6 } })
    assert.equal(boundaryRangeHash([original]), boundaryRangeHash([updated]))
    Object.assign(updated.parts[1], { text: "Different conclusion" })
    assert.notEqual(boundaryRangeHash([original]), boundaryRangeHash([updated]))
})

test("accumulated user-message diff summaries do not change the provider prefix", () => {
    const original: WithParts = {
        info: {
            id: "message",
            sessionID: "session",
            role: "user",
            time: { created: 1 },
            summary: { diffs: [] },
        } as WithParts["info"],
        parts: [
            {
                id: "part",
                messageID: "message",
                sessionID: "session",
                type: "text",
                text: "Keep the original request",
            },
        ],
    }
    const later = structuredClone(original)
    Object.assign(later.info, {
        summary: { diffs: [{ file: "src/parser.ts", patch: "updated", additions: 2 }] },
    })
    assert.equal(boundaryRangeHash([original]), boundaryRangeHash([later]))
    const text = later.parts[0] as { text: string }
    text.text = "Changed user instruction"
    assert.notEqual(boundaryRangeHash([original]), boundaryRangeHash([later]))
})
