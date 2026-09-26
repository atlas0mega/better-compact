import assert from "node:assert/strict"
import test from "node:test"
import { createRuntimeState, saveSessionState, type WithParts } from "../lib/state"
import { createBoundaryJob } from "../lib/boundary/progress"
import { Logger } from "../lib/logger"

function userMessage(sessionID: string): WithParts {
    return {
        info: {
            id: `user-${sessionID}`,
            sessionID,
            role: "user",
            agent: "assistant",
            model: { providerID: "test", modelID: "model" },
            time: { created: 1 },
        } as WithParts["info"],
        parts: [],
    }
}

test("runtime state has stable isolated ownership per session", () => {
    const runtime = createRuntimeState({}, new Logger(false))

    const first = runtime.get("session-a")
    const same = runtime.get("session-a")
    const second = runtime.get("session-b")

    assert.equal(first, same)
    assert.notEqual(first, second)
    assert.equal(first.sessionId, "session-a")
    assert.equal(second.sessionId, "session-b")
})

test("runtime state coalesces concurrent initialization without crossing sessions", async () => {
    const calls = new Map<string, number>()
    const client = {
        session: {
            get: async ({ path }: { path: { id: string } }) => {
                calls.set(path.id, (calls.get(path.id) ?? 0) + 1)
                await new Promise((resolve) => setTimeout(resolve, 5))
                return { data: { parentID: null } }
            },
        },
    }
    const runtime = createRuntimeState(client, new Logger(false))

    const [first, same, second] = await Promise.all([
        runtime.prepare("session-a", [userMessage("session-a")]),
        runtime.prepare("session-a", [userMessage("session-a")]),
        runtime.prepare("session-b", [userMessage("session-b")]),
    ])

    assert.equal(first, same)
    assert.notEqual(first, second)
    assert.equal(calls.get("session-a"), 1)
    assert.equal(calls.get("session-b"), 1)
})

test("runtime state keeps model limits and scratch sessions in their own namespaces", () => {
    const runtime = createRuntimeState({}, new Logger(false))
    runtime.setModelLimit("provider-a", "model", 1_000_000)
    runtime.setModelLimit("provider-b", "model", 200_000)
    const untrack = runtime.trackScratch("scratch")

    assert.equal(runtime.getModelLimit("provider-a", "model"), 1_000_000)
    assert.equal(runtime.getModelLimit("provider-b", "model"), 200_000)
    assert.equal(runtime.isScratch("scratch"), true)

    untrack()
    assert.equal(runtime.isScratch("scratch"), false)
})

test("runtime state allows only one compaction per session", async () => {
    const runtime = createRuntimeState({}, new Logger(false))
    let release!: () => void
    const pending = new Promise<void>((resolve) => {
        release = resolve
    })

    assert.equal(
        runtime.startCompaction("session-a", () => pending),
        true,
    )
    assert.equal(
        runtime.startCompaction("session-a", async () => {}),
        false,
    )
    assert.equal(
        runtime.startCompaction("session-b", async () => {}),
        true,
    )

    release()
    await runtime.activeCompaction("session-a")
    assert.equal(
        runtime.startCompaction("session-a", async () => {}),
        true,
    )
})

test("a restart marks orphaned compaction progress interrupted without losing a saved plan or queue", async () => {
    const sessionID = `ses-restart-job-${process.pid}-${Date.now()}`
    const logger = new Logger(false)
    const client = { session: { get: async () => ({ data: { parentID: null } }) } }
    const runtime = createRuntimeState(client, logger)
    const state = runtime.get(sessionID)
    const plan = { rangeHash: "committed-plan" } as NonNullable<typeof state.boundary.activePlan>
    state.boundary.activePlan = plan
    state.boundary.job = createBoundaryJob({ sessionId: sessionID })
    state.boundary.queuedManual = { requestedAt: Date.now(), lastUserMessageId: "prior-user" }
    await saveSessionState(state, logger)

    const restarted = createRuntimeState(client, logger)
    const recovered = await restarted.prepare(sessionID, [userMessage(sessionID)])
    assert.equal(recovered.boundary.job?.status, "failed")
    assert.match(recovered.boundary.job?.error ?? "", /Interrupted by OpenCode restart/)
    assert.deepEqual(recovered.boundary.activePlan, plan)
    assert.ok(recovered.boundary.queuedManual)
    const reloaded = await createRuntimeState(client, logger).prepare(sessionID, [
        userMessage(sessionID),
    ])
    assert.equal(reloaded.boundary.job?.status, "failed", "interruption is persisted once")
})

test("a new manual intent survives a crash before its job replaces an older failed job", async () => {
    const sessionID = `ses-restart-before-job-${process.pid}-${Date.now()}`
    const logger = new Logger(false)
    const client = { session: { get: async () => ({ data: { parentID: null } }) } }
    const state = createRuntimeState(client, logger).get(sessionID)
    state.boundary.job = createBoundaryJob({ sessionId: sessionID, id: "bc_old_failed" })
    state.boundary.job.status = "failed"
    state.boundary.queuedManual = {
        requestedAt: Date.now(),
        phase: "running",
        jobId: "bc_new_pending",
        lastUserMessageId: `user-${sessionID}`,
    }
    await saveSessionState(state, logger)

    const resumed = await createRuntimeState(client, logger).prepare(sessionID, [
        userMessage(sessionID),
    ])
    assert.equal(resumed.boundary.queuedManual?.phase, "recovery")
    assert.equal(resumed.boundary.queuedManual?.jobId, "bc_new_pending")
    assert.equal(resumed.boundary.job?.status, "failed")
})
