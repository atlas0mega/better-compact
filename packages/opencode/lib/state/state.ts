import type { SessionState, WithParts } from "./types"
import type { Logger } from "../logger"
import { loadSessionState, saveSessionState } from "./persistence"
import { completeBoundaryJob, failBoundaryJob } from "../boundary/progress"
import {
    isSubAgentSession,
    findLastCompactionTimestamp,
    countTurns,
    resetOnCompaction,
} from "./utils"
export async function refreshSessionState(
    state: SessionState,
    messages: WithParts[],
    logger: Logger,
): Promise<void> {
    const lastCompactionTimestamp = findLastCompactionTimestamp(messages)
    if (lastCompactionTimestamp > state.lastCompaction) {
        state.lastCompaction = lastCompactionTimestamp
        resetOnCompaction(state)
        logger.info("Detected compaction - reset stale state", {
            timestamp: lastCompactionTimestamp,
        })

        saveSessionState(state, logger).catch((error) => {
            logger.warn("Failed to persist state reset after compaction", {
                error: error instanceof Error ? error.message : String(error),
            })
        })
    }

    state.currentTurn = countTurns(state, messages)
}

export function createSessionState(sessionId: string | null = null): SessionState {
    return {
        sessionId,
        isSubAgent: false,
        compressPermission: undefined,
        boundary: {
            job: null,
            activePlan: null,
        },
        lastCompaction: 0,
        currentTurn: 0,
        modelContextLimit: undefined,
    }
}

export async function initializeSessionState(
    client: any,
    state: SessionState,
    sessionId: string,
    logger: Logger,
    messages: WithParts[],
): Promise<void> {
    if (state.sessionId !== null && state.sessionId !== sessionId) {
        throw new Error(`Session state ${state.sessionId} cannot be initialized as ${sessionId}`)
    }

    state.sessionId = sessionId

    const isSubAgent = await isSubAgentSession(client, sessionId)
    state.isSubAgent = isSubAgent

    state.lastCompaction = findLastCompactionTimestamp(messages)
    state.currentTurn = countTurns(state, messages)

    const persisted = await loadSessionState(sessionId, logger)
    if (persisted === null) {
        return
    }

    state.boundary.activePlan = persisted.boundary?.activePlan ?? null
    state.boundary.job = persisted.boundary?.job ?? null
    state.boundary.automaticCheck = persisted.boundary?.automaticCheck
    state.boundary.queuedManual = persisted.boundary?.queuedManual
    state.boundary.lastIdleUsageMessageId = persisted.boundary?.lastIdleUsageMessageId
    state.boundary.lastPlannedUsageMessageId = persisted.boundary?.lastPlannedUsageMessageId
    // The intent is persisted before manual work begins. A stored plan and
    // completed store stage are committed in the SAME state write: a crash
    // after that point must not repeat the compaction or change its prefix.
    const job = state.boundary.job
    const intent = state.boundary.queuedManual
    const sameJob = intent?.phase === "running" && intent.jobId === job?.id
    if (
        sameJob &&
        job?.status === "running" &&
        job.stages.some((stage) => stage.id === "store" && stage.status === "completed") &&
        state.boundary.activePlan
    ) {
        completeBoundaryJob(state, "Recovered stored Better Compact plan")
        state.boundary.queuedManual = undefined
        await saveSessionState(state, logger)
    } else if (sameJob && job?.status === "completed") {
        // The job finished before the process exited, but clearing its intent
        // was the next write. Never run an already completed manual job again.
        state.boundary.queuedManual = undefined
        await saveSessionState(state, logger)
    } else if (intent?.phase === "running" && (!sameJob || job?.status !== "failed")) {
        if (sameJob && job?.status === "running") {
            failBoundaryJob(state, "Interrupted by OpenCode restart; queued to resume.")
        }
        state.boundary.queuedManual = { ...intent, phase: "recovery" }
        await saveSessionState(state, logger)
    } else if (sameJob && job?.status === "failed") {
        // A completed failure is not a crash; avoid an infinite retry loop.
        state.boundary.queuedManual = undefined
        await saveSessionState(state, logger)
    } else if (job?.status === "running") {
        // Older plugin builds did not persist a resumable manual intent.
        failBoundaryJob(
            state,
            "Interrupted by OpenCode restart; run Better Compact again if needed.",
        )
        await saveSessionState(state, logger)
    }
}
