import {
    createEngine,
    resolveCompactionProfile,
    type BoundaryContextPlan,
    type BoundarySummaryJob,
    type EnginePorts,
    type PlanSnapshot,
} from "@better-compact/core"
import type { PluginConfig } from "../config"
import type { Logger } from "../logger"
import { openCodeCodec, openCodeSpec, sessionKeyOf } from "../codec"
import { saveSessionState, type SessionState, type WithParts } from "../state"
import { boundaryRangeHash } from "./fingerprint"
import { isSyndicatePluginInjection } from "../messages/injection"
import { createTranscriptStore } from "./transcripts"
import { adaptiveTailUserTurns } from "./context"

// The auto transform path: replay the session's cached plan when it still
// holds, otherwise build, persist, and apply a fresh one. Mutates the
// messages array in place only when the engine changed anything, and only
// after the new plan is durably persisted (a failed save must not leave a
// transformed request without its plan). Returns the freshly built plan so
// the caller can surface it, or null when nothing new was planned.
export async function processBoundaryTransform(input: {
    state: SessionState
    logger: Logger
    config: PluginConfig
    directory: string
    messages: WithParts[]
    providerReportedTokens?: number
    summariesAllowed?: boolean
    summarize?: (jobs: BoundarySummaryJob[]) => Promise<Record<string, string>>
    summarizePrefix?: (
        plan: BoundaryContextPlan,
        turns: import("@better-compact/core").Turn[],
    ) => Promise<string | null | undefined>
}): Promise<BoundaryContextPlan | null> {
    let prefixChunkAttempted = false
    const oldPlan = input.state.boundary.activePlan
    const ports: EnginePorts = {
        transcripts: createTranscriptStore(input.directory),
        plans: {
            load: () => input.state.boundary.activePlan,
            save: async (_sessionKey, snapshot) => {
                const previous = input.state.boundary.activePlan
                input.state.boundary.activePlan = snapshot
                    ? stampForkIdentity(
                          {
                              ...snapshot,
                              ...(prefixChunkAttempted ||
                              (oldPlan?.rangeHash === snapshot.rangeHash &&
                                  oldPlan.prefixChunkAttempted)
                                  ? { prefixChunkAttempted: true as const }
                                  : {}),
                          },
                          input.messages,
                      )
                    : null
                try {
                    await saveSessionState(input.state, input.logger)
                } catch (error) {
                    input.state.boundary.activePlan = previous
                    throw error
                }
            },
        },
        logger: input.logger,
    }
    const profile = resolveCompactionProfile(input.config)
    const oldTailIndex = oldPlan
        ? input.messages.findIndex((message) => message.info.id === oldPlan.rawTailStartMessageId)
        : -1
    const migratePluginInjections =
        !!oldPlan &&
        oldPlan.pluginInjectionPruning !== true &&
        oldTailIndex > 0 &&
        input.messages.slice(0, oldTailIndex).some(isSyndicatePluginInjection)
    const migrateUnboundedPrefix =
        !!oldPlan?.requiresCustomCompaction &&
        oldPlan.afterPruneTokens > oldPlan.targetTokens &&
        oldPlan.prefixChunkAttempted !== true &&
        (oldPlan.prefixSummary?.match(/^- Resume from prior assistant progress: /gm)?.length ??
            0) >= 12 &&
        profile.prefixSummary &&
        input.summariesAllowed !== false &&
        !!input.summarizePrefix
    const engine = createEngine(openCodeSpec, ports)
    const result = await engine.process({
        sessionKey: sessionKeyOf(input.messages),
        turns: openCodeCodec.encode(input.messages),
        contextLimit: input.state.modelContextLimit,
        triggerRatio: profile.triggerPercent / 100,
        targetRatio: profile.targetPercent / 100,
        triggerTokens: input.config.compaction.triggerTokens ?? undefined,
        targetTokens: input.config.compaction.targetTokens ?? undefined,
        recentToolResultBudgetTokens: profile.recentToolTokens,
        minTailUserTurns: adaptiveTailUserTurns(
            input.messages,
            input.state.modelContextLimit ?? 1,
            profile.targetPercent,
            input.config.compaction.targetTokens,
        ),
        prefixSummaryAllowed: profile.prefixSummary,
        collapsePercent: profile.collapsePercent,
        providerReportedTokens: input.providerReportedTokens,
        summariesAllowed: input.summariesAllowed,
        summarize: input.summariesAllowed === false ? undefined : input.summarize,
        summarizePrefix:
            input.summariesAllowed === false || !input.summarizePrefix
                ? undefined
                : async (plan, turns) => {
                      prefixChunkAttempted = true
                      return input.summarizePrefix!(plan, turns)
                  },
        force: migratePluginInjections || migrateUnboundedPrefix,
    })
    if (result.outcome === "unchanged") return null
    const decoded = openCodeCodec.decode(result.turns, input.messages)
    input.messages.length = 0
    input.messages.push(...decoded)
    return result.outcome === "planned" ? result.plan : null
}

function stampForkIdentity(snapshot: PlanSnapshot, messages: WithParts[]) {
    const tagged = messages.some(isSyndicatePluginInjection)
        ? { ...snapshot, pluginInjectionPruning: true as const }
        : snapshot
    if (snapshot.rawTailItemBoundary !== undefined) return tagged
    const tailIndex = messages.findIndex(
        (message) => message.info.id === snapshot.rawTailStartMessageId,
    )
    if (tailIndex <= 0) return tagged
    const prefix = messages.slice(0, tailIndex)
    return {
        ...tagged,
        prefixFingerprint: boundaryRangeHash(prefix),
        compactedMessageCount: prefix.length,
    }
}
