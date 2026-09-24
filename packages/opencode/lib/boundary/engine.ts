import {
    createEngine,
    buildPlan,
    prefixUserMessages,
    resolveCompactionProfile,
    type BoundaryContextPlan,
    type BoundarySummaryJob,
    type EnginePorts,
    type PlanSnapshot,
} from "@better-compact/core"
import type { PluginConfig } from "../config"
import type { Logger } from "../logger"
import { openCodeCodec, openCodeConventions, openCodeSpec, sessionKeyOf } from "../codec"
import { saveSessionState, type SessionState, type WithParts } from "../state"
import { boundaryRangeHash } from "./fingerprint"
import { isSyndicatePluginInjection } from "../messages/injection"
import { createTranscriptStore } from "./transcripts"
import { adaptiveTailUserTurns, efficientAgenticTailBudget } from "./context"
import {
    archiveBoundaryDelta,
    archiveOversizedSummary,
    eligibleRetirementThrough,
    expireArchives,
    liveArchiveDescriptions,
    loadArchiveCatalog,
    nativeMessageFingerprint,
    recordArchiveFailure,
    saveArchiveCatalog,
} from "./archive-catalog"
import type { ArchiveSummaryResult } from "./archive-summarizer"

export const PREFIX_CHUNK_VERSION = 2

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
    forceOverflow?: boolean
    onOutcome?: (outcome: "planned" | "replayed" | "unchanged") => void
    summariesAllowed?: boolean
    summarize?: (jobs: BoundarySummaryJob[]) => Promise<Record<string, string>>
    summarizePrefix?: (
        plan: BoundaryContextPlan,
        turns: import("@better-compact/core").Turn[],
    ) => Promise<string | null | undefined>
    summarizeArchive?: (
        plan: BoundaryContextPlan,
        turns: import("@better-compact/core").Turn[],
        catalog: Awaited<ReturnType<typeof loadArchiveCatalog>>,
    ) => Promise<ArchiveSummaryResult>
}): Promise<BoundaryContextPlan | null> {
    let prefixChunkAttempted = false
    const oldPlan = input.state.boundary.activePlan
    const changedArchivedPrefix =
        !!oldPlan?.prefixFingerprint &&
        oldPlan.compactedMessageCount !== undefined &&
        (oldPlan.compactedMessageCount > input.messages.length ||
            boundaryRangeHash(input.messages.slice(0, oldPlan.compactedMessageCount)) !==
                oldPlan.prefixFingerprint)
    let catalog = await expireArchives(input.directory, input.state.sessionId ?? "unknown-session")
    let newArchiveEntry = false
    const validatedEntry = catalog.validatedCheckpointId
        ? catalog.entries.find((entry) => entry.id === catalog.validatedCheckpointId)
        : undefined
    const eligibleMigrationRetirement = validatedEntry
        ? eligibleRetirementThrough(catalog, validatedEntry.sequence)
        : undefined
    let migrationHandoff: string | undefined
    let migrationRetirement: number | undefined
    const prefixChunkModel = input.config.compaction.summaryModel ?? "inherit"
    const currentPrefixAttempt =
        oldPlan?.prefixChunkAttempted === true &&
        oldPlan.prefixChunkVersion === PREFIX_CHUNK_VERSION &&
        oldPlan.prefixChunkModel === prefixChunkModel
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
                              (oldPlan?.rangeHash === snapshot.rangeHash && currentPrefixAttempt)
                                  ? {
                                        prefixChunkAttempted: true as const,
                                        prefixChunkVersion: PREFIX_CHUNK_VERSION,
                                        prefixChunkModel,
                                    }
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
        archive: async (plan) => {
            const receipt = await archiveBoundaryDelta({
                directory: input.directory,
                sessionId: plan.sessionId,
                plan,
                originalMessages: input.messages,
            })
            catalog = receipt.catalog
            if (
                migrationRetirement !== undefined &&
                (catalog.retirementThrough ?? 0) < migrationRetirement
            ) {
                catalog.retirementThrough = migrationRetirement
                await saveArchiveCatalog(input.directory, catalog)
            }
            newArchiveEntry = receipt.entry !== null
            plan.archiveGeneration = catalog.entries.length
            plan.archiveCatalogText = liveArchiveDescriptions(catalog)
            plan.retirementThrough = catalog.retirementThrough
            if (receipt.entry) {
                input.logger.info("Archived Better Compact delta", {
                    sessionId: plan.sessionId,
                    archiveId: receipt.entry.id,
                    messages: Object.keys(receipt.entry.fingerprints).length,
                })
            }
        },
    }
    const profile = resolveCompactionProfile(input.config)
    if (
        oldPlan &&
        catalog.checkpoint &&
        validatedEntry &&
        eligibleMigrationRetirement !== undefined &&
        (catalog.retirementThrough ?? 0) < eligibleMigrationRetirement
    ) {
        const turns = openCodeCodec.encode(input.messages)
        const common = {
            sessionKey: oldPlan.sessionId,
            citablePath: (key: string, hash: string) =>
                createTranscriptStore(input.directory).citablePath(key, hash),
            contextLimit: input.state.modelContextLimit,
            triggerRatio: profile.triggerPercent / 100,
            targetRatio: profile.targetPercent / 100,
            triggerTokens: input.config.compaction.triggerTokens ?? undefined,
            targetTokens: input.config.compaction.targetTokens ?? undefined,
            recentToolResultBudgetTokens: profile.recentToolTokens,
            recentReasoningBudgetTokens: profile.recentReasoningTokens,
            recentAssistantOutputs: 5,
            prefixSummaryAllowed: profile.prefixSummary,
            collapsePercent: profile.collapsePercent,
            archiveCatalogText: liveArchiveDescriptions(catalog),
            archiveGeneration: catalog.entries.length,
            providerReportedTokens: input.providerReportedTokens,
            priorPlan: oldPlan,
            force: true,
            preservePrefixBudgets: true,
        }
        const baseline = buildPlan(turns, common, openCodeSpec)
        const handoff =
            baseline &&
            retainArchivedUserText(
                catalog.checkpoint,
                turns,
                baseline.rawTailStartIndex,
                catalog,
                eligibleMigrationRetirement,
                input.messages,
            )
        const candidate =
            handoff &&
            buildPlan(
                turns,
                {
                    ...common,
                    prefixSummary: handoff,
                    retirementThrough: eligibleMigrationRetirement,
                },
                openCodeSpec,
            )
        if (
            baseline &&
            candidate &&
            candidate.requiresCustomCompaction &&
            candidate.afterPruneTokens < baseline.afterPruneTokens
        ) {
            migrationHandoff = handoff
            migrationRetirement = eligibleMigrationRetirement
        }
    }
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
        !currentPrefixAttempt &&
        (oldPlan.prefixSummary?.match(/^- Resume from prior assistant progress: /gm)?.length ??
            0) >= 12 &&
        profile.prefixSummary &&
        input.summariesAllowed !== false &&
        !!input.summarizePrefix
    const minTailUserTurns = adaptiveTailUserTurns(
        input.messages,
        input.state.modelContextLimit ?? 1,
        profile.targetPercent,
        input.config.compaction.targetTokens,
    )
    const tailBudgetTokens = efficientAgenticTailBudget(input.messages, {
        contextLimit: input.state.modelContextLimit,
        triggerRatio: profile.triggerPercent / 100,
        targetRatio: profile.targetPercent / 100,
        triggerTokens: input.config.compaction.triggerTokens ?? undefined,
        targetTokens: input.config.compaction.targetTokens ?? undefined,
        recentToolResultBudgetTokens: profile.recentToolTokens,
        recentReasoningBudgetTokens: profile.recentReasoningTokens,
        recentAssistantOutputs: 5,
        preservePrefixBudgets: true,
        minTailUserTurns,
        prefixSummaryAllowed: profile.prefixSummary,
        collapsePercent: profile.collapsePercent,
        archiveCatalogText: liveArchiveDescriptions(catalog),
        archiveGeneration: catalog.entries.length,
        retirementThrough: migrationRetirement ?? catalog.retirementThrough,
        ...(migrationHandoff ? { prefixSummary: migrationHandoff } : {}),
        providerReportedTokens: input.providerReportedTokens,
        priorPlan: oldPlan ?? undefined,
    })
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
        recentReasoningBudgetTokens: profile.recentReasoningTokens,
        recentAssistantOutputs: 5,
        preservePrefixBudgets: true,
        minTailUserTurns,
        tailBudgetTokens,
        prefixSummaryAllowed: profile.prefixSummary,
        collapsePercent: profile.collapsePercent,
        archiveCatalogText: liveArchiveDescriptions(catalog),
        archiveGeneration: catalog.entries.length,
        retirementThrough: migrationRetirement ?? catalog.retirementThrough,
        ...(migrationHandoff ? { prefixSummary: migrationHandoff } : {}),
        providerReportedTokens: input.providerReportedTokens,
        triggerFromProviderOnly: true,
        summariesAllowed: input.summariesAllowed,
        summarize: input.summariesAllowed === false ? undefined : input.summarize,
        summarizePrefix:
            input.summariesAllowed === false || !input.summarizePrefix || !!input.summarizeArchive
                ? undefined
                : async (plan, turns) => {
                      prefixChunkAttempted = true
                      return input.summarizePrefix!(plan, turns)
                  },
        summarizeArchive:
            input.summariesAllowed === false || !input.summarizeArchive
                ? undefined
                : async (plan, turns) => {
                      // Catalog backlog is retried only when new native content
                      // advances the boundary; replay/settings changes are free.
                      if (!newArchiveEntry) return null
                      if (currentPrefixAttempt && oldPlan?.rangeHash === plan.rangeHash) return null
                      prefixChunkAttempted = true
                      const result = await input.summarizeArchive!(plan, turns, catalog)
                      const lastPending = catalog.entries.findLast(
                          (entry) => entry.status === "pending",
                      )
                      if (!result.ok) {
                          await recordArchiveFailure(input.directory, catalog, result.reason)
                          input.logger.warn("Better Compact archive synthesis unavailable", {
                              sessionId: plan.sessionId,
                              reason: result.reason,
                              calls: result.calls,
                          })
                          if (result.oversized && lastPending) {
                              await archiveOversizedSummary(
                                  input.directory,
                                  catalog,
                                  lastPending.id,
                                  result.oversized,
                              )
                          }
                          return null
                      }
                      const candidate = structuredClone(catalog)
                      const last = candidate.entries.at(-1)
                      if (!last) return null
                      candidate.checkpoint = result.handoff
                      candidate.validatedCheckpointId = last.id
                      candidate.retirementThrough = eligibleRetirementThrough(
                          candidate,
                          last.sequence,
                      )
                      return {
                          handoff: retainArchivedUserText(
                              result.handoff,
                              turns,
                              plan.rawTailStartIndex,
                              candidate,
                              candidate.retirementThrough,
                              input.messages,
                          ),
                          catalogText: liveArchiveDescriptions(candidate),
                          retirementThrough: candidate.retirementThrough,
                          async commit() {
                              const latest = await loadArchiveCatalog(
                                  input.directory,
                                  plan.sessionId,
                              )
                              if (
                                  !candidate.entries.every((entry) =>
                                      latest.entries.some(
                                          (stored) =>
                                              stored.id === entry.id &&
                                              stored.checksum === entry.checksum,
                                      ),
                                  )
                              )
                                  throw new Error("Archive catalog changed during handoff")
                              latest.checkpoint = result.handoff
                              latest.validatedCheckpointId = last.id
                              latest.retirementThrough = candidate.retirementThrough
                              await saveArchiveCatalog(input.directory, latest)
                              catalog = latest
                          },
                          async discard() {
                              await recordArchiveFailure(
                                  input.directory,
                                  catalog,
                                  "valid_but_not_smaller",
                              )
                              await archiveOversizedSummary(
                                  input.directory,
                                  catalog,
                                  last.id,
                                  result.handoff,
                              )
                          },
                      }
                  },
        force:
            input.forceOverflow === true ||
            migrationHandoff !== undefined ||
            changedArchivedPrefix ||
            migratePluginInjections ||
            migrateUnboundedPrefix ||
            (oldPlan !== null &&
                !!input.providerReportedTokens &&
                input.providerReportedTokens >=
                    (input.config.compaction.triggerTokens ??
                        Math.floor(
                            ((input.state.modelContextLimit ?? 0) * profile.triggerPercent) / 100,
                        ))),
        reuseStablePrefixOnForce:
            !changedArchivedPrefix &&
            !migratePluginInjections &&
            !migrateUnboundedPrefix &&
            migrationHandoff === undefined,
    })
    input.onOutcome?.(result.outcome)
    if (result.outcome === "unchanged") return null
    const decoded = openCodeCodec.decode(result.turns, input.messages)
    input.messages.length = 0
    input.messages.push(...decoded)
    return result.outcome === "planned" ? result.plan : null
}

/** First archival preserves exact user wording alongside the model handoff. */
export function retainArchivedUserText(
    handoff: string,
    turns: import("@better-compact/core").Turn[],
    rawTailStart: number,
    catalog?: Awaited<ReturnType<typeof loadArchiveCatalog>>,
    retirementThrough?: number,
    nativeMessages?: WithParts[],
): string {
    // A split/updated message may retain its ID while acquiring a new payload
    // and a new archive delta. A prior description of the old ID cannot retire
    // the newly revised user wording. Missing native provenance fails open to
    // keeping the exact wording live.
    const userVersions = new Map(
        (catalog && retirementThrough !== undefined ? (nativeMessages ?? []) : [])
            .filter((message) => message.info.role === "user")
            .map((message) => [message.info.id, nativeMessageFingerprint(message)]),
    )
    const retiredUserIds = new Set(
        [...userVersions]
            .filter(([id, fingerprint]) =>
                catalog?.entries.some(
                    (entry) =>
                        retirementThrough !== undefined &&
                        entry.sequence <= retirementThrough &&
                        entry.fingerprints[id] === fingerprint,
                ),
            )
            .map(([id]) => id),
    )
    const exact = prefixUserMessages(
        turns.slice(0, rawTailStart).filter((turn) => !retiredUserIds.has(turn.key)),
        openCodeConventions,
        turns.slice(rawTailStart),
    )
    // A grounded model handoff may already quote the required original wording.
    // Do not append a second copy just to satisfy first-boundary retention.
    const missing = exact.filter((text) => !handoff.includes(text))
    if (missing.length === 0) return handoff
    const boundary = handoff.lastIndexOf("\n## Next step\n")
    if (boundary < 0) return handoff
    return `${handoff.slice(0, boundary)}\n${missing.map((text) => `- ${text}`).join("\n")}\n${handoff.slice(boundary + 1)}`
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
