import { resolveCompactionProfile, type CompactionConfig } from "@better-compact/core"
import type { RuntimeState, SessionState, WithParts } from "./state"
import type { Logger } from "./logger"
import { resolveModelConfig, type PluginConfig } from "./config"
import { stripHallucinations, stripHallucinationsFromString } from "./messages"
import { filterMessages, filterMessagesInPlace } from "./messages/shape"
import { handleContextCommand, handleHelpCommand, handleStatsCommand } from "./commands"
import { type HostPermissionSnapshot } from "./host-permissions"
import { compressPermission, syncCompressPermissionState } from "./compress-permission"
import { loadSessionState, saveSessionState } from "./state"
import {
    buildBoundaryContextPlan,
    buildPrefixChunks,
    adaptiveTailUserTurns,
    efficientAgenticTailBudget,
    findMatchingBoundaryPlan,
    formatBoundaryReport,
    appendBoundaryLog,
    applyBoundaryPlanSnapshot,
    completeBoundaryJob,
    failBoundaryJob,
    processBoundaryTransform,
    providerAlignedHistoryTokens,
    setBoundaryStage,
    startBoundaryJob,
    storeBoundaryPlan,
    summarizeBoundaryJobs,
    summarizePrefixChunks,
    toBoundaryPlanSnapshot,
    updateBoundaryCounters,
    updateBoundaryPercent,
    writeBoundaryTranscript,
    type BoundaryContextPlan,
} from "./boundary"
import { openCodeCodec } from "./codec"
import {
    archiveBoundaryDelta,
    archiveOversizedSummary,
    eligibleRetirementThrough,
    expireArchives,
    liveArchiveDescriptions,
    loadArchiveCatalog,
    pendingArchiveSessionIds,
    projectArchiveRoots,
    recordArchiveFailure,
    saveArchiveCatalog,
} from "./boundary/archive-catalog"
import {
    summarizeArchiveBoundary,
    summarizePendingArchiveDescriptions,
} from "./boundary/archive-summarizer"
import { PREFIX_CHUNK_VERSION, retainArchivedUserText } from "./boundary/engine"
import { boundaryRangeHash } from "./boundary/fingerprint"
import { getCurrentParams, getCurrentTokenUsage, getCurrentUsageMessageId } from "./token-utils"
import { sendIgnoredMessage } from "./ui/notification"
import { getLastUserMessage, isIgnoredUserMessage } from "./messages/query"
import { isSyndicatePluginInjection } from "./messages/injection"

/** Exceptions from the provider or filesystem may embed private prompt text. */
export function safeCompactionFailure(error: unknown): string {
    const message = error instanceof Error ? error.message : ""
    if (
        /^(?:model_limit_unknown|input_does_not_fit|permission_denied|archive_io_error|invalid_output|missing_chunk|valid_but_not_smaller)$/.test(
            message,
        )
    )
        return message
    return "unexpected_error"
}

function queuedStepFingerprint(messages: WithParts[]): string {
    return boundaryRangeHash(
        messages.filter(
            (message) => !isIgnoredUserMessage(message) && !isSyndicatePluginInjection(message),
        ),
    )
}

async function sessionIsBusy(client: any, sessionId: string): Promise<boolean> {
    if (typeof client?.session?.status !== "function") return false
    try {
        const response = await client.session.status()
        return (
            (response?.data ?? response)?.[sessionId]?.type === "busy" ||
            (response?.data ?? response)?.[sessionId]?.type === "retry"
        )
    } catch {
        // Older hosts without a working status endpoint keep the legacy path.
        return false
    }
}

async function queueManualIfBusy(
    client: any,
    state: SessionState,
    messages: WithParts[],
    logger: Logger,
    jobId?: string,
    jobStartedAt?: number,
    params?: {
        providerId: string | undefined
        modelId: string | undefined
        agent: string | undefined
        variant: string | undefined
    },
    overrides?: Pick<
        NonNullable<SessionState["boundary"]["queuedManual"]>,
        "compaction" | "contextLimit" | "currentTokens" | "summaryVariant"
    >,
): Promise<boolean> {
    if (!state.sessionId || !(await sessionIsBusy(client, state.sessionId))) return false
    state.boundary.queuedManual ??= {
        requestedAt: Date.now(),
        jobId,
        jobStartedAt,
        lastUserMessageId: getLastUserMessage(messages)?.info.id,
        lastEligibleFingerprint: queuedStepFingerprint(messages),
        params,
        ...overrides,
    }
    await saveSessionState(state, logger)
    return true
}

async function runQueuedManual(input: {
    client: any
    runtime: RuntimeState
    state: SessionState
    logger: Logger
    config: PluginConfig
    directory: string
    sessionId: string
    messages: WithParts[]
    beforeRequest: boolean
}): Promise<void> {
    const queued = input.state.boundary.queuedManual
    if (!queued) return
    const started = input.runtime.startCompaction(input.sessionId, async () => {
        input.state.boundary.queuedManual = undefined
        await saveSessionState(input.state, input.logger)
        await runBetterCompact({
            client: input.client,
            runtime: input.runtime,
            state: input.state,
            logger: input.logger,
            config: input.config,
            workingDirectory: input.directory,
            sessionId: input.sessionId,
            messages: input.messages,
            params: queued.params,
            compaction: queued.compaction,
            contextLimit: queued.contextLimit,
            currentTokens: queued.currentTokens,
            summaryVariant: queued.summaryVariant,
            jobId: queued.jobId,
            jobStartedAt: queued.jobStartedAt,
            silent: input.beforeRequest,
        })
    })
    if (!started) {
        await input.runtime.activeCompaction(input.sessionId)?.catch(() => {})
        return
    }
    await input.runtime.activeCompaction(input.sessionId)
}

export function createSystemPromptHandler(
    runtime: RuntimeState,
    logger: Logger,
    config: PluginConfig,
) {
    return async (
        input: {
            sessionID?: string
            model: { id?: string; providerID?: string; limit: { context: number } }
        },
        output: { system: string[] },
    ) => {
        if (input.model?.limit?.context) {
            if (input.model.providerID && input.model.id) {
                runtime.setModelLimit(
                    input.model.providerID,
                    input.model.id,
                    input.model.limit.context,
                )
            }
            if (input.sessionID) {
                runtime.get(input.sessionID).modelContextLimit = input.model.limit.context
            }
            logger.debug("Cached model context limit", { limit: input.model.limit.context })
        }
    }
}

export function createChatMessageTransformHandler(
    client: any,
    runtime: RuntimeState,
    logger: Logger,
    config: PluginConfig,
    hostPermissions: HostPermissionSnapshot,
    workingDirectory = process.cwd(),
    loadConfig: () => PluginConfig = () => config,
) {
    // The incoming array is narrowed to WithParts by filterMessagesInPlace,
    // the single trust boundary between the host SDK's message types and ours.
    return async (_input: {}, output: { messages: unknown[] }) => {
        let currentConfig = loadConfig()
        if (!currentConfig.enabled) return
        const receivedMessages = Array.isArray(output.messages) ? output.messages.length : 0
        const messages = filterMessagesInPlace(output.messages)
        if (messages.length !== receivedMessages) {
            logger.warn("Skipping messages with unexpected shape during chat transform", {
                received: receivedMessages,
                usable: messages.length,
            })
        }

        const sessionId = messages.find((message) => typeof message.info?.sessionID === "string")
            ?.info.sessionID
        if (!sessionId || runtime.isScratch(sessionId)) {
            return
        }

        const state = await runtime.prepare(sessionId, messages)
        const archiveCatalog = await expireArchives(workingDirectory, sessionId).catch((error) => {
            logger.warn("Could not expire Better Compact archives before request", {
                sessionId,
                error: error instanceof Error ? error.name : "unknown",
            })
            return null
        })
        if (
            archiveCatalog &&
            state.boundary.activePlan &&
            state.boundary.activePlan.archiveCatalogText !== undefined &&
            archiveCatalog.entries.some(
                (entry) =>
                    entry.status === "expired" &&
                    state.boundary.activePlan?.archiveCatalogText?.includes(entry.id),
            )
        ) {
            state.boundary.activePlan.archiveCatalogText = liveArchiveDescriptions(archiveCatalog)
            await saveSessionState(state, logger)
        }
        const queued = state.boundary.queuedManual
        if (
            queued &&
            (getLastUserMessage(messages)?.info.id !== queued.lastUserMessageId ||
                (queued.lastEligibleFingerprint !== undefined &&
                    queuedStepFingerprint(messages) !== queued.lastEligibleFingerprint))
        ) {
            syncCompressPermissionState(state, currentConfig, hostPermissions, messages)
            if (
                compressPermission(state, currentConfig) === "allow" &&
                (!state.isSubAgent || currentConfig.experimental.allowSubAgents)
            ) {
                try {
                    await runQueuedManual({
                        client,
                        runtime,
                        state,
                        logger,
                        config: currentConfig,
                        directory: workingDirectory,
                        sessionId,
                        messages,
                        beforeRequest: true,
                    })
                } catch (error) {
                    logger.warn("Queued Better Compact failed before request; continuing safely", {
                        sessionId,
                        error: error instanceof Error ? error.name : "unknown",
                    })
                }
            } else {
                state.boundary.queuedManual = undefined
                await saveSessionState(state, logger)
            }
        }
        const currentParams = getCurrentParams(state, messages, logger)
        currentConfig = resolveModelConfig(
            currentConfig,
            currentParams.providerId,
            currentParams.modelId,
        )
        if (currentParams.providerId && currentParams.modelId) {
            state.modelContextLimit = await runtime.resolveModelLimit(
                currentParams.providerId,
                currentParams.modelId,
            )
        }

        syncCompressPermissionState(state, currentConfig, hostPermissions, messages)
        const preTransformEstimate = openCodeCodec.estimateTurns(openCodeCodec.encode(messages))

        if (state.isSubAgent && !currentConfig.experimental.allowSubAgents) {
            await recordAutomaticCheck(
                state,
                logger,
                "subagent_disabled",
                messages,
                currentConfig,
                preTransformEstimate,
            )
            return
        }

        stripHallucinations(messages)

        const effectivePermission = compressPermission(state, currentConfig)
        if (effectivePermission !== "deny" && !state.boundary.activePlan && messages.length >= 3) {
            const inherited = await findMatchingBoundaryPlan(
                sessionId,
                messages,
                workingDirectory,
                logger,
                async (id) => {
                    const response = await client.session.get({
                        path: { id },
                        query: { directory: workingDirectory },
                    })
                    const info = response?.data ?? response
                    return typeof info?.title === "string" ? info.title : undefined
                },
            )
            if (inherited) {
                state.boundary.activePlan = inherited
                await saveSessionState(state, logger).catch((error) => {
                    logger.warn("Failed to persist inherited Better Compact plan", {
                        error: safeCompactionFailure(error),
                    })
                })
            }
        }

        // Idle normally builds a plan from the provider's completed usage.
        // The pre-provider seam must also be able to compact a long tool loop
        // when the host has not emitted idle since that response.
        const originalMessages = state.boundary.activePlan ? structuredClone(messages) : undefined
        let replayed = false
        if (state.boundary.activePlan && effectivePermission !== "deny") {
            replayed = applyBoundaryPlanSnapshot(messages, state.boundary.activePlan, {
                allowRegrown: true,
            })
        }
        const providerTokens = getCurrentTokenUsage(state, originalMessages ?? messages)
        const usageMessageId = getCurrentUsageMessageId(state, originalMessages ?? messages)
        const contextLimit = state.modelContextLimit ?? 0
        const profile = resolveCompactionProfile(currentConfig)
        const triggerTokens =
            currentConfig.compaction.triggerTokens ??
            Math.floor((contextLimit * profile.triggerPercent) / 100)
        const targetTokens =
            currentConfig.compaction.targetTokens ??
            Math.floor((contextLimit * profile.targetPercent) / 100)
        const cached = state.boundary.activePlan
        const policyChanged =
            !!cached &&
            (cached.contextLimit !== contextLimit ||
                cached.triggerTokens !== triggerTokens ||
                cached.targetTokens !== targetTokens ||
                cached.recentReasoningBudgetTokens !== profile.recentReasoningTokens ||
                cached.recentAssistantOutputs !== 5 ||
                cached.prefixSummaryAllowed !== profile.prefixSummary ||
                cached.collapsePercent !== profile.collapsePercent)
        const outgoingEstimate = openCodeCodec.estimateTurns(openCodeCodec.encode(messages))
        const needsPreRequestPlan =
            currentConfig.compaction.automatic &&
            effectivePermission === "allow" &&
            contextLimit > 0 &&
            // Provider usage is the ordinary trigger. A request estimated to
            // exceed the model window or a stale-policy plan must be settled
            // before the next provider request, without lowering the trigger.
            (policyChanged ||
                (providerTokens >= triggerTokens &&
                    usageMessageId !== state.boundary.lastPlannedUsageMessageId) ||
                outgoingEstimate >= contextLimit)
        let outcome = replayed
            ? "plan_replayed"
            : effectivePermission === "deny"
              ? "permission_denied"
              : effectivePermission === "ask"
                ? "permission_ask"
                : currentConfig.compaction.automatic && contextLimit <= 0
                  ? "model_limit_unknown"
                  : currentConfig.compaction.automatic
                    ? "awaiting_idle"
                    : "automatic_off"
        if (needsPreRequestPlan) {
            if (originalMessages) messages.splice(0, messages.length, ...originalMessages)
            outcome = await runAutomaticTransform({
                client,
                runtime,
                state,
                logger,
                config: currentConfig,
                workingDirectory,
                sessionId,
                messages,
                params: currentParams,
            })
            if (outcome === "engine_error" && originalMessages && state.boundary.activePlan)
                applyBoundaryPlanSnapshot(messages, state.boundary.activePlan, {
                    allowRegrown: true,
                })
            const finalEstimate = openCodeCodec.estimateTurns(openCodeCodec.encode(messages))
            if (finalEstimate >= contextLimit) {
                await recordAutomaticCheck(
                    state,
                    logger,
                    "overflow_unresolved",
                    messages,
                    currentConfig,
                    finalEstimate,
                )
                throw new Error(
                    `Better Compact could not fit the outgoing context into ${contextLimit} tokens; the provider request was stopped.`,
                )
            }
        }
        await recordAutomaticCheck(
            state,
            logger,
            outcome,
            messages,
            currentConfig,
            outgoingEstimate,
        )

        if (state.sessionId) {
            await logger.saveContext(state.sessionId, messages)
        }
    }
}

// One automatic compaction at a time per session: the winner builds and
// commits the plan; a concurrent transform waits and replays the committed
// plan onto its own request. Any failure degrades to an unpruned request.
async function runAutomaticTransform(input: {
    client: any
    runtime: RuntimeState
    state: SessionState
    logger: Logger
    config: PluginConfig
    workingDirectory: string
    sessionId: string
    messages: WithParts[]
    params: ReturnType<typeof getCurrentParams>
}): Promise<string> {
    try {
        const usageMessageId = getCurrentUsageMessageId(input.state, input.messages)
        let liveHandoffCalls = 0
        let planned: BoundaryContextPlan | null = null
        let replayed = false
        const started = input.runtime.startCompaction(input.sessionId, async () => {
            planned = await processBoundaryTransform({
                state: input.state,
                logger: input.logger,
                config: input.config,
                directory: input.workingDirectory,
                messages: input.messages,
                providerReportedTokens: getCurrentTokenUsage(input.state, input.messages),
                onOutcome: (outcome) => {
                    replayed = outcome === "replayed"
                },
                summariesAllowed: true,
                summarizeArchive: async (plan, _turns, catalog) => {
                    const result = await summarizeArchiveBoundary({
                        client: input.client,
                        runtime: input.runtime,
                        logger: input.logger,
                        directory: input.workingDirectory,
                        sessionId: input.sessionId,
                        catalog,
                        messages: input.messages,
                        plan,
                        params: input.params,
                        summaryModel: input.config.compaction.summaryModel,
                        summaryEffort: "high",
                    })
                    liveHandoffCalls = result.calls
                    return result
                },
            })
        })
        const active = input.runtime.activeCompaction(input.sessionId)
        if (!started) {
            await active?.catch(() => {})
            const latestPlan = input.state.boundary.activePlan
            return latestPlan &&
                applyBoundaryPlanSnapshot(input.messages, latestPlan, {
                    allowRegrown: true,
                })
                ? "compaction_inflight"
                : "replay_invalid"
        }
        if (active) await active
        if ((planned || replayed) && usageMessageId) {
            input.state.boundary.lastPlannedUsageMessageId = usageMessageId
            await saveSessionState(input.state, input.logger)
        }
        if (planned)
            scheduleArchiveDescriptions({
                client: input.client,
                runtime: input.runtime,
                logger: input.logger,
                directory: input.workingDirectory,
                sessionId: input.sessionId,
                params: input.params,
                summaryModel: input.config.compaction.summaryModel,
                maxCalls: Math.max(0, 7 - liveHandoffCalls),
            })
        if (planned) await showAutomaticCompactionToast(input.client, planned)
        if (planned) return "planned"
        if (replayed) return "plan_replayed"
        return input.state.boundary.activePlan
            ? applyBoundaryPlanSnapshot(input.messages, input.state.boundary.activePlan, {
                  allowRegrown: true,
              })
                ? "plan_replayed"
                : "replay_invalid"
            : "no_new_plan"
    } catch (error) {
        const message = safeCompactionFailure(error)
        input.logger.error("Automatic Better Compact failed; request continues unpruned", {
            error: message,
        })
        try {
            await input.client.tui.showToast({
                body: {
                    title: "Better Compact failed",
                    message,
                    variant: "error",
                    duration: 7000,
                },
            })
        } catch {}
        return "engine_error"
    }
}

const backgroundDescriptions = new Map<string, Promise<void>>()

function scheduleArchiveDescriptions(
    input: Parameters<typeof summarizePendingArchiveDescriptions>[0],
): Promise<void> {
    if (input.maxCalls <= 0) return Promise.resolve()
    const key = `${input.directory}\0${input.sessionId}`
    const existing = backgroundDescriptions.get(key)
    if (existing) return existing
    const work = summarizePendingArchiveDescriptions(input)
        .then(() => {})
        .catch((error) => {
            input.logger.warn("Background archive description failed", {
                sessionId: input.sessionId,
                error: error instanceof Error ? error.name : "unknown",
            })
        })
        .finally(() => backgroundDescriptions.delete(key))
    backgroundDescriptions.set(key, work)
    return work
}

/** Best-effort, bounded startup backfill. Never mutates a live plan mid-turn. */
export async function startArchiveDescriptionBackfill(input: {
    client: any
    runtime: RuntimeState
    logger: Logger
    directory: string
    summaryModel?: string | null
}): Promise<void> {
    if (!input.summaryModel) return
    const projects = await projectArchiveRoots(input.directory)
    const sessions = (
        await Promise.all(
            projects.map(async (directory) =>
                (await pendingArchiveSessionIds(directory)).map((sessionId) => ({
                    directory,
                    sessionId,
                })),
            ),
        )
    ).flat()
    let next = 0
    await Promise.all(
        Array.from({ length: Math.min(2, sessions.length) }, async () => {
            while (next < sessions.length) {
                const { directory, sessionId } = sessions[next++]
                await scheduleArchiveDescriptions({
                    ...input,
                    directory,
                    sessionId,
                    params: {
                        providerId: undefined,
                        modelId: undefined,
                        agent: undefined,
                        variant: undefined,
                    },
                    maxCalls: 7,
                })
            }
        }),
    )
}

/** Persist only numbers and reason codes; never log message text or model output. */
async function recordAutomaticCheck(
    state: SessionState,
    logger: Logger,
    reason: string,
    messages: WithParts[],
    config: PluginConfig,
    estimatedTokens: number,
    seam: "pre_request" | "idle" = "pre_request",
): Promise<void> {
    const previous = state.boundary.automaticCheck
    const providerTokens = getCurrentTokenUsage(state, messages)
    const contextLimit = state.modelContextLimit ?? null
    const profile = resolveCompactionProfile(config)
    const triggerTokens =
        config.compaction.triggerTokens ??
        (contextLimit ? Math.floor((contextLimit * profile.triggerPercent) / 100) : 0)
    const effectiveReason =
        reason === "no_new_plan" && !contextLimit
            ? "model_limit_unknown"
            : reason === "no_new_plan" && Math.max(providerTokens, estimatedTokens) < triggerTokens
              ? "below_trigger"
              : reason === "no_new_plan"
                ? "no_eligible_boundary"
                : reason
    state.boundary.automaticCheck = {
        at: new Date().toISOString(),
        count: (previous?.count ?? 0) + 1,
        seam,
        reason: effectiveReason,
        providerTokens,
        estimatedTokens,
        triggerTokens,
        contextLimit,
    }
    if (
        previous?.reason !== effectiveReason ||
        previous?.seam !== seam ||
        (Math.max(providerTokens, estimatedTokens) >= triggerTokens &&
            state.boundary.automaticCheck.count % 8 === 0)
    ) {
        await saveSessionState(state, logger).catch(() => {})
    }
    logger.debug("Automatic compaction check", state.boundary.automaticCheck)
}

async function showAutomaticCompactionToast(client: any, plan: BoundaryContextPlan): Promise<void> {
    try {
        await client.tui.showToast({
            body: {
                title: "Better Compact applied",
                message: `${formatCompactTokens(plan.beforeTokens)} → ${formatCompactTokens(plan.afterPruneTokens)} projected context`,
                variant: "success",
                duration: 5000,
            },
        })
    } catch {}
}

export function createCommandExecuteHandler(
    client: any,
    runtime: RuntimeState,
    logger: Logger,
    config: PluginConfig,
    workingDirectory: string,
    hostPermissions: HostPermissionSnapshot,
    loadConfig: () => PluginConfig = () => config,
) {
    return async (
        input: { command: string; sessionID: string; arguments: string },
        output: { parts: any[] },
    ) => {
        const currentConfig = loadConfig()
        if (!currentConfig.enabled || !currentConfig.commands.enabled) {
            return
        }

        if (input.command === "better-compact" || input.command === "better-compact-settings") {
            const messagesResponse = await client.session.messages({
                path: { id: input.sessionID },
            })
            const messages = filterMessages(messagesResponse.data || messagesResponse)

            const state = await runtime.prepare(input.sessionID, messages)

            syncCompressPermissionState(state, currentConfig, hostPermissions, messages)

            const effectivePermission = compressPermission(state, currentConfig)
            if (effectivePermission === "deny") {
                output.parts.length = 0
                return
            }

            const args = (input.arguments || "").trim().split(/\s+/).filter(Boolean)
            const subcommand =
                input.command === "better-compact-settings"
                    ? "settings"
                    : args[0]?.toLowerCase() || "compress"

            const commandCtx = {
                client,
                state,
                config: currentConfig,
                logger,
                sessionId: input.sessionID,
                messages,
            }

            if (subcommand === "context") {
                await handleContextCommand(commandCtx)
                output.parts.length = 0
                return
            }

            if (subcommand === "stats") {
                await handleStatsCommand(commandCtx)
                output.parts.length = 0
                return
            }

            if (subcommand === "help") {
                await handleHelpCommand(commandCtx)
                output.parts.length = 0
                return
            }

            if (subcommand === "settings") {
                const params = getCurrentParams(state, messages, logger)
                await sendIgnoredMessage(
                    client,
                    input.sessionID,
                    "Open Better Compact settings from the command palette with /better-compact-settings.",
                    params,
                    logger,
                )
                output.parts.length = 0
                return
            }

            if (subcommand === "compress") {
                if (await queueManualIfBusy(client, state, messages, logger)) {
                    try {
                        await client.tui.showToast({
                            body: {
                                title: "Better Compact queued",
                                message: "Will compact after this turn finishes.",
                                variant: "info",
                                duration: 5000,
                            },
                        })
                    } catch {}
                    output.parts.length = 0
                    return
                }
                const started = runtime.startCompaction(input.sessionID, async () => {
                    try {
                        await runBetterCompact({
                            client,
                            runtime,
                            state,
                            logger,
                            config: currentConfig,
                            workingDirectory,
                            sessionId: input.sessionID,
                            messages,
                        })
                    } catch (error) {
                        logger.error("Better Compact command job failed", {
                            error: safeCompactionFailure(error),
                        })
                    }
                })
                if (!started) {
                    const params = getCurrentParams(state, messages, logger)
                    await sendIgnoredMessage(
                        client,
                        input.sessionID,
                        "Better Compact is already running for this session.",
                        params,
                        logger,
                    )
                }
                output.parts.length = 0
                return
            }

            await handleHelpCommand(commandCtx)
            output.parts.length = 0
            return
        }
    }
}

export function createChatMessageHandler(
    client: any,
    runtime: RuntimeState,
    logger: Logger,
    config: PluginConfig,
    workingDirectory: string,
    hostPermissions: HostPermissionSnapshot,
    loadConfig: () => PluginConfig = () => config,
) {
    return async (
        input: {
            sessionID: string
            agent?: string
            model?: { providerID?: string; modelID?: string }
            variant?: string
        },
        output: { message: any; parts: any[] },
    ) => {
        const sentinel = output.parts.find(
            (part) =>
                part?.type === "text" &&
                part?.ignored === true &&
                part?.metadata?.betterCompact === "run",
        )
        if (!sentinel) return
        const currentConfig = loadConfig()
        if (!currentConfig.enabled) return

        const messagesResponse = await client.session.messages({
            path: { id: input.sessionID },
        })
        const messages = filterMessages(messagesResponse.data || messagesResponse)
        const state = await runtime.prepare(input.sessionID, messages)
        const jobId = validBoundaryJobId(sentinel.metadata?.jobId)
        const jobStartedAt = validBoundaryJobStartedAt(sentinel.metadata?.jobStartedAt)
        const contextLimit = validBoundaryCounter(sentinel.metadata?.contextLimit)
        const currentTokens = validBoundaryCounter(sentinel.metadata?.currentTokens)
        const targetTokens = validBoundaryCounter(sentinel.metadata?.targetTokens)
        const requestedSummaryVariant = validSummaryVariant(sentinel.metadata?.summaryVariant)
        const messageModel = output.message?.model
        const providerID = input.model?.providerID ?? messageModel?.providerID
        const modelID = input.model?.modelID ?? messageModel?.modelID
        const requestedChatVariant = validSummaryVariant(sentinel.metadata?.chatVariant)
        const chatVariant =
            requestedChatVariant &&
            providerID === sentinel.metadata?.chatProviderID &&
            modelID === sentinel.metadata?.chatModelID
                ? requestedChatVariant
                : (input.variant ?? output.message?.variant)
        const summaryVariant =
            requestedSummaryVariant &&
            providerID === sentinel.metadata?.summaryProviderID &&
            modelID === sentinel.metadata?.summaryModelID
                ? requestedSummaryVariant
                : undefined
        syncCompressPermissionState(state, currentConfig, hostPermissions, messages)
        if (compressPermission(state, currentConfig) === "deny") {
            startBoundaryJob(state, {
                id: jobId,
                sessionId: input.sessionID,
                startedAt: jobStartedAt,
                counters: {
                    beforeTokens: currentTokens,
                    currentTokens,
                    targetTokens,
                    contextLimit,
                },
            })
            failBoundaryJob(state, "Compression is denied by OpenCode permissions.")
            await saveSessionState(state, logger)
            return
        }

        if (
            await queueManualIfBusy(
                client,
                state,
                messages,
                logger,
                jobId,
                jobStartedAt,
                {
                    providerId: providerID,
                    modelId: modelID,
                    agent: input.agent ?? output.message?.agent,
                    variant: chatVariant,
                },
                {
                    compaction: sentinel.metadata?.compaction as
                        Partial<CompactionConfig> | undefined,
                    contextLimit,
                    currentTokens,
                    summaryVariant,
                },
            )
        ) {
            try {
                await client.tui.showToast({
                    body: {
                        title: "Better Compact queued",
                        message: "Will compact after this turn finishes.",
                        variant: "info",
                        duration: 5000,
                    },
                })
            } catch {}
            return
        }

        const started = runtime.startCompaction(input.sessionID, async () => {
            try {
                await runBetterCompact({
                    client,
                    runtime,
                    state,
                    logger,
                    config: currentConfig,
                    workingDirectory,
                    sessionId: input.sessionID,
                    messages,
                    params: {
                        providerId: providerID,
                        modelId: modelID,
                        agent: input.agent ?? output.message?.agent,
                        variant: chatVariant,
                    },
                    compaction: sentinel.metadata?.compaction as
                        Partial<CompactionConfig> | undefined,
                    contextLimit,
                    currentTokens,
                    jobId,
                    jobStartedAt,
                    summaryVariant,
                })
            } catch (error) {
                logger.error("Better Compact TUI job failed", {
                    error: safeCompactionFailure(error),
                })
            }
        })
        if (!started) {
            try {
                await client.tui.showToast({
                    body: {
                        title: "Better Compact already running",
                        message: "Wait for the active compaction to finish.",
                        variant: "warning",
                        duration: 5000,
                    },
                })
            } catch {}
        }
    }
}

async function runBetterCompact(input: {
    client: any
    runtime: RuntimeState
    state: SessionState
    logger: Logger
    config: PluginConfig
    workingDirectory: string
    sessionId: string
    messages: WithParts[]
    params?: {
        providerId: string | undefined
        modelId: string | undefined
        agent: string | undefined
        variant: string | undefined
    }
    compaction?: Partial<CompactionConfig>
    contextLimit?: number
    currentTokens?: number
    jobId?: string
    jobStartedAt?: number
    summaryVariant?: string
    silent?: boolean
}): Promise<void> {
    const params = input.params ?? {
        ...getCurrentParams(input.state, input.messages, input.logger),
        // A previous user message cannot prove the session's current TUI
        // variant after an intervening variant switch.
        variant: undefined,
    }
    const effectiveConfig = resolveModelConfig(input.config, params.providerId, params.modelId)
    const profile = resolveCompactionProfile(effectiveConfig, input.compaction)
    const summariesAllowed = true
    const contextLimit =
        input.contextLimit && input.contextLimit > 0
            ? input.contextLimit
            : (input.state.modelContextLimit ?? 200_000)
    const reportedCurrentTokens =
        input.currentTokens && input.currentTokens > 0
            ? input.currentTokens
            : getCurrentTokenUsage(input.state, input.messages)
    const providerHistoryTokens = providerAlignedHistoryTokens(
        input.state,
        input.messages,
        reportedCurrentTokens,
    )
    startBoundaryJob(input.state, {
        id: input.jobId,
        sessionId: input.sessionId,
        startedAt: input.jobStartedAt,
        counters: {
            beforeTokens: reportedCurrentTokens,
            currentTokens: reportedCurrentTokens,
            targetTokens:
                effectiveConfig.compaction.targetTokens ??
                Math.round((contextLimit * profile.targetPercent) / 100),
            contextLimit,
            stageClearedTokens: 0,
            clearedTokens: 0,
        },
    })
    updateBoundaryCounters(input.state, {
        messages: input.messages.length,
        beforeTokens: reportedCurrentTokens,
        currentTokens: reportedCurrentTokens,
        contextLimit,
        stageClearedTokens: 0,
        clearedTokens: 0,
    })
    const saveProgress = async () => {
        updateBoundaryPercent(input.state)
        await saveSessionState(input.state, input.logger)
    }

    const previousActivePlan = input.state.boundary.activePlan
    let liveHandoffCalls = 0
    try {
        let catalog = await expireArchives(input.workingDirectory, input.sessionId)
        setBoundaryStage(input.state, "load", "running", "Reading current OpenCode session history")
        updateBoundaryCounters(input.state, { messages: input.messages.length })
        appendBoundaryLog(
            input.state,
            `Loaded ${input.messages.length} messages from current session.`,
        )
        setBoundaryStage(
            input.state,
            "load",
            "completed",
            `${input.messages.length} messages loaded`,
        )
        await saveProgress()

        setBoundaryStage(
            input.state,
            "scan",
            "running",
            "Estimating context and selecting pruning stages",
        )
        await saveProgress()
        const minTailUserTurns = adaptiveTailUserTurns(
            input.messages,
            contextLimit,
            profile.targetPercent,
            effectiveConfig.compaction.targetTokens,
        )
        const planOptions: Parameters<typeof buildBoundaryContextPlan>[1] = {
            contextLimit,
            force: true,
            triggerRatio: profile.triggerPercent / 100,
            targetRatio: profile.targetPercent / 100,
            triggerTokens: effectiveConfig.compaction.triggerTokens ?? undefined,
            targetTokens: effectiveConfig.compaction.targetTokens ?? undefined,
            recentToolResultBudgetTokens: profile.recentToolTokens,
            recentReasoningBudgetTokens: profile.recentReasoningTokens,
            minTailUserTurns,
            prefixSummaryAllowed: profile.prefixSummary,
            collapsePercent: profile.collapsePercent,
            providerReportedTokens: reportedCurrentTokens,
            providerHistoryTokens,
            summariesAllowed,
            archiveCatalogText: liveArchiveDescriptions(catalog),
            archiveGeneration: catalog.entries.length,
            retirementThrough: catalog.retirementThrough,
            priorPlan: input.state.boundary.activePlan ?? undefined,
        }
        const modelFirst = summariesAllowed && profile.prefixSummary
        const tailBudgetTokens = efficientAgenticTailBudget(input.messages, {
            ...planOptions,
            prefixSummaryAllowed: modelFirst ? false : profile.prefixSummary,
        })
        const plan = buildBoundaryContextPlan(input.messages, {
            ...planOptions,
            prefixSummaryAllowed: modelFirst ? false : profile.prefixSummary,
            deferPrefixConsolidation:
                modelFirst && input.state.boundary.activePlan?.requiresCustomCompaction === true,
            tailBudgetTokens,
        })
        if (!plan) {
            setBoundaryStage(input.state, "scan", "skipped", "No eligible historical context found")
            appendBoundaryLog(input.state, "Better Compact did not find enough context to prune.")
            completeBoundaryJob(input.state, "No pruning needed")
            await saveSessionState(input.state, input.logger)
            await sendIgnoredMessage(
                input.client,
                input.sessionId,
                "Better Compact did not find enough context to prune.",
                params,
                input.logger,
            )
            return
        }

        updateBoundaryCounters(input.state, {
            beforeTokens: plan.beforeTokens,
            afterTokens: plan.afterPruneTokens,
            currentTokens: plan.beforeTokens,
            targetTokens: plan.targetTokens,
            contextLimit: plan.contextLimit,
            stageClearedTokens: 0,
            clearedTokens: Math.max(0, plan.beforeTokens - plan.afterPruneTokens),
        })
        setBoundaryStage(
            input.state,
            "scan",
            "completed",
            `Projected ${formatCompactTokens(plan.beforeTokens)} -> ${formatCompactTokens(plan.afterPruneTokens)}`,
        )
        appendBoundaryLog(
            input.state,
            `Projected context reduction: ${formatCompactTokens(plan.beforeTokens)} -> ${formatCompactTokens(plan.afterPruneTokens)}.`,
        )
        await saveProgress()

        setBoundaryStage(input.state, "transcript", "running", "Writing raw transcript reference")
        await saveProgress()
        await writeBoundaryTranscript(input.workingDirectory, plan, input.logger)
        const archived = await archiveBoundaryDelta({
            directory: input.workingDirectory,
            sessionId: input.sessionId,
            plan,
            originalMessages: input.messages,
        })
        catalog = archived.catalog
        plan.archiveGeneration = catalog.entries.length
        plan.archiveCatalogText = liveArchiveDescriptions(catalog)
        updateBoundaryCounters(input.state, { archivedMessages: plan.transcript.messageIds.length })
        setBoundaryStage(
            input.state,
            "transcript",
            "completed",
            `${plan.transcript.messageIds.length} messages archived`,
        )
        appendBoundaryLog(input.state, `Transcript written: ${plan.transcript.relativePath}`)
        await saveProgress()

        const appliedStageIds = new Set<string>(plan.stages.map((stage) => stage.name))
        for (const stage of plan.stages) {
            if (
                plan.summaryJobs.length > 0 &&
                (stage.name === "assistant-runs" || stage.name === "prefix-summary")
            ) {
                continue
            }
            const status = stage.status === "skipped" ? "skipped" : "completed"
            setBoundaryStage(
                input.state,
                stage.name,
                status,
                stage.clearedTokens > 0
                    ? `Cleared ${formatCompactTokens(stage.clearedTokens)}`
                    : stage.status === "skipped"
                      ? "No matching context"
                      : "Applied",
                {
                    beforeTokens: stage.beforeTokens,
                    afterTokens: stage.afterTokens,
                    clearedTokens: stage.clearedTokens,
                    changedMessages: stage.changedMessages,
                    changedParts: stage.changedParts,
                },
            )
            updateBoundaryCounters(input.state, {
                currentTokens: stage.afterTokens,
                stageClearedTokens: stage.clearedTokens,
                clearedTokens: Math.max(0, plan.beforeTokens - stage.afterTokens),
            })
            appendBoundaryLog(
                input.state,
                `${stage.label}: ${formatCompactTokens(stage.clearedTokens)} cleared.`,
            )
            await saveProgress()
        }

        for (const skippedStage of [
            "skills",
            "supersede-reads",
            "purge-error-inputs",
            "tools-old",
            "reasoning",
            "tools-remaining",
            "assistant-runs",
            "prefix-summary",
        ]) {
            if (appliedStageIds.has(skippedStage)) continue
            setBoundaryStage(input.state, skippedStage, "skipped", "Not needed")
        }
        await saveProgress()

        let finalPlan = plan
        // The archive synthesis has one shared seven-call budget for source
        // chunks, descriptions and handoff. Never also schedule the legacy
        // per-turn/prefix calls during the same compaction.
        const prefixChunks: ReturnType<typeof buildPrefixChunks> = []
        let prefixAttempted = false
        if (
            summariesAllowed &&
            profile.prefixSummary &&
            plan.afterPruneTokens > Math.floor(plan.targetTokens * 1.15) &&
            catalog.entries.some((entry) => entry.status === "pending") &&
            // A repeated command or a settings-only replan of the same raw
            // range must not re-bill Luna for an already attempted boundary.
            !!archived.entry
        ) {
            prefixAttempted = true
            setBoundaryStage(
                input.state,
                "prefix-summary",
                "running",
                "Synthesizing archived task state and descriptions",
            )
            await saveProgress()
            const result = await summarizeArchiveBoundary({
                client: input.client,
                runtime: input.runtime,
                logger: input.logger,
                directory: input.workingDirectory,
                sessionId: input.sessionId,
                catalog,
                messages: input.messages,
                plan,
                params: { ...params, variant: input.summaryVariant ?? params.variant },
                summaryModel: effectiveConfig.compaction.summaryModel,
                summaryEffort: "high",
            })
            liveHandoffCalls = result.calls
            updateBoundaryCounters(input.state, {
                summaryJobsTotal: result.calls,
                summaryJobsDone: result.calls,
                summaryJobsSucceeded: result.ok ? result.calls : 0,
                summaryJobsFailed: result.ok ? 0 : result.calls,
            })
            if (result.ok) {
                const candidate = structuredClone(catalog)
                candidate.checkpoint = result.handoff
                candidate.validatedCheckpointId = candidate.entries.at(-1)?.id
                const newest = candidate.entries.at(-1)
                if (newest)
                    candidate.retirementThrough = eligibleRetirementThrough(
                        candidate,
                        newest.sequence,
                    )
                const rebuilt = buildBoundaryContextPlan(input.messages, {
                    contextLimit,
                    force: true,
                    prefixSummary: retainArchivedUserText(
                        result.handoff,
                        openCodeCodec.encode(input.messages),
                        plan.rawTailStartIndex,
                        candidate,
                        candidate.retirementThrough,
                        input.messages,
                    ),
                    archiveCatalogText: liveArchiveDescriptions(candidate),
                    archiveGeneration: candidate.entries.length,
                    retirementThrough: candidate.retirementThrough,
                    triggerRatio: profile.triggerPercent / 100,
                    targetRatio: profile.targetPercent / 100,
                    triggerTokens: effectiveConfig.compaction.triggerTokens ?? undefined,
                    targetTokens: effectiveConfig.compaction.targetTokens ?? undefined,
                    recentToolResultBudgetTokens: profile.recentToolTokens,
                    recentReasoningBudgetTokens: profile.recentReasoningTokens,
                    minTailUserTurns,
                    tailBudgetTokens,
                    prefixSummaryAllowed: profile.prefixSummary,
                    collapsePercent: profile.collapsePercent,
                    providerReportedTokens: reportedCurrentTokens,
                    providerHistoryTokens,
                    summariesAllowed,
                    priorPlan: modelFirst
                        ? (input.state.boundary.activePlan ?? undefined)
                        : toBoundaryPlanSnapshot(plan, input.messages),
                })
                if (
                    rebuilt?.requiresCustomCompaction &&
                    rebuilt.afterPruneTokens < plan.afterPruneTokens
                ) {
                    const latest = await loadArchiveCatalog(input.workingDirectory, input.sessionId)
                    if (
                        !candidate.entries.every((entry) =>
                            latest.entries.some(
                                (stored) =>
                                    stored.id === entry.id && stored.checksum === entry.checksum,
                            ),
                        )
                    )
                        throw new Error("Archive catalog changed during handoff")
                    latest.checkpoint = result.handoff
                    latest.validatedCheckpointId = newest!.id
                    latest.retirementThrough = candidate.retirementThrough
                    await saveArchiveCatalog(input.workingDirectory, latest)
                    catalog = latest
                    finalPlan = rebuilt
                } else if (candidate.entries.at(-1)) {
                    await recordArchiveFailure(
                        input.workingDirectory,
                        catalog,
                        "valid_but_not_smaller",
                    )
                    await archiveOversizedSummary(
                        input.workingDirectory,
                        catalog,
                        candidate.entries.at(-1)!.id,
                        result.handoff,
                    )
                }
            } else {
                await recordArchiveFailure(input.workingDirectory, catalog, result.reason)
                appendBoundaryLog(
                    input.state,
                    `Archive synthesis unavailable: ${result.reason}; ${result.calls} calls.`,
                )
                if (result.oversized && catalog.entries.at(-1))
                    await archiveOversizedSummary(
                        input.workingDirectory,
                        catalog,
                        catalog.entries.at(-1)!.id,
                        result.oversized,
                    )
            }
            setBoundaryStage(
                input.state,
                "prefix-summary",
                finalPlan === plan ? "failed" : "completed",
                finalPlan === plan
                    ? "Luna handoff unavailable; deterministic fallback remains available"
                    : `Applied archive handoff: ${formatCompactTokens(plan.afterPruneTokens)} -> ${formatCompactTokens(finalPlan.afterPruneTokens)}`,
            )
            await saveProgress()
        }
        if (
            modelFirst &&
            finalPlan === plan &&
            finalPlan.afterPruneTokens > Math.floor(finalPlan.targetTokens * 1.15)
        ) {
            const fallback = buildBoundaryContextPlan(input.messages, {
                ...planOptions,
                tailBudgetTokens,
                prefixSummaryAllowed: profile.prefixSummary,
                archiveCatalogText: liveArchiveDescriptions(catalog),
                archiveGeneration: catalog.entries.length,
                retirementThrough: catalog.retirementThrough,
            })
            if (fallback && fallback.afterPruneTokens < finalPlan.afterPruneTokens)
                finalPlan = fallback
        }
        if (modelFirst) finalPlan.prefixSummaryAllowed = profile.prefixSummary
        if (prefixChunks.length > 0) {
            prefixAttempted = true
            setBoundaryStage(
                input.state,
                "prefix-summary",
                "running",
                `Consolidating ${prefixChunks.reduce((sum, chunk) => sum + chunk.count, 0)} progress entries in ${prefixChunks.length} parallel chunks`,
            )
            updateBoundaryCounters(input.state, {
                summaryJobsTotal: prefixChunks.length,
                summaryJobsDone: 0,
                summaryJobsSucceeded: 0,
                summaryJobsFailed: 0,
            })
            await saveProgress()
            const consolidated = await summarizePrefixChunks({
                client: input.client,
                runtime: input.runtime,
                logger: input.logger,
                parentSessionId: input.sessionId,
                plan,
                turns: openCodeCodec.encode(input.messages),
                params: { ...params, variant: input.summaryVariant ?? params.variant },
                summaryModel: effectiveConfig.compaction.summaryModel,
                summaryEffort:
                    input.summaryVariant && !effectiveConfig.compaction.summaryModel
                        ? "inherit"
                        : (input.compaction?.summaryEffort ??
                          effectiveConfig.compaction.summaryEffort),
                concurrency: profile.summarizerConcurrency,
                onProgress: async (event) => {
                    updateBoundaryCounters(input.state, {
                        summaryJobsTotal: event.total,
                        summaryJobsDone: event.done,
                        summaryJobsSucceeded: event.succeeded,
                        summaryJobsFailed: event.failed,
                    })
                    await saveProgress()
                },
            })
            if (consolidated) {
                const rebuilt = buildBoundaryContextPlan(input.messages, {
                    contextLimit,
                    force: true,
                    prefixSummary: consolidated,
                    triggerRatio: profile.triggerPercent / 100,
                    targetRatio: profile.targetPercent / 100,
                    triggerTokens: effectiveConfig.compaction.triggerTokens ?? undefined,
                    targetTokens: effectiveConfig.compaction.targetTokens ?? undefined,
                    recentToolResultBudgetTokens: profile.recentToolTokens,
                    recentReasoningBudgetTokens: profile.recentReasoningTokens,
                    minTailUserTurns,
                    prefixSummaryAllowed: profile.prefixSummary,
                    collapsePercent: profile.collapsePercent,
                    providerReportedTokens: reportedCurrentTokens,
                    providerHistoryTokens,
                    summariesAllowed,
                    priorPlan: toBoundaryPlanSnapshot(plan, input.messages),
                })
                if (
                    rebuilt?.requiresCustomCompaction &&
                    rebuilt.afterPruneTokens < plan.afterPruneTokens
                )
                    finalPlan = rebuilt
            }
            setBoundaryStage(
                input.state,
                "prefix-summary",
                finalPlan === plan ? "failed" : "completed",
                finalPlan === plan
                    ? "Keeping original prefix; chunk synthesis incomplete or not smaller"
                    : `Consolidated prefix: ${formatCompactTokens(plan.afterPruneTokens)} -> ${formatCompactTokens(finalPlan.afterPruneTokens)}`,
            )
            await saveProgress()
        }
        const activeJobs: typeof plan.summaryJobs = []
        if (activeJobs.length > 0) {
            const summaryStage = activeJobs.some((job) => !job.key.startsWith("prefix-summary:"))
                ? "assistant-runs"
                : "prefix-summary"
            setBoundaryStage(
                input.state,
                summaryStage,
                "running",
                `${activeJobs.length} summaries selected for up to 5 grouped calls`,
            )
            updateBoundaryCounters(input.state, {
                summaryJobsTotal: activeJobs.length,
                summaryJobsDone: 0,
                summaryJobsSucceeded: 0,
                summaryJobsFailed: 0,
                stageClearedTokens: 0,
            })
            appendBoundaryLog(
                input.state,
                `Distributing ${activeJobs.length} selected summaries across up to 5 concurrent scratch calls.`,
            )
            await saveProgress()
            const assistantSummaries = await summarizeBoundaryJobs({
                summaryEffort:
                    input.summaryVariant &&
                    (!effectiveConfig.compaction.summaryModel ||
                        effectiveConfig.compaction.summaryModel ===
                            `${params.providerId}/${params.modelId}`)
                        ? "inherit"
                        : (input.compaction?.summaryEffort ??
                          effectiveConfig.compaction.summaryEffort),
                summaryModel: effectiveConfig.compaction.summaryModel,
                client: input.client,
                runtime: input.runtime,
                logger: input.logger,
                parentSessionId: input.sessionId,
                jobs: activeJobs,
                params: {
                    ...params,
                    variant: input.summaryVariant ?? params.variant,
                },
                concurrency: profile.summarizerConcurrency,
                onProgress: async (event) => {
                    updateBoundaryCounters(input.state, {
                        summaryJobsTotal: event.total,
                        summaryJobsDone: event.done,
                        summaryJobsSucceeded: event.succeeded,
                        summaryJobsFailed: event.failed,
                    })
                    appendBoundaryLog(
                        input.state,
                        event.ok
                            ? `Summarized assistant turn ${event.done}/${event.total}: ${event.rangeStartMessageId} -> ${event.rangeEndMessageId}.`
                            : `Assistant turn summary failed ${event.done}/${event.total}: ${event.rangeStartMessageId} -> ${event.rangeEndMessageId}.`,
                    )
                    await saveProgress()
                },
            })
            if (Object.keys(assistantSummaries).length > 0) {
                const rebuilt = buildBoundaryContextPlan(input.messages, {
                    contextLimit,
                    force: true,
                    assistantSummaries,
                    triggerRatio: profile.triggerPercent / 100,
                    targetRatio: profile.targetPercent / 100,
                    triggerTokens: effectiveConfig.compaction.triggerTokens ?? undefined,
                    targetTokens: effectiveConfig.compaction.targetTokens ?? undefined,
                    recentToolResultBudgetTokens: profile.recentToolTokens,
                    recentReasoningBudgetTokens: profile.recentReasoningTokens,
                    minTailUserTurns,
                    prefixSummaryAllowed: profile.prefixSummary,
                    collapsePercent: profile.collapsePercent,
                    providerReportedTokens: reportedCurrentTokens,
                    providerHistoryTokens,
                    summariesAllowed,
                    priorPlan: input.state.boundary.activePlan ?? undefined,
                })
                if (rebuilt && rebuilt.afterPruneTokens <= plan.afterPruneTokens) {
                    finalPlan = rebuilt
                } else if (rebuilt) {
                    appendBoundaryLog(
                        input.state,
                        `Retained smaller plan: model summaries projected ${formatCompactTokens(rebuilt.afterPruneTokens)} versus ${formatCompactTokens(plan.afterPruneTokens)}.`,
                    )
                }
            }
            const appliedSummaries = Math.max(
                0,
                Object.keys(finalPlan.assistantSummaries).length -
                    Object.keys(plan.assistantSummaries).length,
            )
            setBoundaryStage(
                input.state,
                summaryStage,
                "completed",
                `${appliedSummaries}/${activeJobs.length} summaries applied (${Object.keys(assistantSummaries).length} valid; ${input.state.boundary.job?.counters.summaryJobsDone ?? 0} attempted; remaining turns use deterministic fallback)`,
            )
            updateBoundaryCounters(input.state, {
                currentTokens: finalPlan.afterPruneTokens,
                stageClearedTokens: Math.max(0, plan.beforeTokens - finalPlan.afterPruneTokens),
                clearedTokens: Math.max(0, finalPlan.beforeTokens - finalPlan.afterPruneTokens),
            })
            const finalPrefixStage = finalPlan.stages.find(
                (stage) => stage.name === "prefix-summary",
            )
            if (finalPrefixStage) {
                setBoundaryStage(
                    input.state,
                    "prefix-summary",
                    finalPrefixStage.status === "failed" ? "failed" : "completed",
                    finalPrefixStage.clearedTokens > 0
                        ? `Cleared ${formatCompactTokens(finalPrefixStage.clearedTokens)}`
                        : "Applied",
                    {
                        beforeTokens: finalPrefixStage.beforeTokens,
                        afterTokens: finalPrefixStage.afterTokens,
                        clearedTokens: finalPrefixStage.clearedTokens,
                        changedMessages: finalPrefixStage.changedMessages,
                        changedParts: finalPrefixStage.changedParts,
                    },
                )
            } else {
                setBoundaryStage(input.state, "prefix-summary", "skipped", "Not needed")
            }
            await saveProgress()
        }

        setBoundaryStage(input.state, "store", "running", "Persisting virtual context plan")
        await saveProgress()
        storeBoundaryPlan(input.state, finalPlan, input.messages)
        if (
            input.state.boundary.activePlan &&
            (prefixAttempted ||
                (previousActivePlan?.rangeHash === finalPlan.rangeHash &&
                    previousActivePlan.prefixChunkAttempted))
        ) {
            input.state.boundary.activePlan.prefixChunkAttempted = true
            input.state.boundary.activePlan.prefixChunkVersion = prefixAttempted
                ? PREFIX_CHUNK_VERSION
                : previousActivePlan?.prefixChunkVersion
            input.state.boundary.activePlan.prefixChunkModel = prefixAttempted
                ? (effectiveConfig.compaction.summaryModel ?? "inherit")
                : previousActivePlan?.prefixChunkModel
        }
        updateBoundaryCounters(input.state, {
            afterTokens: finalPlan.afterPruneTokens,
            currentTokens: finalPlan.afterPruneTokens,
            targetTokens: finalPlan.targetTokens,
            clearedTokens: Math.max(0, finalPlan.beforeTokens - finalPlan.afterPruneTokens),
        })
        setBoundaryStage(input.state, "store", "completed", "Virtual context plan stored")
        appendBoundaryLog(input.state, "Stored Better Compact plan for future model requests.")
        await saveProgress()

        setBoundaryStage(input.state, "report", "running", "Publishing final report")
        await saveProgress()
        if (!input.silent)
            await sendIgnoredMessage(
                input.client,
                input.sessionId,
                formatBoundaryReport(
                    finalPlan,
                    getCurrentTokenUsage(input.state, input.messages),
                    input.state.boundary.job?.counters.summaryJobsDone ?? 0,
                ),
                params,
                input.logger,
            )
        setBoundaryStage(input.state, "report", "completed", "Final report published")
        completeBoundaryJob(input.state, "Complete")
        await saveSessionState(input.state, input.logger)
        scheduleArchiveDescriptions({
            client: input.client,
            runtime: input.runtime,
            logger: input.logger,
            directory: input.workingDirectory,
            sessionId: input.sessionId,
            params,
            summaryModel: effectiveConfig.compaction.summaryModel,
            maxCalls: Math.max(0, 7 - liveHandoffCalls),
        })
        input.logger.info("Better Compact virtual compaction plan stored", {
            sessionId: input.sessionId,
            rangeHash: finalPlan.rangeHash,
            requiresCustomCompaction: finalPlan.requiresCustomCompaction,
        })
    } catch (error) {
        input.state.boundary.activePlan = previousActivePlan
        const message = safeCompactionFailure(error)
        appendBoundaryLog(input.state, `Failed: ${message}`)
        failBoundaryJob(input.state, message)
        await saveSessionState(input.state, input.logger).catch(() => {})
        if (!input.silent)
            await sendIgnoredMessage(
                input.client,
                input.sessionId,
                `Better Compact failed: ${message}`,
                params,
                input.logger,
            )
        throw error
    }
}

function validBoundaryJobId(value: unknown): string | undefined {
    return typeof value === "string" && /^bc_[a-zA-Z0-9]{1,64}$/.test(value) ? value : undefined
}

function validBoundaryJobStartedAt(value: unknown): number | undefined {
    return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : undefined
}

function validBoundaryCounter(value: unknown): number | undefined {
    return typeof value === "number" &&
        Number.isFinite(value) &&
        value >= 0 &&
        value <= Number.MAX_SAFE_INTEGER
        ? value
        : undefined
}

function validSummaryVariant(value: unknown): string | undefined {
    return typeof value === "string" && /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(value)
        ? value
        : undefined
}

function formatCompactTokens(tokens: number): string {
    if (tokens >= 1000) return `${(tokens / 1000).toFixed(1).replace(/\.0$/, "")}K`
    return String(tokens)
}

export function createTextCompleteHandler() {
    return async (
        _input: { sessionID: string; messageID: string; partID: string },
        output: { text: string },
    ) => {
        output.text = stripHallucinationsFromString(output.text)
    }
}

export function createEventHandler(
    runtime: RuntimeState,
    logger: Logger,
    client?: any,
    config?: PluginConfig,
    directory?: string,
    hostPermissions?: HostPermissionSnapshot,
    loadConfig: () => PluginConfig = () => config!,
) {
    return async (input: { event: any }) => {
        if (
            input.event.type === "session.idle" &&
            client &&
            config &&
            directory &&
            hostPermissions
        ) {
            const sessionId = input.event.properties?.sessionID
            if (typeof sessionId !== "string" || runtime.isScratch(sessionId)) return
            try {
                const response = await client.session.messages({ path: { id: sessionId } })
                const messages = filterMessages(response.data ?? response)
                const state = await runtime.prepare(sessionId, messages)
                await expireArchives(directory, sessionId)
                const currentConfig = loadConfig()
                syncCompressPermissionState(state, currentConfig, hostPermissions, messages)
                if (state.boundary.queuedManual) {
                    if (compressPermission(state, currentConfig) !== "allow") {
                        state.boundary.queuedManual = undefined
                        await saveSessionState(state, logger)
                        return
                    }
                    await runQueuedManual({
                        client,
                        runtime,
                        state,
                        logger,
                        config: currentConfig,
                        directory,
                        sessionId,
                        messages,
                        beforeRequest: false,
                    })
                    return
                }
                if (
                    !currentConfig.enabled ||
                    !currentConfig.compaction.automatic ||
                    compressPermission(state, currentConfig) !== "allow" ||
                    (state.isSubAgent && !currentConfig.experimental.allowSubAgents)
                )
                    return
                const usageMessage = [...messages]
                    .reverse()
                    .find(
                        (message) =>
                            message.info.role === "assistant" &&
                            (message.info.tokens?.output ?? 0) > 0,
                    )
                if (!usageMessage || usageMessage.info.id === state.boundary.lastIdleUsageMessageId)
                    return
                const providerTokens = getCurrentTokenUsage(state, messages)
                const params = getCurrentParams(state, messages, logger)
                const resolved = resolveModelConfig(
                    currentConfig,
                    params.providerId,
                    params.modelId,
                )
                const contextLimit =
                    params.providerId && params.modelId
                        ? await runtime.resolveModelLimit(params.providerId, params.modelId)
                        : undefined
                state.modelContextLimit = contextLimit
                if (!contextLimit || !providerTokens) {
                    await recordAutomaticCheck(
                        state,
                        logger,
                        !contextLimit ? "model_limit_unknown" : "provider_usage_unavailable",
                        messages,
                        resolved,
                        providerTokens,
                        "idle",
                    )
                    return
                }
                const profile = resolveCompactionProfile(resolved)
                const trigger =
                    resolved.compaction.triggerTokens ??
                    Math.floor((contextLimit * profile.triggerPercent) / 100)
                state.boundary.lastIdleUsageMessageId = usageMessage.info.id
                if (providerTokens < trigger) {
                    await recordAutomaticCheck(
                        state,
                        logger,
                        "below_trigger",
                        messages,
                        resolved,
                        providerTokens,
                        "idle",
                    )
                    await saveSessionState(state, logger)
                    return
                }
                await saveSessionState(state, logger)
                const outcome = await runAutomaticTransform({
                    client,
                    runtime,
                    state,
                    logger,
                    config: resolved,
                    workingDirectory: directory,
                    sessionId,
                    messages,
                    params,
                })
                await recordAutomaticCheck(
                    state,
                    logger,
                    outcome,
                    messages,
                    resolved,
                    openCodeCodec.estimateTurns(openCodeCodec.encode(messages)),
                    "idle",
                )
            } catch (error) {
                logger.warn("Better Compact could not run at idle", {
                    sessionId,
                    error: error instanceof Error ? error.name : "unknown",
                })
            }
            return
        }
        if (input.event.type === "session.compacted") {
            const sessionId = input.event.properties?.sessionID
            const state = typeof sessionId === "string" ? runtime.peek(sessionId) : undefined
            if (!state) return
            // Native compaction rewrote this session's history; the stored
            // plan and job describe context that no longer exists.
            state.boundary.activePlan = null
            state.boundary.job = null
            await saveSessionState(state, logger).catch((error) => {
                logger.warn("Failed to persist state reset after native compaction", {
                    error: safeCompactionFailure(error),
                })
            })
            return
        }

        if (input.event.type === "session.deleted") {
            const sessionId = input.event.properties?.info?.id
            if (typeof sessionId === "string") runtime.evict(sessionId)
        }
    }
}
