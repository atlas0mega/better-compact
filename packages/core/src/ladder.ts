import { countTokens, estimateTurns, type Estimator } from "./estimate"
import { rangeHash } from "./identity"
import type { CodecOps, Conventions, Item, Turn } from "./ir"
import {
    toPlanSnapshot,
    type BoundaryContextOptions,
    type BoundaryContextPlan,
    type BoundaryStageReport,
    type BoundarySummaryJob,
    type PlanSnapshot,
    type RawTailItemBoundary,
} from "./plan"
import type { EnginePorts } from "./ports"
import { formatPrefixSummaryPrompt } from "./summarize"
import {
    assistantGroups,
    dedupeRepeatableUserTextInSummary,
    extendPrefixSummary,
    findLatestTodoCallId,
    findRawTailStartIndex,
    findBudgetTailStartIndex,
    findRecentToolCallTail,
    formatPrefixSummary,
    primaryToolTarget,
    transformCompactedPrefix,
    type Stage,
    type StageContext,
    type StageMutationResult,
} from "./stages"
import { writeTranscript } from "./transcript"

const TRIGGER_RATIO = 0.85
const TARGET_RATIO = 0.3
const MIN_TAIL_MESSAGES = 3
const MIN_TAIL_USER_TURNS = 2
const RECENT_TOOL_RESULT_BUDGET_TOKENS = 40_000

interface TailBoundary {
    turnIndex: number
    itemIndex: number
}

interface PartitionedTurns {
    turns: Turn[]
    compactedRange: Turn[]
    rawTailStartIndex: number
    rawTailKey?: string
    finalize(turns: Turn[]): Turn[]
}

// A platform adapter: its codec, its conventions, and its declared ladder
// stage order. Composition is data; absence from the array is the only
// conditionality.
export interface LadderSpec {
    codec: CodecOps
    conventions: Conventions
    stages: Stage[]
}

export interface BuildPlanInputs extends BoundaryContextOptions {
    sessionKey: string
    citablePath(sessionKey: string, rangeHash: string): string
}

export function buildPlan(
    turns: Turn[],
    inputs: BuildPlanInputs,
    spec: LadderSpec,
): BoundaryContextPlan | null {
    const contextLimit = inputs.contextLimit
    if (!contextLimit || contextLimit <= 0 || turns.length === 0) return null

    const triggerRatio = inputs.triggerRatio ?? TRIGGER_RATIO
    const targetRatio = inputs.targetRatio ?? TARGET_RATIO
    const rawEstimateTokens = spec.codec.estimateTurns(turns)
    const providerReportedTokens =
        inputs.providerReportedTokens && inputs.providerReportedTokens > 0
            ? inputs.providerReportedTokens
            : 0
    // The provider total comes from the *previous* request. New user/tool
    // content in the current history was not in that request; subtracting it
    // to infer system/schema overhead understates the next outgoing context.
    // When supplied, compare the provider reading to its own aligned history.
    const alignedOverhead =
        providerReportedTokens > 0 &&
        inputs.providerHistoryTokens !== undefined &&
        Number.isFinite(inputs.providerHistoryTokens)
            ? Math.max(0, providerReportedTokens - inputs.providerHistoryTokens)
            : 0
    const overheadTokens =
        providerReportedTokens > 0
            ? Math.max(0, providerReportedTokens - rawEstimateTokens, alignedOverhead)
            : 0
    const estimator: Estimator = { overheadTokens }
    const beforeTokens = providerReportedTokens > 0 ? providerReportedTokens : rawEstimateTokens
    const triggerTokens = inputs.triggerTokens ?? Math.floor(contextLimit * triggerRatio)
    // Either scale crossing the trigger means the request is in danger: the
    // provider total sees overhead the estimate cannot, and the estimate sees
    // fresh turns the provider has not priced yet.
    if (!inputs.force && Math.max(beforeTokens, rawEstimateTokens) < triggerTokens) return null

    const targetTokens = inputs.targetTokens ?? Math.floor(contextLimit * targetRatio)
    // A token budget replaces the count-based tail outright: its ceiling is
    // the cap that keeps a long tool loop from staying raw in full. The raw
    // tail can never exceed the target, or the target is unreachable by
    // construction on small windows.
    const tailStartIndex = inputs.tailBudgetTokens
        ? findBudgetTailStartIndex(
              turns,
              boundedTailBudget(inputs.tailBudgetTokens, targetTokens),
              spec.codec,
          )
        : findRawTailStartIndex(
              turns,
              Math.max(1, inputs.minTailMessages ?? MIN_TAIL_MESSAGES),
              Math.max(1, inputs.minTailUserTurns ?? MIN_TAIL_USER_TURNS),
          )
    const selectedBoundary = selectTailBoundary(
        turns,
        tailStartIndex,
        triggerTokens,
        targetTokens,
        spec.codec,
    )
    const prior = inputs.priorPlan
    const priorBoundary = prior ? resolveTailBoundary(turns, prior) : null
    const boundary =
        priorBoundary && compareBoundaries(selectedBoundary, priorBoundary) < 0
            ? priorBoundary
            : selectedBoundary
    const partition = partitionTurns(turns, boundary)
    const rawTailStartIndex = boundary.turnIndex
    const compactedRange = partition.compactedRange
    if (compactedRange.length === 0) return null

    const compactedRangeHash = rangeHash(compactedRange)
    const transcriptRelativePath = inputs.citablePath(inputs.sessionKey, compactedRangeHash)
    const working = partition.turns
    const stages: BoundaryStageReport[] = []
    const summaryJobs: BoundarySummaryJob[] = []
    const expandedPrefix = priorBoundary !== null && compareBoundaries(boundary, priorBoundary) > 0
    const priorPrefixSummary = prior?.prefixSummary
        ? dedupeRepeatableUserTextInSummary(
              stripTranscriptReference(prior.prefixSummary, prior.transcriptRelativePath),
              turns,
              spec.conventions,
          )
        : undefined
    const prefixSummaryResultKey = `prefix-summary:${compactedRangeHash}`
    const prefixSummaryJobKey =
        expandedPrefix && priorPrefixSummary ? prefixSummaryResultKey : undefined
    // A persisted prefix already contains the older turns' task state. Their
    // per-turn summaries are no longer replayed and must not be re-injected
    // (or paid for again) when the prefix grows on a later compaction.
    const assistantSummaries = {
        ...(prior?.requiresCustomCompaction ? {} : (prior?.assistantSummaries ?? {})),
        ...(inputs.assistantSummaries ?? {}),
    }
    const rolledPrefixSummary = assistantSummaries[prefixSummaryResultKey]
    delete assistantSummaries[prefixSummaryResultKey]
    const preservedToolCallIds = findRecentToolCallTail(
        compactedRange,
        // A configured tool allowance larger than the entire target cannot
        // require a single huge old result to remain live. Its exact bytes
        // are already in the private archive; keep the provider request safe.
        inputs.preservePrefixBudgets
            ? Math.min(
                  inputs.recentToolResultBudgetTokens ?? RECENT_TOOL_RESULT_BUDGET_TOKENS,
                  targetTokens,
              )
            : (inputs.recentToolResultBudgetTokens ?? RECENT_TOOL_RESULT_BUDGET_TOKENS),
        spec.codec,
        spec.conventions,
        inputs.preservePrefixBudgets === true,
    )
    const anchored = inputs.recentAssistantOutputs
        ? selectAnchoredConversation(
              turns,
              compactedRange,
              partition.turns.slice(partition.rawTailStartIndex),
              inputs.recentAssistantOutputs,
              contextLimit,
              overheadTokens,
              inputs.recentReasoningBudgetTokens ?? 0,
              preservedToolCallIds,
              spec.codec,
          )
        : undefined
    applyPreservationFloor(
        preservedToolCallIds,
        priorBoundary ? partitionTurns(turns, priorBoundary).compactedRange : [],
        prior,
    )
    const priorStages = new Set(
        (prior?.stages ?? [])
            .filter((stage) => stage.status !== "skipped" && stage.status !== "failed")
            .map((stage) => stage.name),
    )
    const ctx: StageContext = {
        codec: spec.codec,
        conventions: spec.conventions,
        estimator,
        rawTailStartIndex: partition.rawTailStartIndex,
        transcriptRelativePath,
        archiveCatalogText: inputs.archiveCatalogText ?? prior?.archiveCatalogText,
        preservedToolCallIds,
        protectRecentTools: inputs.preservePrefixBudgets,
        preservedReasoningItemKeys: anchored?.reasoningKeys ?? new Set<string>(),
        protectedAssistantItemKeys: anchored?.textKeys,
        latestTodoCallId: findLatestTodoCallId(compactedRange, spec.conventions),
        assistantSummaries,
        assistantSummaryKeys: new Set<string>(
            prior?.requiresCustomCompaction ? [] : (prior?.assistantSummaryKeys ?? []),
        ),
        summaryJobs,
        selectRuns: true,
        sourceTurns: new Map(compactedRange.map((turn) => [turn.key, turn])),
        targetTokens,
        referenceTokens: 0,
        collapsePercent: inputs.collapsePercent,
        summariesAllowed: inputs.summariesAllowed !== false,
    }
    const prefixSummaryAllowed = inputs.prefixSummaryAllowed !== false
    // The applied output always carries a reference message; gates must account
    // for it so a "trigger met" claim holds for the real transformed context.
    const reference = synthesizeReferenceTurn(turns, compactedRange, ctx, compactedRangeHash)
    ctx.referenceTokens = reference ? spec.codec.estimateTurns([reference]) : 0
    const projectedTokens = () =>
        estimateTurns(working, spec.codec, estimator) + ctx.referenceTokens
    let nativePrefixSource: Turn[] | undefined

    // Escalation chases the TARGET, not the trigger: the trigger decides when
    // compaction happens, the target decides how deep it goes. Stopping at
    // first-under-trigger left sessions at ~50-80% and re-compacting every few
    // turns instead of dropping to the profile's target.
    for (const stage of spec.stages) {
        if (!stage.always && projectedTokens() <= targetTokens && !priorStages.has(stage.name)) {
            markTargetMet(stages)
            continue
        }
        if (stage.name === "reasoning") {
            const withoutOlderReasoning = working.map((turn, index) => ({
                ...turn,
                items:
                    index < partition.rawTailStartIndex
                        ? turn.items.filter((item) => item.kind !== "reasoning")
                        : turn.items,
            }))
            const nonReasoningProjection =
                estimateTurns(withoutOlderReasoning, spec.codec, estimator) + ctx.referenceTokens
            ctx.preservedReasoningItemKeys = anchored
                ? anchored.reasoningKeys
                : selectRecentReasoningKeys(
                      compactedRange,
                      inputs.recentReasoningBudgetTokens ?? 0,
                      targetTokens,
                      nonReasoningProjection,
                      spec.codec,
                      priorBoundary ? partitionTurns(turns, priorBoundary).compactedRange : [],
                      prior,
                  )
        }
        if (stage.name === "assistant-runs" && inputs.preservePrefixBudgets)
            nativePrefixSource = cloneTurns(working)
        runStage(stages, working, estimator, spec.codec, stage, ctx)
    }

    let requiresCustomCompaction = false
    let preservedPrefixTurnKeys: string[] = []
    let prefixSummary =
        inputs.prefixSummary ??
        rolledPrefixSummary ??
        (expandedPrefix ? undefined : priorPrefixSummary)
    // The target is best-effort. Within 15% of it, do not replace an entire
    // older prefix merely to shave the last few tokens. A previously applied
    // prefix still carries historical state and must remain monotonic.
    if (
        prefixSummaryAllowed &&
        (projectedTokens() > Math.floor(targetTokens * 1.15) ||
            prior?.requiresCustomCompaction ||
            inputs.prefixSummary !== undefined)
    ) {
        const newlyCompactedTurns =
            expandedPrefix && priorBoundary
                ? turnsBetweenBoundaries(turns, priorBoundary, boundary)
                : []
        if (
            expandedPrefix &&
            priorPrefixSummary &&
            inputs.prefixSummary === undefined &&
            rolledPrefixSummary === undefined
        ) {
            const extended = extendPrefixSummary(
                priorPrefixSummary,
                newlyCompactedTurns,
                spec.conventions,
                partition.turns.slice(partition.rawTailStartIndex),
            )
            if (extended)
                prefixSummary = dedupeRepeatableUserTextInSummary(extended, turns, spec.conventions)
        }
        if (
            ctx.summariesAllowed !== false &&
            prefixSummaryJobKey &&
            priorPrefixSummary &&
            inputs.prefixSummary === undefined &&
            rolledPrefixSummary === undefined &&
            newlyCompactedTurns.length > 0
        ) {
            summaryJobs.push({
                key: prefixSummaryJobKey,
                rangeStartMessageId: newlyCompactedTurns[0].key,
                rangeEndMessageId: newlyCompactedTurns.at(-1)?.key ?? newlyCompactedTurns[0].key,
                transcriptRelativePath,
                prompt: formatPrefixSummaryPrompt(
                    priorPrefixSummary,
                    newlyCompactedTurns,
                    transcriptRelativePath,
                    spec.codec,
                ),
            })
        }
        const latestTodoState = formatLatestTodoState(compactedRange, ctx)
        if (latestTodoState && !(prefixSummary ?? "").includes(latestTodoState)) {
            prefixSummary = `${prefixSummary ?? formatPrefixSummary(compactedRange, spec.conventions, partition.turns.slice(partition.rawTailStartIndex))}\n- ${latestTodoState}`
        }
        const recentNotes = [
            ...new Set(
                compactedRange.flatMap((turn) =>
                    turn.items
                        .map((item) => spec.conventions.itemNote?.(item))
                        .filter((note): note is string => !!note),
                ),
            ),
        ].slice(-5)
        for (const note of recentNotes) {
            const concise = referencePhrase(note, 240)
            if ((prefixSummary ?? "").includes(concise)) continue
            prefixSummary = `${prefixSummary ?? formatPrefixSummary(compactedRange, spec.conventions, partition.turns.slice(partition.rawTailStartIndex))}\n- ${concise}`
        }
        const currentTailStartIndex = partition.rawTailKey
            ? working.findIndex((turn) => turn.key === partition.rawTailKey)
            : working.length
        const beforePrefix = projectedTokens()
        // Replacing the entire prefix can overshoot a best-effort target by
        // orders of magnitude. Keep the newest complete, pruned turns native
        // in the space the handoff leaves available. Never resurrect turns
        // already inside a prior checkpoint: only the newly covered suffix is
        // eligible. The complete projected context, not summary length alone,
        // determines this allowance.
        if (inputs.preservePrefixBudgets) {
            const source = nativePrefixSource ?? working
            const preview = cloneTurns(source)
            applyPrefixSummary(
                preview,
                currentTailStartIndex > 0 ? currentTailStartIndex : partition.rawTailStartIndex,
                transcriptRelativePath,
                prefixSummary,
                compactedRangeHash,
                spec.conventions,
                ctx.preservedReasoningItemKeys,
                ctx.preservedToolCallIds,
                new Set(),
                undefined,
                anchored?.textKeys,
            )
            let headroom = targetTokens - estimateTurns(preview, spec.codec, estimator)
            const eligibleStart = priorBoundary
                ? partitionTurns(turns, priorBoundary).compactedRange.length
                : 0
            const prefixEnd = currentTailStartIndex > 0
                ? currentTailStartIndex
                : partition.rawTailStartIndex
            const previouslyNative = new Set(prior?.preservedPrefixTurnKeys ?? [])
            // Earlier native prefix turns are a continuity floor until a new,
            // validated handoff can retire the archive cohort containing them.
            // Carrying them unconditionally after that point forces a valid
            // replacement to coexist with its own source and can make the
            // *complete* outgoing context larger rather than smaller.
            const retirePreviousNative =
                inputs.prefixSummary !== undefined &&
                inputs.retirementThrough !== undefined &&
                inputs.retirementThrough > (prior?.retirementThrough ?? 0)
            for (const candidate of source.slice(0, prefixEnd)) {
                if (!previouslyNative.has(candidate.key) || retirePreviousNative) continue
                preservedPrefixTurnKeys.push(candidate.key)
                headroom -= spec.codec.estimateTurns([candidate])
            }
            for (let index = prefixEnd - 1; index >= eligibleStart; index--) {
                const candidate = source[index]
                if (previouslyNative.has(candidate.key)) continue
                const cost = spec.codec.estimateTurns([candidate])
                if (cost > headroom) break
                preservedPrefixTurnKeys.unshift(candidate.key)
                headroom -= cost
            }
        }
        const result = applyPrefixSummary(
            working,
            currentTailStartIndex > 0 ? currentTailStartIndex : partition.rawTailStartIndex,
            transcriptRelativePath,
            prefixSummary,
            compactedRangeHash,
            spec.conventions,
            inputs.preservePrefixBudgets ? ctx.preservedReasoningItemKeys : new Set(),
            inputs.preservePrefixBudgets ? ctx.preservedToolCallIds : new Set(),
            new Set(preservedPrefixTurnKeys),
            nativePrefixSource,
            inputs.preservePrefixBudgets ? anchored?.textKeys : undefined,
        )
        const afterPrefix = estimateTurns(working, spec.codec, estimator)
        prefixSummary = result.prefixSummary
        requiresCustomCompaction = result.changedTurns.size > 0
        stages.push({
            name: "prefix-summary",
            label: "Last-resort prefix summary",
            beforeTokens: beforePrefix,
            afterTokens: afterPrefix,
            clearedTokens: Math.max(0, beforePrefix - afterPrefix),
            changedMessages: result.changedTurns.size,
            changedParts: result.changedItems,
            status: afterPrefix <= targetTokens ? "applied" : "failed",
        })
    }

    // Per-turn summaries are not visible once the old prefix has become a
    // single checkpoint. Keep only a rolling prefix job; do not charge for
    // assistant-turn calls that cannot enter the applied context.
    if (requiresCustomCompaction) {
        for (let index = summaryJobs.length - 1; index >= 0; index--) {
            if (!summaryJobs[index].key.startsWith("prefix-summary:")) summaryJobs.splice(index, 1)
        }
    }

    const plan: BoundaryContextPlan = {
        sessionId: inputs.sessionKey,
        rangeHash: compactedRangeHash,
        contextLimit,
        beforeTokens,
        afterPruneTokens: estimateTurns(working, spec.codec, estimator),
        overheadTokens,
        triggerTokens,
        targetTokens,
        ...(inputs.prefixSummaryAllowed !== undefined
            ? { prefixSummaryAllowed: inputs.prefixSummaryAllowed }
            : {}),
        ...(inputs.collapsePercent !== undefined
            ? { collapsePercent: inputs.collapsePercent }
            : {}),
        ...(inputs.minTailUserTurns !== undefined
            ? { minTailUserTurns: inputs.minTailUserTurns }
            : {}),
        rawTailStartIndex,
        rawTailStartMessageId: turns[boundary.turnIndex]?.key ?? turns.at(-1)?.key ?? "",
        rawTailItemBoundary: recordedItemBoundary(turns, boundary),
        requiresCustomCompaction,
        preservedToolCallIds: [...ctx.preservedToolCallIds],
        toolSurvivesPrefix:
            !!inputs.preservePrefixBudgets &&
            requiresCustomCompaction &&
            ctx.preservedToolCallIds.size > 0,
        ...(inputs.recentReasoningBudgetTokens
            ? { preservedReasoningItemKeys: [...ctx.preservedReasoningItemKeys] }
            : {}),
        ...(inputs.recentAssistantOutputs !== undefined
            ? { recentAssistantOutputs: inputs.recentAssistantOutputs }
            : {}),
        ...(anchored?.textKeys.size
            ? { protectedAssistantItemKeys: [...anchored.textKeys] }
            : {}),
        ...(anchored?.limited ? { anchorReasoningLimited: true } : {}),
        assistantSurvivesPrefix:
            !!inputs.preservePrefixBudgets &&
            requiresCustomCompaction &&
            !!anchored?.textKeys.size,
        reasoningSurvivesPrefix:
            !!inputs.preservePrefixBudgets &&
            requiresCustomCompaction &&
            ctx.preservedReasoningItemKeys.size > 0,
        ...(preservedPrefixTurnKeys.length > 0 ? { preservedPrefixTurnKeys } : {}),
        recentReasoningBudgetTokens: inputs.recentReasoningBudgetTokens,
        preservePrefixBudgets: inputs.preservePrefixBudgets,
        transcript: {
            relativePath: transcriptRelativePath,
            content: "",
            messageIds: compactedRange.map((turn) => turn.key),
            turns: compactedRange,
        },
        stages,
        summaryJobs,
        // Once the consolidated prefix is materialized, its per-turn inputs
        // are absorbed. Keep caches only when replay still needs those turns.
        assistantSummaryKeys: requiresCustomCompaction ? [] : [...ctx.assistantSummaryKeys],
        assistantSummaries: requiresCustomCompaction ? {} : ctx.assistantSummaries,
        prefixSummary,
        archiveCatalogText: inputs.archiveCatalogText ?? prior?.archiveCatalogText,
        archiveGeneration: inputs.archiveGeneration ?? prior?.archiveGeneration,
        retirementThrough: inputs.retirementThrough ?? prior?.retirementThrough,
    }
    const applied = transformTurns(turns, rawTailStartIndex, plan, spec)
    plan.afterPruneTokens = estimateTurns(applied, spec.codec, estimator)
    const estimateComponent = (component: Turn[]) =>
        component.length ? spec.codec.estimateTurns(component) : 0
    const rawTailTokens = estimateComponent(partition.turns.slice(partition.rawTailStartIndex))
    const protectedPartTokens = estimateComponent(
        preservedPrefixTurns(
            applied,
            new Set(plan.preservedReasoningItemKeys ?? []),
            new Set(plan.preservedToolCallIds),
        ),
    )
    const handoffTokens = estimateComponent(
        applied.filter(
            (turn) =>
                turn.key.startsWith("better_compact_summary_") ||
                turn.key.startsWith("better_compact_context_"),
        ),
    )
    plan.residual = {
        rawTailTokens,
        protectedPartTokens,
        handoffTokens,
        otherTokens: Math.max(
            0,
            plan.afterPruneTokens -
                overheadTokens -
                rawTailTokens -
                protectedPartTokens -
                handoffTokens,
        ),
        overheadTokens,
    }
    return plan
}

export function transformTurns(
    turns: Turn[],
    rawTailStartIndex: number,
    plan: BoundaryContextPlan,
    spec: LadderSpec,
): Turn[] {
    const resolvedBoundary = resolveTailBoundary(turns, plan)
    if (!resolvedBoundary && plan.rawTailItemBoundary !== undefined) return turns
    const boundary = resolvedBoundary ?? {
        turnIndex: rawTailStartIndex,
        itemIndex: 0,
    }
    const partition = partitionTurns(turns, boundary)
    const originalPrefix = partition.compactedRange
    // Replay the recorded strip stages exactly as the planner simulated them,
    // then summarize assistant runs over the stripped prefix. This keeps the
    // applied output identical to the simulation used for the plan's numbers.
    const stageNames = new Set<string>(
        plan.stages
            .filter((stage) => stage.status !== "skipped" && stage.status !== "failed")
            .map((stage) => stage.name),
    )
    const working = partition.turns
    const ctx: StageContext = {
        codec: spec.codec,
        conventions: spec.conventions,
        estimator: { overheadTokens: plan.overheadTokens },
        rawTailStartIndex: partition.rawTailStartIndex,
        transcriptRelativePath: plan.transcript.relativePath,
        archiveCatalogText: plan.archiveCatalogText,
        preservedToolCallIds: new Set(plan.preservedToolCallIds),
        protectRecentTools: plan.preservePrefixBudgets,
        preservedReasoningItemKeys: new Set(plan.preservedReasoningItemKeys ?? []),
        protectedAssistantItemKeys: new Set(plan.protectedAssistantItemKeys ?? []),
        latestTodoCallId: findLatestTodoCallId(originalPrefix, spec.conventions),
        assistantSummaries: plan.assistantSummaries,
        assistantSummaryKeys: new Set(plan.assistantSummaryKeys),
        summaryJobs: [],
        selectRuns: false,
        sourceTurns: new Map(originalPrefix.map((turn) => [turn.key, turn])),
        targetTokens: plan.targetTokens,
        referenceTokens: 0,
        // Replay never queues jobs; gate them off so a stray push cannot leak.
        summariesAllowed: false,
    }
    for (const stage of spec.stages) {
        if (stage.name === "assistant-runs") continue
        if (stageNames.has(stage.name)) stage.run(working, ctx)
    }
    if (plan.requiresCustomCompaction) {
        const preserved = preservedPrefixTurns(
            working.slice(0, partition.rawTailStartIndex),
            plan.reasoningSurvivesPrefix
                ? new Set(plan.preservedReasoningItemKeys ?? [])
                : new Set(),
            plan.toolSurvivesPrefix ? new Set(plan.preservedToolCallIds) : new Set(),
            new Set(plan.preservedPrefixTurnKeys ?? []),
            plan.assistantSurvivesPrefix
                ? new Set(plan.protectedAssistantItemKeys ?? [])
                : new Set(),
        )
        const nativeKeys = new Set(plan.preservedPrefixTurnKeys)
        const native = working
            .slice(0, partition.rawTailStartIndex)
            .filter((turn) => nativeKeys.has(turn.key))
        return partition.finalize([
            synthesizeSummaryTurn(
                originalPrefix,
                plan.prefixSummary ||
                    formatPrefixSummary(
                        originalPrefix,
                        spec.conventions,
                        partition.turns.slice(partition.rawTailStartIndex),
                    ),
                plan.transcript.relativePath,
                plan.rangeHash,
                plan.archiveCatalogText,
            ),
            ...preserved,
            ...native,
            ...working.slice(partition.rawTailStartIndex),
        ])
    }
    let prefix = working.slice(0, partition.rawTailStartIndex)
    if (stageNames.has("assistant-runs")) {
        prefix = transformCompactedPrefix(prefix, ctx)
    }
    const result = [...prefix]
    const reference = synthesizeReferenceTurn(turns, originalPrefix, ctx, plan.rangeHash)
    if (reference) result.push(reference)
    result.push(...working.slice(partition.rawTailStartIndex))
    return partition.finalize(result)
}

export interface ReplayOptions {
    // Apply the plan even when the pruned context has regrown past the
    // trigger. Hosts that cannot rebuild (automatic compaction disabled or
    // denied) prefer a stale-but-valid plan over sending raw history.
    allowRegrown?: boolean
}

export function replayPlanSnapshot(
    turns: Turn[],
    snapshot: PlanSnapshot,
    spec: LadderSpec,
    options: ReplayOptions = {},
): Turn[] | null {
    const boundary = resolveTailBoundary(turns, snapshot)
    if (!boundary || !matchesPlanSnapshot(turns, snapshot)) return null
    const rawTailStartIndex = boundary.turnIndex
    const overheadTokens = snapshot.overheadTokens ?? 0
    const prefixSummary = snapshot.prefixSummary
        ? dedupeRepeatableUserTextInSummary(snapshot.prefixSummary, turns, spec.conventions)
        : undefined
    const transformed = transformTurns(
        turns,
        rawTailStartIndex,
        {
            sessionId: snapshot.sessionId,
            rangeHash: snapshot.rangeHash,
            contextLimit:
                snapshot.contextLimit ?? Math.max(snapshot.beforeTokens, snapshot.targetTokens, 1),
            beforeTokens: snapshot.beforeTokens,
            afterPruneTokens: snapshot.afterPruneTokens,
            overheadTokens,
            triggerTokens: snapshot.triggerTokens,
            targetTokens: snapshot.targetTokens,
            rawTailStartIndex,
            rawTailStartMessageId: snapshot.rawTailStartMessageId,
            rawTailItemBoundary: snapshot.rawTailItemBoundary,
            requiresCustomCompaction: snapshot.requiresCustomCompaction,
            preservedToolCallIds: snapshot.preservedToolCallIds ?? [],
            toolSurvivesPrefix: snapshot.toolSurvivesPrefix,
            preservedReasoningItemKeys: snapshot.preservedReasoningItemKeys ?? [],
            protectedAssistantItemKeys: snapshot.protectedAssistantItemKeys,
            assistantSurvivesPrefix: snapshot.assistantSurvivesPrefix,
            recentAssistantOutputs: snapshot.recentAssistantOutputs,
            anchorReasoningLimited: snapshot.anchorReasoningLimited,
            reasoningSurvivesPrefix: snapshot.reasoningSurvivesPrefix,
            preservedPrefixTurnKeys: snapshot.preservedPrefixTurnKeys,
            recentReasoningBudgetTokens: snapshot.recentReasoningBudgetTokens,
            preservePrefixBudgets: snapshot.preservePrefixBudgets,
            assistantSummaryKeys:
                snapshot.assistantSummaryKeys ?? Object.keys(snapshot.assistantSummaries ?? {}),
            transcript: {
                relativePath: snapshot.transcriptRelativePath,
                content: "",
                messageIds: [],
            },
            stages: (snapshot.stages ?? []) as BoundaryStageReport[],
            summaryJobs: [],
            assistantSummaries: snapshot.assistantSummaries ?? {},
            prefixSummary,
            archiveCatalogText: snapshot.archiveCatalogText,
            archiveGeneration: snapshot.archiveGeneration,
            retirementThrough: snapshot.retirementThrough,
        },
        spec,
    )
    // Once new turns regrow the context past the trigger, the frozen plan no
    // longer suffices; refuse so the caller rebuilds with a fresh boundary.
    if (
        !options.allowRegrown &&
        spec.codec.estimateTurns(transformed) + overheadTokens >= snapshot.triggerTokens
    ) {
        return null
    }
    return transformed
}

export function matchesPlanSnapshot(turns: Turn[], snapshot: PlanSnapshot): boolean {
    const boundary = resolveTailBoundary(turns, snapshot)
    if (!boundary) return false
    const compactedRange = partitionTurns(turns, boundary).compactedRange
    if (compactedRange.length === 0) return false
    return rangeHash(compactedRange) === snapshot.rangeHash
}

export type ProcessResult =
    | { outcome: "unchanged" }
    | { outcome: "replayed"; turns: Turn[] }
    | { outcome: "planned"; turns: Turn[]; plan: BoundaryContextPlan }

export interface Engine {
    process(request: {
        sessionKey: string
        turns: Turn[]
        contextLimit?: number
        triggerRatio?: number
        targetRatio?: number
        triggerTokens?: number
        targetTokens?: number
        recentToolResultBudgetTokens?: number
        recentReasoningBudgetTokens?: number
        recentAssistantOutputs?: number
        providerReportedTokens?: number
        providerHistoryTokens?: number
        tailBudgetTokens?: { floor: number; ceiling: number }
        minTailUserTurns?: number
        summariesAllowed?: boolean
        prefixSummaryAllowed?: boolean
        collapsePercent?: number
        archiveCatalogText?: string
        archiveGeneration?: number
        retirementThrough?: number
        prefixSummary?: string
        force?: boolean
        preservePrefixBudgets?: boolean
        /** OpenCode can check another provider response without advancing its archive boundary. */
        reuseStablePrefixOnForce?: boolean
        // Side-model summary results for the automatic path. When a
        // fresh plan queues summary jobs, the engine runs them and rebuilds
        // the plan with the accepted summaries before persisting it.
        summarize?: (jobs: BoundarySummaryJob[]) => Promise<Record<string, string>>
        summarizePrefix?: (
            plan: BoundaryContextPlan,
            turns: Turn[],
        ) => Promise<string | null | undefined>
        summarizeArchive?: (
            plan: BoundaryContextPlan,
            turns: Turn[],
        ) => Promise<
            | {
                  handoff: string
                  catalogText: string
                  retirementThrough?: number
                  commit(): Promise<void>
                  discard?(): Promise<void>
              }
            | null
            | undefined
        >
    }): Promise<ProcessResult>
}

// The boundary-time transform: replay the cached plan when it still holds,
// otherwise discard it and build, persist, and apply a fresh one.
export function createEngine(spec: LadderSpec, ports: EnginePorts): Engine {
    return {
        async process({
            sessionKey,
            turns,
            contextLimit,
            triggerRatio,
            targetRatio,
            triggerTokens,
            targetTokens,
            recentToolResultBudgetTokens,
            recentReasoningBudgetTokens,
            recentAssistantOutputs,
            providerReportedTokens,
            providerHistoryTokens,
            tailBudgetTokens,
            minTailUserTurns,
            summariesAllowed,
            prefixSummaryAllowed,
            collapsePercent,
            archiveCatalogText,
            archiveGeneration,
            retirementThrough,
            prefixSummary,
            force,
            preservePrefixBudgets,
            reuseStablePrefixOnForce,
            summarize,
            summarizePrefix,
            summarizeArchive,
        }) {
            let staleSnapshotCleared = false
            let tailPolicyChanged = false
            let compatibleCachedPlan = false
            let priorPlan: PlanSnapshot | undefined
            const cached = await ports.plans.load(sessionKey)
            if (cached && cached.sessionId === sessionKey) {
                const cleanedSummary = cached.prefixSummary
                    ? dedupeRepeatableUserTextInSummary(
                          cached.prefixSummary,
                          turns,
                          spec.conventions,
                      )
                    : undefined
                const normalizedCached =
                    cleanedSummary !== cached.prefixSummary
                        ? { ...cached, prefixSummary: cleanedSummary }
                        : cached
                tailPolicyChanged = cached.minTailUserTurns !== minTailUserTurns
                const budgetsMatch =
                    cached.contextLimit === contextLimit &&
                    cached.triggerTokens ===
                        (triggerTokens ??
                            Math.floor((contextLimit ?? 0) * (triggerRatio ?? TRIGGER_RATIO))) &&
                    cached.targetTokens ===
                        (targetTokens ??
                            Math.floor((contextLimit ?? 0) * (targetRatio ?? TARGET_RATIO))) &&
                    cached.prefixSummaryAllowed === prefixSummaryAllowed &&
                    cached.collapsePercent === collapsePercent &&
                    cached.archiveCatalogText === archiveCatalogText &&
                    cached.archiveGeneration === archiveGeneration &&
                    cached.retirementThrough === retirementThrough &&
                    cached.recentReasoningBudgetTokens === recentReasoningBudgetTokens &&
                    cached.recentAssistantOutputs === recentAssistantOutputs &&
                    cached.preservePrefixBudgets === preservePrefixBudgets &&
                    !tailPolicyChanged
                compatibleCachedPlan = budgetsMatch && normalizedCached === cached
                const replayed =
                    force || !budgetsMatch
                        ? null
                        : replayPlanSnapshot(turns, normalizedCached, spec)
                if (replayed) {
                    if (normalizedCached !== cached)
                        await ports.plans.save(sessionKey, normalizedCached)
                    return { outcome: "replayed", turns: replayed }
                }
                staleSnapshotCleared = true
                priorPlan = cached
            }

            const inputs: BuildPlanInputs = {
                contextLimit,
                triggerRatio,
                targetRatio,
                triggerTokens,
                targetTokens,
                recentToolResultBudgetTokens,
                recentReasoningBudgetTokens,
                recentAssistantOutputs,
                providerReportedTokens,
                providerHistoryTokens,
                tailBudgetTokens,
                minTailUserTurns,
                summariesAllowed,
                prefixSummaryAllowed,
                collapsePercent,
                archiveCatalogText,
                archiveGeneration,
                retirementThrough,
                prefixSummary,
                preservePrefixBudgets,
                force: force || tailPolicyChanged,
                priorPlan,
                sessionKey,
                citablePath: ports.transcripts.citablePath,
            }
            let plan = buildPlan(turns, inputs, spec)
            if (!plan) {
                if (staleSnapshotCleared) await ports.plans.save(sessionKey, null)
                return { outcome: "unchanged" }
            }
            if (
                force &&
                reuseStablePrefixOnForce &&
                compatibleCachedPlan &&
                priorPlan &&
                prefixSummary === undefined &&
                plan.rangeHash === priorPlan.rangeHash &&
                plan.rawTailStartMessageId === priorPlan.rawTailStartMessageId &&
                JSON.stringify(plan.rawTailItemBoundary) ===
                    JSON.stringify(priorPlan.rawTailItemBoundary) &&
                plan.afterPruneTokens >= priorPlan.afterPruneTokens &&
                (providerReportedTokens ?? 0) < (contextLimit ?? 0)
            ) {
                const replayed = replayPlanSnapshot(turns, priorPlan, spec, { allowRegrown: true })
                if (
                    replayed &&
                    spec.codec.estimateTurns(replayed) + plan.overheadTokens < (contextLimit ?? 0)
                )
                    return { outcome: "replayed", turns: replayed }
            }
            // The original source must be durable before a model summary can
            // replace any portion of it in the outgoing request.
            await ports.archive?.(plan)
            // Once a prefix summary replaces all old turns, individual turn
            // summaries are no longer visible. Only a rolling prefix job can
            // improve that plan; avoid charging for discarded turn jobs.
            let prefixAttempted = false
            if (
                (summarizeArchive || summarizePrefix) &&
                plan.requiresCustomCompaction &&
                plan.afterPruneTokens > Math.floor(plan.targetTokens * 1.15)
            ) {
                try {
                    const proposal = summarizeArchive
                        ? await summarizeArchive(plan, turns)
                        : undefined
                    const replacement = summarizeArchive
                        ? proposal?.handoff
                        : await summarizePrefix!(plan, turns)
                    prefixAttempted = replacement !== undefined
                    if (replacement) {
                        const rebuilt = buildPlan(
                            turns,
                            {
                                ...inputs,
                                priorPlan: toPlanSnapshot(plan),
                                prefixSummary: replacement,
                                archiveCatalogText:
                                    proposal?.catalogText ?? inputs.archiveCatalogText,
                                retirementThrough:
                                    proposal?.retirementThrough ?? inputs.retirementThrough,
                            },
                            spec,
                        )
                        if (
                            rebuilt?.requiresCustomCompaction &&
                            rebuilt.afterPruneTokens < plan.afterPruneTokens
                        ) {
                            await proposal?.commit()
                            plan = rebuilt
                        } else {
                            await proposal?.discard?.()
                        }
                    }
                } catch (error) {
                    prefixAttempted = true
                    ports.logger.warn(
                        "Chunked prefix summary failed; retaining deterministic plan",
                        {
                            sessionId: sessionKey,
                            error: "summary_failure",
                        },
                    )
                }
            }
            const activeJobs =
                prefixAttempted || (summarizeArchive && plan.requiresCustomCompaction)
                    ? []
                    : plan.requiresCustomCompaction
                      ? plan.summaryJobs.filter((job) => job.key.startsWith("prefix-summary:"))
                      : plan.summaryJobs
            if (summarize && activeJobs.length > 0) {
                try {
                    const assistantSummaries = await summarize(activeJobs)
                    if (Object.keys(assistantSummaries).length > 0) {
                        const rebuilt = buildPlan(
                            turns,
                            {
                                ...inputs,
                                priorPlan: toPlanSnapshot(plan),
                                assistantSummaries,
                            },
                            spec,
                        )
                        // Structured per-turn summaries can cost more than the
                        // deterministic previews they replace. Keep the smaller
                        // plan when the whole batch would enlarge live context.
                        if (rebuilt && rebuilt.afterPruneTokens <= plan.afterPruneTokens) {
                            plan = rebuilt
                        } else if (rebuilt) {
                            ports.logger.info("Retained smaller plan after summary expansion", {
                                sessionId: sessionKey,
                                projectedTokens: plan.afterPruneTokens,
                                summarizedTokens: rebuilt.afterPruneTokens,
                            })
                        }
                    }
                } catch (error) {
                    ports.logger.warn("Summary scheduling failed; using deterministic fallback", {
                        sessionId: sessionKey,
                        error: "summary_failure",
                    })
                }
            }

            // The complete outgoing plan includes synthetic wrappers, native
            // protection and provider overhead. A tiny eligible prefix can
            // cost MORE after adding its reference even above the trigger.
            // Keep raw context (or a compatible smaller replay) rather than
            // persisting an expanding compaction and reporting false savings.
            const rawProjection = spec.codec.estimateTurns(turns) + plan.overheadTokens
            if (preservePrefixBudgets === true && plan.afterPruneTokens >= rawProjection) {
                const replayed =
                    compatibleCachedPlan && priorPlan
                        ? replayPlanSnapshot(turns, priorPlan, spec, { allowRegrown: true })
                        : null
                const replayProjection = replayed
                    ? spec.codec.estimateTurns(replayed) + plan.overheadTokens
                    : Infinity
                if (
                    replayed &&
                    replayProjection < rawProjection &&
                    replayProjection < (contextLimit ?? 0)
                )
                    return { outcome: "replayed", turns: replayed }
                if (staleSnapshotCleared) await ports.plans.save(sessionKey, null)
                ports.logger.info("Skipped expanding Better Compact plan", {
                    sessionId: sessionKey,
                    projectedTokens: plan.afterPruneTokens,
                    rawTokens: rawProjection,
                })
                return { outcome: "unchanged" }
            }

            await writeTranscript(plan, {
                transcripts: ports.transcripts,
                logger: ports.logger,
                codec: spec.codec,
            })
            const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
            await ports.plans.save(sessionKey, toPlanSnapshot(plan))
            ports.logger.info("Applied Better Compact staged pruning", {
                sessionId: plan.sessionId,
                beforeTokens: plan.beforeTokens,
                afterPruneTokens: plan.afterPruneTokens,
                transcript: plan.transcript.relativePath,
                stages: plan.stages.map((stage) => stage.name),
            })
            return { outcome: "planned", turns: transformed, plan }
        },
    }
}

function boundedTailBudget(
    budget: { floor: number; ceiling: number },
    targetTokens: number,
): { floor: number; ceiling: number } {
    const ceiling = Math.min(budget.ceiling, targetTokens)
    return { floor: Math.min(budget.floor, ceiling), ceiling }
}

function selectTailBoundary(
    turns: Turn[],
    wholeTurnStartIndex: number,
    triggerTokens: number,
    targetTokens: number,
    codec: CodecOps,
): TailBoundary {
    let selected: TailBoundary = { turnIndex: wholeTurnStartIndex, itemIndex: 0 }
    for (let turnIndex = wholeTurnStartIndex; turnIndex < turns.length; turnIndex++) {
        const turn = turns[turnIndex]
        if (turn.items.length === 0 || codec.estimateTurns([turn]) <= triggerTokens) continue
        const itemIndex = rawSuffixItemIndex(turn, targetTokens, codec)
        if (itemIndex > 0) selected = { turnIndex, itemIndex }
    }
    return selected
}

function rawSuffixItemIndex(turn: Turn, targetTokens: number, codec: CodecOps): number {
    let firstRawIndex = turn.items.length
    for (let index = turn.items.length - 1; index >= 0; index--) {
        const candidate = { ...turn, items: turn.items.slice(index) }
        if (codec.estimateTurns([candidate]) <= targetTokens) {
            firstRawIndex = index
            continue
        }
        if (firstRawIndex < turn.items.length) break
        if (turn.items[index].kind === "tool") return turn.items.length
        return index > 0 ? index : 0
    }
    return firstRawIndex > 0 ? firstRawIndex : 0
}

function resolveTailBoundary(
    turns: Turn[],
    recorded: Pick<PlanSnapshot, "rawTailStartMessageId" | "rawTailItemBoundary">,
): TailBoundary | null {
    const turnIndex = turns.findIndex((turn) => turn.key === recorded.rawTailStartMessageId)
    if (turnIndex < 0) return null
    const itemBoundary = recorded.rawTailItemBoundary
    if (itemBoundary === undefined) {
        return turnIndex > 0 ? { turnIndex, itemIndex: 0 } : null
    }
    const turn = turns[turnIndex]
    const itemIndex = turn.items.findIndex((item) => item.key === itemBoundary.itemKey)
    if (itemIndex < 0) return null
    const boundaryIndex = itemBoundary.side === "after" ? itemIndex + 1 : itemIndex
    return boundaryIndex > 0 ? { turnIndex, itemIndex: boundaryIndex } : null
}

function recordedItemBoundary(
    turns: Turn[],
    boundary: TailBoundary,
): RawTailItemBoundary | undefined {
    if (boundary.itemIndex === 0) return undefined
    const turn = turns[boundary.turnIndex]
    const firstRawItem = turn.items[boundary.itemIndex]
    if (firstRawItem) return { itemKey: firstRawItem.key, side: "before" }
    const lastCompactedItem = turn.items[boundary.itemIndex - 1]
    return lastCompactedItem ? { itemKey: lastCompactedItem.key, side: "after" } : undefined
}

function partitionTurns(turns: Turn[], boundary: TailBoundary): PartitionedTurns {
    const source = turns[boundary.turnIndex]
    if (!source || boundary.itemIndex === 0) {
        const cloned = cloneTurns(turns)
        return {
            turns: cloned,
            compactedRange: cloneTurns(turns.slice(0, boundary.turnIndex)),
            rawTailStartIndex: boundary.turnIndex,
            rawTailKey: cloned[boundary.turnIndex]?.key,
            finalize: (transformed) => transformed,
        }
    }

    const compactedItems = source.items.slice(0, boundary.itemIndex)
    const rawItems = source.items.slice(boundary.itemIndex)
    const fragmentKey = JSON.stringify(compactedItems.map((item) => item.key))
    const compactedFragment: Turn = {
        ...source,
        items: compactedItems,
        fragmentKey,
    }
    const before = cloneTurns(turns.slice(0, boundary.turnIndex))
    const after = cloneTurns(turns.slice(boundary.turnIndex + 1))

    if (rawItems.length === 0) {
        const partitioned = [...before, compactedFragment, ...after]
        const rawTailStartIndex = before.length + 1
        return {
            turns: partitioned,
            compactedRange: cloneTurns(partitioned.slice(0, rawTailStartIndex)),
            rawTailStartIndex,
            rawTailKey: after[0]?.key,
            finalize: (transformed) => transformed,
        }
    }

    const rawTailKey = `${source.key}:better-compact-raw:${rawItems[0].key}`
    const rawFragment: Turn = { ...source, key: rawTailKey, items: rawItems }
    const partitioned = [...before, compactedFragment, rawFragment, ...after]
    const rawTailStartIndex = before.length + 1
    return {
        turns: partitioned,
        compactedRange: cloneTurns(partitioned.slice(0, rawTailStartIndex)),
        rawTailStartIndex,
        rawTailKey,
        finalize(transformed) {
            const rawIndex = transformed.findIndex((turn) => turn.key === rawTailKey)
            if (rawIndex < 0) return cloneTurns(turns)
            const compactedIndex = transformed.findIndex(
                (turn, index) => index < rawIndex && turn.key === source.key,
            )
            const compactedItems = compactedIndex >= 0 ? transformed[compactedIndex].items : []
            const result = transformed.filter((_turn, index) => index !== compactedIndex)
            const adjustedRawIndex = compactedIndex >= 0 ? rawIndex - 1 : rawIndex
            result[adjustedRawIndex] = {
                ...source,
                items: [...compactedItems, ...transformed[rawIndex].items],
            }
            return result
        },
    }
}

function compareBoundaries(left: TailBoundary, right: TailBoundary): number {
    return left.turnIndex === right.turnIndex
        ? left.itemIndex - right.itemIndex
        : left.turnIndex - right.turnIndex
}

function turnsBetweenBoundaries(turns: Turn[], start: TailBoundary, end: TailBoundary): Turn[] {
    if (compareBoundaries(end, start) <= 0) return []
    const delta: Turn[] = []
    for (let turnIndex = start.turnIndex; turnIndex <= end.turnIndex; turnIndex++) {
        const turn = turns[turnIndex]
        const startItem = turnIndex === start.turnIndex ? start.itemIndex : 0
        const endItem = turnIndex === end.turnIndex ? end.itemIndex : turn.items.length
        if (endItem <= startItem) continue
        delta.push({
            ...turn,
            items: turn.items.slice(startItem, endItem),
            fragmentKey: JSON.stringify(
                turn.items.slice(startItem, endItem).map((item) => item.key),
            ),
        })
    }
    return delta
}

function runStage(
    stages: BoundaryStageReport[],
    working: Turn[],
    estimator: Estimator,
    codec: CodecOps,
    stage: Stage,
    ctx: StageContext,
): void {
    const beforeTokens = estimateTurns(working, codec, estimator)
    const result = stage.run(working, ctx)
    const afterTokens = estimateTurns(working, codec, estimator)
    stages.push({
        name: stage.name,
        label: stage.label,
        beforeTokens,
        afterTokens,
        clearedTokens: Math.max(0, beforeTokens - afterTokens),
        changedMessages: result.changedTurns.size,
        changedParts: result.changedItems,
        status: result.changedTurns.size > 0 || result.changedItems > 0 ? "applied" : "skipped",
    })
}

// Tool results the model already lost to the prior plan must not resurface
// in the replacement: drop newly-preserved call ids that live inside the
// prior plan's compacted prefix unless the prior plan preserved them too.
function applyPreservationFloor(
    preservedToolCallIds: Set<string>,
    priorCompactedRange: Turn[],
    prior: PlanSnapshot | undefined,
): void {
    if (!prior || priorCompactedRange.length === 0) return
    const previouslyPreserved = new Set(
        prior.requiresCustomCompaction && !prior.toolSurvivesPrefix
            ? []
            : (prior.preservedToolCallIds ?? []),
    )
    for (const turn of priorCompactedRange) {
        if (
            turn.prunableToolLike &&
            preservedToolCallIds.has(turn.key) &&
            !previouslyPreserved.has(turn.key)
        ) {
            preservedToolCallIds.delete(turn.key)
        }
        for (const item of turn.items) {
            if (
                item.kind === "tool" &&
                item.callId &&
                preservedToolCallIds.has(item.callId) &&
                !previouslyPreserved.has(item.callId)
            ) {
                preservedToolCallIds.delete(item.callId)
            }
        }
    }
}

/** Preserve coherent assistant output/reasoning spans instead of isolated thoughts.
 * Tool calls are intentionally excluded: their separate allowance and exact
 * archive cover those bytes. This OpenCode policy is opt-in via the profile. */
function selectAnchoredConversation(
    turns: Turn[],
    compacted: Turn[],
    rawTail: Turn[],
    outputCount: number,
    contextLimit: number,
    overheadTokens: number,
    fallbackReasoningBudget: number,
    protectedTools: ReadonlySet<string>,
    codec: CodecOps,
): { textKeys: Set<string>; reasoningKeys: Set<string>; limited: boolean } {
    const textKeys = new Set<string>()
    const reasoningKeys = new Set<string>()
    const selected: number[] = []
    for (let index = turns.length - 1; index >= 0 && selected.length < outputCount; index--) {
        const turn = turns[index]
        if (
            turn.role === "assistant" &&
            turn.items.some((item) => item.kind === "text" && item.text.trim().length > 0)
        ) selected.unshift(index)
    }
    if (!selected.length) return { textKeys, reasoningKeys, limited: false }
    const covered = new Set(compacted.flatMap((turn) => turn.items.map((item) => item.key)))
    for (const index of selected) {
        for (const item of turns[index].items) {
            if (item.kind === "text" && covered.has(item.key)) textKeys.add(item.key)
        }
    }
    const reasoning = turns
        .slice(selected[0], selected.at(-1)! + 1)
        .flatMap((turn) => turn.items)
        .filter((item) => item.kind === "reasoning" && covered.has(item.key))
    const itemCost = (item: Item) =>
        codec.estimateTurns([{ key: "protected", stamp: 0, role: "assistant", items: [item] }])
    const textCost = compacted.reduce(
        (total, turn) =>
            total +
            turn.items.reduce(
                (sum, item) => sum + (textKeys.has(item.key) ? itemCost(item) : 0),
                0,
            ),
        0,
    )
    const toolCost = compacted.reduce(
        (total, turn) =>
            total +
            turn.items.reduce(
                (sum, item) =>
                    sum +
                    (item.kind === "tool" && protectedTools.has(item.callId)
                        ? codec.estimateItem(item)
                        : 0),
                0,
            ),
        0,
    )
    // Reserve room for the next provider response and unpriced host wrappers;
    // the user's 28k reasoning setting is a fallback when the whole interval
    // cannot safely share the provider window with the five outputs.
    const buffer = Math.min(
        Math.ceil(contextLimit * 0.2),
        Math.max(8_192, Math.ceil(contextLimit * 0.1)),
    )
    const available = Math.max(
        0,
        contextLimit -
            overheadTokens -
            codec.estimateTurns(rawTail) -
            textCost -
            toolCost -
            buffer -
            Math.max(512, Math.ceil(contextLimit * 0.01)),
    )
    const needed = reasoning.reduce((total, item) => total + itemCost(item), 0)
    const limited = needed > available
    let remaining = limited ? Math.min(available, fallbackReasoningBudget) : available
    for (const item of [...reasoning].reverse()) {
        const cost = itemCost(item)
        if (cost > remaining) break
        reasoningKeys.add(item.key)
        remaining -= cost
    }
    return { textKeys, reasoningKeys, limited }
}

function selectRecentReasoningKeys(
    compacted: Turn[],
    baseBudget: number,
    targetTokens: number,
    nonReasoningProjection: number,
    codec: CodecOps,
    previouslyCompacted: Turn[],
    prior?: PlanSnapshot,
): Set<string> {
    const selected = new Set<string>()
    if (!Number.isFinite(baseBudget) || baseBudget <= 0) return selected
    const totalReasoning = compacted.reduce(
        (sum, turn) =>
            sum +
            turn.items.reduce(
                (partSum, item) =>
                    partSum +
                    (item.kind === "reasoning" ? countTokens(codec.transcriptLine(item)) : 0),
                0,
            ),
        0,
    )
    const growthCap = Math.max(0, targetTokens - nonReasoningProjection - baseBudget)
    const growth = Math.min(Math.max(0, Math.ceil(totalReasoning * 0.25) - baseBudget), growthCap)
    const budget = baseBudget + growth
    const previousKeys = new Set(
        previouslyCompacted.flatMap((turn) => turn.items.map((item) => item.key)),
    )
    const previouslyVisible = new Set(
        prior?.requiresCustomCompaction && !prior.reasoningSurvivesPrefix
            ? []
            : (prior?.preservedReasoningItemKeys ?? []),
    )
    let remaining = budget
    for (const turn of [...compacted].reverse()) {
        for (const item of [...turn.items].reverse()) {
            if (item.kind !== "reasoning") continue
            const size = countTokens(codec.transcriptLine(item))
            // Never resurrect reasoning that a prior plan already removed.
            if (previousKeys.has(item.key) && !previouslyVisible.has(item.key)) continue
            if (size > remaining) return selected
            selected.add(item.key)
            remaining -= size
        }
    }
    return selected
}

function markTargetMet(stages: BoundaryStageReport[]): void {
    const last = stages.at(-1)
    if (!last || last.status === "target-met") return
    last.status = "target-met"
}

function applyPrefixSummary(
    working: Turn[],
    rawTailStartIndex: number,
    transcriptRelativePath: string,
    prefixSummary?: string,
    compactedRangeHash?: string,
    conventions?: Conventions,
    preservedReasoningItemKeys: ReadonlySet<string> = new Set(),
    preservedToolCallIds: ReadonlySet<string> = new Set(),
    preservedPrefixTurnKeys: ReadonlySet<string> = new Set(),
    nativePrefixSource?: Turn[],
    protectedAssistantItemKeys: ReadonlySet<string> = new Set(),
): StageMutationResult & { prefixSummary: string } {
    if (rawTailStartIndex <= 0) {
        return {
            changedTurns: new Set<string>(),
            changedItems: 0,
            prefixSummary: prefixSummary ?? "",
        }
    }
    const compacted = working.slice(0, rawTailStartIndex)
    const summary = stripTranscriptReference(
        prefixSummary?.trim() ||
            formatPrefixSummary(compacted, conventions, working.slice(rawTailStartIndex)),
        transcriptRelativePath,
    )
    const summaryTurn = synthesizeSummaryTurn(
        compacted,
        summary,
        transcriptRelativePath,
        compactedRangeHash,
    )
    const preserved = preservedPrefixTurns(
        compacted,
        preservedReasoningItemKeys,
        preservedToolCallIds,
        preservedPrefixTurnKeys,
        protectedAssistantItemKeys,
    )
    const native = (nativePrefixSource ?? compacted)
        .slice(0, rawTailStartIndex)
        .filter((turn) => preservedPrefixTurnKeys.has(turn.key))
    const changedTurns = new Set(compacted.map((turn) => turn.key))
    const changedItems = compacted.reduce((sum, turn) => sum + turn.items.length, 0)
    const tail = working.slice(rawTailStartIndex)
    working.length = 0
    working.push(summaryTurn, ...preserved, ...native, ...tail)
    return { changedTurns, changedItems, prefixSummary: summary }
}

function preservedPrefixTurns(
    compacted: Turn[],
    reasoning: ReadonlySet<string>,
    tools: ReadonlySet<string>,
    native: ReadonlySet<string> = new Set(),
    assistantText: ReadonlySet<string> = new Set(),
): Turn[] {
    if (reasoning.size === 0 && tools.size === 0 && assistantText.size === 0) return []
    return compacted.flatMap((turn) => {
        if (native.has(turn.key)) return []
        const items = turn.items.filter(
            (item) =>
                (item.kind === "reasoning" && reasoning.has(item.key)) ||
                (item.kind === "tool" && tools.has(item.callId)) ||
                (item.kind === "text" && assistantText.has(item.key)),
        )
        return items.length ? [{ ...turn, items }] : []
    })
}

function synthesizeReferenceTurn(
    turns: Turn[],
    compacted: Turn[],
    ctx: StageContext,
    compactedRangeHash = rangeHash(compacted),
): Turn | null {
    if (!turns.some((turn) => turn.role === "user")) return null

    const hash = compactedRangeHash
    const first = compacted[0]?.key ?? "unknown"
    const last = compacted.at(-1)?.key ?? "unknown"
    const key = `better_compact_context_${hash}`
    const runIndex =
        ctx.archiveCatalogText === undefined
            ? assistantGroups(compacted, ctx.conventions).map((group) =>
                  formatReferenceRun(group.turns, ctx),
              )
            : []
    const latestTodoState = formatLatestTodoState(compacted, ctx)
    return {
        key,
        stamp: 0,
        role: "user",
        items: [
            {
                kind: "synthetic",
                key,
                text: [
                    "[Better Compact context pruning applied]",
                    `Older assistant/tool-heavy context was compactified for this request. Raw messages ${first} through ${last} are preserved in the reference transcript below.`,
                    "",
                    ...(ctx.archiveCatalogText === undefined
                        ? [
                              "## Compacted Assistant Runs",
                              ...(runIndex.length > 0 ? runIndex : ["- (none)"]),
                              "",
                              "## Reference Files",
                              `- "${ctx.transcriptRelativePath}"`,
                              "",
                              "If exact prior wording, raw tool output, or omitted implementation detail is needed, inspect the reference file instead of guessing.",
                          ]
                        : [
                              "Exact older details are available through better_compact_recall when needed; use it sparingly.",
                              ...(ctx.archiveCatalogText
                                  ? ["", "## Ready archives", ctx.archiveCatalogText]
                                  : []),
                          ]),
                    ...(latestTodoState ? ["", latestTodoState] : []),
                ].join("\n"),
            },
        ],
    }
}

function formatReferenceRun(group: Turn[], ctx: StageContext): string {
    const first = group[0]?.key ?? "unknown"
    const last = group.at(-1)?.key ?? first
    const idRange = first === last ? first : `${first} through ${last}`
    const touched = new Set<string>()
    for (const item of group.flatMap((turn) => turn.items)) {
        if (item.kind === "tool") {
            const details = ctx.conventions.tool?.(item)
            const name = referencePhrase(details?.name || "tool", 80)
            const target = primaryToolTarget(details?.input)?.normalized
            touched.add(target ? `${name} ${referencePhrase(target, 160)}` : name)
        }
        const note = ctx.conventions.itemNote?.(item)
        if (note) touched.add(referencePhrase(note, 240))
    }
    const topicItem = group
        .flatMap((turn) => turn.items)
        .find((item) => (item.kind === "text" || item.kind === "synthetic") && item.text.trim())
    const topic =
        topicItem && (topicItem.kind === "text" || topicItem.kind === "synthetic")
            ? referenceTopic(topicItem.text)
            : "(no assistant text)"
    const touchedText = formatTouchedReferences([...touched])
    return `- ${idRange} — ${touchedText} — ${topic}`
}

function formatLatestTodoState(compacted: Turn[], ctx: StageContext): string | null {
    if (!ctx.latestTodoCallId || !ctx.conventions.todo) return null
    for (const item of compacted.flatMap((turn) => turn.items)) {
        if (
            item.kind === "tool" &&
            item.callId === ctx.latestTodoCallId &&
            ctx.conventions.todo.isTodoItem(item)
        ) {
            return `Latest todo state preserved: ${oneLineReference(ctx.conventions.todo.format(item))}`
        }
    }
    return null
}

function referenceTopic(value: string): string {
    const firstLine = value
        .split(/\r\n|\n|\r/)
        .map((line) => line.trim())
        .find(Boolean)
    if (!firstLine) return "(no assistant text)"
    const firstSentence = firstLine.match(/^.*?[.!?](?=\s|$)/)?.[0] ?? firstLine
    return referencePhrase(firstSentence, 160)
}

function formatTouchedReferences(values: string[]): string {
    if (values.length === 0) return "(no files/tools)"
    const selected: string[] = []
    let length = 0
    for (const value of values) {
        const separatorLength = selected.length > 0 ? 2 : 0
        if (selected.length > 0 && length + separatorLength + value.length > 240) {
            selected.push("…")
            break
        }
        const entry = referencePhrase(value, 240)
        selected.push(entry)
        length += separatorLength + entry.length
    }
    return selected.join(", ")
}

function referencePhrase(value: string, maxChars: number): string {
    const normalized = oneLineReference(value)
    if (normalized.length <= maxChars) return normalized
    const prefix = normalized.slice(0, Math.max(0, maxChars - 1))
    const wordBoundary = prefix.lastIndexOf(" ")
    return `${(wordBoundary > maxChars / 2 ? prefix.slice(0, wordBoundary) : prefix).trimEnd()}…`
}

function oneLineReference(value: string): string {
    return value.replace(/\s+/g, " ").trim()
}

function synthesizeSummaryTurn(
    compacted: Turn[],
    summary: string,
    transcriptRelativePath: string,
    compactedRangeHash = rangeHash(compacted),
    archiveCatalogText?: string,
): Turn {
    const key = `better_compact_summary_${compactedRangeHash}`
    const referenceBlock = `## Reference Files\n- "${transcriptRelativePath}"`
    const normalizedSummary = stripTranscriptReference(summary, transcriptRelativePath)
    return {
        key,
        stamp: 0,
        role: "user",
        items: [
            {
                kind: "synthetic",
                key,
                text:
                    archiveCatalogText === undefined
                        ? ["[Context Summary]", normalizedSummary, "", referenceBlock].join("\n")
                        : [
                              "[Context Summary]",
                              normalizedSummary,
                              "",
                              "Exact older details are available through better_compact_recall when needed; use it sparingly.",
                              ...(archiveCatalogText
                                  ? ["", "## Ready archives", archiveCatalogText]
                                  : []),
                          ].join("\n"),
            },
        ],
    }
}

function stripTranscriptReference(summary: string, transcriptRelativePath: string): string {
    const trimmed = summary.trim()
    const referenceBlock = `## Reference Files\n- "${transcriptRelativePath}"`
    return trimmed.endsWith(referenceBlock)
        ? trimmed.slice(0, -referenceBlock.length).trimEnd()
        : trimmed
}

function cloneTurns(turns: Turn[]): Turn[] {
    return turns.map((turn) => ({ ...turn, items: [...turn.items] }))
}
