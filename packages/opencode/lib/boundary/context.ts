import { existsSync } from "node:fs"
import { join } from "node:path"
import {
    buildPlan,
    findRawTailStartIndex,
    rangeHash,
    replayPlanSnapshot,
    toPlanSnapshot,
    transformTurns,
    writeTranscript,
    type BoundaryContextOptions,
    type BoundaryContextPlan,
    type ReplayOptions,
} from "@better-compact/core"
import type { Logger } from "../logger"
import { openCodeCodec, openCodeSpec, sessionKeyOf } from "../codec"
import {
    loadPersistedBoundaryPlans,
    type BoundaryPlanSnapshot,
    type SessionState,
    type WithParts,
} from "../state"
import { boundaryRangeHash, boundarySnapshotHash, PREFIX_FINGERPRINT_VERSION } from "./fingerprint"
import { isSyndicatePluginInjection } from "../messages/injection"
import { createTranscriptStore, transcriptCitablePath } from "./transcripts"
import { inheritArchiveCatalog } from "./archive-catalog"

export type {
    BoundaryContextOptions,
    BoundaryContextPlan,
    BoundaryStageName,
    BoundaryStageReport,
    BoundarySummaryJob,
    BoundaryTranscriptArtifact,
} from "@better-compact/core"

// Two real user turns normally frame the current task. In an agentic loop,
// that span may exceed the entire target: keep the newest user turn raw and
// move older complete turns into the citable transcript instead.
export function adaptiveTailUserTurns(
    messages: WithParts[],
    contextLimit: number,
    targetPercent: number,
    targetTokens?: number | null,
): 1 | 2 {
    const turns = openCodeCodec.encode(messages)
    const start = findRawTailStartIndex(turns, 3, 2)
    const twoUserTail = openCodeCodec.estimateTurns(turns.slice(start))
    const target = targetTokens ?? Math.floor((contextLimit * targetPercent) / 100)
    return twoUserTail > target ? 1 : 2
}

/** Move complete older assistant/tool turns after a sparse user turn only when
 * their full archived handoff actually makes the outgoing plan smaller. */
export function efficientAgenticTailBudget(
    messages: WithParts[],
    options: BoundaryContextOptions,
): { floor: number; ceiling: number } | undefined {
    const target =
        options.targetTokens ??
        Math.floor((options.contextLimit ?? 0) * (options.targetRatio ?? 0.3))
    if (target <= 0) return undefined
    const turns = openCodeCodec.encode(messages)
    const lastUser = turns.findLastIndex((turn) => turn.role === "user" && !turn.ephemeral)
    if (lastUser < 0 || openCodeCodec.estimateTurns(turns.slice(lastUser)) <= target)
        return undefined
    const floor = Math.min(4_096, Math.max(1, Math.floor(target * 0.25)))
    const budget = { floor, ceiling: Math.max(floor, Math.floor(target * 0.6)) }
    const baseline = buildBoundaryContextPlan(messages, { ...options, force: true })
    const candidate = buildBoundaryContextPlan(messages, {
        ...options,
        force: true,
        tailBudgetTokens: budget,
    })
    if (!candidate || (baseline && candidate.afterPruneTokens >= baseline.afterPruneTokens))
        return undefined
    return budget
}

export function buildBoundaryContextPlan(
    messages: WithParts[],
    options: BoundaryContextOptions = {},
): BoundaryContextPlan | null {
    return buildPlan(
        openCodeCodec.encode(messages),
        {
            ...options,
            preservePrefixBudgets: true,
            recentAssistantOutputs: options.recentAssistantOutputs ?? 5,
            sessionKey: sessionKeyOf(messages),
            citablePath: transcriptCitablePath,
        },
        openCodeSpec,
    )
}

export function applyBoundaryContextPlan(messages: WithParts[], plan: BoundaryContextPlan): void {
    const transformed = transformTurns(
        openCodeCodec.encode(messages),
        plan.rawTailStartIndex,
        plan,
        openCodeSpec,
    )
    replaceMessages(messages, openCodeCodec.decode(transformed, messages))
}

export function applyBoundaryPlanSnapshot(
    messages: WithParts[],
    snapshot: BoundaryPlanSnapshot,
    options: ReplayOptions = {},
): boolean {
    if (isAppliedBoundaryPlanSnapshot(messages, snapshot)) return true
    if (
        snapshot.prefixFingerprint &&
        snapshot.compactedMessageCount !== undefined &&
        (snapshot.compactedMessageCount > messages.length ||
            boundarySnapshotHash(messages.slice(0, snapshot.compactedMessageCount), snapshot) !==
                snapshot.prefixFingerprint)
    )
        return false
    const replayed = replayPlanSnapshot(
        openCodeCodec.encode(messages),
        snapshot,
        openCodeSpec,
        options,
    )
    if (!replayed) return false
    replaceMessages(messages, openCodeCodec.decode(replayed, messages))
    return true
}

/** Upgrade a validated legacy plan in place without rebuilding the provider prefix. */
export function upgradeBoundaryPlanFingerprint(
    messages: WithParts[],
    snapshot: BoundaryPlanSnapshot,
): BoundaryPlanSnapshot | null {
    if (
        snapshot.prefixFingerprintVersion === 2 ||
        !snapshot.prefixFingerprint ||
        !snapshot.compactedMessageCount ||
        snapshot.compactedMessageCount > messages.length
    )
        return null
    const prefix = messages.slice(0, snapshot.compactedMessageCount)
    if (boundarySnapshotHash(prefix, snapshot) !== snapshot.prefixFingerprint) return null
    return {
        ...snapshot,
        prefixFingerprintVersion: PREFIX_FINGERPRINT_VERSION,
        prefixFingerprint: boundaryRangeHash(prefix),
    }
}

/** A second invocation on the same outgoing array must not compact our own handoff. */
export function isAppliedBoundaryPlanSnapshot(
    messages: WithParts[],
    snapshot: BoundaryPlanSnapshot,
): boolean {
    const name = `better_compact_${snapshot.requiresCustomCompaction ? "summary" : "context"}_${snapshot.rangeHash}`
    const expectedText = snapshot.requiresCustomCompaction
        ? "[Context Summary]"
        : "[Better Compact context pruning applied]"
    return messages.some(
        (message) =>
            message.info.id === `msg_${name}` &&
            message.info.sessionID === snapshot.sessionId &&
            message.info.role === "user" &&
            message.parts.some(
                (part) =>
                    part.type === "text" &&
                    part.id === `prt_${name}` &&
                    part.synthetic === true &&
                    part.text.startsWith(expectedText),
            ),
    )
}

// Core snapshots carry the id-based rangeHash; the OpenCode layer adds a
// content-addressed prefix identity so forked sessions can inherit the plan.
export function toBoundaryPlanSnapshot(
    plan: BoundaryContextPlan,
    messages: WithParts[],
): BoundaryPlanSnapshot {
    const snapshot = {
        ...toPlanSnapshot(plan),
        ...(messages.some(isSyndicatePluginInjection)
            ? { pluginInjectionPruning: true as const }
            : {}),
    }
    if (plan.rawTailItemBoundary !== undefined) return snapshot
    const prefix = messages.slice(0, plan.rawTailStartIndex)
    return {
        ...snapshot,
        prefixFingerprintVersion: PREFIX_FINGERPRINT_VERSION,
        prefixFingerprint: boundaryRangeHash(prefix),
        compactedMessageCount: prefix.length,
    }
}

export function storeBoundaryPlan(
    state: SessionState,
    plan: BoundaryContextPlan,
    messages: WithParts[],
): void {
    state.boundary.activePlan = toBoundaryPlanSnapshot(plan, messages)
}

// A forked session copies message content but mints new ids. Match this
// session's prefix against persisted plans by content fingerprint and rebase
// the winning snapshot onto the fork's ids so core replay validation holds.
export async function findMatchingBoundaryPlan(
    sessionId: string,
    messages: WithParts[],
    directory: string,
    logger: Logger,
    resolveTitle?: (sessionId: string) => Promise<string | undefined>,
): Promise<BoundaryPlanSnapshot | null> {
    // OpenCode's session.fork clones message content but does not set parentID.
    // The host does mark forks with a derived title. Require that marker plus
    // content identity so an independent session with a common short prefix
    // cannot acquire the first matching owner's private archive link.
    if (!resolveTitle) return null
    const forkTitle = await resolveTitle(sessionId).catch(() => undefined)
    if (!forkTitle || !/ \(fork #\d+\)$/.test(forkTitle)) return null
    const plans = await loadPersistedBoundaryPlans(logger)
    const hashes = new Map<string, string>()
    for (const plan of plans) {
        if (plan.rawTailItemBoundary !== undefined) continue
        const compactedCount = plan.compactedMessageCount
        if (!plan.prefixFingerprint || !compactedCount || compactedCount >= messages.length)
            continue
        const key = `${compactedCount}:${plan.prefixFingerprintVersion ?? 1}`
        const hash =
            hashes.get(key) ?? boundarySnapshotHash(messages.slice(0, compactedCount), plan)
        hashes.set(key, hash)
        if (hash !== plan.prefixFingerprint) continue
        const ownerTitle = await resolveTitle(plan.sessionId).catch(() => undefined)
        if (!ownerTitle || forkTitleOf(ownerTitle) !== forkTitle) continue
        if (!existsSync(join(directory, plan.transcriptRelativePath))) continue
        await inheritArchiveCatalog(directory, sessionId, plan.sessionId, hash)
        return {
            ...plan,
            sessionId,
            rawTailStartMessageId: messages[compactedCount].info.id,
            rangeHash: rangeHash(openCodeCodec.encode(messages.slice(0, compactedCount))),
        }
    }
    return null
}

function forkTitleOf(title: string): string {
    const previous = title.match(/^(.+) \(fork #(\d+)\)$/)
    return previous ? `${previous[1]} (fork #${Number(previous[2]) + 1})` : `${title} (fork #1)`
}

export async function writeBoundaryTranscript(
    directory: string,
    plan: BoundaryContextPlan,
    logger: Logger,
): Promise<void> {
    await writeTranscript(plan, {
        transcripts: createTranscriptStore(directory),
        logger,
        codec: openCodeCodec,
    })
}

function replaceMessages(messages: WithParts[], next: WithParts[]): void {
    messages.length = 0
    messages.push(...next)
}
